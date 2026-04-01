import os
import re
import socket
import threading
import time

import requests
import uvicorn
from fastapi import FastAPI
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, generate_latest
from starlette.responses import Response


INSTANCE = socket.gethostname()
PORT = int(os.getenv("ORCH_PORT", "9702"))
POLL_INTERVAL_SECONDS = float(os.getenv("ORCH_POLL_INTERVAL_SECONDS", "5"))
SERVING_STATUS_URL = os.getenv("SERVING_STATUS_URL", "http://gpu_serving_sim:9700/status")
LEARNING_CONTROL_URL = os.getenv("LEARNING_CONTROL_URL", "http://gpu_learning_sim:9701/control")
LEARNING_STATUS_URL = os.getenv("LEARNING_STATUS_URL", "http://gpu_learning_sim:9701/status")
DCGM_METRICS_URL = os.getenv("DCGM_METRICS_URL", "http://dcgm_exporter:9400/metrics")
FALLBACK_METRICS_URL = os.getenv("FALLBACK_METRICS_URL", "http://gpu_metrics_exporter:9500/metrics")
ACTIVE_PAUSE_THRESHOLD = int(os.getenv("ORCH_SERVING_ACTIVE_PAUSE_THRESHOLD", "1"))
QUEUE_PAUSE_THRESHOLD = int(os.getenv("ORCH_SERVING_QUEUE_PAUSE_THRESHOLD", "1"))
LATENCY_WARN_SECONDS = float(os.getenv("ORCH_SERVING_LATENCY_WARN_SECONDS", "1.2"))
AGGRESSIVE_MAX_UTIL = float(os.getenv("ORCH_GPU_AGGRESSIVE_MAX_UTIL", "45"))
BALANCED_MAX_UTIL = float(os.getenv("ORCH_GPU_BALANCED_MAX_UTIL", "70"))
GPU_MEMORY_MAX_PERCENT = float(os.getenv("ORCH_GPU_MEMORY_MAX_PERCENT", "80"))

CURRENT_MODE = "balanced"
CURRENT_REASON = "startup"
LAST_ERROR = ""
LAST_GPU_UTIL = 0.0
LAST_GPU_MEMORY = 0.0
LAST_SERVING_ACTIVE = 0
LAST_SERVING_QUEUE = 0
LAST_SERVING_LATENCY = 0.0

MODE_GAUGE = Gauge("gpu_orchestrator_mode", "Current mode selected by the orchestrator", ["instance", "mode"])
MODE_CHANGES = Counter("gpu_orchestrator_mode_changes_total", "Mode changes issued by the orchestrator", ["instance", "mode"])
GPU_UTIL_GAUGE = Gauge("gpu_orchestrator_observed_gpu_utilization_percent", "Observed GPU utilization", ["instance"])
GPU_MEMORY_GAUGE = Gauge("gpu_orchestrator_observed_gpu_memory_percent", "Observed GPU memory usage percent", ["instance"])
SERVING_ACTIVE_GAUGE = Gauge("gpu_orchestrator_serving_active_requests", "Observed active requests in serving lane", ["instance"])
SERVING_QUEUE_GAUGE = Gauge("gpu_orchestrator_serving_queue_depth", "Observed queue depth in serving lane", ["instance"])
SERVING_LATENCY_GAUGE = Gauge("gpu_orchestrator_serving_avg_request_seconds", "Observed serving average latency", ["instance"])
DECISIONS_TOTAL = Counter("gpu_orchestrator_decisions_total", "Total scheduler decisions", ["instance", "mode", "reason"])
ERRORS_TOTAL = Counter("gpu_orchestrator_errors_total", "Errors observed by the orchestrator", ["instance", "type"])

app = FastAPI(title="GPU Orchestrator", version="1.0.0")


def set_mode(mode: str, reason: str) -> None:
    global CURRENT_MODE, CURRENT_REASON
    if mode == CURRENT_MODE and reason == CURRENT_REASON:
        return
    response = requests.post(LEARNING_CONTROL_URL, json={"mode": mode, "reason": reason}, timeout=5)
    response.raise_for_status()
    CURRENT_MODE = mode
    CURRENT_REASON = reason
    MODE_CHANGES.labels(instance=INSTANCE, mode=mode).inc()
    DECISIONS_TOTAL.labels(instance=INSTANCE, mode=mode, reason=reason).inc()
    for known_mode in ("paused", "trickle", "balanced", "aggressive"):
        MODE_GAUGE.labels(instance=INSTANCE, mode=known_mode).set(1 if known_mode == mode else 0)


def fetch_json(url: str) -> dict:
    response = requests.get(url, timeout=5)
    response.raise_for_status()
    return response.json()


def parse_metric(text: str, pattern: str) -> float:
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        return 0.0
    return float(match.group(1))


def read_gpu_metrics() -> tuple[float, float]:
    try:
        dcgm_text = requests.get(DCGM_METRICS_URL, timeout=5).text
        util = parse_metric(dcgm_text, r"^DCGM_FI_DEV_GPU_UTIL\{[^\n]*\}\s+([0-9.]+)$")
        used = parse_metric(dcgm_text, r"^DCGM_FI_DEV_FB_USED\{[^\n]*\}\s+([0-9.]+)$")
        free = parse_metric(dcgm_text, r"^DCGM_FI_DEV_FB_FREE\{[^\n]*\}\s+([0-9.]+)$")
        total = used + free
        memory_percent = (used / total) * 100 if total > 0 else 0.0
        return util, memory_percent
    except Exception:
        fallback_text = requests.get(FALLBACK_METRICS_URL, timeout=5).text
        util = parse_metric(fallback_text, r"^nvidia_smi_gpu_utilization_percent\{[^\n]*\}\s+([0-9.]+)$")
        memory_percent = parse_metric(fallback_text, r"^nvidia_smi_gpu_memory_used_percent\{[^\n]*\}\s+([0-9.]+)$")
        return util, memory_percent


def decide_mode(serving: dict, gpu_util: float, gpu_memory: float) -> tuple[str, str]:
    active = int(serving.get("active_requests", 0))
    queue = int(serving.get("queue_depth", 0))
    avg_latency = float(serving.get("avg_request_seconds", 0.0))
    if active >= ACTIVE_PAUSE_THRESHOLD or queue >= QUEUE_PAUSE_THRESHOLD:
        return "paused", "serving_hot"
    if avg_latency >= LATENCY_WARN_SECONDS:
        return "trickle", "serving_latency_guard"
    if gpu_memory >= GPU_MEMORY_MAX_PERCENT:
        return "trickle", "gpu_memory_guard"
    if gpu_util <= AGGRESSIVE_MAX_UTIL:
        return "aggressive", "gpu_has_headroom"
    if gpu_util <= BALANCED_MAX_UTIL:
        return "balanced", "gpu_balanced"
    return "trickle", "gpu_busy"


def orchestrator_loop() -> None:
    global LAST_ERROR, LAST_GPU_UTIL, LAST_GPU_MEMORY, LAST_SERVING_ACTIVE, LAST_SERVING_QUEUE, LAST_SERVING_LATENCY
    while True:
        try:
            serving = fetch_json(SERVING_STATUS_URL)
            gpu_util, gpu_memory = read_gpu_metrics()
            mode, reason = decide_mode(serving, gpu_util, gpu_memory)
            LAST_GPU_UTIL = gpu_util
            LAST_GPU_MEMORY = gpu_memory
            LAST_SERVING_ACTIVE = int(serving.get("active_requests", 0))
            LAST_SERVING_QUEUE = int(serving.get("queue_depth", 0))
            LAST_SERVING_LATENCY = float(serving.get("avg_request_seconds", 0.0))
            GPU_UTIL_GAUGE.labels(instance=INSTANCE).set(gpu_util)
            GPU_MEMORY_GAUGE.labels(instance=INSTANCE).set(gpu_memory)
            SERVING_ACTIVE_GAUGE.labels(instance=INSTANCE).set(LAST_SERVING_ACTIVE)
            SERVING_QUEUE_GAUGE.labels(instance=INSTANCE).set(LAST_SERVING_QUEUE)
            SERVING_LATENCY_GAUGE.labels(instance=INSTANCE).set(LAST_SERVING_LATENCY)
            set_mode(mode, reason)
            LAST_ERROR = ""
        except Exception as exc:
            LAST_ERROR = str(exc)
            ERRORS_TOTAL.labels(instance=INSTANCE, type="loop").inc()
        time.sleep(POLL_INTERVAL_SECONDS)


@app.on_event("startup")
def startup() -> None:
    for known_mode in ("paused", "trickle", "balanced", "aggressive"):
        MODE_GAUGE.labels(instance=INSTANCE, mode=known_mode).set(1 if known_mode == CURRENT_MODE else 0)
    thread = threading.Thread(target=orchestrator_loop, daemon=True)
    thread.start()


@app.get("/healthz")
def healthz() -> dict[str, object]:
    return {"ok": True, "mode": CURRENT_MODE, "reason": CURRENT_REASON}


@app.get("/status")
def status() -> dict[str, object]:
    learning = {}
    try:
        learning = fetch_json(LEARNING_STATUS_URL)
    except Exception as exc:
        learning = {"error": str(exc)}
    return {
        "instance": INSTANCE,
        "mode": CURRENT_MODE,
        "reason": CURRENT_REASON,
        "gpu_utilization_percent": LAST_GPU_UTIL,
        "gpu_memory_percent": LAST_GPU_MEMORY,
        "serving_active_requests": LAST_SERVING_ACTIVE,
        "serving_queue_depth": LAST_SERVING_QUEUE,
        "serving_avg_request_seconds": LAST_SERVING_LATENCY,
        "learning_status": learning,
        "last_error": LAST_ERROR,
    }


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
