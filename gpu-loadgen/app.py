import os
import socket
import threading
import time
from typing import Optional

import requests
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from starlette.responses import Response


INSTANCE = socket.gethostname()
PORT = int(os.getenv("LOADGEN_PORT", "9703"))
TARGET_URL = os.getenv("LOADGEN_TARGET_URL", "http://gpu_serving_sim:9700/infer")
DEFAULT_MATRIX_SIZE = int(os.getenv("LOADGEN_DEFAULT_MATRIX_SIZE", "2048"))
DEFAULT_STEPS = int(os.getenv("LOADGEN_DEFAULT_STEPS", "2"))
DEFAULT_BATCHES = int(os.getenv("LOADGEN_DEFAULT_BATCHES", "1"))
DEFAULT_PRECISION = os.getenv("LOADGEN_DEFAULT_PRECISION", "float16")
DEFAULT_WORKERS = int(os.getenv("LOADGEN_DEFAULT_WORKERS", "0"))
DEFAULT_INTERVAL_SECONDS = float(os.getenv("LOADGEN_DEFAULT_INTERVAL_SECONDS", "0.4"))
REQUEST_TIMEOUT_SECONDS = float(os.getenv("LOADGEN_REQUEST_TIMEOUT_SECONDS", "90"))

STATE_LOCK = threading.Lock()
WORKER_THREADS: list[threading.Thread] = []
RUN_GENERATION = False
CURRENT_PROFILE = "idle"
CURRENT_WORKERS = 0
CURRENT_INTERVAL_SECONDS = DEFAULT_INTERVAL_SECONDS
CURRENT_PAYLOAD = {
    "matrix_size": DEFAULT_MATRIX_SIZE,
    "steps": DEFAULT_STEPS,
    "batches": DEFAULT_BATCHES,
    "precision": DEFAULT_PRECISION,
}
LAST_ERROR = ""

REQUESTS_TOTAL = Counter("gpu_loadgen_requests_total", "Requests emitted by the load generator", ["instance", "status", "profile"])
REQUEST_DURATION = Histogram(
    "gpu_loadgen_request_seconds",
    "Load generator request latency",
    ["instance", "profile"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32),
)
RUNNING_GAUGE = Gauge("gpu_loadgen_running", "Whether the load generator is currently running", ["instance"])
WORKERS_GAUGE = Gauge("gpu_loadgen_workers", "Number of worker threads emitting traffic", ["instance"])
INTERVAL_GAUGE = Gauge("gpu_loadgen_interval_seconds", "Sleep interval between requests", ["instance"])
INFLIGHT_GAUGE = Gauge("gpu_loadgen_inflight_requests", "Number of in-flight requests", ["instance"])
PROFILE_GAUGE = Gauge("gpu_loadgen_profile", "Current named profile", ["instance", "profile"])
LAST_LATENCY_GAUGE = Gauge("gpu_loadgen_last_request_seconds", "Latency of the last load generator request", ["instance"])

app = FastAPI(title="GPU Serving Load Generator", version="1.0.0")


class StartRequest(BaseModel):
    workers: Optional[int] = None
    interval_seconds: Optional[float] = None
    matrix_size: Optional[int] = None
    steps: Optional[int] = None
    batches: Optional[int] = None
    precision: Optional[str] = None
    profile: Optional[str] = None


def set_profile(name: str) -> None:
    for known in ("idle", "light", "medium", "heavy", "custom"):
        PROFILE_GAUGE.labels(instance=INSTANCE, profile=known).set(1 if known == name else 0)


def snapshot() -> dict[str, object]:
    return {
        "running": RUN_GENERATION,
        "profile": CURRENT_PROFILE,
        "workers": CURRENT_WORKERS,
        "interval_seconds": CURRENT_INTERVAL_SECONDS,
        "payload": dict(CURRENT_PAYLOAD),
        "last_error": LAST_ERROR,
    }


def locked_snapshot() -> dict[str, object]:
    with STATE_LOCK:
        return snapshot()


def apply_state(profile: str, workers: int, interval_seconds: float, payload: dict[str, object]) -> None:
    global RUN_GENERATION, CURRENT_PROFILE, CURRENT_WORKERS, CURRENT_INTERVAL_SECONDS, CURRENT_PAYLOAD
    RUN_GENERATION = workers > 0
    CURRENT_PROFILE = profile
    CURRENT_WORKERS = workers
    CURRENT_INTERVAL_SECONDS = interval_seconds
    CURRENT_PAYLOAD = payload
    RUNNING_GAUGE.labels(instance=INSTANCE).set(1 if RUN_GENERATION else 0)
    WORKERS_GAUGE.labels(instance=INSTANCE).set(CURRENT_WORKERS)
    INTERVAL_GAUGE.labels(instance=INSTANCE).set(CURRENT_INTERVAL_SECONDS)
    set_profile(profile)


def worker_loop(worker_id: int) -> None:
    global LAST_ERROR
    session = requests.Session()
    while True:
        with STATE_LOCK:
            should_run = RUN_GENERATION and worker_id < CURRENT_WORKERS
            payload = dict(CURRENT_PAYLOAD)
            interval_seconds = CURRENT_INTERVAL_SECONDS
            profile = CURRENT_PROFILE
        if not should_run:
            time.sleep(0.2)
            continue

        started = time.perf_counter()
        INFLIGHT_GAUGE.labels(instance=INSTANCE).inc()
        try:
            response = session.post(
                TARGET_URL,
                json=payload,
                timeout=REQUEST_TIMEOUT_SECONDS,
                headers={
                    "X-Sim-Source": "gpu-loadgen",
                    "X-Sim-Profile": profile,
                },
            )
            duration = time.perf_counter() - started
            LAST_LATENCY_GAUGE.labels(instance=INSTANCE).set(duration)
            REQUEST_DURATION.labels(instance=INSTANCE, profile=profile).observe(duration)
            if response.ok:
                REQUESTS_TOTAL.labels(instance=INSTANCE, status="success", profile=profile).inc()
                LAST_ERROR = ""
            else:
                REQUESTS_TOTAL.labels(instance=INSTANCE, status="error", profile=profile).inc()
                LAST_ERROR = f"HTTP {response.status_code}: {response.text[:200]}"
        except Exception as exc:
            REQUESTS_TOTAL.labels(instance=INSTANCE, status="error", profile=profile).inc()
            LAST_ERROR = str(exc)
        finally:
            INFLIGHT_GAUGE.labels(instance=INSTANCE).dec()
        time.sleep(interval_seconds)


def ensure_threads(target_workers: int) -> None:
    while len(WORKER_THREADS) < target_workers:
        worker_id = len(WORKER_THREADS)
        thread = threading.Thread(target=worker_loop, args=(worker_id,), daemon=True)
        thread.start()
        WORKER_THREADS.append(thread)


def start_profile(payload: StartRequest) -> dict[str, object]:
    workers = max(0, int(payload.workers if payload.workers is not None else DEFAULT_WORKERS))
    interval_seconds = max(0.0, float(payload.interval_seconds if payload.interval_seconds is not None else DEFAULT_INTERVAL_SECONDS))
    matrix_size = int(payload.matrix_size if payload.matrix_size is not None else DEFAULT_MATRIX_SIZE)
    steps = int(payload.steps if payload.steps is not None else DEFAULT_STEPS)
    batches = int(payload.batches if payload.batches is not None else DEFAULT_BATCHES)
    precision = payload.precision or DEFAULT_PRECISION
    profile = payload.profile or "custom"

    ensure_threads(workers)
    with STATE_LOCK:
        apply_state(
            profile=profile,
            workers=workers,
            interval_seconds=interval_seconds,
            payload={
                "matrix_size": matrix_size,
                "steps": steps,
                "batches": batches,
                "precision": precision,
            },
        )
        return snapshot()


@app.on_event("startup")
def startup() -> None:
    set_profile("idle")
    RUNNING_GAUGE.labels(instance=INSTANCE).set(0)
    WORKERS_GAUGE.labels(instance=INSTANCE).set(0)
    INTERVAL_GAUGE.labels(instance=INSTANCE).set(DEFAULT_INTERVAL_SECONDS)
    INFLIGHT_GAUGE.labels(instance=INSTANCE).set(0)
    LAST_LATENCY_GAUGE.labels(instance=INSTANCE).set(0)
    if DEFAULT_WORKERS > 0:
        ensure_threads(DEFAULT_WORKERS)
        with STATE_LOCK:
            apply_state("custom", DEFAULT_WORKERS, DEFAULT_INTERVAL_SECONDS, dict(CURRENT_PAYLOAD))


@app.get("/healthz")
def healthz() -> dict[str, object]:
    return {"ok": True, **locked_snapshot()}


@app.get("/status")
def status() -> dict[str, object]:
    return {"instance": INSTANCE, "target_url": TARGET_URL, **locked_snapshot()}


@app.post("/start")
def start(request: StartRequest) -> dict[str, object]:
    return start_profile(request)


@app.post("/preset/light")
def preset_light() -> dict[str, object]:
    return start_profile(StartRequest(profile="light", workers=1, interval_seconds=1.1, matrix_size=2048, steps=2, batches=1))


@app.post("/preset/medium")
def preset_medium() -> dict[str, object]:
    return start_profile(StartRequest(profile="medium", workers=2, interval_seconds=0.45, matrix_size=2560, steps=2, batches=2))


@app.post("/preset/heavy")
def preset_heavy() -> dict[str, object]:
    return start_profile(StartRequest(profile="heavy", workers=3, interval_seconds=0.15, matrix_size=3072, steps=3, batches=2))


@app.post("/stop")
def stop() -> dict[str, object]:
    with STATE_LOCK:
        apply_state("idle", 0, DEFAULT_INTERVAL_SECONDS, dict(CURRENT_PAYLOAD))
        return snapshot()


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
