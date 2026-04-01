import os
import socket
import threading
import time

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from starlette.responses import Response


INSTANCE = socket.gethostname()
PORT = int(os.getenv("LEARNING_PORT", "9701"))
PRECISION = os.getenv("LEARNING_PRECISION", "float16").lower()
MODE_CONFIG = {
    "aggressive": {
        "matrix_size": int(os.getenv("LEARNING_AGGRESSIVE_MATRIX_SIZE", "6144")),
        "batches": int(os.getenv("LEARNING_AGGRESSIVE_BATCHES", "5")),
        "sleep_seconds": float(os.getenv("LEARNING_AGGRESSIVE_SLEEP", "0.15")),
    },
    "balanced": {
        "matrix_size": int(os.getenv("LEARNING_BALANCED_MATRIX_SIZE", "4096")),
        "batches": int(os.getenv("LEARNING_BALANCED_BATCHES", "3")),
        "sleep_seconds": float(os.getenv("LEARNING_BALANCED_SLEEP", "0.75")),
    },
    "trickle": {
        "matrix_size": int(os.getenv("LEARNING_TRICKLE_MATRIX_SIZE", "3072")),
        "batches": int(os.getenv("LEARNING_TRICKLE_BATCHES", "1")),
        "sleep_seconds": float(os.getenv("LEARNING_TRICKLE_SLEEP", "1.50")),
    },
    "paused": {"matrix_size": 0, "batches": 0, "sleep_seconds": 1.0},
}
CURRENT_MODE = os.getenv("LEARNING_MODE", "balanced")
CURRENT_REASON = "startup"
CURRENT_LOCK = threading.Lock()
LAST_DURATION = 0.0


MODE_GAUGE = Gauge("gpu_learning_mode", "Current learning lane mode", ["instance", "mode"])
CUDA_GAUGE = Gauge("gpu_learning_cuda_available", "CUDA visibility for learning lane", ["instance"])
ACTIVE_GAUGE = Gauge("gpu_learning_active", "Whether learning lane is actively consuming the GPU", ["instance", "gpu", "uuid"])
LAST_DURATION_GAUGE = Gauge("gpu_learning_last_iteration_seconds", "Duration of the last learning iteration", ["instance", "gpu", "uuid", "mode"])
ALLOCATED_MEMORY = Gauge("gpu_learning_allocated_memory_bytes", "Torch allocated memory for the learning lane", ["instance", "gpu", "uuid"])
RESERVED_MEMORY = Gauge("gpu_learning_reserved_memory_bytes", "Torch reserved memory for the learning lane", ["instance", "gpu", "uuid"])
MODE_MATRIX = Gauge("gpu_learning_target_matrix_size", "Matrix size selected for the current mode", ["instance", "mode"])
MODE_BATCHES = Gauge("gpu_learning_target_batches", "Batch count selected for the current mode", ["instance", "mode"])
MODE_SLEEP = Gauge("gpu_learning_target_sleep_seconds", "Sleep interval selected for the current mode", ["instance", "mode"])
ITERATIONS = Counter("gpu_learning_iterations_total", "Completed learning iterations", ["instance", "gpu", "uuid", "mode"])
CONTROL_CHANGES = Counter("gpu_learning_control_changes_total", "Number of orchestrator or manual control changes", ["instance", "mode"])
ITERATION_DURATION = Histogram(
    "gpu_learning_iteration_seconds",
    "Learning lane iteration duration",
    ["instance", "mode"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16),
)

app = FastAPI(title="GPU Learning Simulation", version="1.0.0")


class ControlRequest(BaseModel):
    mode: str
    reason: str = "manual"


def dtype() -> torch.dtype:
    if PRECISION == "float32":
        return torch.float32
    if PRECISION == "bfloat16":
        return torch.bfloat16
    return torch.float16


def device_labels(device_index: int) -> dict[str, str]:
    props = torch.cuda.get_device_properties(device_index)
    uuid = getattr(props, "uuid", f"gpu-{device_index}")
    if uuid and not str(uuid).startswith("GPU-"):
        uuid = f"GPU-{uuid}"
    return {"instance": INSTANCE, "gpu": str(device_index), "uuid": uuid}


def set_mode(mode: str, reason: str) -> None:
    global CURRENT_MODE, CURRENT_REASON
    if mode not in MODE_CONFIG:
        raise ValueError(f"unsupported mode: {mode}")
    with CURRENT_LOCK:
        CURRENT_MODE = mode
        CURRENT_REASON = reason
        CONTROL_CHANGES.labels(instance=INSTANCE, mode=mode).inc()
        for known_mode, cfg in MODE_CONFIG.items():
            MODE_GAUGE.labels(instance=INSTANCE, mode=known_mode).set(1 if known_mode == mode else 0)
            MODE_MATRIX.labels(instance=INSTANCE, mode=known_mode).set(cfg["matrix_size"])
            MODE_BATCHES.labels(instance=INSTANCE, mode=known_mode).set(cfg["batches"])
            MODE_SLEEP.labels(instance=INSTANCE, mode=known_mode).set(cfg["sleep_seconds"])


def current_mode() -> tuple[str, str]:
    with CURRENT_LOCK:
        return CURRENT_MODE, CURRENT_REASON


def learning_loop() -> None:
    global LAST_DURATION
    set_mode(CURRENT_MODE, CURRENT_REASON)
    while True:
        available = torch.cuda.is_available()
        CUDA_GAUGE.labels(instance=INSTANCE).set(1 if available else 0)
        mode, _ = current_mode()
        if not available or mode == "paused":
            time.sleep(MODE_CONFIG["paused"]["sleep_seconds"])
            continue

        config = MODE_CONFIG[mode]
        device_index = 0
        labels = device_labels(device_index)
        device = torch.device(f"cuda:{device_index}")
        ACTIVE_GAUGE.labels(**labels).set(1)
        started = time.perf_counter()
        with torch.inference_mode():
            accumulator = torch.zeros((1,), device=device, dtype=torch.float32)
            for _ in range(config["batches"]):
                a_tensor = torch.randn((config["matrix_size"], config["matrix_size"]), device=device, dtype=dtype())
                b_tensor = torch.randn((config["matrix_size"], config["matrix_size"]), device=device, dtype=dtype())
                result = torch.matmul(a_tensor, b_tensor)
                accumulator += result.float().mean()
            torch.cuda.synchronize(device)
        LAST_DURATION = time.perf_counter() - started
        ITERATION_DURATION.labels(instance=INSTANCE, mode=mode).observe(LAST_DURATION)
        ITERATIONS.labels(instance=INSTANCE, gpu=labels["gpu"], uuid=labels["uuid"], mode=mode).inc()
        LAST_DURATION_GAUGE.labels(instance=INSTANCE, gpu=labels["gpu"], uuid=labels["uuid"], mode=mode).set(LAST_DURATION)
        ALLOCATED_MEMORY.labels(**labels).set(torch.cuda.memory_allocated(device))
        RESERVED_MEMORY.labels(**labels).set(torch.cuda.memory_reserved(device))
        ACTIVE_GAUGE.labels(**labels).set(0)
        del accumulator
        time.sleep(config["sleep_seconds"])


@app.on_event("startup")
def startup() -> None:
    thread = threading.Thread(target=learning_loop, daemon=True)
    thread.start()


@app.get("/healthz")
def healthz() -> dict[str, object]:
    mode, reason = current_mode()
    return {"ok": True, "mode": mode, "reason": reason}


@app.get("/status")
def status() -> dict[str, object]:
    mode, reason = current_mode()
    return {
        "instance": INSTANCE,
        "mode": mode,
        "reason": reason,
        "cuda_available": torch.cuda.is_available(),
        "last_iteration_seconds": round(LAST_DURATION, 4),
        "modes": MODE_CONFIG,
    }


@app.post("/control")
def control(payload: ControlRequest) -> dict[str, object]:
    set_mode(payload.mode, payload.reason)
    mode, reason = current_mode()
    return {"ok": True, "mode": mode, "reason": reason}


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
