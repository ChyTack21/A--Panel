import os
import socket
import threading
import time
from collections import deque
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from starlette.responses import Response


INSTANCE = socket.gethostname()
PORT = int(os.getenv("SERVING_PORT", "9700"))
DEFAULT_MATRIX_SIZE = int(os.getenv("SERVING_DEFAULT_MATRIX_SIZE", "3584"))
DEFAULT_STEPS = int(os.getenv("SERVING_DEFAULT_STEPS", "3"))
DEFAULT_BATCHES = int(os.getenv("SERVING_DEFAULT_BATCHES", "2"))
DEFAULT_PRECISION = os.getenv("SERVING_DEFAULT_PRECISION", "float16").lower()
CONCURRENCY = max(1, int(os.getenv("SERVING_CONCURRENCY", "1")))
RECENT_LATENCIES = deque(maxlen=200)
RECENT_LOCK = threading.Lock()
SEMAPHORE = threading.Semaphore(CONCURRENCY)
STATE_LOCK = threading.Lock()
TOTAL_REQUESTS = 0
ACTIVE_REQUESTS = 0
PENDING_REQUESTS = 0
LAST_ERROR = ""


REQUESTS_TOTAL = Counter("gpu_serving_requests_total", "Total serving lane requests", ["instance", "status"])
REQUEST_DURATION = Histogram(
    "gpu_serving_request_seconds",
    "Serving lane end-to-end request latency",
    ["instance"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16),
)
ACTIVE_GAUGE = Gauge("gpu_serving_active_requests", "Requests currently executing", ["instance"])
QUEUE_GAUGE = Gauge("gpu_serving_queue_depth", "Requests waiting for execution", ["instance"])
CUDA_GAUGE = Gauge("gpu_serving_cuda_available", "CUDA visible to serving lane", ["instance"])
LAST_MATRIX_GAUGE = Gauge("gpu_serving_last_matrix_size", "Last matrix size used by serving lane", ["instance", "gpu", "uuid"])
LAST_BATCH_GAUGE = Gauge("gpu_serving_last_batches", "Last batch count used by serving lane", ["instance", "gpu", "uuid"])
LAST_STEPS_GAUGE = Gauge("gpu_serving_last_steps", "Last steps count used by serving lane", ["instance", "gpu", "uuid"])
LAST_REQUEST_SECONDS = Gauge("gpu_serving_last_request_seconds", "Latency of the last request", ["instance", "gpu", "uuid"])
LAST_ALLOCATED_MEMORY = Gauge("gpu_serving_allocated_memory_bytes", "Torch allocated memory after the last serving request", ["instance", "gpu", "uuid"])
LAST_RESERVED_MEMORY = Gauge("gpu_serving_reserved_memory_bytes", "Torch reserved memory after the last serving request", ["instance", "gpu", "uuid"])
TOTAL_COMPLETED = Counter("gpu_serving_completed_total", "Completed serving requests", ["instance", "gpu", "uuid"])

app = FastAPI(title="GPU Serving Simulation", version="1.0.0")


class InferenceRequest(BaseModel):
    matrix_size: Optional[int] = None
    steps: Optional[int] = None
    batches: Optional[int] = None
    precision: Optional[str] = None


def torch_dtype(name: str) -> torch.dtype:
    normalized = (name or DEFAULT_PRECISION).lower()
    if normalized == "float32":
        return torch.float32
    if normalized == "bfloat16":
        return torch.bfloat16
    return torch.float16


def device_labels(device_index: int) -> dict[str, str]:
    props = torch.cuda.get_device_properties(device_index)
    uuid = getattr(props, "uuid", f"gpu-{device_index}")
    if uuid and not str(uuid).startswith("GPU-"):
        uuid = f"GPU-{uuid}"
    return {"instance": INSTANCE, "gpu": str(device_index), "uuid": uuid}


def update_state(active_delta: int = 0, pending_delta: int = 0) -> None:
    global ACTIVE_REQUESTS, PENDING_REQUESTS
    with STATE_LOCK:
        ACTIVE_REQUESTS += active_delta
        PENDING_REQUESTS += pending_delta
        ACTIVE_GAUGE.labels(instance=INSTANCE).set(ACTIVE_REQUESTS)
        QUEUE_GAUGE.labels(instance=INSTANCE).set(max(PENDING_REQUESTS, 0))


def infer_on_gpu(payload: InferenceRequest) -> dict[str, object]:
    global TOTAL_REQUESTS, LAST_ERROR
    matrix_size = payload.matrix_size or DEFAULT_MATRIX_SIZE
    steps = payload.steps or DEFAULT_STEPS
    batches = payload.batches or DEFAULT_BATCHES
    precision_name = payload.precision or DEFAULT_PRECISION

    if matrix_size < 256 or steps < 1 or batches < 1:
        raise HTTPException(status_code=400, detail="matrix_size >= 256, steps >= 1 and batches >= 1 are required")

    update_state(pending_delta=1)
    started = time.perf_counter()
    acquired = False
    try:
        SEMAPHORE.acquire()
        acquired = True
        update_state(active_delta=1, pending_delta=-1)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available inside gpu-serving-sim")

        TOTAL_REQUESTS += 1
        device_index = 0
        labels = device_labels(device_index)
        device = torch.device(f"cuda:{device_index}")
        dtype = torch_dtype(precision_name)

        with torch.inference_mode():
            accumulator = torch.zeros((1,), device=device, dtype=torch.float32)
            for _ in range(steps):
                for _ in range(batches):
                    a_tensor = torch.randn((matrix_size, matrix_size), device=device, dtype=dtype)
                    b_tensor = torch.randn((matrix_size, matrix_size), device=device, dtype=dtype)
                    result = torch.matmul(a_tensor, b_tensor)
                    accumulator += result.float().mean()
            torch.cuda.synchronize(device)

        duration = time.perf_counter() - started
        with RECENT_LOCK:
            RECENT_LATENCIES.append(duration)

        REQUESTS_TOTAL.labels(instance=INSTANCE, status="success").inc()
        REQUEST_DURATION.labels(instance=INSTANCE).observe(duration)
        TOTAL_COMPLETED.labels(**labels).inc()
        LAST_REQUEST_SECONDS.labels(**labels).set(duration)
        LAST_MATRIX_GAUGE.labels(**labels).set(matrix_size)
        LAST_BATCH_GAUGE.labels(**labels).set(batches)
        LAST_STEPS_GAUGE.labels(**labels).set(steps)
        LAST_ALLOCATED_MEMORY.labels(**labels).set(torch.cuda.memory_allocated(device))
        LAST_RESERVED_MEMORY.labels(**labels).set(torch.cuda.memory_reserved(device))
        LAST_ERROR = ""
        del accumulator
        return {
            "ok": True,
            "gpu": labels["gpu"],
            "uuid": labels["uuid"],
            "matrix_size": matrix_size,
            "steps": steps,
            "batches": batches,
            "precision": precision_name,
            "duration_seconds": round(duration, 4),
        }
    except Exception as exc:
        LAST_ERROR = str(exc)
        REQUESTS_TOTAL.labels(instance=INSTANCE, status="error").inc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if acquired:
            update_state(active_delta=-1)
            SEMAPHORE.release()
        else:
            update_state(pending_delta=-1)


@app.get("/healthz")
def healthz() -> dict[str, object]:
    CUDA_GAUGE.labels(instance=INSTANCE).set(1 if torch.cuda.is_available() else 0)
    return {"ok": True, "cuda_available": torch.cuda.is_available()}


@app.get("/status")
def status() -> dict[str, object]:
    with RECENT_LOCK:
        latencies = list(RECENT_LATENCIES)
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    return {
        "instance": INSTANCE,
        "cuda_available": torch.cuda.is_available(),
        "active_requests": ACTIVE_REQUESTS,
        "queue_depth": max(PENDING_REQUESTS, 0),
        "total_requests": TOTAL_REQUESTS,
        "avg_request_seconds": round(avg_latency, 4),
        "last_error": LAST_ERROR,
        "defaults": {
            "matrix_size": DEFAULT_MATRIX_SIZE,
            "steps": DEFAULT_STEPS,
            "batches": DEFAULT_BATCHES,
            "precision": DEFAULT_PRECISION,
            "concurrency": CONCURRENCY,
        },
    }


@app.post("/infer")
def infer(payload: InferenceRequest) -> dict[str, object]:
    return infer_on_gpu(payload)


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
