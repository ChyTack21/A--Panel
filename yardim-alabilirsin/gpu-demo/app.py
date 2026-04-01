import logging
import os
import socket
import time

import torch
from prometheus_client import Gauge, start_http_server


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("gpu-demo")

INSTANCE = socket.gethostname()
METRICS_PORT = int(os.getenv("GPU_DEMO_METRICS_PORT", "9600"))
MATRIX_SIZE = int(os.getenv("GPU_DEMO_MATRIX_SIZE", "4096"))
BATCHES = int(os.getenv("GPU_DEMO_BATCHES", "4"))
SLEEP_SECONDS = float(os.getenv("GPU_DEMO_SLEEP_SECONDS", "2"))
WARMUP_SECONDS = float(os.getenv("GPU_DEMO_WARMUP_SECONDS", "5"))
PRECISION = os.getenv("GPU_DEMO_PRECISION", "float16").lower()

ITERATIONS = Gauge("gpu_demo_iterations_total", "Completed GPU demo iterations", ["instance", "gpu", "uuid"])
LAST_ITERATION_SECONDS = Gauge("gpu_demo_last_iteration_seconds", "Duration of the last workload iteration", ["instance", "gpu", "uuid"])
CUDA_AVAILABLE = Gauge("gpu_demo_cuda_available", "Whether CUDA is available", ["instance"])
ACTIVE_GPU = Gauge("gpu_demo_active", "Whether the GPU demo is actively dispatching work", ["instance", "gpu", "uuid"])
ALLOCATED_MEMORY = Gauge("gpu_demo_allocated_memory_bytes", "Torch allocated memory for demo workload", ["instance", "gpu", "uuid"])
RESERVED_MEMORY = Gauge("gpu_demo_reserved_memory_bytes", "Torch reserved memory for demo workload", ["instance", "gpu", "uuid"])


def torch_dtype():
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
    return {
        "instance": INSTANCE,
        "gpu": str(device_index),
        "uuid": uuid,
    }


def workload(device_index: int) -> None:
    dtype = torch_dtype()
    labels = device_labels(device_index)
    device = torch.device(f"cuda:{device_index}")
    matrix_shape = (MATRIX_SIZE, MATRIX_SIZE)
    logger.info("dispatching workload gpu=%s uuid=%s matrix=%s batches=%s dtype=%s", labels["gpu"], labels["uuid"], MATRIX_SIZE, BATCHES, dtype)
    ACTIVE_GPU.labels(**labels).set(1)
    start = time.perf_counter()
    accumulator = torch.zeros((1,), device=device, dtype=torch.float32)
    for _ in range(BATCHES):
        a_tensor = torch.randn(matrix_shape, device=device, dtype=dtype)
        b_tensor = torch.randn(matrix_shape, device=device, dtype=dtype)
        result = torch.matmul(a_tensor, b_tensor)
        accumulator += result.float().mean()
    torch.cuda.synchronize(device)
    duration = time.perf_counter() - start
    ITERATIONS.labels(**labels).inc()
    LAST_ITERATION_SECONDS.labels(**labels).set(duration)
    ALLOCATED_MEMORY.labels(**labels).set(torch.cuda.memory_allocated(device))
    RESERVED_MEMORY.labels(**labels).set(torch.cuda.memory_reserved(device))
    ACTIVE_GPU.labels(**labels).set(0)
    del accumulator
    torch.cuda.empty_cache()


def main() -> None:
    start_http_server(METRICS_PORT)
    logger.info("gpu demo metrics available at :%s/metrics", METRICS_PORT)
    if WARMUP_SECONDS > 0:
        logger.info("warming up for %.1f seconds", WARMUP_SECONDS)
        time.sleep(WARMUP_SECONDS)

    while True:
        available = torch.cuda.is_available()
        CUDA_AVAILABLE.labels(instance=INSTANCE).set(1 if available else 0)
        if not available:
            logger.warning("cuda is not available inside gpu-demo container")
            time.sleep(max(SLEEP_SECONDS, 5))
            continue

        device_count = torch.cuda.device_count()
        for device_index in range(device_count):
            try:
                workload(device_index)
            except RuntimeError:
                logger.exception("gpu workload failed on device %s", device_index)
            time.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    main()
