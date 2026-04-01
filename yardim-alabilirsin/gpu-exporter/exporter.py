import csv
import os
import socket
import subprocess
import time
from io import StringIO

from prometheus_client import Gauge, start_http_server


PORT = int(os.getenv("EXPORTER_PORT", "9500"))
SCRAPE_INTERVAL_SECONDS = float(os.getenv("SCRAPE_INTERVAL_SECONDS", "5"))
INSTANCE = os.getenv("EXPORTER_INSTANCE", socket.gethostname())

GPU_UP = Gauge("nvidia_smi_up", "Whether nvidia-smi exporter is collecting GPU metrics")
GPU_COUNT = Gauge("nvidia_smi_gpu_count", "Visible GPU count reported by nvidia-smi")
GPU_UTIL = Gauge(
    "nvidia_smi_gpu_utilization_percent",
    "GPU utilization percent",
    ["instance", "gpu", "uuid", "name"],
)
GPU_MEM_UTIL = Gauge(
    "nvidia_smi_gpu_memory_used_percent",
    "GPU memory usage percent",
    ["instance", "gpu", "uuid", "name"],
)
GPU_MEM_USED = Gauge(
    "nvidia_smi_gpu_memory_used_megabytes",
    "GPU memory used in megabytes",
    ["instance", "gpu", "uuid", "name"],
)
GPU_MEM_TOTAL = Gauge(
    "nvidia_smi_gpu_memory_total_megabytes",
    "GPU memory total in megabytes",
    ["instance", "gpu", "uuid", "name"],
)
GPU_TEMP = Gauge(
    "nvidia_smi_gpu_temperature_celsius",
    "GPU temperature in Celsius",
    ["instance", "gpu", "uuid", "name"],
)
GPU_POWER = Gauge(
    "nvidia_smi_gpu_power_draw_watts",
    "GPU power draw in watts",
    ["instance", "gpu", "uuid", "name"],
)
GPU_CLOCK_SM = Gauge(
    "nvidia_smi_gpu_sm_clock_mhz",
    "GPU SM clock in MHz",
    ["instance", "gpu", "uuid", "name"],
)
GPU_CLOCK_MEM = Gauge(
    "nvidia_smi_gpu_memory_clock_mhz",
    "GPU memory clock in MHz",
    ["instance", "gpu", "uuid", "name"],
)
GPU_FAN = Gauge(
    "nvidia_smi_gpu_fan_speed_percent",
    "GPU fan speed percent",
    ["instance", "gpu", "uuid", "name"],
)
GPU_PSTATE = Gauge(
    "nvidia_smi_gpu_pstate",
    "GPU performance state as numeric value",
    ["instance", "gpu", "uuid", "name"],
)


def as_float(value: str) -> float:
    cleaned = (value or "").strip().strip("[]")
    if not cleaned or cleaned in {"N/A", "[Not Supported]"}:
        return float("nan")
    return float(cleaned)


def as_pstate(value: str) -> float:
    cleaned = (value or "").strip().upper().replace("P", "")
    if not cleaned or cleaned == "N/A":
        return float("nan")
    return float(cleaned)


def collect() -> None:
    fields = [
        "index",
        "uuid",
        "name",
        "utilization.gpu",
        "utilization.memory",
        "memory.used",
        "memory.total",
        "temperature.gpu",
        "power.draw",
        "clocks.sm",
        "clocks.mem",
        "fan.speed",
        "pstate",
    ]
    command = [
        "nvidia-smi",
        f"--query-gpu={','.join(fields)}",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        GPU_UP.set(0)
        GPU_COUNT.set(0)
        return

    rows = list(csv.reader(StringIO(result.stdout)))
    GPU_UP.set(1)
    GPU_COUNT.set(len(rows))

    for row in rows:
        if len(row) != len(fields):
            continue
        index, uuid, name, util_gpu, _, mem_used, mem_total, temp, power, sm_clock, mem_clock, fan, pstate = row
        labels = {"instance": INSTANCE, "gpu": index.strip(), "uuid": uuid.strip(), "name": name.strip()}
        mem_used_value = as_float(mem_used)
        mem_total_value = as_float(mem_total)
        mem_ratio = float("nan")
        if mem_total_value and mem_total_value == mem_total_value:
            mem_ratio = (mem_used_value / mem_total_value) * 100 if mem_total_value else float("nan")

        GPU_UTIL.labels(**labels).set(as_float(util_gpu))
        GPU_MEM_UTIL.labels(**labels).set(mem_ratio)
        GPU_MEM_USED.labels(**labels).set(mem_used_value)
        GPU_MEM_TOTAL.labels(**labels).set(mem_total_value)
        GPU_TEMP.labels(**labels).set(as_float(temp))
        GPU_POWER.labels(**labels).set(as_float(power))
        GPU_CLOCK_SM.labels(**labels).set(as_float(sm_clock))
        GPU_CLOCK_MEM.labels(**labels).set(as_float(mem_clock))
        GPU_FAN.labels(**labels).set(as_float(fan))
        GPU_PSTATE.labels(**labels).set(as_pstate(pstate))


def main() -> None:
    start_http_server(PORT)
    while True:
        collect()
        time.sleep(SCRAPE_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
