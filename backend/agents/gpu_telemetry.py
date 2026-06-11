import time
import threading
from dataclasses import dataclass, field


try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False


@dataclass
class GPUSnapshot:
    timestamp: float
    utilization_pct: float      
    memory_used_mb: float      
    memory_total_mb: float      
    memory_free_mb: float       
    temperature_c: float        


@dataclass
class GPUReport:
    snapshots: list[GPUSnapshot] = field(default_factory=list)
    peak_memory_mb: float = 0.0
    avg_utilization_pct: float = 0.0
    avg_temperature_c: float = 0.0
    bottlenecks: list[str] = field(default_factory=list)
    gpu_available: bool = False
    mock_data: bool = False


class GPUTelemetryAgent:
    """
    Monitors GPU metrics while a workload runs.
    Usage:
        agent = GPUTelemetryAgent(poll_interval=0.5)
        agent.start()
        # ... run your pytorch code ...
        report = agent.stop()
    """

    def __init__(self, poll_interval: float = 0.5, gpu_index: int = 0):
        self.poll_interval = poll_interval  # seconds between readings
        self.gpu_index = gpu_index
        self._snapshots: list[GPUSnapshot] = []
        self._running = False
        self._thread: threading.Thread | None = None

   
    def start(self):
        self._snapshots = []
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

   
    def stop(self) -> GPUReport:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        return self._build_report()

    def _poll_loop(self):
        while self._running:
            snapshot = self._read_gpu() if NVML_AVAILABLE else self._mock_gpu()
            self._snapshots.append(snapshot)
            time.sleep(self.poll_interval)

   
    def _read_gpu(self) -> GPUSnapshot:
        handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        temp = pynvml.nvmlDeviceGetTemperature(
            handle, pynvml.NVML_TEMPERATURE_GPU
        )
        return GPUSnapshot(
            timestamp=time.time(),
            utilization_pct=float(util.gpu),
            memory_used_mb=mem.used / 1024 / 1024,
            memory_total_mb=mem.total / 1024 / 1024,
            memory_free_mb=mem.free / 1024 / 1024,
            temperature_c=float(temp),
        )


    def _mock_gpu(self) -> GPUSnapshot:
        import math
        import random
        t = time.time() % 60
        utilization = 60 + 30 * math.sin(t / 10) + random.uniform(-5, 5)
        memory_used = 4000 + 2000 * math.sin(t / 15) + random.uniform(-100, 100)
        return GPUSnapshot(
            timestamp=time.time(),
            utilization_pct=max(0, min(100, utilization)),
            memory_used_mb=max(0, memory_used),
            memory_total_mb=8192,
            memory_free_mb=max(0, 8192 - memory_used),
            temperature_c=65 + random.uniform(-3, 3),
        )

    def _build_report(self) -> GPUReport:
        if not self._snapshots:
            return GPUReport(gpu_available=NVML_AVAILABLE, mock_data=not NVML_AVAILABLE)

        peak_memory = max(s.memory_used_mb for s in self._snapshots)
        avg_util = sum(s.utilization_pct for s in self._snapshots) / len(self._snapshots)
        avg_temp = sum(s.temperature_c for s in self._snapshots) / len(self._snapshots)

        bottlenecks = []

        total = self._snapshots[0].memory_total_mb
        if peak_memory / total > 0.90:
            bottlenecks.append(
                f"MEMORY_PRESSURE: Peak memory usage was "
                f"{peak_memory:.0f}MB / {total:.0f}MB "
                f"({peak_memory/total*100:.1f}%). Risk of OOM errors. "
                f"Consider reducing batch size or using gradient checkpointing."
            )

      
        if avg_util < 30:
            bottlenecks.append(
                f"LOW_GPU_UTILIZATION: Average utilization was {avg_util:.1f}%. "
                f"GPU is likely waiting on CPU/DataLoader. "
                f"Increase num_workers or batch size."
            )

        
        if avg_temp > 85:
            bottlenecks.append(
                f"HIGH_TEMPERATURE: Average GPU temperature was {avg_temp:.1f}C. "
                f"Check cooling and consider reducing workload."
            )

        return GPUReport(
            snapshots=self._snapshots,
            peak_memory_mb=peak_memory,
            avg_utilization_pct=avg_util,
            avg_temperature_c=avg_temp,
            bottlenecks=bottlenecks,
            gpu_available=NVML_AVAILABLE,
            mock_data=not NVML_AVAILABLE,
        )