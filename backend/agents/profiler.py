import time
import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
from dataclasses import dataclass, field


@dataclass
class OpStat:
    name: str
    cpu_time_ms: float        
    cuda_time_ms: float       
    calls: int                
    cpu_memory_mb: float      


@dataclass
class ProfilerReport:
    top_ops: list[OpStat] = field(default_factory=list)
    total_cpu_time_ms: float = 0.0
    total_cuda_time_ms: float = 0.0
    cuda_available: bool = False
    bottleneck_device: str = ""   
    slowest_op: str = ""
    warnings: list[str] = field(default_factory=list)


def profile_model(
    model: nn.Module,
    sample_input: torch.Tensor,
    warmup_steps: int = 2,
    profile_steps: int = 5,
) -> ProfilerReport:
    """
    Profile a PyTorch model's forward pass.

    Args:
        model:         the nn.Module to profile
        sample_input:  a sample input tensor
        warmup_steps:  number of warmup runs before profiling (warms up CUDA)
        profile_steps: number of runs to profile and average over
    """
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")

    model = model.to(device)
    sample_input = sample_input.to(device)
    model.eval()

  
    with torch.no_grad():
        for _ in range(warmup_steps):
            _ = model(sample_input)

   
    activities = [ProfilerActivity.CPU]
    if cuda_available:
        activities.append(ProfilerActivity.CUDA)

    with torch.no_grad():
        with profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as prof:
            for _ in range(profile_steps):
                with record_function("model_inference"):
                    _ = model(sample_input)

   
    top_ops = []
    key_averages = prof.key_averages()

    for event in key_averages:
        if event.key.startswith("aten::") or event.key == "model_inference":
            top_ops.append(OpStat(
                name=event.key,
                cpu_time_ms=event.self_cpu_time_total / 1000 / profile_steps,
                cuda_time_ms=getattr(event, 'self_cuda_time_total', getattr(event, 'self_device_time_total', 0)) / 1000 / profile_steps,
                calls=event.count // profile_steps,
                cpu_memory_mb=event.self_cpu_memory_usage / 1024 / 1024,
            ))

   
    top_ops.sort(key=lambda x: x.cpu_time_ms + x.cuda_time_ms, reverse=True)
    top_ops = top_ops[:10]

    
    total_cpu = sum(op.cpu_time_ms for op in top_ops)
    total_cuda = sum(op.cuda_time_ms for op in top_ops)

    bottleneck_device = "GPU" if total_cuda > total_cpu else "CPU"
    slowest_op = top_ops[0].name if top_ops else "unknown"

  
    warnings = []

    if cuda_available and total_cuda == 0:
        warnings.append(
            "CUDA_NOT_USED: GPU is available but no CUDA time was recorded. "
            "Verify your model and tensors are on the correct device."
        )

    if top_ops and top_ops[0].cpu_time_ms > 100:
        warnings.append(
            f"SLOW_OP: '{top_ops[0].name}' is the slowest operation at "
            f"{top_ops[0].cpu_time_ms:.1f}ms CPU time. "
            f"Consider optimizing or replacing this operation."
        )

    if not cuda_available:
        warnings.append(
            "CPU_ONLY: No GPU detected. Running on CPU only. "
            "For production workloads, use a CUDA-enabled GPU."
        )

    return ProfilerReport(
        top_ops=top_ops,
        total_cpu_time_ms=total_cpu,
        total_cuda_time_ms=total_cuda,
        cuda_available=cuda_available,
        bottleneck_device=bottleneck_device,
        slowest_op=slowest_op,
        warnings=warnings,
    )