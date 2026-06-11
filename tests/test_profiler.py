import torch
import torch.nn as nn
from backend.agents.profiler import profile_model



class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(32 * 4 * 4, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def test_profiler():
    model = SimpleModel()

    sample_input = torch.randn(4, 3, 32, 32)

    print("\nProfiling model...")
    report = profile_model(model, sample_input, warmup_steps=2, profile_steps=5)

    print(f"\nProfiler Report:")
    print(f"  CUDA available: {report.cuda_available}")
    print(f"  Bottleneck device: {report.bottleneck_device}")
    print(f"  Slowest op: {report.slowest_op}")
    print(f"  Total CPU time: {report.total_cpu_time_ms:.2f} ms")
    print(f"  Total CUDA time: {report.total_cuda_time_ms:.2f} ms")

    print(f"\nTop {len(report.top_ops)} operations:")
    for i, op in enumerate(report.top_ops):
        print(f"  {i+1}. {op.name}")
        print(f"     CPU: {op.cpu_time_ms:.2f}ms  |  "
              f"CUDA: {op.cuda_time_ms:.2f}ms  |  "
              f"Calls: {op.calls}")

    if report.warnings:
        print(f"\nWarnings:")
        for w in report.warnings:
            print(f"  ⚠️  {w}")

    # Assertions
    assert len(report.top_ops) > 0, "Should have profiled some ops"
    assert report.total_cpu_time_ms > 0, "Should have CPU time"
    assert report.bottleneck_device in ("CPU", "GPU"), "Should identify bottleneck"
    assert report.slowest_op != "", "Should identify slowest op"

    print("\nAll tests passed!")


if __name__ == "__main__":
    test_profiler()