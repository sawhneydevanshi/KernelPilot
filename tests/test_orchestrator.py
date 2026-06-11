import torch
import torch.nn as nn
from backend.core.orchestrator import run_orchestration


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


BAD_CODE = """
import torch
from torch.utils.data import DataLoader

model = torch.nn.Linear(10, 1)
model.eval()

dataset = []
loader = DataLoader(dataset, batch_size=32)

for batch in loader:
    out = model(batch)
    loss = out.sum()
    print(loss.item())

    x = torch.zeros(10)
"""


def test_orchestration():
    model = SimpleModel()
    sample_input = torch.randn(4, 3, 32, 32)

    print("\nRunning full orchestration (all 3 agents in parallel)...")
    report = run_orchestration(
        source_code=BAD_CODE,
        model=model,
        sample_input=sample_input,
    )

   
    print(f"\n{'='*55}")
    print(f"  KERNELPILOT DIAGNOSIS REPORT")
    print(f"{'='*55}")
    print(f"  Completed in {report.total_time_seconds:.2f} seconds")
    print(f"{'='*55}\n")

    print(f"STATIC ANALYSIS — {len(report.static_issues)} issue(s) found")
    for issue in report.static_issues:
        line = f"line {issue.line}" if issue.line > 0 else "global"
        print(f"  [{issue.code}] {line}: {issue.message[:60]}...")

    print(f"\nPROFILER — top op: {report.profiler_report.slowest_op if report.profiler_report else 'N/A'}")
    if report.profiler_report:
        print(f"  Bottleneck device: {report.profiler_report.bottleneck_device}")
        print(f"  Total CPU time: {report.profiler_report.total_cpu_time_ms:.2f}ms")

    print(f"\nGPU TELEMETRY — {'mock' if report.gpu_report and report.gpu_report.mock_data else 'real'} data")
    if report.gpu_report:
        print(f"  Peak memory: {report.gpu_report.peak_memory_mb:.0f}MB")
        print(f"  Avg utilization: {report.gpu_report.avg_utilization_pct:.1f}%")

    print(f"\n{'='*55}")
    print(f"  UNIFIED SUMMARY")
    print(f"{'='*55}")
    for line in report.summary:
        print(f"  {line}")

    if report.errors:
        print(f"\nErrors: {report.errors}")

   
    assert len(report.static_issues) > 0, "Should find static issues"
    assert report.profiler_report is not None, "Should have profiler report"
    assert report.gpu_report is not None, "Should have GPU report"
    assert report.total_time_seconds < 60, "Should complete in under 60 seconds"
    assert len(report.errors) == 0, f"Should have no errors: {report.errors}"

    print(f"\nAll tests passed!")


if __name__ == "__main__":
    test_orchestration()