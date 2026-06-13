import requests

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

def test_api():
    # Test root
    r = requests.get("http://127.0.0.1:8000")
    assert r.status_code == 200
    print("GET / ✓")

    # Test health
    r = requests.get("http://127.0.0.1:8000/health")
    assert r.status_code == 200
    print("GET /health ✓")

    # Test analyze
    print("\nPOST /analyze — running full diagnosis...")
    r = requests.post(
        "http://127.0.0.1:8000/analyze",
        json={
            "source_code": BAD_CODE,
            "model_type": "simple_cnn",
            "batch_size": 4,
        }
    )

    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()

    print(f"\n  Completed in: {data['total_time_seconds']:.2f}s")
    print(f"  Static issues: {len(data['static_issues'])}")
    print(f"  Slowest op: {data['profiler']['slowest_op']}")
    print(f"  Bottleneck device: {data['profiler']['bottleneck_device']}")
    print(f"  Peak GPU memory: {data['gpu_telemetry']['peak_memory_mb']:.0f}MB")

    print(f"\n  Summary:")
    for line in data['summary']:
        print(f"    {line}")

    assert len(data['static_issues']) > 0
    assert data['profiler']['slowest_op'] != ""
    assert len(data['summary']) > 0

    print("\nAll API tests passed!")


if __name__ == "__main__":
    test_api()