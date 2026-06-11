from backend.agents.static_analyzer import analyze


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

def test_finds_issues():
    issues = analyze(BAD_CODE)
    codes = [i.code for i in issues]
    print("\nDetected issues:")
    for issue in issues:
        print(f"  Line {issue.line} [{issue.severity}] {issue.code}: {issue.message}")
    
    assert "ITEM_IN_LOOP" in codes
    assert "NO_GRAD_MISSING" in codes
    assert "DATALOADER_NO_WORKERS" in codes

if __name__ == "__main__":
    test_finds_issues()
    print("\nAll tests passed!")