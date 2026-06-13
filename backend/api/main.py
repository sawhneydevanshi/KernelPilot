import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.core.orchestrator import run_orchestration, OrchestrationReport

app = FastAPI(
    title="KernelPilot",
    description="AI-powered PyTorch bottleneck diagnosis platform",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)




class AnalyzeRequest(BaseModel):
    source_code: str                  
    model_type: str = "simple_cnn"    
    batch_size: int = 4               

class StaticIssueResponse(BaseModel):
    line: int
    severity: str
    code: str
    message: str


class ProfilerOpResponse(BaseModel):
    name: str
    cpu_time_ms: float
    cuda_time_ms: float
    calls: int


class GPUTelemetryResponse(BaseModel):
    peak_memory_mb: float
    avg_utilization_pct: float
    avg_temperature_c: float
    bottlenecks: list[str]
    mock_data: bool


class AnalyzeResponse(BaseModel):
    total_time_seconds: float
    static_issues: list[StaticIssueResponse]
    profiler: dict
    gpu_telemetry: GPUTelemetryResponse
    summary: list[str]
    errors: dict




class SimpleCNN(nn.Module):
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


class SimpleLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


MODEL_REGISTRY = {
    "simple_cnn": {
        "model": SimpleCNN,
        "input_shape": (3, 32, 32),   
    },
    "simple_linear": {
        "model": SimpleLinear,
        "input_shape": (128,),
    },
}




@app.get("/")
def root():
    return {
        "name": "KernelPilot",
        "status": "running",
        "version": "1.0.0",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest):
 
    if request.model_type not in MODEL_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model_type '{request.model_type}'. "
                   f"Choose from: {list(MODEL_REGISTRY.keys())}"
        )

    
    if not request.source_code.strip():
        raise HTTPException(
            status_code=400,
            detail="source_code cannot be empty."
        )

    
    registry_entry = MODEL_REGISTRY[request.model_type]
    model = registry_entry["model"]()
    input_shape = (request.batch_size,) + registry_entry["input_shape"]
    sample_input = torch.randn(*input_shape)

    report: OrchestrationReport = run_orchestration(
        source_code=request.source_code,
        model=model,
        sample_input=sample_input,
    )

    return AnalyzeResponse(
        total_time_seconds=report.total_time_seconds,
        static_issues=[
            StaticIssueResponse(
                line=issue.line,
                severity=issue.severity,
                code=issue.code,
                message=issue.message,
            )
            for issue in report.static_issues
        ],
        profiler={
            "bottleneck_device": report.profiler_report.bottleneck_device if report.profiler_report else "unknown",
            "slowest_op": report.profiler_report.slowest_op if report.profiler_report else "unknown",
            "total_cpu_time_ms": report.profiler_report.total_cpu_time_ms if report.profiler_report else 0,
            "total_cuda_time_ms": report.profiler_report.total_cuda_time_ms if report.profiler_report else 0,
            "top_ops": [
                {
                    "name": op.name,
                    "cpu_time_ms": op.cpu_time_ms,
                    "cuda_time_ms": op.cuda_time_ms,
                    "calls": op.calls,
                }
                for op in (report.profiler_report.top_ops if report.profiler_report else [])
            ],
            "warnings": report.profiler_report.warnings if report.profiler_report else [],
        },
        gpu_telemetry=GPUTelemetryResponse(
            peak_memory_mb=report.gpu_report.peak_memory_mb if report.gpu_report else 0,
            avg_utilization_pct=report.gpu_report.avg_utilization_pct if report.gpu_report else 0,
            avg_temperature_c=report.gpu_report.avg_temperature_c if report.gpu_report else 0,
            bottlenecks=report.gpu_report.bottlenecks if report.gpu_report else [],
            mock_data=report.gpu_report.mock_data if report.gpu_report else True,
        ),
        summary=report.summary,
        errors=report.errors,
    )