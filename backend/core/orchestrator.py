import asyncio
import time
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

from backend.agents.static_analyzer import analyze, Issue
from backend.agents.gpu_telemetry import GPUTelemetryAgent, GPUReport
from backend.agents.profiler import profile_model, ProfilerReport


@dataclass
class OrchestrationReport:
 
    static_issues: list[Issue] = field(default_factory=list)
    gpu_report: GPUReport = None
    profiler_report: ProfilerReport = None


    total_time_seconds: float = 0.0

   
    errors: dict = field(default_factory=dict)


    summary: list[str] = field(default_factory=list)


def run_orchestration(
    source_code: str,
    model: nn.Module,
    sample_input: torch.Tensor,
    gpu_poll_interval: float = 0.5,
    profile_steps: int = 5,
) -> OrchestrationReport:
    """
    Run all three agents in parallel and merge their results.

    Args:
        source_code:   raw Python source code string to static analyze
        model:         PyTorch model to profile
        sample_input:  sample input tensor for the model
    """
    report = OrchestrationReport()
    start_time = time.time()

   
    gpu_agent = GPUTelemetryAgent(poll_interval=gpu_poll_interval)
    gpu_agent.start()

    
    with ThreadPoolExecutor(max_workers=2) as executor:

        
        static_future = executor.submit(_run_static_analysis, source_code)
        profiler_future = executor.submit(
            _run_profiler, model, sample_input, profile_steps
        )

        
        for future in as_completed([static_future, profiler_future]):
            if future is static_future:
                try:
                    report.static_issues = future.result()
                except Exception as e:
                    report.errors["static_analyzer"] = str(e)
                    report.static_issues = []
            elif future is profiler_future:
                try:
                    report.profiler_report = future.result()
                except Exception as e:
                    report.errors["profiler"] = str(e)
                    report.profiler_report = None

   
    try:
        report.gpu_report = gpu_agent.stop()
    except Exception as e:
        report.errors["gpu_telemetry"] = str(e)

    report.total_time_seconds = time.time() - start_time

   
    report.summary = _build_summary(report)

    return report


def _run_static_analysis(source_code: str) -> list[Issue]:
    return analyze(source_code)


def _run_profiler(
    model: nn.Module,
    sample_input: torch.Tensor,
    profile_steps: int,
) -> ProfilerReport:
    return profile_model(model, sample_input, profile_steps=profile_steps)


def _build_summary(report: OrchestrationReport) -> list[str]:
    """
    Combine findings from all three agents into a prioritized list
    of actionable insights.
    """
    summary = []

   
    if report.static_issues:
        summary.append(
            f"STATIC ANALYSIS: Found {len(report.static_issues)} issue(s) in your code:"
        )
        for issue in report.static_issues:
            line_info = f" (line {issue.line})" if issue.line > 0 else ""
            summary.append(f"  [{issue.severity.upper()}]{line_info} {issue.message}")


    if report.profiler_report:
        pr = report.profiler_report
        summary.append(
            f"PROFILER: Bottleneck is on {pr.bottleneck_device}. "
            f"Slowest op: '{pr.slowest_op}' "
            f"({pr.top_ops[0].cpu_time_ms:.2f}ms CPU time)."
        )
        for warning in pr.warnings:
            summary.append(f"  {warning}")

  
    if report.gpu_report:
        gr = report.gpu_report
        if gr.mock_data:
            summary.append(
                f"GPU TELEMETRY: Running on Mac (mock data). "
                f"Peak memory: {gr.peak_memory_mb:.0f}MB, "
                f"Avg utilization: {gr.avg_utilization_pct:.1f}%."
            )
        else:
            summary.append(
                f"GPU TELEMETRY: Peak memory {gr.peak_memory_mb:.0f}MB, "
                f"Avg utilization {gr.avg_utilization_pct:.1f}%."
            )
        for bottleneck in gr.bottlenecks:
            summary.append(f"  {bottleneck}")


    if report.errors:
        for agent, error in report.errors.items():
            summary.append(f"ERROR in {agent}: {error}")

   
    if not summary:
        summary.append("No issues detected across all agents.")

    return summary