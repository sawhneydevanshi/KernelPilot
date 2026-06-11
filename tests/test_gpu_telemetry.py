import time
from backend.agents.gpu_telemetry import GPUTelemetryAgent

def test_gpu_telemetry():
    agent = GPUTelemetryAgent(poll_interval=0.2)

    
    agent.start()

   
    print("\nMonitoring GPU for 2 seconds...")
    time.sleep(2)

    
    report = agent.stop()

    print(f"\nGPU Report:")
    print(f"  GPU Available (NVIDIA): {report.gpu_available}")
    print(f"  Using mock data (Mac): {report.mock_data}")
    print(f"  Snapshots collected: {len(report.snapshots)}")
    print(f"  Peak memory: {report.peak_memory_mb:.1f} MB")
    print(f"  Avg utilization: {report.avg_utilization_pct:.1f}%")
    print(f"  Avg temperature: {report.avg_temperature_c:.1f}C")

    if report.bottlenecks:
        print(f"\nBottlenecks detected:")
        for b in report.bottlenecks:
            print(f"  ⚠️  {b}")
    else:
        print(f"\nNo bottlenecks detected.")


    assert len(report.snapshots) > 0, "Should have collected snapshots"
    assert report.peak_memory_mb > 0, "Peak memory should be > 0"
    assert 0 <= report.avg_utilization_pct <= 100, "Utilization should be 0-100"

    print("\nAll tests passed!")

if __name__ == "__main__":
    test_gpu_telemetry()