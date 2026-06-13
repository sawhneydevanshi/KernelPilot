from backend.rag.retrieval_pipeline import retrieve_for_diagnosis
from backend.rag.synthesizer import synthesize_fix

SAMPLE_DIAGNOSIS = {
    "static_issues": [
        {"code": "NO_GRAD_MISSING", "message": "model.eval() detected but torch.no_grad() context manager is missing."},
        {"code": "DATALOADER_NO_WORKERS", "message": "DataLoader created without num_workers."},
        {"code": "ITEM_IN_LOOP", "message": ".item() inside a loop forces a GPU-CPU sync on every iteration."},
    ],
    "profiler": {
        "slowest_op": "aten::_slow_conv2d_forward",
        "bottleneck_device": "CPU",
        "total_cpu_time_ms": 0.93,
        "total_cuda_time_ms": 0.0,
        "warnings": ["CPU_ONLY: No GPU detected."],
    },
    "gpu_telemetry": {
        "peak_memory_mb": 4500,
        "avg_utilization_pct": 31.0,
        "bottlenecks": [],
    },
    "summary": [],
}


def test_synthesizer():
    
    print("\nStep 1 — Retrieving relevant docs...")
    retrieved_docs = retrieve_for_diagnosis(
        SAMPLE_DIAGNOSIS,
        top_k_per_query=3,
        max_final_results=4,
        use_query_expansion=False,  
    )
    print(f"  Retrieved {len(retrieved_docs)} docs")

  
    print("\nStep 2 — Synthesizing explanation and fix...")
    result = synthesize_fix(SAMPLE_DIAGNOSIS, retrieved_docs)


    print(f"\n{'='*60}")
    print("  KERNELPILOT — AI-GENERATED FIX")
    print(f"{'='*60}")

    print(f"\nEXPLANATION:\n{result['explanation']}")
    print(f"\nSUGGESTED FIX:\n{result['suggested_fix']}")
    print(f"\nPRIORITY:\n{result['priority']}")

    print(f"\nSOURCE DOCS USED:")
    for doc in result["relevant_docs"]:
        print(f"  - [{doc['relevance_score']:.3f}] {doc['title']}")
        print(f"    {doc['url']}")

    print(f"\nModel: {result['model_used']}")
    print(f"{'='*60}")

    assert result["explanation"], "Should have an explanation"
    assert result["suggested_fix"], "Should have a suggested fix"
    assert len(result["relevant_docs"]) > 0, "Should cite source docs"
    assert result["model_used"] == "gpt-4o-mini"

    print("\nAll tests passed!")


if __name__ == "__main__":
    test_synthesizer()