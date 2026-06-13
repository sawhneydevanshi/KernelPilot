from backend.rag.retrieval_pipeline import (
    build_queries_from_diagnosis,
    retrieve_for_diagnosis,
)


SAMPLE_DIAGNOSIS = {
    "static_issues": [
        {"code": "NO_GRAD_MISSING", "message": "model.eval() without no_grad"},
        {"code": "DATALOADER_NO_WORKERS", "message": "DataLoader missing num_workers"},
        {"code": "ITEM_IN_LOOP", "message": ".item() called in loop"},
    ],
    "profiler": {
        "slowest_op": "aten::_slow_conv2d_forward",
        "bottleneck_device": "CPU",
        "warnings": ["CPU_ONLY: No GPU detected"],
    },
    "gpu_telemetry": {
        "bottlenecks": [],
        "peak_memory_mb": 4000,
        "avg_utilization_pct": 30,
    },
    "summary": [],
}


def test_retrieval_pipeline():
   
    print("\nStep 1 — Building queries from diagnosis...")
    queries = build_queries_from_diagnosis(SAMPLE_DIAGNOSIS)
    print(f"  Generated {len(queries)} queries:")
    for q in queries:
        print(f"    - {q}")
    assert len(queries) > 0, "Should generate at least one query"

   
    print("\nStep 2 — Running full retrieval pipeline...")
    results = retrieve_for_diagnosis(
        SAMPLE_DIAGNOSIS,
        top_k_per_query=3,
        max_final_results=5,
        use_query_expansion=True,
    )

    print(f"\nTop {len(results)} retrieved docs:")
    for i, r in enumerate(results):
        print(f"  {i+1}. [{r['relevance_score']:.3f}] {r['page_title']}")
        print(f"       {r['source_url']}")

    assert len(results) > 0, "Should retrieve some results"
    assert results[0]["relevance_score"] > 0.3, "Top result should be relevant"

    print("\nAll tests passed!")


if __name__ == "__main__":
    test_retrieval_pipeline()