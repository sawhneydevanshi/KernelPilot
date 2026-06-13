from backend.rag.chunker import load_chunks
from backend.rag.vector_store import index_chunks, retrieve


def test_vector_store():
    
    print("\nLoading chunks from disk...")
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks")

    print("\nIndexing into ChromaDB (this calls OpenAI embeddings API)...")
    count = index_chunks(chunks, reset=True)
    print(f"Indexed {count} chunks")

  
    test_queries = [
        "how do I fix slow DataLoader bottleneck",
        "GPU out of memory error during training",
        "how to use torch.no_grad during inference",
        "slow convolution operation in profiler",
        "mixed precision training speedup",
    ]

    print(f"\nTesting retrieval with {len(test_queries)} queries:\n")
    for query in test_queries:
        print(f"Query: '{query}'")
        results = retrieve(query, top_k=3)
        for i, r in enumerate(results):
            print(f"  {i+1}. [{r['relevance_score']:.3f}] {r['page_title']}")
        print()

  
    assert count > 0, "Should have indexed chunks"
    results = retrieve("DataLoader performance", top_k=3)
    assert len(results) > 0, "Should return results"
    assert results[0]["relevance_score"] > 0.3, "Top result should be relevant"
    assert "text" in results[0], "Results should have text"
    assert "source_url" in results[0], "Results should have source_url"

    print("All tests passed!")


if __name__ == "__main__":
    test_vector_store()