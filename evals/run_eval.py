import sys
import json

sys.path.insert(0, ".")

from backend.rag.vector_store import retrieve


def run_eval(
    eval_path: str = "evals/eval_queries.json",
    top_k: int = 3,
) -> dict:
    with open(eval_path) as f:
        queries = json.load(f)

    print(f"\nRunning eval on {len(queries)} queries (top-{top_k} precision)...\n")

    hits = 0
    misses = 0
    miss_details = []

    for i, item in enumerate(queries):
        query = item["query"]
        relevant = set(item["relevant_docs"])

        results = retrieve(query, top_k=top_k)
        retrieved_titles = {r["page_title"] for r in results}

    
        if relevant & retrieved_titles:
            hits += 1
        else:
            misses += 1
            miss_details.append({
                "query": query,
                "expected": list(relevant),
                "got": [r["page_title"] for r in results],
            })
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(queries)}] Running...")

    total = len(queries)
    precision = hits / total * 100

    print(f"\n{'='*55}")
    print(f"  EVAL RESULTS — Top-{top_k} Precision")
    print(f"{'='*55}")
    print(f"  Total queries:  {total}")
    print(f"  Hits:           {hits}")
    print(f"  Misses:         {misses}")
    print(f"  Precision:      {precision:.1f}%")
    print(f"{'='*55}")

    if miss_details:
        print(f"\nMissed queries ({len(miss_details)}):")
        for m in miss_details:
            print(f"  Query: '{m['query']}'")
            print(f"    Expected: {m['expected']}")
            print(f"    Got:      {m['got']}")

    return {
        "total": total,
        "hits": hits,
        "misses": misses,
        "precision": precision,
        "miss_details": miss_details,
    }


if __name__ == "__main__":
    results = run_eval(top_k=3)

    with open("evals/eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to evals/eval_results.json")