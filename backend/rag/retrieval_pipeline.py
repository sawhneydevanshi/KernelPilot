import os
from openai import OpenAI
from dotenv import load_dotenv
from backend.rag.vector_store import retrieve

load_dotenv()


def build_queries_from_diagnosis(diagnosis: dict) -> list[str]:
    """
    Take the orchestration report output and build search queries from it.
    
    diagnosis is a dict with keys:
        - static_issues: list of issue dicts
        - profiler: dict with slowest_op, bottleneck_device, warnings
        - gpu_telemetry: dict with bottlenecks
        - summary: list of strings
    """
    queries = []

  
    slowest_op = diagnosis.get("profiler", {}).get("slowest_op", "")
    bottleneck_device = diagnosis.get("profiler", {}).get("bottleneck_device", "")

    if slowest_op and slowest_op != "unknown":
        queries.append(f"{slowest_op} slow performance optimization")

    if bottleneck_device == "CPU":
        queries.append("CPU bottleneck GPU utilization PyTorch performance")
    elif bottleneck_device == "GPU":
        queries.append("GPU memory optimization CUDA PyTorch")


    static_issues = diagnosis.get("static_issues", [])
    for issue in static_issues:
        code = issue.get("code", "")
        if code == "NO_GRAD_MISSING":
            queries.append("torch no_grad inference mode memory saving")
        elif code == "ITEM_IN_LOOP":
            queries.append("tensor item loop GPU CPU sync avoid")
        elif code == "DATALOADER_NO_WORKERS":
            queries.append("DataLoader num_workers parallel loading performance")
        elif code == "TENSOR_NO_DEVICE":
            queries.append("tensor device cuda CPU GPU mismatch")

    
    gpu_bottlenecks = diagnosis.get("gpu_telemetry", {}).get("bottlenecks", [])
    for b in gpu_bottlenecks:
        if "MEMORY_PRESSURE" in b:
            queries.append("GPU out of memory reduce memory usage")
        elif "LOW_GPU_UTILIZATION" in b:
            queries.append("low GPU utilization DataLoader bottleneck")

   
    seen = set()
    unique_queries = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            unique_queries.append(q)

    return unique_queries


def expand_queries_with_llm(queries: list[str]) -> list[str]:
    """
    Use OpenAI to rephrase each query into 1-2 alternative formulations.
    This improves recall by catching different ways of asking the same thing.
    """
    if not queries:
        return []

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    prompt = f"""You are helping expand search queries for a PyTorch documentation retrieval system.

For each query below, generate 1 alternative rephrasing that might retrieve different but relevant results.
Return ONLY the alternative queries, one per line, in the same order as the input.
Do not number them or add any explanation.

Queries:
{chr(10).join(queries)}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300,
        )
        expanded = response.choices[0].message.content.strip().splitlines()
   
        all_queries = queries + [q.strip() for q in expanded if q.strip()]
        seen = set()
        unique = []
        for q in all_queries:
            if q not in seen:
                seen.add(q)
                unique.append(q)
        return unique
    except Exception as e:
        print(f"  [Warning] Query expansion failed: {e}. Using original queries.")
        return queries


def retrieve_for_diagnosis(
    diagnosis: dict,
    top_k_per_query: int = 3,
    max_final_results: int = 5,
    use_query_expansion: bool = True,
) -> list[dict]:
    """
    Full retrieval pipeline:
    1. Build queries from diagnosis
    2. Optionally expand with LLM
    3. Retrieve chunks for each query
    4. Deduplicate and re-rank by best score
    5. Return top results

    Returns list of dicts with: text, source_url, page_title, relevance_score
    """
    
    queries = build_queries_from_diagnosis(diagnosis)
    print(f"  Built {len(queries)} queries from diagnosis")

    if not queries:
        queries = ["PyTorch performance optimization bottleneck"]

   
    if use_query_expansion and len(queries) > 0:
        queries = expand_queries_with_llm(queries)
        print(f"  Expanded to {len(queries)} queries")

   
    all_results: dict[str, dict] = {}  

    for query in queries:
        try:
            results = retrieve(query, top_k=top_k_per_query)
            for r in results:
                key = r["page_title"] 
                if key not in all_results or r["relevance_score"] > all_results[key]["relevance_score"]:
                    all_results[key] = r
        except Exception as e:
            print(f"  [Warning] Retrieval failed for query '{query}': {e}")

   
    ranked = sorted(all_results.values(), key=lambda x: x["relevance_score"], reverse=True)
    final = ranked[:max_final_results]

    print(f"  Retrieved {len(final)} unique relevant chunks")
    return final