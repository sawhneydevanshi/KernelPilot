import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def synthesize_fix(
    diagnosis: dict,
    retrieved_docs: list[dict],
) -> dict:
   
    if not retrieved_docs:
        return {
            "explanation": "No relevant documentation found for this diagnosis.",
            "suggested_fix": "Please review the static analysis warnings manually.",
            "relevant_docs": [],
            "model_used": "none",
        }


    doc_context = ""
    for i, doc in enumerate(retrieved_docs[:4]):  
        doc_context += f"\n--- Doc {i+1}: {doc['page_title']} ---\n"
        doc_context += doc["text"][:800]  
        doc_context += "\n"

    static_issues = diagnosis.get("static_issues", [])
    profiler = diagnosis.get("profiler", {})
    gpu = diagnosis.get("gpu_telemetry", {})

    diagnosis_text = "DIAGNOSIS FINDINGS:\n"

    if static_issues:
        diagnosis_text += f"\nStatic Analysis ({len(static_issues)} issues):\n"
        for issue in static_issues:
            diagnosis_text += f"  - [{issue.get('code')}] {issue.get('message', '')[:120]}\n"

    if profiler.get("slowest_op"):
        diagnosis_text += f"\nProfiler:\n"
        diagnosis_text += f"  - Slowest op: {profiler['slowest_op']}\n"
        diagnosis_text += f"  - Bottleneck device: {profiler.get('bottleneck_device', 'unknown')}\n"
        diagnosis_text += f"  - Total CPU time: {profiler.get('total_cpu_time_ms', 0):.2f}ms\n"

    if gpu.get("peak_memory_mb"):
        diagnosis_text += f"\nGPU Telemetry:\n"
        diagnosis_text += f"  - Peak memory: {gpu['peak_memory_mb']:.0f}MB\n"
        diagnosis_text += f"  - Avg utilization: {gpu.get('avg_utilization_pct', 0):.1f}%\n"
        for b in gpu.get("bottlenecks", []):
            diagnosis_text += f"  - {b}\n"

  
    prompt = f"""You are a PyTorch performance expert. A developer ran KernelPilot on their PyTorch code and got the following diagnosis.

{diagnosis_text}

Here is relevant PyTorch documentation to help you explain the issues and fixes:
{doc_context}

Based on the diagnosis and documentation above, provide:

1. EXPLANATION: A clear 2-3 sentence explanation of what is causing the performance issues and why.

2. SUGGESTED FIX: Concrete, specific code changes the developer should make. Include actual code snippets where helpful. Be specific to the issues found, not generic advice.

3. PRIORITY: Which single issue should they fix first for the biggest impact?

Keep your response practical and direct. No fluff."""

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=600,
    )

    raw_response = response.choices[0].message.content.strip()

   
    explanation = _extract_section(raw_response, "EXPLANATION")
    suggested_fix = _extract_section(raw_response, "SUGGESTED FIX")
    priority = _extract_section(raw_response, "PRIORITY")

    if not explanation:
        explanation = raw_response  

    return {
        "explanation": explanation,
        "suggested_fix": suggested_fix,
        "priority": priority,
        "relevant_docs": [
            {
                "title": doc["page_title"],
                "url": doc["source_url"],
                "relevance_score": doc["relevance_score"],
            }
            for doc in retrieved_docs
        ],
        "model_used": "gpt-4o-mini",
        "raw_response": raw_response,
    }


def _extract_section(text: str, section_name: str) -> str:
    lines = text.splitlines()
    capture = False
    result_lines = []
    
    section_keywords = ["EXPLANATION", "SUGGESTED FIX", "PRIORITY"]
    
    for line in lines:
        line_upper = line.upper()
        is_header = any(kw in line_upper for kw in section_keywords)
        
        if is_header:
            if section_name.upper() in line_upper:
                capture = True
                continue  
            elif capture:
                break  
        
        if capture:
            result_lines.append(line)
    
    return "\n".join(result_lines).strip()