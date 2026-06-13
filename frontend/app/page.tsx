"use client";

import { useState } from "react";
import dynamic from "next/dynamic";

const MonacoEditor = dynamic(() => import("@monaco-editor/react"), {
  ssr: false,
});

const DEFAULT_CODE = `import torch
import torch.nn as nn
from torch.utils.data import DataLoader

model = nn.Linear(10, 1)
model.eval()

dataset = []
loader = DataLoader(dataset, batch_size=32)

for batch in loader:
    out = model(batch)
    loss = out.sum()
    print(loss.item())

    x = torch.zeros(10)
`;

const SEVERITY_COLORS: Record<string, string> = {
  warning: "text-yellow-400 bg-yellow-400/10 border-yellow-400/30",
  error: "text-red-400 bg-red-400/10 border-red-400/30",
};

const CODE_LABELS: Record<string, string> = {
  NO_GRAD_MISSING: "Missing no_grad",
  DATALOADER_NO_WORKERS: "DataLoader Workers",
  ITEM_IN_LOOP: ".item() in Loop",
  TENSOR_NO_DEVICE: "No Device Arg",
};

export default function Home() {
  const [code, setCode] = useState(DEFAULT_CODE);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  async function handleAnalyze() {
    setLoading(true);
    setResult(null);
    setError(null);

    try {
      const response = await fetch("http://127.0.0.1:8000/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          source_code: code,
          model_type: "simple_cnn",
          batch_size: 4,
        }),
      });

      if (!response.ok) throw new Error(`API error: ${response.status}`);
      const data = await response.json();
      setResult(data);
    } catch (err: any) {
      setError(err.message || "Something went wrong");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen p-8 max-w-5xl mx-auto">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white mb-1">
          Kernel<span className="text-blue-400">Pilot</span>
        </h1>
        <p className="text-gray-500 text-sm">
          AI-powered PyTorch bottleneck diagnosis
        </p>
      </div>

      <div className="mb-4">
        <label className="text-xs text-gray-500 uppercase tracking-widest mb-2 block">
          PyTorch Code
        </label>
        <div className="rounded-lg overflow-hidden border border-gray-800">
          <MonacoEditor
            height="300px"
            language="python"
            theme="vs-dark"
            value={code}
            onChange={(val) => setCode(val || "")}
            options={{
              fontSize: 13,
              minimap: { enabled: false },
              scrollBeyondLastLine: false,
              lineNumbers: "on",
              renderLineHighlight: "none",
              overviewRulerBorder: false,
            }}
          />
        </div>
      </div>

      <button
        onClick={handleAnalyze}
        disabled={loading || !code.trim()}
        className="mb-10 px-5 py-2 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-800 disabled:text-gray-600 text-white text-sm font-medium rounded-lg transition-colors"
      >
        {loading ? "Analyzing..." : "Analyze"}
      </button>

      {loading && (
        <div className="border border-gray-800 rounded-lg p-8 text-center mb-6">
          <div className="inline-block w-5 h-5 border-2 border-blue-400 border-t-transparent rounded-full animate-spin mb-3" />
          <p className="text-gray-500 text-sm">
            Running static analysis, profiler, and GPU telemetry in parallel...
          </p>
        </div>
      )}

      {error && (
        <div className="border border-red-900 bg-red-950/50 rounded-lg p-4 text-red-400 text-sm mb-6">
          {error}
        </div>
      )}

      {result && (
        <div className="space-y-6">
          <div className="flex items-center gap-4 text-sm text-gray-400 border-b border-gray-800 pb-4">
            <span className="text-green-400 font-medium">✓ Analysis complete</span>
            <span>{result.total_time_seconds?.toFixed(2)}s</span>
            <span className="text-gray-600">·</span>
            <span>{result.static_issues?.length} static issues</span>
            <span className="text-gray-600">·</span>
            <span>Bottleneck: <span className="text-white">{result.profiler?.bottleneck_device}</span></span>
          </div>

          <section>
            <h2 className="text-xs text-gray-500 uppercase tracking-widest mb-3">
              Static Analysis
            </h2>
            <div className="space-y-2">
              {result.static_issues?.map((issue: any, i: number) => (
                <div
                  key={i}
                  className={`border rounded-lg p-4 ${SEVERITY_COLORS[issue.severity] || SEVERITY_COLORS.warning}`}
                >
                  <div className="flex items-center gap-3 mb-1">
                    <span className="text-xs font-mono font-semibold">
                      {CODE_LABELS[issue.code] || issue.code}
                    </span>
                    {issue.line > 0 && (
                      <span className="text-xs opacity-60">line {issue.line}</span>
                    )}
                  </div>
                  <p className="text-sm opacity-80">{issue.message}</p>
                </div>
              ))}
            </div>
          </section>

          <section>
            <h2 className="text-xs text-gray-500 uppercase tracking-widest mb-3">
              Profiler — Top Operations
            </h2>
            <div className="border border-gray-800 rounded-lg overflow-hidden">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-gray-800 text-gray-500 text-xs">
                    <th className="text-left p-3 font-normal">Operation</th>
                    <th className="text-right p-3 font-normal">CPU (ms)</th>
                    <th className="text-right p-3 font-normal">CUDA (ms)</th>
                    <th className="text-right p-3 font-normal">Calls</th>
                  </tr>
                </thead>
                <tbody>
                  {result.profiler?.top_ops?.map((op: any, i: number) => (
                    <tr
                      key={i}
                      className={`border-b border-gray-800/50 ${i === 0 ? "text-yellow-400" : "text-gray-300"}`}
                    >
                      <td className="p-3 font-mono text-xs">{op.name}</td>
                      <td className="p-3 text-right font-mono text-xs">{op.cpu_time_ms.toFixed(3)}</td>
                      <td className="p-3 text-right font-mono text-xs">{op.cuda_time_ms.toFixed(3)}</td>
                      <td className="p-3 text-right font-mono text-xs">{op.calls}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="text-xs text-gray-600 mt-2">
              Slowest op highlighted in yellow · Bottleneck device: {result.profiler?.bottleneck_device}
            </p>
          </section>

          <section>
            <h2 className="text-xs text-gray-500 uppercase tracking-widest mb-3">
              GPU Telemetry {result.gpu_telemetry?.mock_data && (
                <span className="normal-case text-gray-600 ml-1">(mock — no NVIDIA GPU)</span>
              )}
            </h2>
            <div className="grid grid-cols-3 gap-3">
              {[
                { label: "Peak Memory", value: `${result.gpu_telemetry?.peak_memory_mb?.toFixed(0)} MB` },
                { label: "Avg Utilization", value: `${result.gpu_telemetry?.avg_utilization_pct?.toFixed(1)}%` },
                { label: "Avg Temperature", value: `${result.gpu_telemetry?.avg_temperature_c?.toFixed(1)}°C` },
              ].map((stat) => (
                <div key={stat.label} className="border border-gray-800 rounded-lg p-4">
                  <p className="text-xs text-gray-500 mb-1">{stat.label}</p>
                  <p className="text-xl font-semibold text-white">{stat.value}</p>
                </div>
              ))}
            </div>
            {result.gpu_telemetry?.bottlenecks?.length > 0 && (
              <div className="mt-3 space-y-2">
                {result.gpu_telemetry.bottlenecks.map((b: string, i: number) => (
                  <div key={i} className="border border-orange-900 bg-orange-950/30 rounded-lg p-3 text-orange-400 text-xs">
                    {b}
                  </div>
                ))}
              </div>
            )}
          </section>

          {result.rag_result?.explanation && (
            <section>
              <h2 className="text-xs text-gray-500 uppercase tracking-widest mb-3">
                AI-Generated Fix
              </h2>
              <div className="border border-blue-900 bg-blue-950/20 rounded-lg p-5 space-y-4">
                <div>
                  <p className="text-xs text-blue-400 uppercase tracking-wide mb-2">What's wrong</p>
                  <p className="text-sm text-gray-300 leading-relaxed">{result.rag_result.explanation}</p>
                </div>
                {result.rag_result.suggested_fix && (
  <div>
    <p className="text-xs text-blue-400 uppercase tracking-wide mb-2">How to fix it</p>
    <div className="text-sm text-gray-300 leading-relaxed whitespace-pre-wrap font-mono bg-gray-900/50 rounded p-3">
      {result.rag_result.suggested_fix.replace(/```python\n?/g, '').replace(/```\n?/g, '')}
    </div>
  </div>
)}
                {result.rag_result.priority && (
                  <div>
                    <p className="text-xs text-blue-400 uppercase tracking-wide mb-2">Fix this first</p>
                    <p className="text-sm text-gray-300 leading-relaxed">{result.rag_result.priority}</p>
                  </div>
                )}
              </div>
            </section>
          )}

          {result.rag_result?.relevant_docs?.length > 0 && (
            <section>
              <h2 className="text-xs text-gray-500 uppercase tracking-widest mb-3">
                Source Documentation
              </h2>
              <div className="space-y-2">
                {result.rag_result.relevant_docs.map((doc: any, i: number) => (
                  <a
                    key={i}
                    href={doc.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center justify-between border border-gray-800 rounded-lg p-3 hover:border-gray-600 transition-colors group"
                  >
                    <div>
                      <p className="text-sm text-gray-300 group-hover:text-white transition-colors">{doc.title}</p>
                      <p className="text-xs text-gray-600 mt-0.5">{doc.url}</p>
                    </div>
                    <span className="text-xs text-gray-600 font-mono ml-4 shrink-0">
                      {(doc.relevance_score * 100).toFixed(0)}% match
                    </span>
                  </a>
                ))}
              </div>
            </section>
          )}

        </div>
      )}
    </main>
  );
}
