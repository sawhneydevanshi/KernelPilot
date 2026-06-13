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

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err: any) {
      setError(err.message || "Something went wrong");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen p-8 max-w-6xl mx-auto">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white mb-1">
          Kernel<span className="text-blue-400">Pilot</span>
        </h1>
        <p className="text-gray-400 text-sm">
          AI-powered PyTorch bottleneck diagnosis
        </p>
      </div>

      {/* Editor */}
      <div className="mb-4">
        <label className="text-xs text-gray-400 uppercase tracking-wide mb-2 block">
          Paste your PyTorch code
        </label>
        <div className="rounded-lg overflow-hidden border border-gray-700">
          <MonacoEditor
            height="340px"
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

      {/* Analyze Button */}
      <button
        onClick={handleAnalyze}
        disabled={loading || !code.trim()}
        className="mb-8 px-6 py-2.5 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 disabled:text-gray-500 text-white text-sm font-medium rounded-lg transition-colors"
      >
        {loading ? "Analyzing..." : "Analyze"}
      </button>

      {/* Loading State */}
      {loading && (
        <div className="border border-gray-700 rounded-lg p-6 text-center">
          <div className="inline-block w-5 h-5 border-2 border-blue-400 border-t-transparent rounded-full animate-spin mb-3" />
          <p className="text-gray-400 text-sm">
            Running static analysis, profiler, and GPU telemetry in parallel...
          </p>
        </div>
      )}

      {/* Error State */}
      {error && (
        <div className="border border-red-800 bg-red-950 rounded-lg p-4 text-red-400 text-sm">
          {error}
        </div>
      )}

      {/* Results — placeholder for Day 12 */}
      {result && (
        <div className="border border-gray-700 rounded-lg p-6">
          <p className="text-green-400 text-sm font-medium mb-2">
            ✓ Analysis complete in {result.total_time_seconds?.toFixed(2)}s
          </p>
          <p className="text-gray-400 text-sm">
            Found {result.static_issues?.length} static issues. Results display
            coming in Day 12.
          </p>
          <pre className="mt-4 text-xs text-gray-500 overflow-auto max-h-40">
            {JSON.stringify(result, null, 2)}
          </pre>
        </div>
      )}
    </main>
  );
}