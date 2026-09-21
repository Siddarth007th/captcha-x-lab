"use client";

import { useState } from "react";
import { UploadCloud, CheckCircle2, AlertCircle, Loader2 } from "lucide-react";
import { MetricCard } from "../../components/MetricCard";
import { API_BASE_URL } from "../../lib/api";

export default function MasterPage() {
  const [file, setFile] = useState<File | null>(null);
  const [textQuery, setTextQuery] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      setFile(e.dataTransfer.files[0]);
    }
  };

  const handlePredict = async () => {
    if (!file) return;
    setIsLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);
    if (textQuery) {
      formData.append("text_query", textQuery);
    }

    try {
      const res = await fetch(`${API_BASE_URL}/inference/predict`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || "Inference failed");
      }

      const data = await res.json();
      if (data.success === false) {
        throw new Error(data.error_information || "Model inference failed inside Master Router");
      }
      setResult(data);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="space-y-8 animate-in fade-in duration-500 h-full flex flex-col">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Master Router (Live Inference)</h1>
        <p className="text-muted-foreground mt-2">Test the orchestrated ensemble model on raw inputs.</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 flex-1">
        
        {/* Upload Section */}
        <div className="bg-card border border-border rounded-xl p-8 flex flex-col shadow-sm h-[500px]">
          <h2 className="text-xl font-semibold mb-4">Input Challenge</h2>
          
          <div 
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleDrop}
            className="flex-1 border-2 border-dashed border-border hover:border-primary/50 transition-colors rounded-xl flex flex-col items-center justify-center text-center p-6 cursor-pointer bg-muted/30"
            onClick={() => document.getElementById("file-upload")?.click()}
          >
            <input 
              id="file-upload" 
              type="file" 
              className="hidden" 
              onChange={(e) => e.target.files && setFile(e.target.files[0])} 
            />
            <UploadCloud size={48} className="text-muted-foreground mb-4" />
            <h3 className="text-lg font-medium">Drag & Drop test sample here</h3>
            <p className="text-sm text-muted-foreground mt-2">Supports Images (.png, .jpg) and Audio (.wav)</p>
            {file && <p className="mt-4 font-mono text-primary bg-primary/10 px-3 py-1 rounded-md">{file.name}</p>}
          </div>

          <div className="mt-6">
            <label className="block text-sm font-medium text-muted-foreground mb-2">Optional Text Query (for Reasoning models)</label>
            <input 
              type="text" 
              placeholder="e.g. 'What number is in the image?'"
              value={textQuery}
              onChange={(e) => setTextQuery(e.target.value)}
              className="w-full bg-background border border-border rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-primary/50"
            />
          </div>

          <button 
            disabled={!file || isLoading}
            onClick={handlePredict}
            className="mt-6 w-full bg-primary hover:bg-primary/90 disabled:opacity-50 text-primary-foreground font-medium py-3 rounded-lg flex items-center justify-center transition-colors"
          >
            {isLoading ? <Loader2 className="animate-spin mr-2" /> : "Run Inference"}
          </button>
        </div>

        {/* Results Section */}
        <div className="bg-card border border-border rounded-xl p-8 shadow-sm h-[500px] flex flex-col overflow-y-auto">
          <h2 className="text-xl font-semibold mb-6">Prediction Results</h2>
          
          {!result && !error && !isLoading && (
            <div className="flex-1 flex flex-col items-center justify-center text-center text-muted-foreground">
              <CheckCircle2 size={48} className="mb-4 opacity-20" />
              <p>Upload a challenge and click Run Inference.</p>
              <p className="text-sm mt-2 opacity-70">The Master Router will lazy-load the appropriate model on the first request.</p>
            </div>
          )}

          {isLoading && (
            <div className="flex-1 flex flex-col items-center justify-center text-center text-primary">
              <Loader2 size={48} className="animate-spin mb-4" />
              <p className="font-medium animate-pulse">Routing request and performing inference...</p>
              <p className="text-xs mt-2 text-muted-foreground">Note: First requests may take 5-10 seconds as the model loads into memory.</p>
            </div>
          )}

          {error && (
            <div className="p-4 bg-red-500/10 border border-red-500/20 rounded-lg flex items-start text-red-500">
              <AlertCircle className="mr-3 mt-0.5 flex-shrink-0" size={20} />
              <div>
                <h4 className="font-medium">Inference Failed</h4>
                <p className="text-sm mt-1">{error}</p>
              </div>
            </div>
          )}

          {result && !isLoading && (
            <div className="space-y-6">
              <div className="grid grid-cols-2 gap-4">
                <MetricCard title="Modality Detected" value={result.challenge_type} />
                <MetricCard title="Routed Model" value={result.model_used} />
              </div>
              
              <div className="p-6 bg-background rounded-lg border border-border">
                <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider mb-3">Final Prediction</h3>
                <p className="text-3xl font-bold tracking-tight text-foreground">{result.prediction}</p>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="p-4 bg-background rounded-lg border border-border">
                  <h3 className="text-xs font-semibold text-muted-foreground uppercase">Confidence</h3>
                  <p className="text-xl font-medium mt-1">{(result.confidence * 100).toFixed(1)}%</p>
                </div>
                <div className="p-4 bg-background rounded-lg border border-border">
                  <h3 className="text-xs font-semibold text-muted-foreground uppercase">Latency</h3>
                  <p className="text-xl font-medium mt-1">{(result.processing_latency * 1000).toFixed(0)} ms</p>
                </div>
              </div>
            </div>
          )}

        </div>
      </div>
    </div>
  );
}
