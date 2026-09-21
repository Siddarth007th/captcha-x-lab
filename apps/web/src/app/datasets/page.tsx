import { Database, ExternalLink, Image as ImageIcon, Mic, BrainCircuit, Activity } from "lucide-react";
import Link from "next/link";

const datasets = [
  {
    id: "vision",
    name: "IIIT-5K",
    type: "Image Text Recognition",
    pipeline: "Vision (TrOCR)",
    icon: <ImageIcon size={24} className="text-blue-500" />,
    description: "A dataset of 5,000 cropped text images from real-world scenes. Includes varied typography, cursive text, and noisy backgrounds.",
    source: "HuggingFace",
    url: "https://huggingface.co/datasets/HuggingFaceM4/IIIT-5K",
    stats: {
      split: "Test (Evaluation)",
      samples: "50 samples tested",
      performance: "28.0% Accuracy",
    },
    bg: "bg-blue-500/10",
    border: "border-blue-500/20"
  },
  {
    id: "audio",
    name: "LibriSpeech (Clean)",
    type: "Speech Recognition",
    pipeline: "Audio (Whisper)",
    icon: <Mic size={24} className="text-emerald-500" />,
    description: "A corpus of approximately 1,000 hours of 16kHz read English speech. Used as the primary benchmark for the ASR pipeline.",
    source: "HuggingFace",
    url: "https://huggingface.co/datasets/openslr/librispeech_asr",
    stats: {
      split: "Test (Evaluation)",
      samples: "Tested via TTS proxy",
      performance: "0.0% Accuracy (Backend Issue)",
    },
    bg: "bg-emerald-500/10",
    border: "border-emerald-500/20"
  },
  {
    id: "reasoning",
    name: "VQA-RAD",
    type: "Visual Question Answering",
    pipeline: "Reasoning (ViLT)",
    icon: <BrainCircuit size={24} className="text-purple-500" />,
    description: "A dataset of clinical questions asked by radiologists on radiology images. Used to test domain-specific abstract reasoning.",
    source: "HuggingFace",
    url: "https://huggingface.co/datasets/flaviagiammarino/vqa-rad",
    stats: {
      split: "Test (Evaluation)",
      samples: "50 samples tested",
      performance: "18.0% Accuracy",
    },
    bg: "bg-purple-500/10",
    border: "border-purple-500/20"
  }
];

export default function DatasetsPage() {
  return (
    <div className="space-y-8 animate-in fade-in duration-500 h-full flex flex-col pb-10">
      <div>
        <h1 className="text-3xl font-bold tracking-tight flex items-center gap-3">
          <Database className="text-primary" size={28} />
          Evaluation Datasets
        </h1>
        <p className="text-muted-foreground mt-2 max-w-2xl">
          The following real-world datasets are used to evaluate the accuracy and robustness of the CAPTCHA-X Lab pipelines. Synthetic toy datasets have been deprecated in favor of these HuggingFace benchmarks.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {datasets.map((ds) => (
          <div key={ds.id} className={`flex flex-col bg-card border rounded-xl overflow-hidden shadow-sm hover:shadow-md transition-all duration-200 hover:-translate-y-1 ${ds.border}`}>
            <div className={`p-6 ${ds.bg} flex items-center gap-4 border-b border-inherit`}>
              <div className="p-3 bg-background/80 backdrop-blur-sm rounded-lg shadow-sm">
                {ds.icon}
              </div>
              <div>
                <h3 className="font-semibold text-lg">{ds.name}</h3>
                <p className="text-sm font-medium opacity-80">{ds.pipeline}</p>
              </div>
            </div>
            
            <div className="p-6 flex-1 flex flex-col">
              <div className="mb-4">
                <span className="inline-block px-2.5 py-0.5 rounded-full bg-secondary text-secondary-foreground text-xs font-medium mb-3">
                  {ds.type}
                </span>
                <p className="text-sm text-muted-foreground leading-relaxed">
                  {ds.description}
                </p>
              </div>
              
              <div className="mt-auto space-y-4">
                <div className="bg-muted/50 rounded-lg p-4 space-y-2 border border-border/50">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-muted-foreground font-medium">Split</span>
                    <span className="font-semibold">{ds.stats.split}</span>
                  </div>
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-muted-foreground font-medium">Samples</span>
                    <span className="font-semibold">{ds.stats.samples}</span>
                  </div>
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-muted-foreground font-medium flex items-center gap-1.5"><Activity size={14}/> Baseline Perf.</span>
                    <span className="font-semibold">{ds.stats.performance}</span>
                  </div>
                </div>
                
                <Link 
                  href={ds.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center justify-center gap-2 w-full py-2.5 px-4 bg-secondary/50 hover:bg-secondary text-secondary-foreground text-sm font-medium rounded-lg transition-colors border border-border/50"
                >
                  View on {ds.source}
                  <ExternalLink size={14} />
                </Link>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
