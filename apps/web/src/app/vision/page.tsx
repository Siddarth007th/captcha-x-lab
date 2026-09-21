import { fetchProjects, fetchExperiments, fetchRuns } from "../../lib/api";
import { MetricCard } from "../../components/MetricCard";
import { Eye, Target, Zap, FileText, AlertTriangle } from "lucide-react";
import { CustomBarChart, CustomLineChart } from "../../components/Charts";
import { FailureList } from "../../components/FailureList";

export default async function VisionPage() {
  const projects = await fetchProjects();
  let visionExperiments: any[] = [];
  
  for (const p of projects) {
    const exps = await fetchExperiments(p.id);
    const vExps = exps.filter(e => e.name.toLowerCase().includes('vision') || e.name.toLowerCase().includes('trocr'));
    
    for (const exp of vExps) {
      const runs = await fetchRuns(exp.id);
      runs.forEach(r => {
        visionExperiments.push({
          expName: exp.name,
          run: r
        });
      });
    }
  }

  const standardRuns: any[] = [];
  const oodRuns: any[] = [];

  for (const item of visionExperiments) {
    const run = item.run;
    if (run.status !== 'completed' || !run.metrics) continue;

    if (run.mixture_config?.is_ood) {
      oodRuns.push(run);
    } else {
      standardRuns.push(run);
    }
  }

  // Find the latest standard run to show key metrics
  let latestRun = null;
  const chartData = [];
  
  // Sort runs by fraction for the chart
  standardRuns.sort((a, b) => {
    const fA = a.mixture_config?.train_fraction || 0;
    const fB = b.mixture_config?.train_fraction || 0;
    return fA - fB;
  });

  if (standardRuns.length > 0) {
    for (const run of standardRuns) {
      if (!latestRun || new Date(run.created_at) > new Date(latestRun.created_at)) {
        latestRun = run;
      }
      const fraction = run.mixture_config?.train_fraction || 1.0;
      chartData.push({
        name: `${(fraction * 100).toFixed(0)}%`,
        cer: run.metrics?.test_cer || 0,
        wer: run.metrics?.test_wer || 0,
      });
    }
  }

  // Find the latest OOD run
  let latestOODRun = null;
  if (oodRuns.length > 0) {
    for (const run of oodRuns) {
      if (!latestOODRun || new Date(run.created_at) > new Date(latestOODRun.created_at)) {
        latestOODRun = run;
      }
    }
  }

  const metrics = latestRun?.metrics || {};
  const failures = metrics.failures || [];
  
  const oodMetrics = latestOODRun?.metrics || {};
  const oodFailures = oodMetrics.failures || [];
  
  const oodChartData = latestRun && latestOODRun ? [
    { name: 'Baseline (In-Dist)', cer: metrics.test_cer, wer: metrics.test_wer },
    { name: 'OOD (Robustness)', cer: oodMetrics.test_cer, wer: oodMetrics.test_wer }
  ] : [];

  return (
    <div className="space-y-8 animate-in fade-in duration-500">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Vision Model Performance</h1>
        <p className="text-muted-foreground mt-2">OCR scaling metrics and failure analysis.</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricCard 
          title="Test CER (Latest)" 
          value={metrics.test_cer !== undefined ? metrics.test_cer.toFixed(4) : 'N/A'} 
          icon={<Target size={24} />} 
          subtitle="Character Error Rate"
        />
        <MetricCard 
          title="Test WER (Latest)" 
          value={metrics.test_wer !== undefined ? metrics.test_wer.toFixed(4) : 'N/A'} 
          icon={<FileText size={24} />} 
          subtitle="Word Error Rate"
        />
        <MetricCard 
          title="Exact Match (Latest)" 
          value={metrics.test_exact_match !== undefined ? (metrics.test_exact_match * 100).toFixed(1) + '%' : 'N/A'} 
          icon={<Eye size={24} />} 
        />
        <MetricCard 
          title="Avg Latency" 
          value={metrics.avg_inference_latency_sec !== undefined ? `${(metrics.avg_inference_latency_sec * 1000).toFixed(0)} ms` : 'N/A'} 
          icon={<Zap size={24} />} 
          subtitle="Per Sample"
        />
      </div>

      {chartData.length > 0 && (
        <div className="mt-8">
          <h2 className="text-xl font-semibold mb-4">Error Rates by Dataset Scale</h2>
          <div className="bg-card border border-border rounded-xl p-6 shadow-sm">
            <CustomLineChart 
              data={chartData} 
              xKey="name" 
              lines={[
                { key: 'cer', color: '#3b82f6', name: 'CER' },
                { key: 'wer', color: '#8b5cf6', name: 'WER' }
              ]} 
            />
          </div>
        </div>
      )}

      {latestOODRun && (
        <div className="mt-8">
          <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <AlertTriangle className="text-amber-500" size={24} /> Robustness & OOD Analysis
          </h2>
          <p className="text-muted-foreground mb-4">Condition: {latestOODRun.mixture_config?.ood_condition || 'Unknown'}</p>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
            <MetricCard 
              title="OOD CER" 
              value={oodMetrics.test_cer !== undefined ? oodMetrics.test_cer.toFixed(4) : 'N/A'} 
              icon={<Target size={24} />} 
              subtitle={`Baseline: ${metrics.test_cer?.toFixed(4) || 'N/A'}`}
            />
            <MetricCard 
              title="OOD WER" 
              value={oodMetrics.test_wer !== undefined ? oodMetrics.test_wer.toFixed(4) : 'N/A'} 
              icon={<FileText size={24} />} 
              subtitle={`Baseline: ${metrics.test_wer?.toFixed(4) || 'N/A'}`}
            />
            <MetricCard 
              title="OOD Exact Match" 
              value={oodMetrics.test_exact_match !== undefined ? (oodMetrics.test_exact_match * 100).toFixed(1) + '%' : 'N/A'} 
              icon={<Eye size={24} />} 
              subtitle={`Baseline: ${metrics.test_exact_match ? (metrics.test_exact_match * 100).toFixed(1) + '%' : 'N/A'}`}
            />
          </div>
          {oodChartData.length > 0 && (
            <div className="bg-card border border-border rounded-xl p-6 shadow-sm mb-6">
              <CustomBarChart 
                data={oodChartData} 
                xKey="name" 
                bars={[
                  { key: 'cer', color: '#3b82f6', name: 'CER' },
                  { key: 'wer', color: '#8b5cf6', name: 'WER' }
                ]} 
              />
            </div>
          )}
          <h3 className="text-lg font-semibold mb-3">OOD Failure Cases</h3>
          <FailureList failures={oodFailures} />
        </div>
      )}

      <div className="mt-8">
        <h2 className="text-xl font-semibold mb-4">Baseline Failure Analysis</h2>
        <FailureList failures={failures} />
      </div>
    </div>
  );
}

