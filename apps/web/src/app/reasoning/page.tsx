import { fetchProjects, fetchExperiments, fetchRuns } from "../../lib/api";
import { MetricCard } from "../../components/MetricCard";
import { BrainCircuit, CheckSquare, ListTree, Zap, AlertTriangle } from "lucide-react";
import { CustomBarChart, CustomLineChart } from "../../components/Charts";
import { FailureList } from "../../components/FailureList";

export default async function ReasoningPage() {
  const projects = await fetchProjects();
  let reasoningExperiments: any[] = [];
  
  for (const p of projects) {
    const exps = await fetchExperiments(p.id);
    const rExps = exps.filter(e => e.name.toLowerCase().includes('reasoning') || e.name.toLowerCase().includes('vilt') || e.name.toLowerCase().includes('vqa'));
    
    for (const exp of rExps) {
      const runs = await fetchRuns(exp.id);
      runs.forEach(r => {
        reasoningExperiments.push({
          expName: exp.name,
          run: r
        });
      });
    }
  }

  const standardRuns: any[] = [];
  const oodRuns: any[] = [];

  for (const item of reasoningExperiments) {
    const run = item.run;
    if (run.status !== 'completed' || !run.metrics) continue;

    if (run.mixture_config?.is_ood) {
      oodRuns.push(run);
    } else {
      standardRuns.push(run);
    }
  }

  // Find the latest standard run
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
        acc: run.metrics?.test_accuracy || 0,
        exactMatch: run.metrics?.test_exact_match || 0
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
    { name: 'Baseline (In-Dist)', acc: metrics.test_accuracy, exactMatch: metrics.test_exact_match },
    { name: 'OOD (Robustness)', acc: oodMetrics.test_accuracy, exactMatch: oodMetrics.test_exact_match }
  ] : [];

  return (
    <div className="space-y-8 animate-in fade-in duration-500">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Reasoning Model Performance</h1>
        <p className="text-muted-foreground mt-2">VQA scaling metrics and failure analysis.</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricCard 
          title="VQA Accuracy (Latest)" 
          value={metrics.test_accuracy !== undefined ? (metrics.test_accuracy * 100).toFixed(1) + '%' : 'N/A'} 
          icon={<BrainCircuit size={24} />} 
          subtitle="Test Set"
        />
        <MetricCard 
          title="Exact Match (Latest)" 
          value={metrics.test_exact_match !== undefined ? (metrics.test_exact_match * 100).toFixed(1) + '%' : 'N/A'} 
          icon={<CheckSquare size={24} />} 
        />
        <MetricCard 
          title="Category Perf (Unknown)" 
          value={metrics.cat_acc_unknown !== undefined ? (metrics.cat_acc_unknown * 100).toFixed(1) + '%' : 'N/A'} 
          icon={<ListTree size={24} />} 
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
          <h2 className="text-xl font-semibold mb-4">Accuracy by Dataset Scale</h2>
          <div className="bg-card border border-border rounded-xl p-6 shadow-sm">
            <CustomLineChart 
              data={chartData} 
              xKey="name" 
              lines={[
                { key: 'acc', color: '#10b981', name: 'Accuracy' },
                { key: 'exactMatch', color: '#6366f1', name: 'Exact Match' }
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
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
            <MetricCard 
              title="OOD Accuracy" 
              value={oodMetrics.test_accuracy !== undefined ? (oodMetrics.test_accuracy * 100).toFixed(1) + '%' : 'N/A'} 
              icon={<BrainCircuit size={24} />} 
              subtitle={`Baseline: ${metrics.test_accuracy ? (metrics.test_accuracy * 100).toFixed(1) + '%' : 'N/A'}`}
            />
            <MetricCard 
              title="OOD Exact Match" 
              value={oodMetrics.test_exact_match !== undefined ? (oodMetrics.test_exact_match * 100).toFixed(1) + '%' : 'N/A'} 
              icon={<CheckSquare size={24} />} 
              subtitle={`Baseline: ${metrics.test_exact_match ? (metrics.test_exact_match * 100).toFixed(1) + '%' : 'N/A'}`}
            />
          </div>
          {oodChartData.length > 0 && (
            <div className="bg-card border border-border rounded-xl p-6 shadow-sm mb-6">
              <CustomBarChart 
                data={oodChartData} 
                xKey="name" 
                bars={[
                  { key: 'acc', color: '#10b981', name: 'Accuracy' },
                  { key: 'exactMatch', color: '#6366f1', name: 'Exact Match' }
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
