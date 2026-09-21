import { fetchProjects, fetchExperiments, fetchRuns } from "../../lib/api";
import { CustomBarChart } from "../../components/Charts";
import { ExportDataButton } from "../../components/ExportDataButton";

export default async function ExperimentsPage() {
  const projects = await fetchProjects();
  let allRuns: any[] = [];
  
  for (const p of projects) {
    const exps = await fetchExperiments(p.id);
    for (const exp of exps) {
      const runs = await fetchRuns(exp.id);
      runs.forEach(run => {
        allRuns.push({
          id: run.id,
          expName: exp.name,
          date: new Date(run.created_at).toLocaleDateString(),
          metrics: run.metrics || {},
          fraction: run.mixture_config?.train_fraction || 1.0,
          model: run.mixture_config?.model_architecture || "unknown",
          status: run.status
        });
      });
    }
  }

  // Filter out runs that are not completed
  const completedRuns = allRuns.filter(r => r.status === 'completed' && Object.keys(r.metrics).length > 0);

  // Sort by fraction
  completedRuns.sort((a, b) => a.fraction - b.fraction);

  // Example comparison logic: comparing Validation metrics across all runs
  const comparisonData = completedRuns.map(r => ({
    name: r.expName.substring(0, 20),
    trainLoss: r.metrics.train_loss || 0,
    valMetric: r.metrics.test_cer || r.metrics.test_wer || r.metrics.test_accuracy || 0,
    trainingTime: r.metrics.training_time_sec || 0
  }));

  return (
    <div className="space-y-8 animate-in fade-in duration-500">
      <div className="flex justify-between items-end">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Experiment Comparisons</h1>
          <p className="text-muted-foreground mt-2">Ablation studies, scaling effects, and cross-run metrics.</p>
        </div>
        <ExportDataButton />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div className="bg-card border border-border rounded-xl p-6 shadow-sm">
          <h2 className="text-xl font-semibold mb-4">Cross-Model Test Metric vs Train Loss</h2>
          <CustomBarChart 
            data={comparisonData} 
            xKey="name" 
            bars={[
              { key: 'trainLoss', color: '#ef4444', name: 'Train Loss' },
              { key: 'valMetric', color: '#3b82f6', name: 'Test Metric (WER/CER/Acc)' }
            ]} 
          />
        </div>
        <div className="bg-card border border-border rounded-xl p-6 shadow-sm">
          <h2 className="text-xl font-semibold mb-4">Training Time (seconds)</h2>
          <CustomBarChart 
            data={comparisonData} 
            xKey="name" 
            bars={[
              { key: 'trainingTime', color: '#10b981', name: 'Training Time (sec)' }
            ]} 
          />
        </div>
      </div>

      <div className="mt-8">
        <h2 className="text-xl font-semibold mb-4">Run Log & Metadata</h2>
        <div className="bg-card border border-border rounded-xl shadow-sm overflow-hidden">
          <table className="w-full text-left text-sm text-muted-foreground">
            <thead className="bg-muted text-xs uppercase text-foreground">
              <tr>
                <th className="px-6 py-4 font-medium">Experiment</th>
                <th className="px-6 py-4 font-medium">Scale</th>
                <th className="px-6 py-4 font-medium">Train Loss</th>
                <th className="px-6 py-4 font-medium">Test Metric</th>
                <th className="px-6 py-4 font-medium">Latency</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {allRuns.map(run => (
                <tr key={run.id} className="hover:bg-muted/50 transition-colors">
                  <td className="px-6 py-4 font-medium text-foreground">{run.expName}</td>
                  <td className="px-6 py-4">{(run.fraction * 100).toFixed(0)}%</td>
                  <td className="px-6 py-4">{run.metrics.train_loss?.toFixed(4) || 'N/A'}</td>
                  <td className="px-6 py-4">
                    {run.metrics.test_cer?.toFixed(4) || run.metrics.test_wer?.toFixed(4) || run.metrics.test_accuracy?.toFixed(4) || 'N/A'}
                  </td>
                  <td className="px-6 py-4">{run.metrics.avg_inference_latency_sec ? `${(run.metrics.avg_inference_latency_sec * 1000).toFixed(0)} ms` : 'N/A'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
