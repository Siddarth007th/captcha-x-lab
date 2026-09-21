import { fetchProjects, fetchExperiments, fetchRuns } from "../lib/api";
import { MetricCard } from "../components/MetricCard";
import { Activity, Layers, PlayCircle, BarChart2 } from "lucide-react";
import { CustomBarChart } from "../components/Charts";

export default async function OverviewPage() {
  const projects = await fetchProjects();
  
  // Aggregate stats across all projects
  let totalExperiments = 0;
  let totalRuns = 0;
  let latestRunDate = new Date(0);
  
  const chartData = [];

  for (const project of projects) {
    const experiments = await fetchExperiments(project.id);
    totalExperiments += experiments.length;
    
    for (const exp of experiments) {
      const runs = await fetchRuns(exp.id);
      totalRuns += runs.length;
      
      runs.forEach(run => {
        const runDate = new Date(run.created_at);
        if (runDate > latestRunDate) {
          latestRunDate = runDate;
        }
      });

      // Sample data for overview chart (using training loss as a generic metric if it exists)
      if (runs.length > 0) {
        const latestRun = runs[runs.length - 1];
        chartData.push({
          name: exp.name.length > 15 ? exp.name.substring(0, 15) + '...' : exp.name,
          loss: latestRun.metrics?.train_loss || 0
        });
      }
    }
  }

  const hasRuns = totalRuns > 0;

  return (
    <div className="space-y-8 animate-in fade-in duration-500">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Platform Overview</h1>
        <p className="text-muted-foreground mt-2">Central dashboard for CAPTCHA-X Lab telemetry and model performance.</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricCard 
          title="Active Projects" 
          value={projects.length} 
          icon={<Layers size={24} />} 
        />
        <MetricCard 
          title="Total Experiments" 
          value={totalExperiments} 
          icon={<Activity size={24} />} 
        />
        <MetricCard 
          title="Model Runs" 
          value={totalRuns} 
          icon={<PlayCircle size={24} />} 
        />
        <MetricCard 
          title="Last Updated" 
          value={hasRuns ? latestRunDate.toLocaleDateString() : 'N/A'} 
          subtitle={hasRuns ? latestRunDate.toLocaleTimeString() : ''}
          icon={<BarChart2 size={24} />} 
        />
      </div>

      {hasRuns && (
        <div className="mt-8">
          <h2 className="text-xl font-semibold mb-4">Latest Experiment Losses</h2>
          <div className="bg-card border border-border rounded-xl p-6 shadow-sm">
            <CustomBarChart 
              data={chartData} 
              xKey="name" 
              bars={[{ key: 'loss', color: '#3b82f6', name: 'Training Loss' }]} 
            />
          </div>
        </div>
      )}
    </div>
  );
}
