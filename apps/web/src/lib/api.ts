export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface Project {
  id: number;
  name: string;
  description: string | null;
  created_at: string;
}

export interface Experiment {
  id: number;
  project_id: number;
  name: string;
  status: string;
  created_at: string;
}

export interface ExperimentRun {
  id: number;
  experiment_id: number;
  model_version_id: number;
  mixture_config: any;
  metrics: any;
  created_at: string;
}

export async function fetchProjects(): Promise<Project[]> {
  const res = await fetch(`${API_BASE_URL}/projects/`, { cache: 'no-store' });
  if (!res.ok) throw new Error("Failed to fetch projects");
  return res.json();
}

export async function fetchExperiments(projectId: number): Promise<Experiment[]> {
  const res = await fetch(`${API_BASE_URL}/projects/${projectId}/experiments`, { cache: 'no-store' });
  if (!res.ok) throw new Error("Failed to fetch experiments");
  return res.json();
}

export async function fetchRuns(experimentId: number): Promise<ExperimentRun[]> {
  const res = await fetch(`${API_BASE_URL}/experiments/${experimentId}/runs`, { cache: 'no-store' });
  if (!res.ok) throw new Error("Failed to fetch runs");
  return res.json();
}
