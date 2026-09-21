import httpx
from typing import Dict, Any, Optional

API_BASE_URL = "http://localhost:8000"

class ExperimentTracker:
    """
    API client for standardized experiment tracking to our FastAPI backend.
    """
    
    def __init__(self, project_name: str, experiment_name: str, model_version_id: int):
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.model_version_id = model_version_id
        
        self.client = httpx.Client(base_url=API_BASE_URL, timeout=10.0)
        self.project_id = self._get_or_create_project()
        self.experiment_id = self._get_or_create_experiment()
        
        self.run_id = None
        self.metrics = {}
        self.history = []
        self.failure_cases = []

    def _get_or_create_project(self) -> int:
        res = self.client.get("/projects/")
        for p in res.json():
            if p["name"] == self.project_name:
                return p["id"]
        
        res = self.client.post("/projects/", json={"name": self.project_name, "description": "Auto-created by tracker"})
        return res.json()["id"]
        
    def _get_or_create_experiment(self) -> int:
        res = self.client.get(f"/projects/{self.project_id}/experiments")
        for e in res.json():
            if e["name"] == self.experiment_name:
                return e["id"]
                
        res = self.client.post(f"/projects/{self.project_id}/experiments", json={"name": self.experiment_name})
        return res.json()["id"]

    def start_run(self, mixture_config: Dict[str, Any]) -> None:
        """Initialize an ExperimentRun."""
        print(f"Started tracking run for experiment {self.experiment_name}")
        payload = {
            "model_version_id": self.model_version_id,
            "mixture_config": mixture_config,
            "metrics": {}
        }
        res = self.client.post(f"/experiments/{self.experiment_id}/runs", json=payload)
        self.run_id = res.json()["id"]

    def log_metric(self, step: int, metrics: Dict[str, float]) -> None:
        """Log evaluation metrics (accuracy, loss, etc.) at a given step."""
        metrics["step"] = step
        self.history.append(metrics)
        for k, v in metrics.items():
            self.metrics[k] = v
            
        if self.run_id:
            self.client.patch(f"/experiments/runs/{self.run_id}", json={"metrics": self.metrics})

    def log_failure(self, expected: str, predicted: str, meta: Optional[Dict] = None) -> None:
        """Log a failure case for analysis."""
        case = {"expected": expected, "predicted": predicted}
        if meta:
            case.update(meta)
        self.failure_cases.append(case)

    def end_run(self) -> None:
        """Finalize the current tracking run, saving final metrics."""
        if self.failure_cases:
            self.metrics["failures"] = self.failure_cases[:50] # save top 50
            if self.run_id:
                self.client.patch(f"/experiments/runs/{self.run_id}", json={"metrics": self.metrics})
        
        print("Run complete. Final metrics:", {k: v for k, v in self.metrics.items() if k != 'failures'})
