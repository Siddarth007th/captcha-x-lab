import requests
import time
import sys

BASE_URL = "http://localhost:8000"

def run_experiment(fraction):
    print(f"\n1. Creating Project for {fraction*100}% Reasoning Scale...")
    proj_resp = requests.post(f"{BASE_URL}/projects/", json={"name": f"Phase 9 Reasoning", "description": "Scaling experiments"})
    proj_id = proj_resp.json().get("id") or 1 # Fallback if already exists
    
    print(f"2. Creating Experiment in Project {proj_id}...")
    exp_resp = requests.post(f"{BASE_URL}/projects/{proj_id}/experiments/", json={"name": f"Reasoning-Scale-{fraction*100}pct"})
    exp_id = exp_resp.json()["id"]
    
    print(f"3. Triggering Run for Experiment {exp_id}...")
    run_resp = requests.post(f"{BASE_URL}/experiments/{exp_id}/runs", json={
        "model_version_id": 2,
        "mixture_config": {
            "model_architecture": "reasoning",
            "train_fraction": fraction,
            "num_epochs": 1,
            "batch_size": 1, 
            "device": "mps" # Apple Silicon
        }
    })
    run_id = run_resp.json()["id"]
    print(f"   Run triggered! Run ID: {run_id}")
    
    print("4. Status: queued in Celery.")
    return run_id
        
if __name__ == "__main__":
    fraction = float(sys.argv[1]) if len(sys.argv) > 1 else 0.1
    run_experiment(fraction)
