import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'ml')))
from vision.training.train import run_training
from vision.evaluation.evaluate_ood import run_ood_evaluation

def main():
    device = "mps"
    ablations = ["rotation", "blur", "jitter"]
    
    for ablation in ablations:
        print(f"\n======================================")
        print(f"Running Vision Ablation: {ablation}")
        print(f"======================================")
        
        # Train
        config = {
            "train_fraction": 1.0,
            "num_epochs": 1,
            "device": device,
            "ablation": ablation
        }
        run_training(config)
        
        # Evaluate OOD
        run_ood_evaluation(ablation=ablation)

if __name__ == "__main__":
    main()
