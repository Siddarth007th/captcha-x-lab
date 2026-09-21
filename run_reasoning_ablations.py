import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'ml')))
from reasoning.training.train import run_training
from reasoning.evaluation.evaluate_ood import run_ood_evaluation

def main():
    device = "mps"
    fractions = [0.25, 0.5]
    
    for frac in fractions:
        print(f"\n======================================")
        print(f"Running Reasoning Ablation: PathVQA Fraction {frac}")
        print(f"======================================")
        
        # Train
        config = {
            "train_fraction": 1.0,
            "num_epochs": 1,
            "device": device,
            "path_vqa_fraction": frac
        }
        run_training(config)
        
        # Evaluate OOD
        run_ood_evaluation(path_vqa_fraction=frac)

if __name__ == "__main__":
    main()
