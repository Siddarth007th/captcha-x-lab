import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'ml')))
from audio.training.train import run_training
from audio.evaluation.evaluate_ood import run_ood_evaluation

def main():
    device = "mps"
    factors = [0.025, 0.1]
    
    for factor in factors:
        print(f"\n======================================")
        print(f"Running Audio Ablation: Noise Factor {factor}")
        print(f"======================================")
        
        # Train
        config = {
            "train_fraction": 1.0,
            "num_epochs": 1,
            "device": device,
            "noise_factor": factor
        }
        run_training(config)
        
        # Evaluate OOD
        run_ood_evaluation(noise_factor=factor)

if __name__ == "__main__":
    main()
