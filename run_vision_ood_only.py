import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'ml')))
from vision.evaluation.evaluate_ood import run_ood_evaluation

for ablation in ["rotation", "blur", "jitter"]:
    run_ood_evaluation(ablation=ablation)
