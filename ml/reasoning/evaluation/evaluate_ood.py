import torch
import sys
import os
import time
from datasets import load_dataset
from transformers import ViltProcessor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from ml.reasoning.model.baseline import ReasoningBaseline
from ml.shared.evaluation.tracker import ExperimentTracker

def run_ood_evaluation(path_vqa_fraction=None):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"=== Reasoning OOD Evaluation (PathVQA Fraction: {path_vqa_fraction}) ===")
    
    if path_vqa_fraction is not None:
        checkpoint_dir = f"checkpoints/reasoning/vilt_vqarad_pathvqa{int(path_vqa_fraction*100)}_100pct"
    else:
        checkpoint_dir = "checkpoints/reasoning/vilt_vqarad_optimized_100pct"
        
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint {checkpoint_dir} not found. Ensure training is complete.")
        return
        
    model = ReasoningBaseline()
    model.model.from_pretrained(checkpoint_dir)
    model.to(device)
    model.eval()
    processor = ViltProcessor.from_pretrained(checkpoint_dir)
    
    id2label = model.model.config.id2label
    
    print("Loading OOD Dataset (PathVQA subset)...")
    dataset = load_dataset("flaviagiammarino/path-vqa", split="test", streaming=False)
    # Take a small subset to evaluate
    test_data = dataset.shuffle(seed=42).select(range(min(500, len(dataset))))
    
    exp_name = f"Reasoning-OOD-PathVQA-Ablation-{int(path_vqa_fraction*100)}-100pct" if path_vqa_fraction is not None else "Reasoning-OOD-PathVQA-Optimized-100pct"
    tracker = ExperimentTracker(
        project_name="Phase 13: Ablation Analysis",
        experiment_name=exp_name,
        model_version_id=2
    )
    tracker.start_run({
        "dataset": "HuggingFaceM4/PathVQA (subset)",
        "num_test_samples": len(test_data),
        "device": str(device),
        "is_ood": True,
        "ood_condition": "Domain Shift (General Images vs Medical)",
        "path_vqa_fraction": path_vqa_fraction
    })
    
    test_correct = 0
    total_latency = 0.0
    
    print("Evaluating...")
    with torch.no_grad():
        for item in test_data:
            image = item["image"].convert("RGB")
            text = item["question"]
            answer = item.get("answer") or item.get("multiple_choice_answer") or (item.get("answers", [{"answer":""}])[0]["answer"] if item.get("answers") else "")
            
            start_t = time.time()
            encoding = processor(image, text, return_tensors="pt", truncation=True, max_length=40).to(device)
            outputs = model(
                pixel_values=encoding.pixel_values,
                input_ids=encoding.input_ids,
                token_type_ids=encoding.token_type_ids,
                attention_mask=encoding.attention_mask
            )
            idx = outputs.logits.argmax(-1).item()
            pred_answer = id2label[idx]
            total_latency += (time.time() - start_t)
            
            if pred_answer.lower() == str(answer).lower():
                test_correct += 1
            else:
                tracker.log_failure(
                    expected=str(answer), 
                    predicted=pred_answer, 
                    meta={"question": text}
                )
                
    test_acc = test_correct / max(1, len(test_data))
    avg_latency = total_latency / max(1, len(test_data))
    
    final_metrics = {
        "test_accuracy": test_acc, 
        "test_exact_match": test_acc,
        "avg_inference_latency_sec": avg_latency
    }
    
    tracker.log_metric(step=1, metrics=final_metrics)
    tracker.end_run()
    
    print(f"Reasoning OOD Complete. Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    run_ood_evaluation()
