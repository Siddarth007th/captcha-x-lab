import torch
import sys
import os
import time
from typing import Callable, Optional, Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from ml.reasoning.data.pipeline import ReasoningPipeline
from ml.reasoning.model.baseline import ReasoningBaseline
from ml.shared.evaluation.tracker import ExperimentTracker

def run_training(config: Dict[str, Any], progress_callback: Optional[Callable[[int, str], bool]] = None) -> Dict[str, Any]:
    """
    Executes a training run for Reasoning (ViLT).
    progress_callback: A function(samples_processed, stage_name) -> bool. If it returns False, training should cancel.
    """
    train_fraction = config.get("train_fraction", 1.0)
    num_epochs = config.get("num_epochs", 1)
    lr = config.get("lr", 5e-5)
    batch_size = config.get("batch_size", 1)
    device = torch.device(config.get("device", "cpu"))
    path_vqa_fraction = config.get("path_vqa_fraction", 0.0)
    is_optimized = config.get("is_optimized", False)
    
    # Backward compatibility
    if is_optimized and path_vqa_fraction == 0.0:
        path_vqa_fraction = 1.0
    
    print(f"=== Reasoning Training Pipeline ({train_fraction*100}% subset, PathVQA: {path_vqa_fraction}) ===")
    pipeline = ReasoningPipeline(
        raw_dir="data/raw/reasoning",
        processed_dir="data/processed/reasoning",
        config={"train_fraction": train_fraction, "seed": config.get("seed", 42), "path_vqa_fraction": path_vqa_fraction}
    )
    train_data, val_data, test_data = pipeline.run_pipeline()
    processor = pipeline.processor
    
    exp_name = f"ViLT-VQARAD-PathVQA{int(path_vqa_fraction*100)}-{train_fraction*100}pct" if path_vqa_fraction > 0 else f"ViLT-VQARAD-Scale-{train_fraction*100}pct"
    tracker = ExperimentTracker(
        project_name="Phase 13: Ablation Analysis",
        experiment_name=exp_name,
        model_version_id=2
    )
    hyperparams = {
        "dataset": f"flaviagiammarino/vqa-rad + PathVQA({path_vqa_fraction})" if path_vqa_fraction > 0 else "flaviagiammarino/vqa-rad",
        "batch_size": batch_size,
        "lr": lr,
        "train_fraction": train_fraction,
        "num_train_samples": len(train_data),
        "num_val_samples": len(val_data),
        "num_test_samples": len(test_data),
        "epochs": num_epochs,
        "device": str(device),
        "path_vqa_fraction": path_vqa_fraction
    }
    tracker.start_run(hyperparams)
    
    model = ReasoningBaseline().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    id2label = model.model.config.id2label
    label2id = model.model.config.label2id
    
    processed = 0
    start_time = time.time()
    
    print("\nStarting Training Loop...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for i, item in enumerate(train_data):
            if i % 10 == 0 and progress_callback:
                if not progress_callback(processed, f"Epoch {epoch+1}: Training"):
                    tracker.end_run()
                    return {"status": "cancelled"}
                    
            image = item["image"].convert("RGB")
            text = item["question"]
            answer = item.get("target_answer") or item.get("multiple_choice_answer")
            
            label_id = label2id.get(answer, 0)
            labels = torch.zeros(len(id2label)).to(device)
            labels[label_id] = 1.0
            labels = labels.unsqueeze(0)
            
            # Using max_length truncation to prevent crash on long sentences from PathVQA
            encoding = processor(image, text, return_tensors="pt", truncation=True, max_length=40).to(device)
            
            optimizer.zero_grad()
            outputs = model(
                pixel_values=encoding.pixel_values,
                input_ids=encoding.input_ids,
                token_type_ids=encoding.token_type_ids,
                attention_mask=encoding.attention_mask,
                labels=labels
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            processed += 1
            
        import math
        avg_train_loss = total_loss / max(1, len(train_data))
        if math.isnan(avg_train_loss):
            avg_train_loss = 0.0
            
        # Validation
        model.eval()
        correct = 0
        with torch.no_grad():
            for i, item in enumerate(val_data):
                if i % 10 == 0 and progress_callback:
                    if not progress_callback(processed, f"Epoch {epoch+1}: Validation"):
                        tracker.end_run()
                        return {"status": "cancelled"}
                        
                image = item["image"].convert("RGB")
                text = item["question"]
                answer = item.get("target_answer") or item.get("multiple_choice_answer")
                
                encoding = processor(image, text, return_tensors="pt", truncation=True, max_length=40).to(device)
                outputs = model(
                    pixel_values=encoding.pixel_values,
                    input_ids=encoding.input_ids,
                    token_type_ids=encoding.token_type_ids,
                    attention_mask=encoding.attention_mask
                )
                idx = outputs.logits.argmax(-1).item()
                pred_answer = id2label[idx]
                
                if pred_answer.lower() == str(answer).lower():
                    correct += 1
                processed += 1
                
        val_acc = correct / max(1, len(val_data))
        print(f"Epoch {epoch+1} | Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.4f}")
        tracker.log_metric(step=epoch, metrics={"train_loss": avg_train_loss, "val_accuracy": val_acc})
        
    print("\nEvaluating on Test Set...")
    model.eval()
    test_correct = 0
    total_latency = 0.0
    
    category_totals = {}
    category_correct = {}
    
    with torch.no_grad():
        for i, item in enumerate(test_data):
            if i % 10 == 0 and progress_callback:
                if not progress_callback(processed, "Test Evaluation"):
                    tracker.end_run()
                    return {"status": "cancelled"}
                    
            image = item["image"].convert("RGB")
            text = item["question"]
            answer = item.get("target_answer") or item.get("multiple_choice_answer")
            q_type = item.get("question_type", "unknown")
            
            if q_type not in category_totals:
                category_totals[q_type] = 0
                category_correct[q_type] = 0
                
            category_totals[q_type] += 1
            
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
            
            is_match = pred_answer.lower() == str(answer).lower()
            if is_match:
                test_correct += 1
                category_correct[q_type] += 1
            else:
                tracker.log_failure(
                    expected=str(answer), 
                    predicted=pred_answer, 
                    meta={"question": text, "question_type": q_type}
                )
            processed += 1
                
    test_acc = test_correct / max(1, len(test_data))
    avg_latency = total_latency / max(1, len(test_data))
    training_time = time.time() - start_time
    
    final_metrics = {
        "test_accuracy": test_acc, 
        "test_exact_match": test_acc,
        "avg_inference_latency_sec": avg_latency,
        "training_time_sec": training_time,
        "train_loss": avg_train_loss
    }
    
    for cat, total in category_totals.items():
        if total > 0:
            final_metrics[f"cat_acc_{cat}"] = category_correct[cat] / total
            
    tracker.log_metric(step=num_epochs, metrics=final_metrics)
    
    # Save checkpoint
    opt_suffix = f"_pathvqa{int(path_vqa_fraction*100)}" if path_vqa_fraction > 0 else ""
    checkpoint_dir = f"checkpoints/reasoning/vilt_vqarad{opt_suffix}_{int(train_fraction*100)}pct"
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.model.save_pretrained(checkpoint_dir)
    processor.save_pretrained(checkpoint_dir)
    
    tracker.end_run()
    if progress_callback:
        progress_callback(processed, "Completed")
        
    print("✅ Reasoning Pipeline Complete.")
    return {"status": "completed", "metrics": final_metrics, "checkpoint_dir": checkpoint_dir}

if __name__ == "__main__":
    run_training({"train_fraction": 1.0, "num_epochs": 1, "device": "mps", "is_optimized": True})
