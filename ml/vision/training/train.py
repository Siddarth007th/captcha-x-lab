import torch
import sys
import os
import evaluate
import time
import torchvision.transforms as T
from typing import Callable, Optional, Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from ml.vision.data.pipeline import VisionPipeline
from ml.vision.model.baseline import VisionBaseline
from ml.shared.evaluation.tracker import ExperimentTracker

def run_training(config: Dict[str, Any], progress_callback: Optional[Callable[[int, str], bool]] = None) -> Dict[str, Any]:
    """
    Executes a training run.
    progress_callback: A function(samples_processed, stage_name) -> bool. If it returns False, training should cancel.
    """
    train_fraction = config.get("train_fraction", 1.0)
    num_epochs = config.get("num_epochs", 1)
    lr = config.get("lr", 5e-5)
    batch_size = config.get("batch_size", 2)
    device = torch.device(config.get("device", "cpu"))
    ablation = config.get("ablation", None)
    is_optimized = ablation == "all"
    
    print(f"=== Vision Training Pipeline ({train_fraction*100}% subset, Ablation: {ablation}) ===")
    pipeline = VisionPipeline(
        raw_dir="data/raw/vision",
        processed_dir="data/processed/vision",
        config={"train_fraction": train_fraction, "seed": config.get("seed", 42)}
    )
    train_data, val_data, test_data = pipeline.run_pipeline()
    processor = pipeline.processor
    
    exp_name = f"TrOCR-IIIT5K-Ablation-{ablation}-{train_fraction*100}pct" if ablation else f"TrOCR-IIIT5K-Scale-{train_fraction*100}pct"
    tracker = ExperimentTracker(
        project_name="Phase 13: Ablation Analysis",
        experiment_name=exp_name,
        model_version_id=1
    )
    hyperparams = {
        "dataset": "HuggingFaceM4/IIIT-5K",
        "batch_size": batch_size,
        "lr": lr,
        "train_fraction": train_fraction,
        "num_train_samples": len(train_data),
        "num_val_samples": len(val_data),
        "num_test_samples": len(test_data),
        "epochs": num_epochs,
        "device": str(device),
        "ablation": ablation
    }
    tracker.start_run(hyperparams)
    
    model = VisionBaseline().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    cer_metric = evaluate.load("cer")
    wer_metric = evaluate.load("wer")
    
    total_samples_to_process = (len(train_data) * num_epochs) + len(val_data) * num_epochs + len(test_data)
    processed = 0
    
    start_time = time.time()
    
    # Augmentation transform
    transforms_list = []
    if ablation in ["rotation", "all"]:
        transforms_list.append(T.RandomRotation(15))
    if ablation in ["blur", "all"]:
        transforms_list.append(T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)))
    if ablation in ["jitter", "all"]:
        transforms_list.append(T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))
    
    augment = T.Compose(transforms_list) if transforms_list else None
    
    print("\nStarting Training Loop...")
    for epoch in range(num_epochs): 
        model.train()
        total_loss = 0
        
        # We manually batch or iterate depending on requirements. Since batch_size=2 isn't natively handled in this raw loop, 
        # we'll process 1 by 1 for simplicity of progress tracking unless explicitly batched.
        for i, item in enumerate(train_data):
            # Check cancellation every 10 samples
            if i % 10 == 0 and progress_callback:
                if not progress_callback(processed, f"Epoch {epoch+1}: Training"):
                    tracker.end_run()
                    return {"status": "cancelled"}
                    
            image = item["image"].convert("RGB")
            if is_optimized:
                image = augment(image)
            pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
            labels = processor.tokenizer(item["label"], return_tensors="pt").input_ids.to(device)
            
            optimizer.zero_grad()
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            processed += 1
            
        avg_train_loss = total_loss / max(1, len(train_data))
        
        # Validation
        model.eval()
        predictions = []
        references = []
        with torch.no_grad():
            for i, item in enumerate(val_data):
                if i % 10 == 0 and progress_callback:
                    if not progress_callback(processed, f"Epoch {epoch+1}: Validation"):
                        tracker.end_run()
                        return {"status": "cancelled"}
                        
                image = item["image"].convert("RGB")
                pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
                generated_ids = model.generate(pixel_values)
                pred_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                predictions.append(pred_text)
                references.append(item["label"])
                processed += 1
                
        val_cer = cer_metric.compute(predictions=predictions, references=references)
        val_wer = wer_metric.compute(predictions=predictions, references=references)
        
        print(f"Epoch {epoch+1} | Loss: {avg_train_loss:.4f} | Val CER: {val_cer:.4f}")
        tracker.log_metric(step=epoch, metrics={"train_loss": avg_train_loss, "val_cer": val_cer, "val_wer": val_wer})
        
    print("\nEvaluating on Test Set...")
    model.eval()
    test_preds = []
    test_refs = []
    exact_matches = 0
    total_latency = 0.0
    
    with torch.no_grad():
        for i, item in enumerate(test_data):
            if i % 10 == 0 and progress_callback:
                if not progress_callback(processed, "Test Evaluation"):
                    tracker.end_run()
                    return {"status": "cancelled"}
                    
            image = item["image"].convert("RGB")
            
            start_t = time.time()
            pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
            generated_ids = model.generate(pixel_values)
            pred_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            total_latency += (time.time() - start_t)
            
            test_preds.append(pred_text)
            test_refs.append(item["label"])
            
            is_match = pred_text.strip().lower() == item["label"].strip().lower()
            if is_match:
                exact_matches += 1
            else:
                tracker.log_failure(expected=item["label"], predicted=pred_text)
            processed += 1
                
    test_cer = cer_metric.compute(predictions=test_preds, references=test_refs)
    test_wer = wer_metric.compute(predictions=test_preds, references=test_refs)
    test_em = exact_matches / max(1, len(test_data))
    avg_latency = total_latency / max(1, len(test_data))
    training_time = time.time() - start_time
    
    final_metrics = {
        "test_cer": test_cer, 
        "test_wer": test_wer,
        "test_exact_match": test_em,
        "avg_inference_latency_sec": avg_latency,
        "training_time_sec": training_time,
        "train_loss": avg_train_loss
    }
    
    tracker.log_metric(step=num_epochs, metrics=final_metrics)
    
    # Save checkpoint
    if config.get("ablation"):
        checkpoint_dir = f"checkpoints/vision/trocr_iiit5k_ablation_{config['ablation']}_{int(train_fraction*100)}pct"
    else:
        checkpoint_dir = f"checkpoints/vision/trocr_iiit5k_optimized_{int(train_fraction*100)}pct" if is_optimized else f"checkpoints/vision/trocr_iiit5k_{int(train_fraction*100)}pct"
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.model.save_pretrained(checkpoint_dir)
    processor.save_pretrained(checkpoint_dir)
    
    tracker.end_run()
    if progress_callback:
        progress_callback(processed, "Completed")
        
    print("✅ Vision Pipeline Complete.")
    return {"status": "completed", "metrics": final_metrics, "checkpoint_dir": checkpoint_dir}

if __name__ == "__main__":
    # Test block for local execution
    run_training({"train_fraction": 1.0, "num_epochs": 1, "device": "mps", "is_optimized": True})
