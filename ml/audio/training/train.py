import torch
import sys
import os
import evaluate
import time
import numpy as np
from typing import Callable, Optional, Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from ml.audio.data.pipeline import AudioPipeline
from ml.audio.model.baseline import AudioBaseline
from ml.shared.evaluation.tracker import ExperimentTracker

def add_noise(audio_array, noise_factor=0.01):
    noise = np.random.randn(*np.array(audio_array).shape)
    augmented_data = audio_array + noise_factor * noise
    return augmented_data

def run_training(config: Dict[str, Any], progress_callback: Optional[Callable[[int, str], bool]] = None) -> Dict[str, Any]:
    """
    Executes a training run.
    progress_callback: A function(samples_processed, stage_name) -> bool. If it returns False, training should cancel.
    """
    train_fraction = config.get("train_fraction", 1.0)
    num_epochs = config.get("num_epochs", 1)
    lr = config.get("lr", 5e-5)
    is_optimized = config.get("is_optimized", False)
    noise_factor = config.get("noise_factor", 0.0)
    
    # Backward compatibility
    if is_optimized and noise_factor == 0.0:
        noise_factor = 0.05
        
    batch_size = config.get("batch_size", 1)
    device = torch.device(config.get("device", "cpu"))
    
    print(f"=== Audio Training Pipeline ({train_fraction*100}% subset, Noise: {noise_factor}) ===")
    pipeline = AudioPipeline(
        raw_dir="data/raw/audio",
        processed_dir="data/processed/audio",
        config={"train_fraction": train_fraction, "seed": config.get("seed", 42)}
    )
    train_data, val_data, test_data = pipeline.run_pipeline()
    processor = pipeline.processor
    
    exp_name = f"Whisper-LibriSpeech-Noise{noise_factor}-{train_fraction*100}pct" if noise_factor > 0 else f"Whisper-LibriSpeech-Scale-{train_fraction*100}pct"
    tracker = ExperimentTracker(
        project_name="Phase 13: Ablation Analysis",
        experiment_name=exp_name,
        model_version_id=3
    )
    hyperparams = {
        "dataset": "librispeech_asr (clean)",
        "batch_size": batch_size,
        "lr": lr,
        "noise_factor": noise_factor,
        "train_fraction": train_fraction,
        "num_train_samples": len(train_data),
        "num_val_samples": len(val_data),
        "num_test_samples": len(test_data),
        "epochs": num_epochs,
        "device": str(device)
    }
    tracker.start_run(hyperparams)
    
    model = AudioBaseline().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    
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
                    
            audio = item["audio"]["array"]
            if noise_factor > 0:
                audio = add_noise(audio, noise_factor)
                
            text = item["text"]
            
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt").to(device)
            labels = processor.tokenizer(text, return_tensors="pt").input_ids.to(device)
            labels[labels == processor.tokenizer.pad_token_id] = -100
            
            optimizer.zero_grad()
            outputs = model(
                input_features=inputs.input_features,
                attention_mask=getattr(inputs, "attention_mask", None),
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
        predictions = []
        references = []
        with torch.no_grad():
            for i, item in enumerate(val_data):
                if i % 10 == 0 and progress_callback:
                    if not progress_callback(processed, f"Epoch {epoch+1}: Validation"):
                        tracker.end_run()
                        return {"status": "cancelled"}
                        
                audio = item["audio"]["array"]
                text = item["text"]
                
                inputs = processor(audio, sampling_rate=16000, return_tensors="pt").to(device)
                pred_ids = model.generate(
                    inputs.input_features, 
                    attention_mask=getattr(inputs, "attention_mask", None)
                )
                pred_text = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
                
                predictions.append(pred_text.upper())
                references.append(text.upper())
                processed += 1
                
        val_wer = wer_metric.compute(predictions=predictions, references=references)
        val_cer = cer_metric.compute(predictions=predictions, references=references)
        
        print(f"Epoch {epoch+1} | Loss: {avg_train_loss:.4f} | Val WER: {val_wer:.4f} | Val CER: {val_cer:.4f}")
        tracker.log_metric(step=epoch, metrics={"train_loss": avg_train_loss, "val_wer": val_wer, "val_cer": val_cer})
        
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
                    
            audio = item["audio"]["array"]
            text = item["text"]
            
            start_t = time.time()
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt").to(device)
            pred_ids = model.generate(
                inputs.input_features, 
                attention_mask=getattr(inputs, "attention_mask", None)
            )
            pred_text = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
            total_latency += (time.time() - start_t)
            
            test_preds.append(pred_text.upper())
            test_refs.append(text.upper())
            
            if pred_text.strip().upper() == text.strip().upper():
                exact_matches += 1
            else:
                tracker.log_failure(expected=text.upper(), predicted=pred_text.upper())
            processed += 1
            
    test_wer = wer_metric.compute(predictions=test_preds, references=test_refs)
    test_cer = cer_metric.compute(predictions=test_preds, references=test_refs)
    test_em = exact_matches / max(1, len(test_data))
    avg_latency = total_latency / max(1, len(test_data))
    training_time = time.time() - start_time
    
    final_metrics = {
        "test_wer": test_wer, 
        "test_cer": test_cer,
        "test_exact_match": test_em,
        "avg_inference_latency_sec": avg_latency,
        "training_time_sec": training_time,
        "train_loss": avg_train_loss
    }
    
    tracker.log_metric(step=num_epochs, metrics=final_metrics)
    
    # Save checkpoint
    opt_suffix = f"_noise{noise_factor}" if noise_factor > 0 else ""
    checkpoint_dir = f"checkpoints/audio/whisper_librispeech{opt_suffix}_{int(train_fraction*100)}pct"
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.model.save_pretrained(checkpoint_dir)
    processor.save_pretrained(checkpoint_dir)
    
    tracker.end_run()
    if progress_callback:
        progress_callback(processed, "Completed")
        
    print("✅ Audio Pipeline Complete.")
    return {"status": "completed", "metrics": final_metrics, "checkpoint_dir": checkpoint_dir}

if __name__ == "__main__":
    run_training({"train_fraction": 1.0, "num_epochs": 1, "device": "mps", "is_optimized": True})
