import torch
import sys
import os
import time
import evaluate
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from ml.audio.data.pipeline import AudioPipeline
from ml.audio.model.baseline import AudioBaseline
from ml.shared.evaluation.tracker import ExperimentTracker

def add_noise(audio_array, noise_factor=0.05):
    noise = np.random.randn(*np.array(audio_array).shape)
    return audio_array + noise_factor * noise

def run_ood_evaluation(noise_factor=None):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"=== Audio OOD Evaluation (Noise Factor: {noise_factor}) ===")
    
    pipeline = AudioPipeline(
        raw_dir="data/raw/audio",
        processed_dir="data/processed/audio",
        config={"seed": 42}
    )
    _, _, test_data = pipeline.run_pipeline()
    processor = pipeline.processor
    
    if noise_factor is not None:
        checkpoint_dir = f"checkpoints/audio/whisper_librispeech_noise{noise_factor}_100pct"
    else:
        checkpoint_dir = "checkpoints/audio/whisper_librispeech_optimized_100pct"
        
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint {checkpoint_dir} not found. Ensure training is complete.")
        return
        
    model = AudioBaseline()
    model.model.from_pretrained(checkpoint_dir)
    model.to(device)
    model.eval()
    
    exp_name = f"Audio-OOD-Noise-Ablation-{noise_factor}-100pct" if noise_factor is not None else "Audio-OOD-Noise-Optimized-100pct"
    tracker = ExperimentTracker(
        project_name="Phase 13: Ablation Analysis",
        experiment_name=exp_name,
        model_version_id=3
    )
    tracker.start_run({
        "dataset": "librispeech_asr (clean)",
        "num_test_samples": len(test_data),
        "device": str(device),
        "is_ood": True,
        "ood_condition": f"High Gaussian Noise (factor=0.05)",
        "noise_factor_training": noise_factor
    })
    
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    
    test_preds = []
    test_refs = []
    exact_matches = 0
    total_latency = 0.0
    
    print("Evaluating...")
    with torch.no_grad():
        for item in test_data:
            audio = item["audio"]["array"]
            # Apply OOD noise
            audio = add_noise(audio, noise_factor=0.05)
            text = item["text"]
            
            start_t = time.time()
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt").to(device)
            pred_ids = model.model.generate(
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
                
    test_wer = wer_metric.compute(predictions=test_preds, references=test_refs)
    test_cer = cer_metric.compute(predictions=test_preds, references=test_refs)
    test_em = exact_matches / max(1, len(test_data))
    avg_latency = total_latency / max(1, len(test_data))
    
    final_metrics = {
        "test_wer": test_wer, 
        "test_cer": test_cer,
        "test_exact_match": test_em,
        "avg_inference_latency_sec": avg_latency
    }
    
    tracker.log_metric(step=1, metrics=final_metrics)
    tracker.end_run()
    
    print(f"Audio OOD Complete. WER: {test_wer:.4f}, CER: {test_cer:.4f}")

if __name__ == "__main__":
    run_ood_evaluation()
