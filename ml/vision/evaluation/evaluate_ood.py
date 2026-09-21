import torch
import sys
import os
import time
import evaluate
from PIL import ImageFilter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from ml.vision.data.pipeline import VisionPipeline
from ml.vision.model.baseline import VisionBaseline
from ml.shared.evaluation.tracker import ExperimentTracker

def run_ood_evaluation(ablation=None):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"=== Vision OOD Evaluation (Ablation: {ablation}) ===")
    
    pipeline = VisionPipeline(
        raw_dir="data/raw/vision",
        processed_dir="data/processed/vision",
        config={"seed": 42}
    )
    _, _, test_data = pipeline.run_pipeline()
    processor = pipeline.processor
    
    if ablation:
        checkpoint_dir = f"checkpoints/vision/trocr_iiit5k_ablation_{ablation}_100pct"
    else:
        checkpoint_dir = "checkpoints/vision/trocr_iiit5k_optimized_100pct"
        
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint {checkpoint_dir} not found. Ensure 100% training is complete.")
        return
        
    model = VisionBaseline()
    from transformers import VisionEncoderDecoderModel
    model.model = VisionEncoderDecoderModel.from_pretrained(checkpoint_dir)
    model.to(device)
    model.eval()
    
    exp_name = f"Vision-OOD-Blur-Ablation-{ablation}-100pct" if ablation else "Vision-OOD-Blur-Optimized-100pct"
    tracker = ExperimentTracker(
        project_name="Phase 13: Ablation Analysis",
        experiment_name=exp_name,
        model_version_id=1
    )
    tracker.start_run({
        "dataset": "HuggingFaceM4/IIIT-5K",
        "num_test_samples": len(test_data),
        "device": str(device),
        "is_ood": True,
        "ood_condition": "Gaussian Blur (radius=2) & Rotation (5 deg)",
        "ablation": ablation
    })
    
    cer_metric = evaluate.load("cer")
    wer_metric = evaluate.load("wer")
    
    test_preds = []
    test_refs = []
    exact_matches = 0
    total_latency = 0.0
    
    print("Evaluating...")
    with torch.no_grad():
        for item in test_data:
            image = item["image"].convert("RGB")
            # Apply OOD perturbations
            image = image.filter(ImageFilter.GaussianBlur(radius=2))
            image = image.rotate(5)
            
            start_t = time.time()
            pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
            generated_ids = model.model.generate(pixel_values)
            pred_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            total_latency += (time.time() - start_t)
            
            test_preds.append(pred_text)
            test_refs.append(item["label"])
            
            if pred_text.strip().lower() == item["label"].strip().lower():
                exact_matches += 1
            else:
                tracker.log_failure(expected=item["label"], predicted=pred_text)
                
    test_cer = cer_metric.compute(predictions=test_preds, references=test_refs)
    test_wer = wer_metric.compute(predictions=test_preds, references=test_refs)
    test_em = exact_matches / max(1, len(test_data))
    avg_latency = total_latency / max(1, len(test_data))
    
    final_metrics = {
        "test_cer": test_cer, 
        "test_wer": test_wer,
        "test_exact_match": test_em,
        "avg_inference_latency_sec": avg_latency
    }
    
    tracker.log_metric(step=1, metrics=final_metrics)
    tracker.end_run()
    
    print(f"Vision OOD Complete. CER: {test_cer:.4f}, WER: {test_wer:.4f}")

if __name__ == "__main__":
    run_ood_evaluation()
