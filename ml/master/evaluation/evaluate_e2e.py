import sys
import os
import time
import io
import torch
import numpy as np
import soundfile as sf
from PIL import Image, ImageFilter
import evaluate

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from ml.master.router.router import MasterRouter
from ml.vision.data.pipeline import VisionPipeline
from ml.reasoning.data.pipeline import ReasoningPipeline
from ml.audio.data.pipeline import AudioPipeline
from ml.shared.evaluation.tracker import ExperimentTracker

def add_audio_noise(audio_array, noise_factor=0.1):
    noise = np.random.randn(*np.array(audio_array).shape)
    augmented_data = audio_array + noise_factor * noise
    return augmented_data

def evaluate_e2e():
    print("=== Final Master Router & End-to-End System Evaluation ===")
    router = MasterRouter()
    
    tracker = ExperimentTracker(
        project_name="Phase 14: End-to-End Evaluation",
        experiment_name="Master-Router-Final-Eval",
        model_version_id=3
    )
    tracker.start_run({"description": "E2E Evaluation across Vision, Reasoning, and Audio pipelines"})

    metrics = {
        "routing_correct": 0,
        "routing_total": 0,
        
        "vision_cer": 0,
        "vision_wer": 0,
        "vision_exact": 0,
        "vision_total": 0,
        
        "reasoning_exact": 0,
        "reasoning_total": 0,
        
        "audio_wer": 0,
        "audio_cer": 0,
        "audio_total": 0,
        
        "total_latency": 0.0,
        "specialist_latency": 0.0,
    }
    
    cer_metric = evaluate.load("cer")
    wer_metric = evaluate.load("wer")
    
    # ------------------
    # 1. Vision OOD
    # ------------------
    print("\n--- Evaluating Vision OOD (IIIT-5K + Blur/Rotation) ---")
    v_pipeline = VisionPipeline("data/raw/vision", "data/processed/vision", {"seed": 42})
    _, _, v_test_data = v_pipeline.run_pipeline()
    
    v_preds = []
    v_refs = []
    
    for i, item in enumerate(v_test_data):
        # 10% sample to keep E2E evaluation fast, 200 samples
        if i >= 200: 
            break
            
        # Augment image
        image = item["image"].convert("RGB").filter(ImageFilter.GaussianBlur(radius=2)).rotate(5)
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        file_bytes = img_byte_arr.getvalue()
        
        label = item["label"]
        
        start_time = time.time()
        result = router.infer(file_bytes=file_bytes, mime_type="image/png")
        e2e_time = time.time() - start_time
        
        metrics["total_latency"] += e2e_time
        metrics["specialist_latency"] += result.get("processing_latency", e2e_time)
        
        metrics["routing_total"] += 1
        metrics["vision_total"] += 1
        if result["challenge_type"] == "image_text_captcha":
            metrics["routing_correct"] += 1
            
        pred = result["prediction"]
        v_preds.append(pred)
        v_refs.append(label)
        if pred.strip().upper() == label.strip().upper():
            metrics["vision_exact"] += 1
        else:
            tracker.log_failure(expected=label, predicted=pred, meta={"modality": "vision"})
            
        if i % 50 == 0:
            print(f"Vision Processed {i}/200")
            
    metrics["vision_cer"] = cer_metric.compute(predictions=v_preds, references=v_refs)
    metrics["vision_wer"] = wer_metric.compute(predictions=v_preds, references=v_refs)

    # ------------------
    # 2. Reasoning OOD
    # ------------------
    print("\n--- Evaluating Reasoning OOD (PathVQA) ---")
    r_pipeline = ReasoningPipeline("data/raw/reasoning", "data/processed/reasoning", {"pathvqa_fraction": 1.0, "seed": 42})
    _, _, r_test_data = r_pipeline.run_pipeline()
    
    for i, item in enumerate(r_test_data):
        if i >= 200:
            break
            
        image = item["image"]
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        file_bytes = img_byte_arr.getvalue()
        
        question = item["question"]
        answers = [a["answer"] for a in item["answers"]] if "answers" in item else [item["answer"]]
        
        start_time = time.time()
        result = router.infer(file_bytes=file_bytes, mime_type="image/png", text_query=question)
        e2e_time = time.time() - start_time
        
        metrics["total_latency"] += e2e_time
        metrics["specialist_latency"] += result.get("processing_latency", e2e_time)
        
        metrics["routing_total"] += 1
        metrics["reasoning_total"] += 1
        if result["challenge_type"] == "vqa_captcha":
            metrics["routing_correct"] += 1
            
        pred = result["prediction"]
        is_correct = any(pred.lower().strip() == ans.lower().strip() for ans in answers)
        if is_correct:
            metrics["reasoning_exact"] += 1
        else:
            tracker.log_failure(expected=str(answers), predicted=pred, meta={"modality": "reasoning"})
            
        if i % 50 == 0:
            print(f"Reasoning Processed {i}/200")

    # ------------------
    # 3. Audio OOD
    # ------------------
    print("\n--- Evaluating Audio OOD (LibriSpeech + Noise 0.1) ---")
    a_pipeline = AudioPipeline("data/raw/audio", "data/processed/audio", {"seed": 42})
    _, _, a_test_data = a_pipeline.run_pipeline()
    
    a_preds = []
    a_refs = []
    
    for i, item in enumerate(a_test_data):
        if i >= 200:
            break
            
        audio = item["audio"]["array"]
        audio_noisy = add_audio_noise(audio, noise_factor=0.1)
        
        audio_byte_arr = io.BytesIO()
        sf.write(audio_byte_arr, audio_noisy, 16000, format='WAV')
        file_bytes = audio_byte_arr.getvalue()
        
        label = item["text"]
        
        start_time = time.time()
        result = router.infer(file_bytes=file_bytes, mime_type="audio/wav")
        e2e_time = time.time() - start_time
        
        metrics["total_latency"] += e2e_time
        metrics["specialist_latency"] += result.get("processing_latency", e2e_time)
        
        metrics["routing_total"] += 1
        metrics["audio_total"] += 1
        if result["challenge_type"] == "audio_captcha":
            metrics["routing_correct"] += 1
            
        pred = result["prediction"]
        a_preds.append(pred.upper())
        a_refs.append(label.upper())
        if pred.strip().upper() != label.strip().upper():
            tracker.log_failure(expected=label.upper(), predicted=pred.upper(), meta={"modality": "audio"})
            
        if i % 50 == 0:
            print(f"Audio Processed {i}/200")
            
    metrics["audio_cer"] = cer_metric.compute(predictions=a_preds, references=a_refs)
    metrics["audio_wer"] = wer_metric.compute(predictions=a_preds, references=a_refs)

    # ------------------
    # Compile Final Stats
    # ------------------
    final_metrics = {
        "routing_accuracy": metrics["routing_correct"] / max(1, metrics["routing_total"]),
        "vision_ood_cer": metrics["vision_cer"],
        "vision_ood_wer": metrics["vision_wer"],
        "vision_exact_match": metrics["vision_exact"] / max(1, metrics["vision_total"]),
        "reasoning_accuracy": metrics["reasoning_exact"] / max(1, metrics["reasoning_total"]),
        "audio_ood_wer": metrics["audio_wer"],
        "audio_ood_cer": metrics["audio_cer"],
        "avg_e2e_latency_sec": metrics["total_latency"] / max(1, metrics["routing_total"]),
        "avg_specialist_latency_sec": metrics["specialist_latency"] / max(1, metrics["routing_total"]),
    }
    
    tracker.log_metric(step=1, metrics=final_metrics)
    tracker.end_run()
    
    print("\n✅ Final E2E Evaluation Complete!")
    for k, v in final_metrics.items():
        print(f"  {k}: {v:.4f}")

if __name__ == "__main__":
    evaluate_e2e()
