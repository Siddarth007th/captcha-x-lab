import time
import torch
from io import BytesIO
import librosa
import soundfile as sf
import os
from PIL import Image

# Import models
from ml.vision.model.baseline import VisionBaseline
from ml.reasoning.model.baseline import ReasoningBaseline
from ml.audio.model.baseline import AudioBaseline

# Import Processors
from transformers import TrOCRProcessor, ViltProcessor, WhisperProcessor

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))

class ModelManager:
    """
    Lazy loads and manages model instances for inference, keeping them in memory 
    only when requested.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance.models = {}
            cls._instance.processors = {}
            cls._instance.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        return cls._instance

    def _load_vision(self):
        if "vision" not in self.models:
            print(f"Loading Vision Model onto {self.device}...")
            vision_ckpt = os.path.join(BASE_DIR, "checkpoints/vision/trocr_iiit5k_optimized_100pct")
            self.processors["vision"] = TrOCRProcessor.from_pretrained(vision_ckpt)
            # Load the Phase 12 Optimized Checkpoint directly using the underlying architecture
            from transformers import VisionEncoderDecoderModel
            model = VisionEncoderDecoderModel.from_pretrained(vision_ckpt)
            model.to(self.device)
            model.eval()
            self.models["vision"] = model
        return self.models["vision"], self.processors["vision"]

    def _load_reasoning(self):
        if "reasoning" not in self.models:
            print(f"Loading Reasoning Model onto {self.device}...")
            reasoning_ckpt = os.path.join(BASE_DIR, "checkpoints/reasoning/vilt_vqarad_pathvqa25_100pct")
            self.processors["reasoning"] = ViltProcessor.from_pretrained(reasoning_ckpt)
            from transformers import ViltForQuestionAnswering
            model = ViltForQuestionAnswering.from_pretrained(reasoning_ckpt)
            model.to(self.device)
            model.eval()
            self.models["reasoning"] = model
        return self.models["reasoning"], self.processors["reasoning"]

    def _load_audio(self):
        if "audio" not in self.models:
            print(f"Loading Audio Model onto {self.device}...")
            # We use Phase 12 checkpoint for Audio as it balanced clean and noisy performance
            audio_ckpt = os.path.join(BASE_DIR, "checkpoints/audio/whisper_librispeech_optimized_100pct")
            self.processors["audio"] = WhisperProcessor.from_pretrained(audio_ckpt)
            from transformers import WhisperForConditionalGeneration
            model = WhisperForConditionalGeneration.from_pretrained(audio_ckpt)
            model.to(self.device)
            model.eval()
            self.models["audio"] = model
        return self.models["audio"], self.processors["audio"]

    def unload_model(self, model_name: str):
        if model_name in self.models:
            del self.models[model_name]
            del self.processors[model_name]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()


class MasterRouter:
    """
    Orchestration layer that heuristically determines input modality 
    and routes to the appropriate sub-model for inference.
    """
    def __init__(self):
        self.manager = ModelManager()
        self.device = self.manager.device

    def infer(self, file_bytes: bytes, mime_type: str, text_query: str = None) -> dict:
        start_time = time.time()
        result = {
            "model_used": "unknown",
            "challenge_type": "unknown",
            "prediction": "",
            "confidence": 0.0,
            "processing_latency": 0.0,
            "success": False,
            "error_information": None,
            "model_version": "v1.0-baseline"
        }

        try:
            if "audio" in mime_type:
                result = self._route_audio(file_bytes, result)
            elif "image" in mime_type:
                if text_query:
                    result = self._route_reasoning(file_bytes, text_query, result)
                else:
                    result = self._route_vision(file_bytes, result)
            else:
                raise ValueError(f"Unsupported MIME type: {mime_type}")
            
            result["success"] = True

        except Exception as e:
            result["success"] = False
            result["error_information"] = str(e)
            print(f"Inference Error: {e}")
        
        result["processing_latency"] = time.time() - start_time
        return result

    def _route_audio(self, file_bytes: bytes, result: dict) -> dict:
        result["challenge_type"] = "audio_captcha"
        result["model_used"] = "whisper-tiny-librispeech"
        
        model, processor = self.manager._load_audio()
        
        # Parse audio bytes
        audio, rate = sf.read(BytesIO(file_bytes))
        
        # If stereo, convert to mono first
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
            
        # Resample to 16kHz which Whisper expects
        if rate != 16000:
            audio = librosa.resample(audio, orig_sr=rate, target_sr=16000)

        # Truncate to 30 seconds maximum (480000 samples at 16kHz) to avoid Whisper length errors
        max_samples = 30 * 16000
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        inputs = processor(audio, sampling_rate=16000, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = model.generate(
                inputs.input_features, 
                max_new_tokens=128,
                return_dict_in_generate=True,
                output_scores=True
            )
            predicted_ids = outputs.sequences
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            
            transition_scores = model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            )
            probs = torch.exp(transition_scores[0])
            confidence = probs.mean().item() if len(probs) > 0 else 0.0
        
        result["prediction"] = transcription
        result["confidence"] = confidence
        return result

    def _route_vision(self, file_bytes: bytes, result: dict) -> dict:
        result["challenge_type"] = "image_text_captcha"
        result["model_used"] = "trocr-small-printed"
        
        model, processor = self.manager._load_vision()
        
        image = Image.open(BytesIO(file_bytes)).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        with torch.no_grad():
            outputs = model.generate(
                pixel_values,
                max_new_tokens=32,
                return_dict_in_generate=True,
                output_scores=True
            )
            generated_ids = outputs.sequences
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            transition_scores = model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            )
            probs = torch.exp(transition_scores[0])
            confidence = probs.mean().item() if len(probs) > 0 else 0.0

        result["prediction"] = generated_text
        result["confidence"] = confidence
        return result

    def _route_reasoning(self, file_bytes: bytes, text_query: str, result: dict) -> dict:
        result["challenge_type"] = "vqa_captcha"
        result["model_used"] = "vilt-b32-finetuned-vqa"
        
        model, processor = self.manager._load_reasoning()
        
        image = Image.open(BytesIO(file_bytes)).convert("RGB")
        encoding = processor(image, text_query, return_tensors="pt")
        # Ensure all are moved to device
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        
        # ViLT processor output has 'pixel_mask' but the baseline forward doesn't accept it, filter it out
        forward_args = {k: v for k, v in encoding.items() if k != 'pixel_mask'}

        with torch.no_grad():
            outputs = model(**forward_args)
            logits = outputs.logits
            idx = logits.argmax(-1).item()
            answer = model.config.id2label[idx]
            
            probs = torch.nn.functional.softmax(logits, dim=-1)
            confidence = probs[0, idx].item()
        
        result["prediction"] = answer
        result["confidence"] = confidence
        return result
