from abc import ABC, abstractmethod
from typing import Any, Dict, List

class BaseEvaluator(ABC):
    """
    Stable interface for all model evaluators in CAPTCHA-X Lab.
    Ensures future models implement the exact same lifecycle methods.
    """
    
    @abstractmethod
    def load(self) -> None:
        """Load the model weights and set to evaluation mode."""
        pass

    @abstractmethod
    def predict(self, sample: Any) -> Any:
        """Run inference on a single sample."""
        pass

    @abstractmethod
    def batch_predict(self, samples: List[Any]) -> List[Any]:
        """Run inference on a batch of samples."""
        pass

    @abstractmethod
    def evaluate(self, dataset: Any) -> Dict[str, Any]:
        """
        Evaluate the model on a full dataset.
        Returns a dictionary of metrics (e.g. accuracy, F1, OOD performance).
        """
        pass

class VisionEvaluator(BaseEvaluator):
    """Base class for Vision models (OCR, object detection, classification)."""
    pass

class ReasoningEvaluator(BaseEvaluator):
    """Base class for Vision-Language Reasoning models (VQA, spatial relationships)."""
    pass

class AudioEvaluator(BaseEvaluator):
    """Base class for Speech/Audio models."""
    pass
