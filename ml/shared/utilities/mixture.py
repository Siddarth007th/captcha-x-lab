from typing import Dict, Any

class DatasetMixtureManager:
    """
    Manages the sampling weights and mixtures of different dataset sources.
    Example: 40% Synthetic OCR, 30% Real OCR, 20% COCO, 10% Generated.
    """
    
    def __init__(self, mixture_config: Dict[str, float]):
        """
        Initialize with a dictionary of dataset names to their sampling weights.
        The weights do not strictly need to sum to 1.0 (they can be normalized internally).
        """
        self.mixture_config = mixture_config
        self._normalize_weights()
        
    def _normalize_weights(self) -> None:
        """Normalizes the sampling weights to sum to 1.0."""
        total = sum(self.mixture_config.values())
        if total > 0:
            self.normalized_weights = {k: v / total for k, v in self.mixture_config.items()}
        else:
            self.normalized_weights = {}

    def get_sampler(self) -> Any:
        """
        Returns a PyTorch WeightedRandomSampler (or equivalent) 
        configured according to the normalized weights.
        """
        pass
