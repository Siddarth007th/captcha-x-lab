from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import os

class BaseDatasetPipeline(ABC):
    """
    Abstract base class for CAPTCHA-X Lab dataset pipelines.
    Enforces the 'Validate -> Preprocess -> Split -> Ready' flow.
    """
    def __init__(self, raw_dir: str, processed_dir: str, config: Dict[str, Any]):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.config = config
        self.stats = {
            "total_samples": 0,
            "valid_samples": 0,
            "corrupt_samples": 0,
            "duplicate_samples": 0,
            "splits": {"train": 0, "val": 0, "test": 0}
        }
        
        os.makedirs(self.processed_dir, exist_ok=True)

    @abstractmethod
    def validate_and_clean(self) -> None:
        """
        Check for corrupted files and basic duplicates.
        Update self.stats during the process.
        """
        pass
        
    @abstractmethod
    def preprocess(self) -> None:
        """
        Apply necessary preprocessing (e.g. resizing, normalization, resampling).
        """
        pass
        
    @abstractmethod
    def split(self) -> Tuple[Any, Any, Any]:
        """
        Split into Train, Validation, and Test sets.
        Returns the data splits.
        """
        pass

    def run_pipeline(self) -> Tuple[Any, Any, Any]:
        """
        Executes the full pipeline sequentially.
        """
        print(f"Running pipeline for {self.__class__.__name__}...")
        self.validate_and_clean()
        self.preprocess()
        splits = self.split()
        print("Pipeline Statistics:", self.stats)
        return splits
