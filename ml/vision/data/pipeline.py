import os
import sys
from datasets import load_dataset
from transformers import TrOCRProcessor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from ml.shared.data.pipeline import BaseDatasetPipeline

class VisionPipeline(BaseDatasetPipeline):
    def __init__(self, raw_dir: str, processed_dir: str, config: dict):
        super().__init__(raw_dir, processed_dir, config)
        from transformers import TrOCRProcessor, ViTImageProcessor, XLMRobertaTokenizer
        feature_extractor = ViTImageProcessor.from_pretrained("microsoft/trocr-small-printed")
        tokenizer = XLMRobertaTokenizer.from_pretrained("microsoft/trocr-small-printed")
        self.processor = TrOCRProcessor(image_processor=feature_extractor, tokenizer=tokenizer)
        self.train_pool = None
        self.val_set = None
        self.test_set = None

    def validate_and_clean(self):
        print("Loading Vision datasets (IIIT-5K)...")
        # Ensure consistent seeding
        seed = self.config.get("seed", 42)
        
        # Load IIIT-5K train (2000 samples)
        d_train = load_dataset("HuggingFaceM4/IIIT-5K", split="train", streaming=False)
        self.train_pool = d_train.shuffle(seed=seed)
        
        # Load IIIT-5K test (3000 samples) and split it into validation and test
        d_test_full = load_dataset("HuggingFaceM4/IIIT-5K", split="test", streaming=False)
        d_test_full = d_test_full.shuffle(seed=seed)
        
        self.val_set = d_test_full.select(range(1000))
        self.test_set = d_test_full.select(range(1000, 3000))
        
        self.stats["train_pool_size"] = len(self.train_pool)
        self.stats["val_size"] = len(self.val_set)
        self.stats["test_size"] = len(self.test_set)

    def preprocess(self):
        print("Preprocessing handled on the fly.")
        pass

    def split(self):
        # Calculate how many samples we take from the training pool
        fraction = self.config.get("train_fraction", 1.0)
        num_samples = int(len(self.train_pool) * fraction)
        num_samples = max(1, min(num_samples, len(self.train_pool)))
        
        print(f"Selecting {num_samples} samples ({fraction*100}%) for training...")
        train = self.train_pool.select(range(num_samples))
        
        self.stats["splits"] = {
            "train": len(train),
            "val": len(self.val_set),
            "test": len(self.test_set)
        }
        return train, self.val_set, self.test_set
