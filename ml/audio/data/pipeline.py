import os
import sys
from datasets import load_dataset
from transformers import WhisperProcessor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from ml.shared.data.pipeline import BaseDatasetPipeline

class AudioPipeline(BaseDatasetPipeline):
    def __init__(self, raw_dir: str, processed_dir: str, config: dict):
        super().__init__(raw_dir, processed_dir, config)
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
        self.train_pool = None
        self.val_set = None
        self.test_set = None

    def validate_and_clean(self):
        print("Loading Audio dataset (LibriSpeech clean subset)...")
        seed = self.config.get("seed", 42)
        
        # Loading 'validation' split (about 2703 samples) to act as our training pool for scaling experiments
        d_train = load_dataset("openslr/librispeech_asr", "clean", split="validation", streaming=False)
        self.train_pool = d_train.shuffle(seed=seed)
        
        # Loading 'test' split (about 2620 samples) and split it into validation and test
        d_test_full = load_dataset("openslr/librispeech_asr", "clean", split="test", streaming=False)
        d_test_full = d_test_full.shuffle(seed=seed)
        
        self.val_set = d_test_full.select(range(500))
        self.test_set = d_test_full.select(range(500, min(1500, len(d_test_full))))
        
        self.stats["train_pool_size"] = len(self.train_pool)
        self.stats["val_size"] = len(self.val_set)
        self.stats["test_size"] = len(self.test_set)
        print(f"Loaded train pool: {len(self.train_pool)}, val: {len(self.val_set)}, test: {len(self.test_set)}")

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
