import os
import sys
from datasets import load_dataset
from transformers import ViltProcessor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from ml.shared.data.pipeline import BaseDatasetPipeline

class ReasoningPipeline(BaseDatasetPipeline):
    def __init__(self, raw_dir: str, processed_dir: str, config: dict):
        super().__init__(raw_dir, processed_dir, config)
        self.processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        self.train_pool = None
        self.val_set = None
        self.test_set = None

    def validate_and_clean(self):
        print("Loading Reasoning datasets...")
        seed = self.config.get("seed", 42)
        path_vqa_fraction = self.config.get("path_vqa_fraction", 0.0)
        is_optimized = self.config.get("is_optimized", False)
        
        # Backward compatibility for Phase 12 optimized flag
        if is_optimized and path_vqa_fraction == 0.0:
            path_vqa_fraction = 1.0
            
        # Load VQA-RAD
        d_train = load_dataset("flaviagiammarino/vqa-rad", split="train", streaming=False)
        self.train_pool = d_train.shuffle(seed=seed)
        
        def map_answer(example):
            example["target_answer"] = example["answer"]
            return example
            
        self.train_pool = self.train_pool.map(map_answer)
        
        if path_vqa_fraction > 0:
            print(f"Optimized Mode: Mixing PathVQA (fraction={path_vqa_fraction}) into training pool...")
            path_vqa = load_dataset("flaviagiammarino/path-vqa", split="train", streaming=False)
            num_samples = int(5000 * path_vqa_fraction)
            path_vqa = path_vqa.shuffle(seed=seed).select(range(num_samples)).map(map_answer)
            
            # Combine the pools (features must match, we only need image, question, target_answer)
            def keep_columns(example):
                return {"image": example["image"], "question": example["question"], "target_answer": example["target_answer"]}
            
            self.train_pool = self.train_pool.map(keep_columns)
            path_vqa = path_vqa.map(keep_columns)
            
            from datasets import concatenate_datasets
            self.train_pool = concatenate_datasets([self.train_pool, path_vqa]).shuffle(seed=seed)
        else:
            def keep_columns(example):
                return {"image": example["image"], "question": example["question"], "target_answer": example["target_answer"]}
            self.train_pool = self.train_pool.map(keep_columns)
            
        d_test_full = load_dataset("flaviagiammarino/vqa-rad", split="test", streaming=False)
        d_test_full = d_test_full.shuffle(seed=seed).map(map_answer).map(keep_columns)
        
        self.val_set = d_test_full.select(range(200))
        self.test_set = d_test_full.select(range(200, len(d_test_full)))
        
        self.stats["train_pool_size"] = len(self.train_pool)
        self.stats["val_size"] = len(self.val_set)
        self.stats["test_size"] = len(self.test_set)
        print(f"Loaded train pool: {len(self.train_pool)}, val: {len(self.val_set)}, test: {len(self.test_set)}")

    def preprocess(self):
        print("Preprocessing handled on the fly.")
        pass

    def split(self):
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
