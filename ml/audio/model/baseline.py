import torch
import torch.nn as nn
from transformers import WhisperForConditionalGeneration

class AudioBaseline(nn.Module):
    """
    Wraps the pretrained Whisper model for ASR.
    """
    def __init__(self):
        super().__init__()
        # Load pre-trained Whisper model
        self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")

    def forward(self, input_features, attention_mask=None, labels=None):
        outputs = self.model(
            input_features=input_features,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs
    
    def generate(self, input_features, attention_mask=None):
        # Whisper model's generate method automatically handles encoder-decoder sequence generation
        return self.model.generate(
            input_features=input_features, 
            attention_mask=attention_mask,
            max_new_tokens=128
        )
