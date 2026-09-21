import torch
import torch.nn as nn
from transformers import VisionEncoderDecoderModel

class VisionBaseline(nn.Module):
    """
    Wraps TrOCR for OCR baselining.
    """
    def __init__(self):
        super().__init__()
        self.model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-printed")
        # set special tokens used for creating the decoder_input_ids from the labels
        self.model.config.decoder_start_token_id = self.model.config.decoder.bos_token_id
        self.model.config.pad_token_id = self.model.config.decoder.pad_token_id
        self.model.config.vocab_size = self.model.config.decoder.vocab_size

    def forward(self, pixel_values, labels=None):
        outputs = self.model(pixel_values=pixel_values, labels=labels)
        return outputs
    
    def generate(self, pixel_values):
        return self.model.generate(pixel_values=pixel_values, max_length=20)
