import torch
import torch.nn as nn
from transformers import ViltForQuestionAnswering

class ReasoningBaseline(nn.Module):
    """
    Wraps the pretrained ViLT model for VQA.
    """
    def __init__(self):
        super().__init__()
        # Load pre-trained ViLT model fine-tuned on VQA
        self.model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

    def forward(self, pixel_values, input_ids, token_type_ids, attention_mask, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values=pixel_values,
            labels=labels
        )
        return outputs
