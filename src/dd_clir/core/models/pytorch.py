from pathlib import Path

import torch.nn.functional as F
import torch
from transformers import AutoTokenizer, AutoModel

from dd_clir.core.models.base import Model
from dd_clir.core.utils import pool


class PytorchModel(Model):
    def __init__(self, model_dir: Path, device: str, **kwargs):
        self.load(model_dir, device)

    def load(self, model_dir: str, device: str):
        self.model = AutoModel.from_pretrained(model_dir).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.device = device

    def unload(self):
        self.model.to("cpu")
        del self.model
        del self.tokenizer

    def infer(self, batch: list[str], normalized: bool = True):
        """Infer embeddings for a batch of passages"""
        with torch.inference_mode():
            inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=512).to(self.model.device)
            outputs = self.model(**inputs)
            pooled = pool(outputs.last_hidden_state, inputs.attention_mask, method="avg")
            if normalized:
                pooled = F.normalize(pooled, p=2, dim=1)
        return pooled.cpu().numpy()