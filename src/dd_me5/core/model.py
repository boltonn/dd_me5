from pathlib import Path
from typing import Literal

from loguru import logger
from optimum.onnxruntime import ORTModelForFeatureExtraction
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import transformers

from dd_me5.schemas.enums import Device

transformers.logging.set_verbosity(transformers.logging.CRITICAL)


class MultilingualE5:
    def __init__(self, model_dir: Path, device: Device, warmup: bool = False):
        self.load(model_dir=model_dir, device=device, warmup=warmup)

    def load(self, model_dir: Path, device: Device, warmup: bool = False):
        provider = "CPUExecutionProvider" if device == Device.cpu else "CUDAExecutionProvider"
        logger.debug(f"Loading model from {model_dir} onto {device}")

        self.model = ORTModelForFeatureExtraction.from_pretrained(
            model_dir, 
            local_files_only=True,
            execution_provider=provider,
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, use_fast=True)
        self.model_dir = model_dir
        self.device = self.model.device
        logger.info(f"Loaded model from {model_dir} onto {device}")

        if warmup:
            logger.info("Warming up model")
            self.infer(["Hello world!"])

    def unload(self):
        self.model.to("cpu")
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()

    @staticmethod
    def pool(hidden_states, attention_mask, method: Literal["avg", "cls"] = "avg"):
        """Pooling function for a batch of hidden states"""
        if method == "avg":
            pooled = torch.sum(hidden_states * attention_mask.unsqueeze(-1), dim=1) / torch.sum(attention_mask, dim=1, keepdim=True)
        elif method == "cls":
            pooled = hidden_states[:, 0]
        else:
            raise ValueError(f"Pooling method {method} not supported")
        return pooled
    
    def infer(self, batch: list[str], normalize: bool = True):
        """Infer embeddings for a batch of passages"""
        with torch.inference_mode():
            inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            pooled = MultilingualE5.pool(outputs.last_hidden_state, inputs.attention_mask, method="avg")
            if normalize:
                pooled = F.normalize(pooled, p=2, dim=1)
        return pooled.cpu().numpy()