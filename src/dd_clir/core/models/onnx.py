from pathlib import Path

from loguru import logger
from optimum.onnxruntime import ORTModelForFeatureExtraction
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import transformers

from dd_clir.core.models.base import Model
from dd_clir.core.utils import pool
from dd_clir.schemas.enums import Device

transformers.logging.set_verbosity(transformers.logging.CRITICAL)


class OnnxModel(Model):
    def __init__(self, model_dir: Path, device: Device, warmup: bool = False, quantized: bool = False):
        self.load(model_dir=model_dir, device=device, warmup=warmup, quantized=quantized)

    def load(self, model_dir: Path, device: Device, warmup: bool = False, quantized: bool = False):
        file_name=None
        possible_files = [f.relative_to(model_dir) for f in model_dir.rglob("*.onnx")]
        desired_filename = "model_quantized.onnx" if quantized else "model.onnx"
        assert desired_filename in [f.name for f in possible_files], f"Model {desired_filename} not found in {model_dir}"
        file_relative_path = next(f for f in possible_files if f.name == desired_filename)
        
        provider = "CPUExecutionProvider" if device == Device.cpu else "CUDAExecutionProvider"
        logger.debug(f"Loading model from {model_dir} and {file_name=} onto {device}")

        self.model = ORTModelForFeatureExtraction.from_pretrained(
            model_dir, 
            file_name=file_relative_path,
            local_files_only=True,
            execution_provider=provider,
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, use_fast=True)
        self.model_dir = model_dir
        self.device = self.model.device
        logger.info(f"Loaded model from {model_dir} and {file_name=} onto {device}")

        if warmup:
            logger.info("Warming up model")
            self.infer(["Hello world!"])

    def unload(self):
        self.model.to("cpu")
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()
    
    def infer(self, batch: list[str], normalize: bool = True):
        """Infer embeddings for a batch of passages"""
        with torch.inference_mode():
            inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=512).to(self.device)
            outputs = self.model(**inputs)
            pooled = pool(outputs.last_hidden_state, inputs.attention_mask, method="avg")
            if normalize:
                pooled = F.normalize(pooled, p=2, dim=1)
        return pooled.cpu().numpy()