from pathlib import Path

from dd_clir.core.models.onnx import OnnxModel
from dd_clir.schemas.enums import Device

# providing intelligent defaults based off the model

class BgeM3(OnnxModel):
    def __init__(
        self, 
        model_dir: Path, 
        device: Device = Device.cpu, 
        warmup: bool = False, 
        quantized: bool = True
    ):
        super().__init__(model_dir=model_dir, device=device, warmup=warmup, quantized=quantized)


class MultilingualE5(OnnxModel):
    def __init__(
        self, 
        model_dir: Path, 
        device: Device = Device.cuda, 
        warmup: bool = True, 
        quantized: bool = False
    ):
        super().__init__(model_dir=model_dir, device=device, warmup=warmup, quantized=quantized)