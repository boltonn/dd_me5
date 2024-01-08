import json
import os
from pathlib import Path

from loguru import logger
import numpy as np
from torch.utils.data import Dataset, DataLoader

from dd_me5.core.model import MultilingualE5


os.environ["TOKENIZERS_PARALLELISM"] = "false"

def _get_text_from_json(doc: dict) -> str:
    if doc["data_type"] in ["document", "email", "chat"]:
        if 'summary' in doc:
            return doc['summary'].strip()
        elif 'text' in doc:
            return doc['text']['source'].strip()

def read_json(file_path: Path) -> str:
    with open(file_path, 'rb') as fb:
        data = json.load(fb)
    return data

def load(file_path: Path, min_length: int = 50) -> str:
    text = _get_text_from_json(file_path)
    if text and len(text) > min_length:
        return {"file_path": file_path, "text": text.strip}

class ME5Dataset(Dataset):
    def __init__(self, input_dir: Path):
        file_paths = [p for p in input_dir.glob('**/*.json')]
        self.file_paths = np.array(file_paths)
        logger.info(f"Loaded {len(self.file_paths)} files from {input_dir}")
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        data = load(file_path)
        if data:
            return data

def _collate(output:list[dict]) -> dict:
    # assumes the same keys are in every file
    keys = output[0].keys()
    collated = dict()
    for key in keys:
        collated[key] = [o[key] for o in output]
    return collated


def main(model_dir: Path, input_dir:Path):
    model = MultilingualE5(model_dir=model_dir)
    dataset = ME5Dataset(input_dir)
    dataloader = DataLoader(dataset, batch_size=32, num_workers=8, collate_fn=_collate)
    for batch_num, batch in enumerate(dataloader):
        if batch_num == 0:
            break
