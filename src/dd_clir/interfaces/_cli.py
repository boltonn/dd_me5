import json
import os
from pathlib import Path

from loguru import logger
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from dd_clir.core.models.pytorch import PytorchModel


os.environ["TOKENIZERS_PARALLELISM"] = "false"

def _get_text_from_json(doc: dict) -> str:
    if "data_type" in doc and doc["data_type"] in ["document", "email", "chat"]:
        if 'summary' in doc:
            return doc['summary'].strip()
        elif 'text' in doc and doc['text']['source']:
            return doc['text']['source'].strip()

def read_json(file_path: Path) -> str:
    with open(file_path, 'rb') as fb:
        data = json.load(fb)
    return data


def write_embedding(file_path: Path, embedding: np.ndarray):
    """Append embedding to json file"""
    data = read_json(file_path)
    data["embedding"] = {"text": embedding.tolist()}
    with open(file_path, 'w') as fb:
        json.dump(data, fb)


def load(file_path: Path, min_length: int = 50) -> str:
    data = read_json(file_path)
    text = _get_text_from_json(data)
    if text and len(text) > min_length:
        return {"file_path": file_path, "text": f"passage: {text.strip()}"}

class EmbeddingDataset(Dataset):
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

def collate(output:list[dict]) -> dict:
    output = [o for o in output if o]
    if output:
        return {
            "file_path": [o["file_path"] for o in output],
            "text": [o["text"] for o in output],
        }


def main(
    model_dir: Path, 
    in_dir:Path, 
    device: str,
    batch_size: int = 32,
    num_workers: int = 4,
    quantized: bool = False
):
    
    model = PytorchModel(model_dir=model_dir, device=device, quantized=quantized)
    dataset = EmbeddingDataset(in_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate)

    n_batches = len(dataloader)
    progress_bar = tqdm(dataloader, total=n_batches)
    n_images = 0
    progress_bar.set_description("Processed 0 documents")
    for batch in dataloader:
        if batch:
            batch_embeddings = model.infer(batch=batch["text"])
            for embedding, file_path in zip(batch_embeddings, batch["file_path"]):
                write_embedding(file_path=file_path, embedding=embedding)
                n_images += 1
                progress_bar.set_description(f"Processed {n_images} documents")
        progress_bar.update(1)


if __name__ == "__main__":
    
    import argparse

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--in_dir", type=Path, required=True)
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--quantized", type=str2bool, default=False)
    args = parser.parse_args()
    main(**vars(args))
