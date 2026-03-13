from typing import Dict, List
from omegaconf import DictConfig, OmegaConf
from datasets import load_dataset, Dataset
from rich import print
import torch

def to_example(name: str, x: Dict) -> Dict:
    parts = [f'{name}: {x["input"].strip()}']
    parts.extend(
        f'E{idx % 2 + 1}: {output.strip()}'
        for idx, output in enumerate(x['outputs'])
    )
    x['example'] = '\n'.join(parts)
    x['query'] = parts[0]
    return x

def avg_similarity(data: List[str]):
    from sentence_transformers import SentenceTransformer, util
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(data, normalize_embeddings=True)
    similarity_matrix = util.cos_sim(embeddings, embeddings)

    lower_triangular_values = similarity_matrix[
        torch.tril(torch.ones(similarity_matrix.shape), diagonal=-1) == 1
    ]
    mean_similarity = lower_triangular_values.mean().item()
    print(f'[bold green]Sim mean: [/bold green] {mean_similarity}')

def format_legal_document(doc):
    m = doc.metadata
    return (
        f"Regulation: {m.get('regulation', 'Unknown')}\n"
        f"Article: {m.get('article', 'N/A')} ({m.get('country', 'General')})\n"
        f"Content: {doc.page_content.strip()}\n"
    )

def make_dataset(options: DictConfig) -> Dataset:
    dataset = load_dataset(
        path=options.path,
        name=options.name,
        data_files=options.data_files,
        split=options.split
    )
    if hasattr(options, 'map'):
        dataset = dataset.map(eval(options.map))
    if hasattr(options, 'field'):
        dataset = dataset[options.field]
    return dataset
