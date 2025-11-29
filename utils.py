from typing import Dict, List
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