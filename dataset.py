from yaml import safe_load
from pathlib import Path
from datasets import Dataset, load_dataset

def load_articles(name: str) -> Dataset:
    data_dir = 'data/regulation/'
    data = safe_load(Path(data_dir + name + '.yaml').read_text())
    data = [dict(input=x['content'], outputs=[]) for x in data['articles']]
    return Dataset.from_list(data)

def load_examples(name: str) -> Dataset:
    data_dir = 'data/example/'
    data = safe_load(Path(data_dir + name + '.yaml').read_text())
    return Dataset.from_list(data)

def load_outputs(name: str) -> Dataset:
    data_file = 'output/' + name + '.jsonl'
    return load_dataset('json', data_files=data_file, split='train')