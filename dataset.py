from yaml import safe_load
from pathlib import Path
from datasets import Dataset

def load_articles(name: str) -> Dataset:
    data_dir = 'data/regulation/'
    data = safe_load(Path(data_dir + name + '.yaml').read_text())
    return Dataset.from_list(data['articles'])

def load_examples(name: str) -> Dataset:
    data_dir = 'data/example/'
    data = safe_load(Path(data_dir + name + '.yaml').read_text())
    result = []

    for row in data:
        _input = row['input'].strip()
        if not row['outputs']:
            example = dict(input=_input, output='')
            result.append(example)
        for output in row['outputs']:
            example = dict(input=_input, output=output)
            result.append(example)

    return Dataset.from_list(result)