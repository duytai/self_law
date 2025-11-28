from tqdm import tqdm
from rich import print
from functools import partial
from datasets import Dataset, concatenate_datasets
import dataset, llm, utils

SEED = 42

def extract_violation(name: str, n_shots: int = 3, n_rounds: int = 10) -> Dataset:
    articles = dataset.load_articles(name)
    examples = dataset.load_examples('violation')
    to_violation = partial(utils.to_example, 'Article', 'Violation')

    examples = examples.map(to_violation)
    created = Dataset.from_list([])
    visited = set()

    for _ in tqdm(range(n_rounds)):
        for article in articles['content']:
            if article in visited:
                continue

            merged = concatenate_datasets([examples, created])
            choice = merged.shuffle(SEED).select(range(n_shots))
            few_shot = '\n\n'.join([x['example'] for x in choice])

            violation = llm.create_violation(few_shot, article)
            if violation:
                violation = utils.remove_starting(violation, 'Violation:')
                _item = to_violation(dict(input=article, output=violation))
                created = created.add_item(_item)
                continue

            visited.add(article)

    return created.remove_columns('example')

if __name__ == '__main__':
    for item in extract_violation('audiovisual_media'):
        print(f'[bold green]Violation:[/bold green] {item["output"]}')