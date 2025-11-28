from tqdm import tqdm
from rich import print
from functools import partial
from utils import ParseOptions
from datasets import Dataset, concatenate_datasets as concat
import dataset, llm, utils

def parse_violation(name: str, option: ParseOptions) -> Dataset:
    articles = dataset.load_articles(name)
    examples = dataset.load_examples('violation')
    to_violation = partial(utils.to_example, 'Article', 'Violation')

    examples = examples.map(to_violation)
    result = Dataset.from_list([])
    visited = set()

    for _ in tqdm(range(option.rounds)):
        for article in articles['content']:
            if article in visited:
                continue

            merged = concat([examples, result])
            choice = merged.shuffle(42).select(range(option.shots))
            few_shot = '\n\n'.join([x['example'] for x in choice])

            violation = llm.create_violation(few_shot, article)
            if violation:
                violation = utils.remove_starting(violation, 'Violation:')
                _item = to_violation(dict(input=article, output=violation))
                result = result.add_item(_item)
                continue

            visited.add(article)

    return result.remove_columns('example')

def parse_scenario(name: str, option: ParseOptions):
    examples = dataset.load_examples('scenario')
    to_violation = partial(utils.to_example, 'Violation', 'Scenario')
    for example in examples:
        print(example)

if __name__ == '__main__':
    parse_option = ParseOptions()
    for item in parse_violation('audiovisual_media', parse_option):
        print(f'[bold green]Violation:[/bold green] {item["output"]}')
    parse_scenario('audiovisual_media', parse_option)