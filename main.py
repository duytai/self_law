from tqdm import tqdm
from rich import print
from functools import partial
from utils import ParseOptions
from typing import Set
from datasets import Dataset, concatenate_datasets as concat
import dataset, llm, utils

def parse_violation(articles: Set[str], option: ParseOptions) -> Dataset:
    examples = dataset.load_examples('violation')
    to_violation = partial(utils.to_example, 'Article', 'Violation')

    examples = examples.map(to_violation)
    result = Dataset.from_list([])

    for _ in tqdm(range(option.rounds)):
        for article in sorted(articles):
            merged = concat([examples, result])
            choice = merged.shuffle(42).select(range(option.shots))
            few_shot = '\n\n'.join([x['example'] for x in choice])

            violation = llm.create_violation(few_shot, article)
            if violation:
                violation = utils.remove_starting(violation, 'Violation:')
                _item = to_violation(dict(input=article, output=violation))
                result = result.add_item(_item)
                continue
            articles.remove(article)

    return result.remove_columns('example')

def parse_scenario(violations: Set[str], option: ParseOptions):
    examples = dataset.load_examples('scenario')
    to_scenario = partial(utils.to_example, 'Violation', 'Scenario')

    examples = examples.map(to_scenario)
    result = Dataset.from_list([])

    for violation in sorted(violations):
        print(f'[bold green]Violation: [/bold green] {violation}')

if __name__ == '__main__':
    parse_option = ParseOptions()

    data = set(dataset.load_articles('audiovisual_media')['content'])
    violations = parse_violation(data, parse_option)

    data = set(violations['output'])
    parse_scenario(data, parse_option)