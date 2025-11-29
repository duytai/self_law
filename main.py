from tqdm import tqdm
from functools import partial
from datasets import Dataset
import llm, dataset, utils

def main_loop(name: str, data: Dataset, to_example, prompt) -> Dataset:
    result = []
    examples = dataset.load_examples(name)

    examples = examples.map(to_example)
    data = data.map(to_example)

    for item in tqdm(data):
        choice = examples.shuffle(42).select(range(5))
        few_shot = '\n\n'.join([x['example'] for x in choice])
        result += llm.call(prompt, few_shot, item['query'])

    return Dataset.from_list([
        dict(input=x, outputs=[])
        for x in result
    ])

def main():
    to_example = partial(utils.to_example, 'Article')
    articles = dataset.load_articles('audiovisual_media')
    violations = main_loop('violation', articles, to_example, llm.create_violation_prompt)

    to_example = partial(utils.to_example, 'Violation')
    scenarios = main_loop('scenario', violations, to_example, llm.create_scenario_prompt)

    for scenario in scenarios:
        print(f'{scenario["input"]!r}')

if __name__ == '__main__':
    main()