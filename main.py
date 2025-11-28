from tqdm import tqdm
from functools import partial
from datasets import Dataset
import llm, dataset, utils

def main_loop(data: Dataset, to_example, prompt) -> Dataset:
    result = []
    examples = dataset.load_examples('violation')

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
    to_example = partial(utils.to_example, 'Violation')
    articles = dataset.load_articles('audiovisual_media')
    violations = main_loop(articles, to_example, llm.create_violation_prompt)

    for x in violations:
        print(repr(x))

if __name__ == '__main__':
    main()