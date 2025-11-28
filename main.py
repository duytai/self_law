from tqdm import tqdm
from rich import print
from functools import partial
from datasets import Dataset
import dataset, llm, utils

if __name__ == '__main__':
    articles = dataset.load_articles('audiovisual_media')
    examples = dataset.load_examples('violation')
    to_violation = partial(utils.to_example, 'Article', 'Violation')

    examples = examples.map(to_violation)
    synthesized = Dataset.from_list([])

    print(f'found {len(examples)} examples!')
    for article in tqdm(articles['content']):
        choice = examples.shuffle(seed=42).select(range(3))
        few_shot = '\n\n'.join([x['example'] for x in choice])
        violation = llm.create_violation(few_shot, article)
        if violation:
            violation = utils.remove_starting(violation, 'Violation:')
            item = to_violation(dict(input=article, output=violation))
            synthesized = synthesized.add_item(item)
            # print(f'[bold green]Article: [/bold green]{article}')
            # print(f'[bold green]Violation: [/bold green]{violation}')
            # print('----')
    print(f'created {len(synthesized)}!')