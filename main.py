from typing import Dict
from tqdm import tqdm
import llm, dataset

def to_conversation(x) -> Dict:
    example = f'Article: {x["input"].strip()}'
    for idx, output in enumerate(x['outputs']):
        label = idx % 2 + 1
        example += f'\nE{label}: {output.strip()}'
    x['example'] = example
    return x

def main():
    articles = dataset.load_articles('audiovisual_media')
    examples = dataset.load_examples('violation')
    examples = examples.map(to_conversation)

    for article in tqdm(articles['content']):
        choice = examples.shuffle(42).select(range(5))
        few_shot = '\n\n'.join([x['example'] for x in choice])
        violations = llm.create_violation(few_shot, article)
        for violation in violations:
            print(repr(violation))

if __name__ == '__main__':
    main()