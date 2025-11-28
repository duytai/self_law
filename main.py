from typing import Dict
from tqdm import tqdm
import llm, dataset

def main():
    articles = dataset.load_articles('audiovisual_media')
    examples = dataset.load_examples('violation')
    examples = examples.map(llm.create_violation_example)

    for article in tqdm(articles['content']):
        choice = examples.shuffle(42).select(range(5))
        few_shot = '\n\n'.join([x['example'] for x in choice])
        violations = llm.create_violation_call(few_shot, article)
        for violation in violations:
            print(repr(violation))

if __name__ == '__main__':
    main()