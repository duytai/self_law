from tqdm import tqdm
import llm, dataset

def create_violation(name: str):
    result = []
    articles = dataset.load_articles(name)
    examples = dataset.load_examples('violation')
    examples = examples.map(llm.create_violation_example)

    for article in tqdm(articles['content']):
        choice = examples.shuffle(42).select(range(5))
        few_shot = '\n\n'.join([x['example'] for x in choice])
        result += llm.call(llm.create_violation_prompt, few_shot, article)

    return result

def main():
    violations = create_violation('audiovisual_media')
    for x in violations:
        print(repr(x))

if __name__ == '__main__':
    main()