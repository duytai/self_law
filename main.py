from tqdm import tqdm
from functools import partial
from datasets import Dataset, concatenate_datasets as concat
from langchain_core.prompts import ChatPromptTemplate
from typing import Callable, List
from rich import print
import llm, dataset, utils, math

def filter_loop(
    name: str,
    data: Dataset,
    prompt: ChatPromptTemplate,
    few_shot_size: int = 5,
    selected_label: str = 'True'
) -> Dataset:
    examples = dataset.load_examples(name)
    parts = [
        data['input'][i * few_shot_size:(i + 1) * few_shot_size]
        for i in range(math.ceil(len(data['input']) / few_shot_size))
    ]
    revs = [
        data['prev'][i * few_shot_size:(i + 1) * few_shot_size]
        for i in range(math.ceil(len(data['input']) / few_shot_size))
    ]
    result = []
    for part, prev in tqdm(zip(parts, revs), desc='Filtering'):
        choice = examples.select(range(few_shot_size))
        shots = [
            f'Q: {item["input"]}\nA: {item["outputs"][0]}'
            for item in choice
        ]
        few_shot = '\n\n'.join(shots)
        query = '\n\n'.join(
            f'Q{idx + 1}: {item}'
            for idx, item in enumerate(part)
        )
        labels = llm.call(prompt, few_shot, query)
        if len(labels) == len(part):
            for label, text, prev_ in zip(labels, part, prev):
                if label in labels:
                    if label == selected_label:
                        result.append(dict(input=text, outputs=[], prev=prev_))

    return Dataset.from_list(result)

def classify_loop(
    name: str,
    data: Dataset,
    prompt: ChatPromptTemplate,
    labels: List[str],
    few_shot_size: int = 5,
) -> Dataset:
    examples = dataset.load_examples(name)
    parts = [
        data['input'][i * few_shot_size:(i + 1) * few_shot_size]
        for i in range(math.ceil(len(data['input']) / few_shot_size))
    ]
    revs = [
        data['prev'][i * few_shot_size:(i + 1) * few_shot_size]
        for i in range(math.ceil(len(data['input']) / few_shot_size))
    ]
    result = []
    for part, prev in tqdm(zip(parts, revs), desc='Classifying'):
        choice = examples.select(range(few_shot_size))
        shots = [
            f'Q: {item["input"]}\nA: {item["outputs"][0]}'
            for item in choice
        ]
        few_shot = '\n\n'.join(shots)
        query = '\n\n'.join(
            f'Q{idx + 1}: {item}'
            for idx, item in enumerate(part)
        )
        response = llm.call(prompt, few_shot, query)
        if len(response) == len(part):
            for p, text, prev_ in zip(response, part, prev):
                if p.startswith(tuple(labels)):
                    selected = next((label for label in labels if p.startswith(label)), None)
                    result.append(dict(input=text, outputs=[selected], prev=prev_))

    return Dataset.from_list(result)

def generate_loop(
    name: str,
    data: Dataset,
    to_example: Callable,
    prompt: ChatPromptTemplate,
    seed: int = 42,
    few_shot_size: int = 3,
) -> Dataset:
    result = []
    examples = dataset.load_examples(name)

    examples = examples.map(to_example)
    data = data.map(to_example)

    for item in tqdm(data, desc='Generating'):
        choice = examples.shuffle(seed).select(range(few_shot_size))
        few_shot = '\n\n'.join([x['example'] for x in choice])
        response = llm.call(prompt, few_shot, item['query'])
        # handle response
        prev = [item['input']]
        if 'prev' in item:
            prev += item['prev'] 
        response = [dict(input=x, outputs=[], prev=prev) for x in response]
        result.extend(response)

    return Dataset.from_list(result)

def generate_scenario():
    names = [
        "audiovisual_media",
        "basic_law_of_governance",
        "combating_crimes_of_terrorism_and_its_financing",
        "crime",
        "printed_materials_and_publication",
        "public_decency",
        "shura_council",
    ]
    for name in tqdm(names, desc='Regulation'):
        articles = dataset.load_articles(name)
        print(len(articles))

        to_example = partial(utils.to_example, 'Article')
        violations = generate_loop('violation', articles, to_example, llm.create_violation_prompt)
        print(f'[bold blue]Violation: {len(violations)}[/bold blue]')

        to_example = partial(utils.to_example, 'Violation')
        scenarios = generate_loop('scenario', violations, to_example, llm.create_scenario_prompt)
        print(f'[bold blue]Scenario: {len(scenarios)}[/bold blue]')

        to_example = partial(utils.to_example, 'Scenario')
        feedback = generate_loop('refinement', scenarios, to_example, llm.refine_scenario_prompt)
        print(f'[bold blue]Feedback: {len(feedback)}[/bold blue]')

        filtered = filter_loop('fil_scenario', feedback, llm.filter_scenario_prompt)
        print(f'[bold blue]Refined: {len(filtered)}[/bold blue]')

        scenarios = concat([scenarios, filtered])
        False and utils.avg_similarity([x['input'] for x in scenarios])
        print(f'[bold blue]Raw: {len(scenarios)}[/bold blue]')

        classified = classify_loop(
            'eval_scenario',
            scenarios,
            llm.standard_eval_scenario_prompt,
            ['VIOLATION', 'AMBIGUOUS', 'LEGAL']
        )

        classified.to_json(f'output/{name}.jsonl')
        print(f'[bold blue]Classified: {len(classified)}[/bold blue]')

def adversarial_test():
    pass

def main():
    adversarial_test()
    # generate_scenario()

if __name__ == '__main__':
    main()
