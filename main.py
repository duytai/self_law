from tqdm import tqdm
from functools import partial
from datasets import Dataset, concatenate_datasets as concat
from langchain_core.prompts import ChatPromptTemplate
from typing import Callable, List, Tuple
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
    result = []
    for part in tqdm(parts):
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
        assert len(labels) == len(part)
        for label, text in zip(labels, part):
            assert label in labels
            if label == selected_label:
                result.append(dict(input=text, outputs=[]))

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
    result = []
    for part in tqdm(parts):
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
        assert len(response) == len(part)
        for p, text in zip(response, part):
            assert p.startswith(tuple(labels))
            selected = next((label for label in labels if p.startswith(label)), None)
            result.append(dict(input=text, outputs=[selected]))

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
        response = [dict(input=x, outputs=[]) for x in response]
        result.extend(response)

    return Dataset.from_list(result)

def generate_scenario():
    name  = 'public_decency'
    articles = dataset.load_articles(name)

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
    scenarios = scenarios.filter(lambda x: x['input'].find('refined scenario:') == -1)
    print(f'[bold blue]Raw: {len(scenarios)}[/bold blue]')

    classified = classify_loop(
        'eval_scenario',
        scenarios,
        llm.standard_eval_scenario_prompt,
        ['VIOLATION', 'AMBIGUOUS', 'LEGAL']
    )

    classified.to_json(f'output/{name}.jsonl')
    print(f'[bold blue]Classified: {len(classified)}[/bold blue]')

def main():
    generate_scenario()

if __name__ == '__main__':
    main()
