from tqdm import tqdm
from functools import partial
from datasets import Dataset, concatenate_datasets as concat
from typing import Callable, Tuple
from langchain_core.prompts import ChatPromptTemplate
from rich import print
import llm, dataset, utils

def filter_loop(
    name: str,
    data: Dataset,
    prompt: ChatPromptTemplate,
    few_shot_size: int = 4,
) -> Dataset:
    result = []
    examples = dataset.load_examples(name)
    parts = [
        data['input'][i * few_shot_size: (i + 1) * few_shot_size]
        for i in range(few_shot_size)
        if i * few_shot_size < len(data['input'])
    ]

    choice = examples.select(range(few_shot_size))
    for part in parts:
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
        print(response)
        break

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

def main():
    articles = dataset.load_articles('audiovisual_media')

    articles = articles.select(range(2))

    to_example = partial(utils.to_example, 'Article')
    violations = generate_loop('violation', articles, to_example, llm.create_violation_prompt)
    print(f'[bold blue]Violation: {len(violations)}[/bold blue]')

    violations = violations.select(range(2))

    to_example = partial(utils.to_example, 'Violation')
    scenarios = generate_loop('scenario', violations, to_example, llm.create_scenario_prompt)
    False and utils.avg_similarity([x['input'] for x in scenarios])
    print(f'[bold blue]Scenario: {len(scenarios)}[/bold blue]')

    scenarios = scenarios.select(range(2))
    to_example = partial(utils.to_example, 'Scenario')
    feedback = generate_loop('refinement', scenarios, to_example, llm.refine_scenario_prompt)
    print(f'[bold blue]Feedback: {len(feedback)}[/bold blue]')

    feedback = feedback.select(range(10))
    answers = filter_loop('fil_scenario', feedback, llm.filter_scenario_prompt)
    for r in answers:
        print(f'[bold green]Scenario:[/bold green] {r["input"]}')

    print(f'[bold blue]Final: {len(answers)}[/bold blue]')

if __name__ == '__main__':
    main()