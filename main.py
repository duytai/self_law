from tqdm import tqdm
from functools import partial
from datasets import Dataset, concatenate_datasets as concat
from typing import Callable, Tuple
from langchain_core.prompts import ChatPromptTemplate
from rich import print
import llm, dataset, utils

def refine_loop(
    name: str,
    data: Dataset,
    to_example: Callable,
    prompt: ChatPromptTemplate,
    seed: int = 42,
    few_shot_size: int = 3,
) -> Tuple[Dataset, Dataset]:
    feedback, refine = [], []
    examples = dataset.load_examples(name)

    examples = examples.map(to_example)
    data = data.map(to_example)

    for item in tqdm(data, desc='Refining'):
        choice = examples.shuffle(seed).select(range(few_shot_size))
        few_shot = '\n\n'.join([x['example'] for x in choice])
        response = llm.call(prompt, few_shot, item['query'])

        # handle response
        response = [dict(input=x, outputs=[]) for x in response]
        even_responses = response[::2]
        odd_responses = response[1::2]

        min_size = min(len(even_responses), len(odd_responses))
        feedback.extend(even_responses[:min_size])
        refine.extend(odd_responses[:min_size])

    feedback = Dataset.from_list(feedback)
    refine = Dataset.from_list(refine)

    return feedback, refine

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

    to_example = partial(utils.to_example, 'Article')
    violations = generate_loop('violation', articles, to_example, llm.create_violation_prompt)

    to_example = partial(utils.to_example, 'Violation')
    scenarios = generate_loop('scenario', violations, to_example, llm.create_scenario_prompt)

    False and utils.avg_similarity([x['input'] for x in scenarios])

    scenarios = scenarios.select(range(180))
    to_example = partial(utils.to_example, 'Scenario')
    feedback, refine = refine_loop('refinement', scenarios, to_example, llm.refine_scenario_prompt)

    combined = concat([scenarios, refine])
    for r in combined:
        print(f'[bold green]Scenario: [/bold green] {r["input"]}')
    print(f'ALL: {len(combined)}')

if __name__ == '__main__':
    main()