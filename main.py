from tqdm import tqdm
from rich import print
from functools import partial
from utils import ParseOptions
from typing import List
from datasets import Dataset, concatenate_datasets as concat
import dataset, llm, utils

def main_loop(inputs: List[str], options: ParseOptions):
    fn = partial(utils.to_example, options.example_key, options.example_value)
    examples = dataset.load_examples(options.example_name)
    examples = examples.map(fn)

    created = Dataset.from_list([])
    ignored_inputs = []
    visited_outputs = []

    for _ in tqdm(range(options.rounds)):
        for input in inputs:
            merged = concat([examples, created])
            choice = merged.shuffle(options.seed).select(range(options.shots))
            few_shot = '\n\n'.join([x['example'] for x in choice])

            response = options.llm_call(few_shot, input)
            if response:
                output = utils.remove_starting(response, options.example_starting)
                if output not in visited_outputs:
                    _item = fn(dict(input=input, output=output))
                    created = created.add_item(_item)
                    continue
                visited_outputs.append(output)
                continue
            ignored_inputs.append(input)
        inputs = list(set(inputs) - set(ignored_inputs))

    return created.remove_columns('example')

if __name__ == '__main__':
    parse_option = ParseOptions(
        example_name='violation',
        example_key='Regulation',
        example_value='Violation',
        llm_call=llm.create_violation
    )
    data = dataset.load_articles('audiovisual_media')['content']
    result = main_loop(data, parse_option)

    parse_option = ParseOptions(
        example_name='scenario',
        example_key='Violation',
        example_value='Scenario',
        llm_call=llm.create_scenario
    )
    data = result['output']
    scenarios = main_loop(data, parse_option)

    for scenario in scenarios:
        print(f'[bold green]Scenario: [/bold green]{scenario["output"]}')