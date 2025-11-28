from typing import Dict

def to_example(key: str, value: str, x: Dict):
    x['example'] = f'{key}: {x["input"]}\n{value}: {x["output"]}'
    return x

def remove_starting(value: str, starting: str):
    value = value[len(starting):] if value.startswith(starting) else value
    return value.strip()
