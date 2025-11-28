from typing import Dict, Any, Callable
from pydantic import BaseModel

class ParseOptions(BaseModel):
    example_name: str
    example_key: str
    example_value: str
    llm_call: Callable
    shots: int = 5
    rounds: int = 2
    seed: int = 42
    example_starting: str = None

    def model_post_init(self, context: Any, /) -> None:
        self.example_starting = self.example_value + ':'

def to_example(key: str, value: str, x: Dict):
    x['example'] = f'{key}: {x["input"]}\n{value}: {x["output"]}'
    return x

def remove_starting(value: str, starting: str):
    value = value[len(starting):] if value.startswith(starting) else value
    return value.strip()
