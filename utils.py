from typing import Dict

def to_example(name: str, x: Dict) -> Dict:
    parts = [f'{name}: {x["input"].strip()}']
    parts.extend(
        f'E{idx % 2 + 1}: {output.strip()}'
        for idx, output in enumerate(x['outputs'])
    )
    x['example'] = '\n'.join(parts)
    x['query'] = parts[0]
    return x