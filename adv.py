from omegaconf import DictConfig
from typing import Dict
import llm, hydra, dataset, module

def rewrite_once():
    ...

def dual_verify(scenario: str) -> Dict:
    params = dict(llm=module.gpt4, few_shot='', query=scenario)
    strict_adherence = llm.call_batch(**params, prompt=llm.STRICT_AGREE_VERIFIER)
    risk_adverse = llm.call_batch(**params, prompt=llm.STRICT_DISAGREE_VERIFIER)
    l0, l1 = strict_adherence[0], risk_adverse[0]
    # Label is one of: CONTROVERSIAL, AGREE, DISAGREE
    label = l0 if l0 == l1 else 'CONTROVERSIAL'
    return dict(
        label=label,
        agree_verifier=strict_adherence[1],
        disagree_verifier=risk_adverse[1],
    )

@hydra.main(version_base=None)
def adv_loop(cfg: DictConfig):
    data = dataset.load_outputs('public_decency')
    for row in data.select(range(10)):
        result = dual_verify(scenario=row['input'])
        print(result)

if __name__ == '__main__':
    adv_loop()