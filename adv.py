from omegaconf import DictConfig
from typing import Dict
from rich import print
import llm, hydra, dataset, module

def dual_verify(scenario: str) -> Dict:
    params = dict(llm=module.gpt4, few_shot='', query=scenario)
    strict_adherence = llm.call_batch(**params, prompt=llm.STRICT_AGREE_VERIFIER)
    risk_adverse = llm.call_batch(**params, prompt=llm.STRICT_DISAGREE_VERIFIER)
    l0, l1 = strict_adherence[0], risk_adverse[0]
    # Label is one of: CONTROVERSIAL, AGREE, DISAGREE
    label = l0 if l0 == l1 else 'CONTROVERSIAL'
    return dict(
        label=label,
        agree_verifier=strict_adherence,
        disagree_verifier=risk_adverse,
    )

def adv_feedback(scenario: str, action: str) -> str:
    params = dict(llm=module.gpt4, scenario=scenario, action=action)
    feedbacks = llm.call_batch(**params, prompt=llm.ADV_FEEDBACK)
    return feedbacks[0]

def rewrite_once(scenario: str, action: str) -> str:
    params = dict(llm=module.gpt4, scenario=scenario, action=action)
    actions = llm.call_batch(**params, prompt=llm.ADV_REFINER)
    return actions[0]

@hydra.main(version_base=None)
def adv_loop(cfg: DictConfig):
    data = dataset.load_outputs('public_decency')

    for row in data.select([2]):
        action = ''
        verified = dual_verify(row['input'])
        if verified['label'] == 'AGREE':
            print('[bold green]#SCENARIO:[/bold green]', row['input'])
            for _ in range(10):
                action = adv_feedback(row['input'], action)
                scenario = rewrite_once(row['input'], action)
                print('[bold green]# REFINED[/bold green]')
                verified = dual_verify(scenario)
                print(repr(scenario))
                print(verified)
                if verified['label'] == 'DISAGREE':
                    print('OK')
                    break

if __name__ == '__main__':
    adv_loop()