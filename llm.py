from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models.base import ChatOpenAI
from typing import List
import re

def call(llm: ChatOpenAI, prompt: ChatPromptTemplate, few_shot: str, query: str) -> List[str]:
    chain = prompt | llm
    message = chain.invoke(dict(few_shot=few_shot, query=query))
    pattern = r'([A-Z]+[0-9]*:)'
    splits = re.split(pattern, message.content)
    return [
        v.strip()
        for k, v in zip(splits, splits[1:])
        if re.match(pattern, k) and v.strip()
    ]

CREATE_VIOLATION_PROMPT = ChatPromptTemplate.from_template(
"""
Simulate a conversation between two violation extractors, E1 and E2, who are tasked with extracting violations from a given regulation article.
Each extracted violation is automic and standalone violation. 
Response blank if nothing to extract.

Use the following few-shot examples as guidance:
{{few_shot | trim}}

Now, start the conversation.
{{query | trim}}
""".strip(), template_format='jinja2')

CREATE_SCENARIO_PROMPT = ChatPromptTemplate.from_template(
"""
Simulate a conversation between two scenario proposer, E1 and E2, who are tasked with creating realistic, real-life scenarios that illustrate a given violation. 
Do not add comments, judgments, and explanations.

Use the following few-shot examples as guidance:
{{few_shot | trim}}

Now, start the conversation.
{{query | trim}}
""".strip(), template_format='jinja2')

REFINE_SCENARIO_PROMPT = ChatPromptTemplate.from_template(
"""
Simulate a long conversation between feedback analyst E1 and rewriter E2.
E1: Analyzes the given scenario and provides specific guidance on how to make the violation harder to detect.
E2: Uses the feedback to rewrite and refine the scenario so that the underlying violation still exists, but is less obvious and more difficult to identify.

Use the following few-shot examples as guidance:
{{few_shot | trim}}

Now, start the conversation.
{{query | trim}}
""".strip(), template_format='jinja2')

FILTER_SCENARIO_PROMPT = ChatPromptTemplate.from_template(
"""
Your task is to classify each Q as either a Scenario or Not a Scenario.
Scenario (True):
- The text only describes an event, situation, or narrative with no advice, no evaluation, and no recommendations.
Not a Scenario (False):
- The text contains advice, suggestions, instructions, mitigation strategies, or guidance.
- The text contains evaluation, judgment, critique, or analysis.
- The text mixes description of events with advice or evaluation (this is still False).

Use the following few-shot examples as guidance:
{{few_shot | trim}}

Now classify the following questions:
{{query | trim}}

Answer format:
A1:
A2:
A3:
A4:
...
""".strip(), template_format='jinja2')

STANDARD_EVAL_SCENARIO_PROMPT = ChatPromptTemplate.from_template(
"""
Classify the following scenario under Saudi Arabian law as VIOLATION, LEGAL, or AMBIGUOUS.

Use the following few-shot examples as guidance:
{{few_shot | trim}}

Now classify the following scenarios:
{{query | trim}}

Answer format:
A1:
A2:
A3:
A4:
...
""".strip(), template_format='jinja2')

ADVERSARIAL_REWRITER_PROMPT = """
You are an adversarial scenario rewriter used in a refinement loop.

Objective:
Rewrite the scenario so that the SAME violation still exists, but becomes harder to detect.

Hard Constraints:
- The violation MUST remain.
- The violation MUST be the same.
- Do NOT add new violations.
- Do NOT mention laws or legality.
- Do NOT explain anything.

Adversarial Techniques (use at least ONE):
- Replace explicit violation verbs with neutral language
- Make the violation implicit or procedural
- Frame as routine or justified behavior
- Use omission (leave the violating step implied)

Scenario:
{scenario}

Rewritten Scenario:
""".strip()

VERIFIER_PROMPT = """
You are a violation verifier.

Task:
Determine whether the scenario still contains at least one violation.

Output exactly one token:
AGREE or DISAGREE

Scenario:
{scenario}

Answer:
""".strip()