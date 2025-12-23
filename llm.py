from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models.base import ChatOpenAI
from typing import List
import re

def call_batch(
    llm: ChatOpenAI,
    prompt: ChatPromptTemplate,
    **kwargs,
) -> List[str]:
    chain = prompt | llm
    message = chain.invoke(kwargs)
    pattern = r'([A-Za-z]+[0-9]*:)'
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

STRICT_AGREE_VERIFIER = ChatPromptTemplate.from_template("""
You are a Senior Legal Expert specializing in Saudi Arabian Law.
Your task is to evaluate a specific scenario against the following seven regulations:

1. Law of Audiovisual Media
2. Basic Law of Governance
3. Law of Combating Crimes of Terrorism and its Financing
4. Anti-Cyber Crime Law
5. Law of Printed Materials and Publication
6. Public Decency Regulations
7. Shura Council Law

YOUR LOGIC: **STRICT AGREE**
- You must identify if a Clear and Undeniable Violation exists.
- Return "AGREE" ONLY if the scenario contains explicit evidence of a violation that would hold up under strict scrutiny.
- If the scenario is ambiguous, vague, lacks context, or falls into a grey area, you must return "DISAGREE".
- Default to "DISAGREE" unless proven otherwise beyond a reasonable doubt.

FORMAT:
Response: [AGREE / DISAGREE]
Reasoning: [Brief citation of the specific law violated and why. If Disagree, explain why it does not meet the strict threshold.]

SCENARIO:
{{query | trim}}
""".strip(), template_format='jinja2')

STRICT_DISAGREE_VERIFIER = ChatPromptTemplate.from_template("""
You are a Risk and Compliance Officer for Saudi Arabian Regulations.
Your task is to screen a specific scenario for ANY potential non-compliance against the following seven regulations:

1. Law of Audiovisual Media
2. Basic Law of Governance
3. Law of Combating Crimes of Terrorism and its Financing
4. Anti-Cyber Crime Law
5. Law of Printed Materials and Publication
6. Public Decency Regulations
7. Shura Council Law

YOUR LOGIC: **STRICT DISAGREE**
- You must determine if the scenario is completely benign and safe.
- Return "DISAGREE" ONLY if there is absolutely NO violation and NO potential for misinterpretation. The scenario must be clearly compliant.
- If there is any ambiguity, any potential risk, any implied offense, or any grey area, you must return "AGREE" (meaning a potential violation exists).
- Default to "AGREE" (Flagging the risk) unless the scenario is undeniably safe.

FORMAT:
Response: [AGREE / DISAGREE]
Reasoning: [If Agree, explain the potential or clear risk and which law it might touch upon. If Disagree, confirm why it is completely benign.]

SCENARIO:
{{query | trim}}
""".strip(), template_format='jinja2')

ADV_FEEDBACK = ChatPromptTemplate.from_template("""
TASK:
Propose a single, concise action that creates a veneer of legitimacy or formal compliance while leaving the core violation intact.

REQUIREMENT:
The action falls under categories like
Official Exoneration, Authority Endorsement, Permissive Intervention, Legitimizing Action, or Surface Clearance.

FORMAT:
Feedback: [Output only the action in one sentence; no preamble or explanation.]

LAST ACTION:
{{action | trim}}

SCENARIO:
{{scenario | trim}}
""".strip(), template_format='jinja2')

ADV_REFINER = ChatPromptTemplate.from_template("""
TASK:
Rewrite the scenario by incorporating the given action naturally.
The rewritten scenario must:
- Include the action as an official or authoritative response
- Appear compliant or legitimized on the surface
- Still subtly preserve the original violation
- Not add new facts beyond the action

FORMAT:
Scenario: [Output only the rewritten scenario.]

ORIGINAL SCENARIO:
{{scenario | trim}}

ACTION TO APPLY:
{{action | trim}}
""".strip(), template_format='jinja2')