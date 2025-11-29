from langchain_core.prompts import ChatPromptTemplate
from typing import List
import module, re

def call(prompt: ChatPromptTemplate, few_shot: str, query: str) -> List[str]:
    chain = prompt | module.llm
    message = chain.invoke(dict(few_shot=few_shot, query=query))
    splits = re.split(r'(E[12]:)', message.content)
    return [
        v.strip()
        for k, v in zip(splits, splits[1:])
        if k in {'E1:', 'E2:'} and v.strip()
    ]

create_violation_prompt = ChatPromptTemplate.from_template(
"""
Simulate a conversation between two violation extractors, E1 and E2, who are tasked with extracting violations from a given regulation article.
Each extracted violation is automic and standalone violation. 
Response blank if nothing to extract.

Use the following few-shot examples as guidance:
{{few_shot | trim}}

Now, start the conversation.
{{query}}
""".strip(), template_format='jinja2')

create_scenario_prompt = ChatPromptTemplate.from_template(
"""
Simulate a conversation between two scenario proposer, E1 and E2, who are tasked with creating realistic, real-life scenarios that illustrate a given violation. 
Do not add comments, judgments, and explanations.

Use the following few-shot examples as guidance:
{{few_shot | trim}}

Now, start the conversation.
{{query}}
""".strip(), template_format='jinja2')

refine_scenario_prompt = ChatPromptTemplate.from_template(
"""
Simulate a long conversation between feedback analyst E1 and rewriter E2.
E1: Analyzes the given scenario and provides specific guidance on how to make the violation harder to detect.
E2: Uses the feedback to rewrite and refine the scenario so that the underlying violation still exists, but is less obvious and more difficult to identify.

Use the following few-shot examples as guidance:
{{few_shot | trim}}

Now, start the conversation.
{{query}}
""".strip(), template_format='jinja2')