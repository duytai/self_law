from langchain_core.prompts import ChatPromptTemplate
import module

create_violation_prompt = ChatPromptTemplate.from_template(
"""
{{few_shot | trim}}

Now, from the article, extract one non-duplicate violation.
If no violation is present, produce a blank output.

Article: {{article | trim}}
Violation:
""".strip(), template_format='jinja2')

def create_violation(few_shot: str, article: str) -> str:
    chain = (create_violation_prompt | module.llm)
    message = chain.invoke(dict(few_shot=few_shot, article=article))
    return message.content

create_scenario_prompt = ChatPromptTemplate.from_template(
"""
{{few_shot | trim}}

Now, from the violation, create a real-life scenario.
If no violation is present, produce a blank output.

Violation: {{violation | trim}}
Scenario:
""".strip(), template_format='jinja2')

def create_scenario(few_shot: str, violation: str) -> str:
    chain = (create_scenario_prompt | module.llm)
    message = chain.invoke(dict(few_shot=few_shot, violation=violation))
    return message.content
