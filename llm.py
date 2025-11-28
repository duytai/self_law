from langchain_core.prompts import ChatPromptTemplate
from typing import Dict
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