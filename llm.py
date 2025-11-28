from langchain_core.prompts import ChatPromptTemplate
from typing import List
import module, re

create_violation_prompt = ChatPromptTemplate.from_template(
"""
Simulate a conversation between two violation extractors, E1 and E2, who are tasked with extracting violations from a given regulation article.
Each extracted violation is automic and standalone violation. 
Response blank if nothing to extract.

{{few_shot | trim}}

Now, begin the conversation.
Article: {{article | trim}}
""".strip(), template_format='jinja2')

def create_violation(few_shot: str, article: str) -> List[str]:
    chain = (create_violation_prompt | module.llm)
    message = chain.invoke(dict(few_shot=few_shot, article=article))
    splits = re.split(r'(E[12]:)', message.content)
    return [
        v.strip()
        for k, v in zip(splits, splits[1:])
        if k in {'E1:', 'E2:'} and v.strip()
    ]