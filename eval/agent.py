from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from typing import List
from rich import print
import sys

model_name = 'gpt-4.1-mini'
embeddings = OpenAIEmbeddings(model='text-embedding-3-large')
vectorstore = Chroma(persist_directory='reg_db', embedding_function=embeddings)

scenario = """
A café in Mecca subtly incorporates branding for a new product into its interior design.
The product’s logo and colors appear as part of the wallpaper pattern and on the menu’s decorative borders.
Instead of direct advertisements, the menu features poetic descriptions and imagery that allude to the product’s qualities without naming it explicitly.
Officials notice the branding elements but find no clear-cut promotional materials to justify immediate removal.
""".strip('\n')

@tool
def get_relevant_articles(summarized_scenario: str) -> str:
    """
    Query the database for specific articles of a regulation. 
    Use this repeatedly to gather all necessary legal context.
    """
    documents = vectorstore.similarity_search(summarized_scenario, k=2)
    texts = []
    for document in documents:
        regulation = document.metadata['regulation']
        article = document.metadata['article']
        content = document.page_content
        texts.append(f'Regulation: {regulation}\nArticle {article}: {content}')
    return '\n'.join(texts)

@tool
def filter_relevant_articles(scenario: str, retrieved_articles: str) -> str:
    """
    Select only the articles that are legally relevant to the scenario.
    Also explain briefly WHY each article applies.
    """
    prompt = f"""
    You are a Saudi legal expert.

    Scenario:
    {scenario}

    Retrieved legal articles:
    {retrieved_articles}

    Task:
    1. Select ONLY the articles that are legally relevant.
    2. Discard irrelevant or weakly related ones.
    3. For each selected article, explain in ONE sentence why it applies.

    Output format:
    - Regulation X, Article Y: reason
    """
    llm = ChatOpenAI(model=model_name, temperature=0.0)
    return llm.invoke(prompt).content.strip()

@tool
def suggest_additional_articles(scenario: str, current_legal_context: str) -> str:
    """
    Identify missing legal angles and suggest what types of articles
    should be retrieved next.
    """
    prompt = f"""
    You are a Saudi legal expert.

    Scenario:
    {scenario}

    Current legal context:
    {current_legal_context}

    Task:
    1. Identify any missing legal angles (e.g., indirect advertising, religious sanctity, public decency).
    2. Suggest keywords or article themes that should be searched next.
    3. Do NOT hallucinate article numbers.

    Output:
    - Missing angle: ...
    - Suggested search keywords: ...
    """
    llm = ChatOpenAI(model=model_name, temperature=0.3)
    return llm.invoke(prompt).content.strip()

@tool
def evaluate_individual_legality(scenario: str, legal_context: str, temperature: float) -> str:
    """
    Classify violation status using a specific temperature (0.0, 0.5, or 1.0).
    Output: AGREE (violation) or DISAGREE (no violation).
    """
    voting_prompt = (
        f"You are a Saudi legal expert.\n\n"
        f"Legal context: {legal_context}\n\n"
        f"Scenario: {scenario}\n\n"
        "Task: Determine if there is a violation.\n"
        "Answer: 'AGREE' (violation) or 'DISAGREE' (no violation)."
    )
    # The temperature is passed dynamically to ensure diverse reasoning
    llm = ChatOpenAI(model=model_name, temperature=temperature)
    response = llm.invoke(voting_prompt)
    return response.content.strip()

system_prompt = (
    "You are a Saudi legal expert. Mission: Identify compliance status.\n\n"
    "Steps:\n"
    "1. get_relevant_articles\n"
    "2. filter_relevant_articles\n"
    "3. suggest_additional_articles\n"
    "4. You MUST call 'evaluate_individual_legality' EXACTLY THREE TIMES.\n"
    " - Call 1: temperature 0.0\n"
    " - Call 2: temperature 0.5\n"
    " - Call 3: temperature 1.0\n"
    "5. Perform a majority vote. Output the final decision and the justifying articles."
)

legal_agent = create_agent(
    model=model_name,
    tools=[
        get_relevant_articles,
        filter_relevant_articles,
        suggest_additional_articles,
        evaluate_individual_legality,
    ],
    system_prompt=system_prompt
)

results = legal_agent.invoke({'messages': [{'role': 'user', 'content': f'Scenario: {scenario}'}]})
for message in results['messages']:
    print(message.content)
    print('----')
