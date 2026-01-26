from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pydantic import BaseModel, Field
from typing import List, Literal

model_name = 'gpt-4.1-mini'
embeddings = OpenAIEmbeddings(model='text-embedding-3-large')
vectorstore = Chroma(persist_directory='reg_db', embedding_function=embeddings)

# Optimized formatter: More concise for context window efficiency
def format_legal_document(doc):
    m = doc.metadata
    return (
        f"SOURCE: {m.get('regulation', 'Unknown')}\n"
        f"REF: {m.get('article', 'N/A')} ({m.get('country', 'General')})\n"
        f"TEXT: {doc.page_content.strip()}...\n" # Truncate if necessary
    )

@tool
def search_tool(query: str) -> str:
    '''
    Retrieves internal and external legal regulations. 
    Ideal for finding the 'Rule' in FIRAC analysis.
    '''
    print(f'Searching for: {query}')

    ## Search for internal regulations
    docs = vectorstore.similarity_search(query, k=2)
    internal = '\n'.join([format_legal_document(d) for d in docs])

    # Search for external regulations
    llm = ChatOpenAI(model=model_name, temperature=0.7)
    template = ChatPromptTemplate.from_messages([
        ('human', 'Search for current regulations about: {query}')
    ])
    external = (template | llm).invoke({'query': query}).content
    return (
        '### INTERNAL RECORDS\n'
        f'{internal}\n\n'
        '### LIVE WEB UPDATES\n'
        f'{external}\n\n'
        "INSTRUCTION: Synthesize these into the 'Rule' section. "
        'Prioritize specific statutes over general web info.'
    )


scenario = '''
A café in Mecca subtly incorporates branding for a new product into its interior design.
The product's logo and colors appear as part of the wallpaper pattern and on the menu's decorative borders.
Instead of direct advertisements, the menu features poetic descriptions and imagery that allude to the product's qualities without naming it explicitly.
Officials notice the branding elements but find no clear-cut promotional materials to justify immediate removal.
'''.strip('\n')

legal_agent = create_agent(
    model=model_name,
    tools=[search_tool],
)

template = ChatPromptTemplate.from_messages([
    ('system', (
        'You are a legal analyst assistant. Your task is to apply the FIRAC framework to a scenario.\n\n'
        'INSTRUCTIONS:\n'
        '1. SEARCH: Use the available search tools to find the specific, up-to-date regulations or statutes relevant to the scenario.\n'
        '2. ANALYZE: Apply the FIRAC framework (Facts, Issue, Rule, Application, Conclusion).\n'
        '3. VERDICT: Your Conclusion must use exactly one label:\n'
        '   - AGREE: A legal violation exists.\n'
        '   - DISAGREE: No legal violation exists.\n\n'
        'Always cite the specific regulation or code section you found during your search.'
    )),
    ('human', 'Scenario: {scenario}')
])

output = (template | legal_agent).invoke({'scenario': scenario})
for message in output['messages']:
    print(message.content)
    print('----')
