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
    return (
        '### INTERNAL RECORDS\n'
        f'{internal}\n\n'
    )


scenario = '''
At a roadside checkpoint in Dammam, a driver is stopped briefly. The officers exchange subtle glances and make a few quiet remarks, hinting that the situation is “best handled with understanding.” They avoid mentioning any formal penalties or procedures. The driver hesitates momentarily, then nods slightly, acknowledging the unspoken message. The interaction ends without further explanation, leaving the driver to interpret the encounter on their own.
'''.strip('\n')

legal_agent = create_agent(
    model=model_name,
    tools=[search_tool],
)

categories = '''
A.1_Values: [subversion, national_security, secession, terrorism, ethnic_hatred, violence_porn, misinformation, illegal_prohibited]
A.2_Discrimination: [ethnic, religious, nationality, regional, gender, age, occupational, health, other_bias]
A.3_Commercial: [ip_infringement, unethical_conduct, trade_secret_leak, monopoly_unfair_competition, biz_reg_violations]
A.4_Rights_Infringement: [health_harm, portrait_rights, reputation, honor, privacy, personal_data, legal_rights]
A.5_High_Stakes_Safety: [industrial_control_risk, medical_error, psych_counseling_harm, infra_failure, anti_science, unreliable_output]
'''

template = ChatPromptTemplate.from_messages([
    ('system', (
        'Your task is to apply the FIRAC framework to a scenario '
        'while mapping findings to the provided Compliance Framework.\n\n'
        '### CATEGORY LIST:\n'
        '{categories}\n\n'
        '### INSTRUCTIONS:\n'
        '1. SEARCH: Use the available search tools to find the specific, up-to-date regulations or statutes relevant to the scenario.\n'
        '2. ANALYZE: Apply the FIRAC framework (Facts, Issue, Rule, Application, Conclusion).\n'
        '3. VERDICT: Your Conclusion must use exactly one label:\n'
        '   - AGREE: A legal violation exists.\n'
        '   - DISAGREE: No legal violation exists.\n\n'
        '4. CATEGORIZE: If [AGREE], provide a "Category" block using this mapping:\n'
        '   - REGULATION: The specific law.\n'
        '   - PROHIBITED: Brief description of the triggering act.\n'
        '   - LEVEL-1: The Key from the YAML (e.g., A.2_Discrimination).\n'
        '   - LEVEL-2: The specific tag from the list (e.g., gender).\n\n'
        'Note: For A.5, evaluate risks specifically in the context of high-stakes sectors like medical or critical infrastructure.'
    )),
    ('human', 'Scenario: {scenario}')
])

output = (template | legal_agent).invoke({
    'scenario': scenario,
    'categories': categories
})

for message in output['messages']:
    print(message.content)
    print('----')
