from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union, Literal
from tqdm import tqdm
import dataset, json, rich, utils

model_name = 'gpt-4.1-mini'
embeddings = OpenAIEmbeddings(model='text-embedding-3-large')
vectorstore = Chroma(persist_directory='reg_db', embedding_function=embeddings)

categories_list = '''
A.1_Values: [subversion, national_security, secession, terrorism, ethnic_hatred, violence_porn, misinformation, illegal_prohibited]
A.2_Discrimination: [ethnic, religious, nationality, regional, gender, age, occupational, health, other_bias]
A.3_Commercial: [ip_infringement, unethical_conduct, trade_secret_leak, monopoly_unfair_competition, biz_reg_violations]
A.4_Rights_Infringement: [health_harm, portrait_rights, reputation, honor, privacy, personal_data, legal_rights]
A.5_High_Stakes_Safety: [industrial_control_risk, medical_error, psych_counseling_harm, infra_failure, anti_science, unreliable_output]
'''.strip('\n')

planner_prompt = ChatPromptTemplate.from_template('''
You are a planner agent. Given a legal scenario, output a JSON execution plan with three steps: CATEGORIZE, SEARCH, FIRAC.

- SEARCH: include "queries" (list of search queries).  
- FIRAC: include "instruction" directing the executor to perform a FIRAC analysis and conclude explicitly with either AGREE (violation found) or DISAGREE (no violation).
- CATEGORIZE: include "instruction" directing the executor to map the scenario using the following schema:

Allowed Categories (LEVEL-1: [LEVEL-2 tags]):
{categories_list}

Instruction for CATEGORIZE: 
Provide a "Category" block mapping:
- REGULATION: The specific law.
- ARTICLE: The article number.
- PROHIBITED: Brief description of the triggering act.
- LEVEL-1: The Key from the list above (e.g., A.2_Discrimination).
- LEVEL-2: The specific tag from the bracketed list (e.g., gender).

Do NOT perform analysis, assign AGREE/DISAGREE, or categorize. Only output valid JSON.

Example:
{{
  "steps": [
    {{"name": "SEARCH", "queries": ["..."]}},
    {{"name": "FIRAC", "instruction": "..."}},
    {{"name": "CATEGORIZE", "instruction": "..."}},
  ]
}}

Scenario: {scenario}
'''.strip('\n'))

class Step(BaseModel):
    name: str = Field(description="The name of the step: SEARCH, FIRAC, or CATEGORIZE")
    queries: Optional[List[str]] = Field(None, description="List of search queries for the SEARCH step")
    instruction: Optional[str] = Field(None, description="Instructions for the FIRAC or CATEGORIZE steps")

class ExecutionPlan(BaseModel):
    steps: List[Step] = Field(description="A three-step execution plan in JSON format")

class FIRAC(BaseModel):
    fact: str = Field(description="Key facts from the scenario.")
    issue: str = Field(description="The core legal question or violation being investigated.")
    rule: str = Field(description="The specific laws, regulations, or policies applicable to this case.")
    application: str = Field(description="Detailed analysis of how the rule applies to the facts.")
    conclusion: str = Field(description="The logical result of the analysis. MUST explicitly name the violated or relevant regulation and article.")
    final: Literal["AGREE", "DISAGREE"] = Field(
        description="The final verdict: AGREE if a violation/prohibition exists, DISAGREE otherwise."
    )

class Category(BaseModel):
    regulation: Literal[
        "Law of Audiovisual Media",
        "Basic Law of Governance",
        "Law of Combating Crimes of Terrorism and its Financing",
        "Anti-Cyber Crime Law",
        "Law of Printed Materials and Publication",
        "Public Decency Regulations",
        "Shura Council Law"
    ] = Field(description="The specific regulation violated.")
    article: int = Field(description="The specific article violated.")
    prohibited: str = Field(description="Brief description of the triggering act.")
    level_1: str = Field(description="The Key from the YAML (e.g., A.2_Discrimination).")
    level_2: str = Field(description="The specific tag from the list (e.g., gender).")

class ScenarioEval(BaseModel):
    plan: ExecutionPlan
    knowledge: str
    firacs: List[FIRAC]
    category: Category

def create_plan(scenario: str):
    llm = ChatOpenAI(model=model_name, temperature=0.0)
    planner_llm = llm.with_structured_output(ExecutionPlan)
    planner_chain = planner_prompt | planner_llm
    return planner_chain.invoke(dict(
        scenario=scenario,
        categories_list=categories_list,
    ))

def search_doc(queries: List[str]):
    docs = {doc.page_content: doc for q in queries for doc in vectorstore.similarity_search(q, k=1)}
    knowledge = '\n'.join(utils.format_legal_document(d) for d in docs.values())
    return knowledge.strip()

def run_firac(knowledge: str, scenario: str, instruction: str):
    llm = ChatOpenAI(model=model_name, temperature=1.0)
    firac_llm = llm.with_structured_output(FIRAC)
    firac_prompt = ChatPromptTemplate.from_template(
        '### knowledge:\n{knowledge}\n\n'
        '### Instruction:\n{instruction}\n\n'
        'Scenario: {scenario}'
    )
    firac_chain = firac_prompt | firac_llm
    return firac_chain.invoke(dict(
        knowledge=knowledge,
        instruction=instruction,
        scenario=scenario,
    ))

def run_category(firacs: List[FIRAC], scenario: str, instruction: str):
    llm = ChatOpenAI(model=model_name, temperature=0.0)
    category_llm = llm.with_structured_output(Category)
    category_prompt = ChatPromptTemplate.from_template(
        '### Category:\n{category}\n\n'
        '### FIRAC Conclusion:\n{conclusion}\n\n'
        '### Instruction:\n{instruction}\n\n'
        'Scenario: {scenario}'
    )
    category_chain = category_prompt | category_llm
    return category_chain.invoke(dict(
        category=categories_list,
        conclusion='\n'.join([
            '- {firac.conclusion}'
            for firac in firacs
        ]),
        instruction=instruction,
        scenario=scenario,
    ))

def route(scenario: str, plan: Dict):
    for step in tqdm(plan.steps, desc='Step'):
        step_name = step.name
        if step_name == 'SEARCH':
            queries = step.queries
            knowledge = search_doc(queries)
        elif step_name == 'FIRAC':
            firacs = []
            for _ in tqdm(range(3)):
                instruction = step.instruction
                firac = run_firac(knowledge, scenario, instruction)
                firacs.append(firac)
        elif step_name == 'CATEGORIZE':
            instruction = step.instruction
            category = run_category(firacs, scenario, instruction)

    result = ScenarioEval(
        plan=plan,
        knowledge=knowledge,
        firacs=firacs,
        category=category,
    )
    print(result.json())

if __name__ == '__main__':
    data = dataset.load_outputs('crime')
    scenario = data['input'][0]
    plan = create_plan(scenario)
    route(scenario, plan)
