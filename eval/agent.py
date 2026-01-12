from langchain.agents import create_agent
from langchain_core.tools import tool
from typing import List

scenario = (
    "At a checkpoint in Dammam, a driver is stopped for a minor traffic issue."
    "The officers speak quietly among themselves before telling the driver that the matter could be settled informally, though they don't specify how."
    "They mention that following up at a local office might be necessary but hint that cooperation could make things easier."
    "The driver, unsure of the expectations, offers a polite smile and agrees to comply, while the officers remain composed, leaving the encounter open-ended and subtly charged with unspoken pressure."
)

@tool
def get_relevant_articles(regulation_name: str) -> str:
    """
    Query the database for specific articles of a regulation. 
    Use this repeatedly to gather all necessary legal context.
    """
    # In a real app, this queries your Vector DB or SQL
    db = {
        "Basic Law of Governance": "Article 26: The State protects human rights...",
        "Anti-Bribery Law": "Article 1: Criminalizes seeking a benefit to perform or abstain from an act of duty.",
        "Public Decency Regulations": "Article 5: Officers must act with professional decorum."
    }
    return db.get(regulation_name, f"No detailed articles found for {regulation_name}.")

system_prompt = (
    "You are a Legal Research Assistant. Your goal is to collect ALL relevant articles "
    "from Saudi regulations that might apply to the scenario provided.\n\n"
    "Available Regulations:\n"
    "- Law of Audiovisual Media\n"
    "- Basic Law of Governance\n"
    "- Law of Combating Crimes of Terrorism and its Financing\n"
    "- Anti-Cyber Crime Law\n"
    "- Law of Printed Materials and Publication\n"
    "- Public Decency Regulations\n"
    "- Shura Council Law\n"
    "INSTRUCTIONS:\n"
    "1. Read the scenario.\n"
    "2. Use 'get_relevant_articles' for EACH regulation that seems applicable.\n"
    "3. If an article mentions a related law, search for that as well.\n"
    "4. Do not stop until you have a complete legal profile of the scenario."
)

agent = create_agent(
    model='gpt-4.1-mini',
    tools=[get_relevant_articles],
    system_prompt=system_prompt,
)

result = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": f"Scenario: {scenario}"}
        ]
    }
)

print(result)
