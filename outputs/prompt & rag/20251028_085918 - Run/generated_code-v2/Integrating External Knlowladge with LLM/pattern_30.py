import json
import os

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_core.prompts import PromptTemplate

# Simulated/Placeholder External Knowledge Sources

simulated_ehr_data = {
    "patient_123": {
        "name": "John Doe",
        "age": 55,
        "conditions": ["Hypertension", "Type 2 Diabetes"],
        "medications": ["Metformin", "Lisinopril"],
        "allergies": ["Penicillin"],
        "last_visit": "2023-10-26"
    },
    "patient_456": {
        "name": "Jane Smith",
        "age": 32,
        "conditions": ["Asthma"],
        "medications": ["Albuterol"],
        "allergies": [],
        "last_visit": "2023-11-15"
    }
}

def get_ehr_data(patient_id: str) -> str:
    patient_data = simulated_ehr_data.get(patient_id)
    if patient_data:
        return json.dumps(patient_data, indent=2)
    return f"No EHR data found for patient ID: {patient_id}"

simulated_medical_knowledge = {
    "Hypertension": "High blood pressure. Can lead to heart disease. Treatment often involves lifestyle changes and medication.",
    "Type 2 Diabetes": "A chronic condition that affects the way the body processes blood sugar (glucose). Managed with diet, exercise, and medication.",
    "Asthma": "A condition in which your airways narrow and swell and may produce extra mucus. This can make breathing difficult.",
    "Metformin": "Oral medication used to treat type 2 diabetes. Common side effects include nausea and diarrhea.",
    "Lisinopril": "ACE inhibitor used to treat high blood pressure. Can cause cough or dizziness.",
    "Penicillin": "A common antibiotic. Patients with penicillin allergy should avoid it."
}

def query_medical_knowledge(query: str) -> str:
    query_lower = query.lower()
    results = []
    for key, value in simulated_medical_knowledge.items():
        if query_lower in key.lower() or query_lower in value.lower():
            results.append(f"{key}: {value}")
    if results:
        return "\n".join(results)
    return f"No medical knowledge found for query: {query}"

def search_pubmed(query: str) -> str:
    return f"Placeholder: Searching PubMed for '{query}' would yield recent research papers and clinical trials. For demonstration, consider relevant articles on '{query}' published in the last 6 months."

def check_drug_interactions(drugs: str) -> str:
    drug_list = [d.strip() for d in drugs.split(',')]
    if "Metformin" in drug_list and "Lisinopril" in drug_list:
        return "Placeholder: Potential interaction between Metformin and Lisinopril could include increased risk of kidney issues, especially in patients with pre-existing renal impairment. Monitor kidney function."
    return f"Placeholder: Checking drug interactions for {', '.join(drug_list)} would involve a dedicated drug interaction database. No critical interactions found for this specific combination in the simulated data."

# LLM Integration

# Ensure you have your OpenAI API key set as an environment variable or replace 'os.environ["OPENAI_API_KEY"]' with your actual key.
# For example: os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# External Knowledge Source Abstraction (Tools)

tools = [
    Tool(
        name="EHR System",
        func=get_ehr_data,
        description="Useful for retrieving electronic health record data for a specific patient ID. Input should be a patient ID (e.g., 'patient_123')."
    ),
    Tool(
        name="Medical Knowledge Graph",
        func=query_medical_knowledge,
        description="Useful for querying general medical knowledge, disease information, or basic guidelines. Input should be a medical term or question."
    ),
    Tool(
        name="PubMed Literature Search",
        func=search_pubmed,
        description="Useful for searching real-time medical literature and clinical trial registries. Input should be a medical query or keywords."
    ),
    Tool(
        name="Drug Interaction Checker",
        func=check_drug_interactions,
        description="Useful for checking potential drug-drug interactions. Input should be a comma-separated list of drug names (e.g., 'Metformin, Lisinopril')."
    )
]

# Agentic Orchestration

# Define the prompt for the agent
prompt = PromptTemplate.from_template(
    """You are a helpful AI medical assistant designed to provide clinical decision support. 
    You have access to several specialized tools to gather information. 
    
    When answering a clinical query, first consider if you need patient-specific data from the EHR. 
    Then, consult the Medical Knowledge Graph for general information. 
    For recent research or specific drug interactions, use the PubMed Literature Search or Drug Interaction Checker.
    
    Always provide a comprehensive, evidence-based recommendation, including potential diagnoses, 
    suggested diagnostic tests, personalized treatment options, and potential contraindications or considerations. 
    State explicitly which information came from which source (e.g., "EHR data indicates...", "Medical knowledge suggests...").
    
    Use the following tools:
    
    {tools}
    
    Use the format:
    
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final comprehensive recommendation to the user
    
    Begin!
    
    Question: {input}
    Thought:{agent_scratchpad}"""
)

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# User Interface (Basic CLI)

def main():
    print("MediCounsel AI - Clinical Decision Support System")
    print("Type 'exit' to quit.")
    while True:
        user_query = input("\nEnter your clinical query or patient symptoms: ")
        if user_query.lower() == 'exit':
            break
        try:
            result = agent_executor.invoke({"input": user_query})
            print("\n--- MediCounsel AI Recommendation ---")
            print(result["output"])
            print("-------------------------------------")
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please try rephrasing your query.")

if __name__ == "__main__":
    main()