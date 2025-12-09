from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate

import os

# Mock API Keys/Setup (replace with actual if running locally)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")

# --- External Knowledge Sources (Mock Tools) ---

@tool
def search_medical_literature(query: str) -> str:
    """Searches medical literature databases (e.g., PubMed, Embase) for the latest research papers, clinical trials, and meta-analyses based on the given query."""
    if "COVID-19 treatment" in query:
        return "Latest research suggests Paxlovid and remdesivir are effective for early COVID-19 treatment in high-risk patients. Recent studies also explore long COVID mechanisms."
    elif "diabetes management" in query:
        return "New guidelines emphasize personalized diabetes management plans, incorporating continuous glucose monitoring and GLP-1 receptor agonists for cardiovascular benefits."
    else:
        return f"No specific recent literature found for '{query}'. Please refine your query."

@tool
def get_clinical_guidelines(disease: str) -> str:
    """Retrieves current national and international clinical practice guidelines for a specific disease."""
    if "hypertension" == disease.lower():
        return "ACC/AHA 2017 guidelines recommend blood pressure targets <130/80 mmHg for most adults. Lifestyle modifications are first-line, followed by various classes of antihypertensives."
    elif "asthma" == disease.lower():
        return "GINA guidelines emphasize a stepwise approach to asthma management, starting with as-needed low-dose inhaled corticosteroids for most adults and adolescents."
    else:
        return f"No specific clinical guidelines found for '{disease}'."

@tool
def get_drug_information(drug_name: str) -> str:
    """Provides up-to-date information on drug dosages, interactions, contraindications, and side effects for a given drug name."""
    if "metformin" == drug_name.lower():
        return "Metformin: Oral biguanide for type 2 diabetes. Common side effects: GI upset. Contraindications: severe renal impairment. Dosage varies."
    elif "amoxicillin" == drug_name.lower():
        return "Amoxicillin: Penicillin antibiotic. Common side effects: rash, diarrhea. Interactions: Warfarin. Dosage: 250-500mg every 8 hours."
    else:
        return f"No detailed drug information found for '{drug_name}'."

@tool
def get_patient_ehr_data(patient_id: str, data_type: str = "all") -> str:
    """Fetches patient-specific data from Electronic Health Records (EHR) such as medical history, lab results, imaging reports, and current medications. Requires patient_id and optionally data_type (e.g., 'labs', 'meds'). This is a mock implementation and would require secure, permissioned access in a real system."""
    if patient_id == "P12345":
        if data_type == "labs":
            return "Patient P12345: Recent lab results - HbA1c: 7.2%, Cholesterol: 200 mg/dL."
        elif data_type == "meds":
            return "Patient P12345: Current medications - Metformin 500mg BID, Lisinopril 10mg QD."
        else:
            return "Patient P12345: Medical History - Type 2 Diabetes, Hypertension. Current Meds - Metformin, Lisinopril. Recent Labs - HbA1c: 7.2%."
    else:
        return f"Patient with ID '{patient_id}' not found or no '{data_type}' data available."

@tool
def query_medical_knowledge_graph(entity: str, relationship_type: str = "all") -> str:
    """Queries a medical knowledge graph (e.g., SNOMED CT, UMLS) to understand complex relationships between symptoms, diseases, diagnoses, treatments, etc. Requires an entity and optionally a relationship_type."""
    if entity.lower() == "fever":
        return "Fever is a symptom associated with: Influenza, Common Cold, Bacterial Infection, COVID-19. Related treatments often include antipyretics like paracetamol."
    elif entity.lower() == "migraine":
        return "Migraine is a type of headache characterized by severe throbbing pain or a pulsing sensation. Associated symptoms: nausea, vomiting, sensitivity to light/sound. Treatments: triptans, NSAIDs, CGRP inhibitors."
    else:
        return f"No specific knowledge graph relationships found for '{entity}'."

@tool
def get_disease_outbreak_data(disease: str, region: str = "global") -> str:
    """Retrieves real-time disease outbreak data from sources like CDC/WHO for a specific disease and region."""
    if disease.lower() == "influenza":
        return "Current influenza activity: Moderate in North America, low in Europe. Dominant strains are H3N2 and B/Victoria. Vaccination recommended."
    elif disease.lower() == "dengue":
        return "Dengue fever outbreaks reported in Southeast Asia and parts of South America. Increased vector control measures advised."
    else:
        return f"No real-time outbreak data available for '{disease}' in '{region}'."

# --- LLM and Agent Setup ---

# Initialize the LLM (using OpenAI's model as an example)
# Make sure to set your OPENAI_API_KEY environment variable
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# List of tools available to the agent
tools = [
    search_medical_literature,
    get_clinical_guidelines,
    get_drug_information,
    get_patient_ehr_data,
    query_medical_knowledge_graph,
    get_disease_outbreak_data
]

# Define the agent prompt
prompt = PromptTemplate.from_template("""
Answer the following questions as thoroughly as possible. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
""")

# Create the ReAct agent
agent = create_react_agent(llm, tools, prompt)

# Create the Agent Executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# --- Main Application Logic (Example Usage) ---

def get_medical_recommendation(query: str) -> dict:
    """Processes a medical query using the LLM agent and returns a recommendation based on external knowledge."""
    print(f"\nProcessing query: {query}")
    try:
        response = agent_executor.invoke({"input": query})
        return {"query": query, "recommendation": response["output"]}
    except Exception as e:
        return {"query": query, "error": str(e)}

if __name__ == "__main__":
    # Example Queries
    queries = [
        "What are the latest treatments for COVID-19?",
        "Provide clinical guidelines for managing hypertension.",
        "Tell me about the drug metformin: its uses and side effects.",
        "What is the medical history and current medications for patient P12345?",
        "What diseases are associated with fever?",
        "Are there any current influenza outbreaks globally?",
        "What is the latest research on breast cancer treatment?"
    ]

    for q in queries:
        result = get_medical_recommendation(q)
        print("\n--- Result ---")
        print(f"Query: {result.get('query')}")
        if "recommendation" in result:
            print(f"Recommendation: {result.get('recommendation')}")
        else:
            print(f"Error: {result.get('error')}")
        print("----------------\n")
