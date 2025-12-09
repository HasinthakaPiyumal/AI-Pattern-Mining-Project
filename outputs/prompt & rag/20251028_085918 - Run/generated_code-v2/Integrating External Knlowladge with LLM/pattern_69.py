import os
import gradio as gr
import networkx as nx
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from dotenv import load_dotenv

load_dotenv()

# --- 1. Simulated Knowledge Graph (using networkx) ---
medical_kg = nx.DiGraph()

# Diseases and Symptoms
medical_kg.add_node("Influenza", type="disease")
medical_kg.add_node("Common Cold", type="disease")
medical_kg.add_node("Pneumonia", type="disease")
medical_kg.add_node("Diabetes", type="disease")
medical_kg.add_node("Hypertension", type="disease")

medical_kg.add_node("Fever", type="symptom")
medical_kg.add_node("Cough", type="symptom")
medical_kg.add_node("Sore Throat", type="symptom")
medical_kg.add_node("Runny Nose", type="symptom")
medical_kg.add_node("Fatigue", type="symptom")
medical_kg.add_node("Shortness of Breath", type="symptom")
medical_kg.add_node("High Blood Sugar", type="symptom")
medical_kg.add_node("High Blood Pressure", type="symptom")
medical_kg.add_node("Weight Loss", type="symptom")

# Treatments/Drugs
medical_kg.add_node("Antivirals", type="treatment")
medical_kg.add_node("Antibiotics", type="treatment")
medical_kg.add_node("Pain Relievers", type="treatment")
medical_kg.add_node("Insulin", type="treatment")
medical_kg.add_node("ACE Inhibitors", type="treatment")

# Relationships
medical_kg.add_edge("Influenza", "Fever", relation="has_symptom")
medical_kg.add_edge("Influenza", "Cough", relation="has_symptom")
medical_kg.add_edge("Influenza", "Fatigue", relation="has_symptom")
medical_kg.add_edge("Influenza", "Antivirals", relation="treatable_by")
medical_kg.add_edge("Influenza", "Pain Relievers", relation="treatable_by")

medical_kg.add_edge("Common Cold", "Sore Throat", relation="has_symptom")
medical_kg.add_edge("Common Cold", "Runny Nose", relation="has_symptom")
medical_kg.add_edge("Common Cold", "Cough", relation="has_symptom")
medical_kg.add_edge("Common Cold", "Pain Relievers", relation="treatable_by")

medical_kg.add_edge("Pneumonia", "Fever", relation="has_symptom")
medical_kg.add_edge("Pneumonia", "Cough", relation="has_symptom")
medical_kg.add_edge("Pneumonia", "Shortness of Breath", relation="has_symptom")
medical_kg.add_edge("Pneumonia", "Antibiotics", relation="treatable_by")

medical_kg.add_edge("Diabetes", "High Blood Sugar", relation="has_symptom")
medical_kg.add_edge("Diabetes", "Fatigue", relation="has_symptom")
medical_kg.add_edge("Diabetes", "Weight Loss", relation="has_symptom")
medical_kg.add_edge("Diabetes", "Insulin", relation="treatable_by")

medical_kg.add_edge("Hypertension", "High Blood Pressure", relation="has_symptom")
medical_kg.add_edge("Hypertension", "ACE Inhibitors", relation="treatable_by")

# Drug interactions (simplified)
medical_kg.add_edge("Antivirals", "Pain Relievers", relation="may_interact_with", interaction_info="Consult doctor for combined use.")

# --- 2. KG Tools (exposed to LLM Agent) ---

@tool
def query_disease_symptoms(disease_name: str) -> str:
    """Queries the knowledge graph to find symptoms associated with a given disease."""
    disease_name = disease_name.title()
    symptoms = [target for source, target, data in medical_kg.edges(data=True)
                if source == disease_name and data.get("relation") == "has_symptom"]
    if symptoms:
        return f"Symptoms for {disease_name}: {', '.join(symptoms)}."
    return f"No specific symptoms found for {disease_name} in the knowledge graph."

@tool
def get_drug_interactions(drug_name: str) -> str:
    """Queries the knowledge graph for potential interactions involving a given drug."""
    drug_name = drug_name.title()
    interactions = [f"{source} and {target} ({data.get('interaction_info')})"
                    for source, target, data in medical_kg.edges(data=True)
                    if (source == drug_name or target == drug_name) and data.get("relation") == "may_interact_with"]
    if interactions:
        return f"Potential drug interactions for {drug_name}: {'; '.join(interactions)}."
    return f"No significant interactions found for {drug_name} in the knowledge graph."

@tool
def find_treatment_guidelines(disease_name: str) -> str:
    """Queries the knowledge graph to find general treatment guidelines or common treatments for a disease."""
    disease_name = disease_name.title()
    treatments = [target for source, target, data in medical_kg.edges(data=True)
                  if source == disease_name and data.get("relation") == "treatable_by"]
    if treatments:
        return f"Common treatments for {disease_name}: {', '.join(treatments)}."
    return f"No specific treatment guidelines found for {disease_name} in the knowledge graph."

@tool
def diagnose_disease_from_symptoms(symptoms_list: str) -> str:
    """Attempts to diagnose a disease based on a comma-separated list of symptoms by checking the knowledge graph.
    Example: 'fever, cough, fatigue'"""
    input_symptoms = {s.strip().title() for s in symptoms_list.split(',')}
    possible_diseases = []

    for disease_node in [n for n, data in medical_kg.nodes(data=True) if data.get("type") == "disease"]:
        disease_symptoms = {target for source, target, data in medical_kg.edges(data=True)
                              if source == disease_node and data.get("relation") == "has_symptom"}
        # Check if all input symptoms are present in the disease's symptoms
        if input_symptoms.issubset(disease_symptoms) and len(input_symptoms) > 0:
            possible_diseases.append(disease_node)
            
    if possible_diseases:
        return f"Based on the symptoms '{symptoms_list}', possible diseases include: {', '.join(possible_diseases)}."
    return f"Could not find a direct diagnosis based on the provided symptoms: {symptoms_list}. Consider providing more details or asking about specific diseases."


kg_tools = [query_disease_symptoms, get_drug_interactions, find_treatment_guidelines, diagnose_disease_from_symptoms]

# --- 3. LLM and Agent Setup ---

llm = ChatOpenAI(model="gpt-4o", temperature=0)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a clinical decision support AI assistant. Your goal is to provide accurate medical information, potential diagnoses, and treatment recommendations based on patient input and by leveraging your access to a medical knowledge graph. Always try to use the provided tools to gather information before making a recommendation."),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

agent = create_tool_calling_agent(llm, kg_tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=kg_tools, verbose=True)

# --- 4. Gradio User Interface ---
def predict(patient_query: str) -> str:
    """Function to handle user input and get a response from the LLM agent."""
    try:
        response = agent_executor.invoke({"input": patient_query, "chat_history": []})
        return response["output"]
    except Exception as e:
        return f"An error occurred: {str(e)}. Please try again or rephrase your query."

if __name__ == "__main__":
    demo = gr.Interface(
        fn=predict,
        inputs=gr.Textbox(lines=5, placeholder="Describe the patient's symptoms or ask a medical question..."),
        outputs="text",
        title="Clinical Decision Support System (KG-Agent)",
        description=(
            "I am an AI assistant designed to help with clinical decision support by leveraging a medical knowledge graph. "
            "Ask me about symptoms, diseases, treatments, or drug interactions. "
            "Example queries: 'What are the symptoms of Influenza?', 'How is Diabetes treated?', 'Diagnose based on fever, cough, fatigue.', 'Are there interactions between Antivirals and Pain Relievers?'"
        ),
        examples=[
            "What are the symptoms of Pneumonia?",
            "How can Hypertension be treated?",
            "Diagnose based on sore throat, runny nose, cough.",
            "Tell me about drug interactions for Antivirals.",
            "What diseases cause high blood sugar?"
        ]
    )

    demo.launch()
