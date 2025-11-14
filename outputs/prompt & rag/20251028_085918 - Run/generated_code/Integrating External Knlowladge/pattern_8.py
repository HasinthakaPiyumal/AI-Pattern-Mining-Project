""" 
This module implements an AI-powered Medical Diagnostic Assistant based on the Dynamic Knowledge-Augmented LLMs pattern.
It integrates a RAG system for dynamic knowledge retrieval and external tool orchestration
for real-time information access, aiming to enhance LLM factual accuracy and reasoning capabilities.
"""

import os
import json
from typing import List, Dict, Any

# Langchain imports
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import tool

# Placeholder for LLM - requires API key or local setup
# For demonstration, we'll use a placeholder that mimics an LLM interaction.
# In a real application, you would initialize ChatOpenAI or a similar LLM.
# from langchain_openai import ChatOpenAI


# --- 1. LLM Core Module (Placeholder) ---
# In a real scenario, you would initialize your LLM here, e.g.:
# llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)

class MockLLM:
    """A mock LLM for demonstration purposes."""
    def invoke(self, prompt: str) -> str:
        if "tool_code" in prompt:
            # Simulate tool output processing
            return f"[Mock LLM]: I have processed the tool output. Here is a synthesized response based on: {prompt}"
        return f"[Mock LLM]: I received your query: '{prompt}'. Let me check the information."

    def get_prompts_to_tools(self, messages):
        # This is a simplified mock. In a real scenario, an LLM would parse messages
        # to decide on tool calls. For this mock, we'll assume the agent handles it.
        pass

llm = MockLLM() # Using the mock LLM for this self-contained example

# --- 2. Knowledge Retrieval and Management Module (RAG System) ---

# Sample Medical Documents (replace with actual data loading in production)
medical_docs_content = [
    "Aspirin is commonly used for pain relief, fever reduction, and anti-inflammatory purposes. It can also be prescribed to prevent blood clots.",
    "Diabetes Mellitus Type 2 is characterized by insulin resistance and relative insulin deficiency. Management often includes diet, exercise, and medication like Metformin.",
    "Hypertension, or high blood pressure, increases the risk of heart disease and stroke. Lifestyle changes and antihypertensive drugs are common treatments.",
    "Common symptoms of the flu include fever, cough, sore throat, muscle aches, and fatigue. It is caused by influenza viruses.",
    "The recommended vaccination schedule for children includes vaccines for measles, mumps, rubella (MMR), diphtheria, tetanus, pertussis (DTaP), and polio.",
    "Clinical trials for a new Alzheimer's drug, 'NeuroFix', are currently in Phase 3, showing promising results in reducing cognitive decline.",
    "Latest research suggests a strong link between gut microbiome health and mental well-being, influencing conditions like anxiety and depression.",
    "Anaphylaxis is a severe, life-threatening allergic reaction. Symptoms include difficulty breathing, hives, swelling, and a drop in blood pressure. Epinephrine is the primary treatment.",
    "MRI scans use strong magnetic fields and radio waves to create detailed images of organs and tissues within the body. They are particularly useful for brain and spinal cord imaging.",
    "Chronic Kidney Disease (CKD) is a progressive loss of kidney function over months or years. Dialysis or kidney transplant may be necessary in advanced stages."
]

# Simulate document loading and splitting
documents = []
for i, content in enumerate(medical_docs_content):
    # Using TextLoader for simplicity; in reality, this would load from files/APIs
    # For a self-contained example, we'll create a mock document structure
    documents.append({'page_content': content, 'metadata': {'source': f'medical_journal_{i}.txt'}})

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# Mock `split_documents` as we are using a list of dicts directly
# In a real scenario, `text_splitter.split_documents(docs)` would be used after loading
texts = [doc['page_content'] for doc in documents] # Extracting text content for embedding
metadatas = [doc['metadata'] for doc in documents] # Extracting metadata

# Embedding Model
# Using a lightweight sentence-transformer model
embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Vector Database (Chroma for simplicity)
vectorstore = Chroma.from_texts(texts=texts, embedding=embeddings_model, metadatas=metadatas)
retriever = vectorstore.as_retriever()

# Retrieval Chain for RAG
rag_prompt = ChatPromptTemplate.from_template("""
Answer the user's question based on the below context:

{context}

Question: {input}
""")

document_chain = create_stuff_documents_chain(llm, rag_prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# --- 3. External Tool Integration Module (Mock/Simulated) ---

@tool
def get_pubmed_abstract(query: str) -> str:
    """Fetches a simulated abstract from PubMed based on a medical query."""
    print(f"\n[Tool Call] Simulating PubMed search for: '{query}'")
    # In a real application, this would make an actual API call to PubMed
    mock_results = {
        "Alzheimer's drug": "Clinical trials for 'NeuroFix' show significant reduction in amyloid plaques and improved cognitive scores in early-stage Alzheimer's patients. (Source: Mock NeuroJournal 2023)",
        "Hypertension treatment": "A recent meta-analysis indicates that combined lifestyle interventions and a new class of ACE inhibitors are most effective in managing resistant hypertension. (Source: Mock CardioReview 2024)",
        "Pediatric vaccination": "New guidelines for pediatric vaccination emphasize earlier MMR boosters in high-risk populations. (Source: Mock PediJournal 2023)"
    }
    return mock_results.get(query, f"No specific abstract found for '{query}'. Searching general medical literature...")

@tool
def get_ehr_patient_data(patient_id: str) -> str:
    """Fetches simulated anonymized Electronic Health Record (EHR) data for a given patient ID."""
    print(f"\n[Tool Call] Simulating EHR data retrieval for Patient ID: '{patient_id}'")
    # In a real application, this would securely access an anonymized EHR system
    mock_ehr_data = {
        "P1001": {"age": 65, "gender": "Female", "conditions": ["Hypertension", "Type 2 Diabetes"], "medications": ["Metformin", "Lisinopril"], "allergies": ["Penicillin"]},
        "P1002": {"age": 32, "gender": "Male", "conditions": ["Seasonal Allergies"], "medications": ["Loratadine"], "allergies": []},
        "P1003": {"age": 50, "gender": "Male", "conditions": ["Chronic Back Pain"], "medications": ["Ibuprofen"], "allergies": []}
    }
    return json.dumps(mock_ehr_data.get(patient_id, {"error": "Patient ID not found or unauthorized access."}), indent=2)

@tool
def check_drug_interactions(drug1: str, drug2: str) -> str:
    """Checks for simulated drug-drug interactions between two specified medications."""
    print(f"\n[Tool Call] Simulating drug interaction check for: '{drug1}' and '{drug2}'")
    # This would typically call a drug interaction API like OpenFDA or a proprietary database
    interactions = {
        ("Metformin", "Lisinopril"): "Generally safe, but monitor kidney function as both can affect it.",
        ("Aspirin", "Warfarin"): "Significant risk of bleeding. Concurrent use is generally contraindicated or requires very careful monitoring.",
        ("Loratadine", "Alcohol"): "Increased drowsiness. Advise patient to avoid alcohol."
    }
    return interactions.get((drug1, drug2), interactions.get((drug2, drug1), f"No significant interaction found between {drug1} and {drug2} in simulated database."))

@tool
def query_medical_knowledge_graph(entity: str) -> str:
    """Queries a simulated medical knowledge graph for related entities (e.g., symptoms, diseases, treatments)."""
    print(f"\n[Tool Call] Simulating knowledge graph query for: '{entity}'")
    # In a real system, this would query a graph database (e.g., Neo4j) with medical ontologies
    graph_data = {
        "Diabetes": {"symptoms": ["frequent urination", "increased thirst", "fatigue"], "treatments": ["Metformin", "Insulin", "diet change"], "related_conditions": ["obesity", "heart disease"]},
        "Hypertension": {"symptoms": ["headache", "chest pain"], "treatments": ["Lisinopril", "Amlodipine", "lifestyle change"], "related_conditions": ["stroke", "heart attack"]},
        "Metformin": {"class": "Biguanide", "used_for": "Type 2 Diabetes", "side_effects": ["nausea", "diarrhea"]}
    }
    return json.dumps(graph_data.get(entity, {"error": f"Entity '{entity}' not found in simulated knowledge graph."}), indent=2)

# Combine all tools
medical_tools = [
    get_pubmed_abstract,
    get_ehr_patient_data,
    check_drug_interactions,
    query_medical_knowledge_graph
]

# --- 4. Orchestration and Application Logic Module (Agentic Workflow) ---

# Define the agent prompt
agent_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful Medical Diagnostic Assistant for healthcare professionals. Use the provided tools and retrieved medical knowledge to answer questions thoroughly and accurately. Always cite your sources or tool usage."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ]
)

# Create the agent
agent = create_tool_calling_agent(llm, medical_tools, agent_prompt)

# Create the Agent Executor
agent_executor = AgentExecutor(agent=agent, tools=medical_tools, verbose=True)

# --- Main Application Logic ---

def run_diagnostic_assistant(patient_case: str) -> str:
    """Runs the medical diagnostic assistant with a given patient case."""
    print(f"\n--- Processing Patient Case ---\nPatient Input: {patient_case}")
    
    # The agent executor will decide whether to use RAG or tools based on the prompt.
    # For this simplified example, we'll let the agent decide how to respond.
    # In a more complex scenario, you might explicitly call the retrieval_chain first
    # then pass that context to the agent, or let the agent handle retrieval as a tool.

    # For demonstration, we'll let the agent handle the query directly, 
    # and it will decide if a tool is needed.
    try:
        response = agent_executor.invoke({"input": patient_case})
        # Combine RAG and tool results if both are used by the agent
        final_answer = response.get("output", "Could not process the request.")
        return final_answer
    except Exception as e:
        return f"An error occurred during processing: {e}"

# --- 5. User Interface (Simple CLI) ---

if __name__ == "__main__":
    print("Welcome to the AI-powered Medical Diagnostic Assistant (Mock Version)!")
    print("You can ask about patient symptoms, drug interactions, or general medical knowledge.")
    print("Type 'exit' to quit.")
    
    while True:
        user_input = input("\nHealthcare Professional: ")
        if user_input.lower() == 'exit':
            print("Exiting assistant. Goodbye!")
            break
        
        # Example of how to integrate RAG (if the agent doesn't do it implicitly)
        # For this setup, the agent is designed to use tools and potentially integrate RAG results
        # if the LLM's prompt guides it to do so after initial retrieval.
        
        # Direct agent invocation for simplicity, assuming it handles tool/RAG decision
        assistant_response = run_diagnostic_assistant(user_input)
        print(f"\nAssistant: {assistant_response}")

