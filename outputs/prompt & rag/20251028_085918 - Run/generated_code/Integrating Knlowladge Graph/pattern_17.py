# main.py

import streamlit as st
import uvicorn
import spacy
import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import requests
import json
import threading
import time

# --- 1. Data Models (Pydantic) ---
class PatientInput(BaseModel):
    symptoms: str = Field(..., example="fever, headache, fatigue")
    medical_history: str = Field(..., example="no significant medical history, non-smoker")

class Evidence(BaseModel):
    source: str = Field(..., example="Knowledge Graph")
    content: str = Field(..., example="Fever is a common symptom of Influenza.")

class Diagnosis(BaseModel):
    condition: str = Field(..., example="Influenza")
    probability: float = Field(..., ge=0, le=1, example=0.75)
    reasoning: str = Field(..., example="Patient exhibits fever and headache, which are key symptoms. KG retrieval shows strong association with Influenza.")
    evidence: List[Evidence]

class DiagnosticOutput(BaseModel):
    diagnoses: List[Diagnosis]
    explanation: str = Field(..., example="The system analyzed patient symptoms and medical history, leveraging the medical knowledge graph to identify potential conditions and provide evidence-based reasoning.")

# --- 2. Knowledge Graph (networkx) ---
class MedicalKnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self._initialize_graph()

    def _initialize_graph(self):
        # Diseases
        self.add_entity("Influenza", {"type": "disease", "description": "A common viral infection that can be deadly, especially in high-risk groups."})
        self.add_entity("Common Cold", {"type": "disease", "description": "A viral infectious disease of the upper respiratory tract."})
        self.add_entity("Migraine", {"type": "disease", "description": "A primary headache disorder characterized by recurrent headaches."})
        self.add_entity("Tension Headache", {"type": "disease", "description": "A primary headache disorder that can last from minutes to days."})

        # Symptoms
        self.add_entity("fever", {"type": "symptom"})
        self.add_entity("headache", {"type": "symptom"})
        self.add_entity("fatigue", {"type": "symptom"})
        self.add_entity("cough", {"type": "symptom"})
        self.add_entity("sore throat", {"type": "symptom"})
        self.add_entity("nasal congestion", {"type": "symptom"})
        self.add_entity("nausea", {"type": "symptom"})
        self.add_entity("light sensitivity", {"type": "symptom"})
        self.add_entity("muscle aches", {"type": "symptom"})

        # Relationships
        self.add_relationship("fever", "symptom_of", "Influenza")
        self.add_relationship("headache", "symptom_of", "Influenza")
        self.add_relationship("fatigue", "symptom_of", "Influenza")
        self.add_relationship("cough", "symptom_of", "Influenza")
        self.add_relationship("muscle aches", "symptom_of", "Influenza")

        self.add_relationship("headache", "symptom_of", "Common Cold")
        self.add_relationship("sore throat", "symptom_of", "Common Cold")
        self.add_relationship("nasal congestion", "symptom_of", "Common Cold")
        self.add_relationship("cough", "symptom_of", "Common Cold")

        self.add_relationship("headache", "symptom_of", "Migraine")
        self.add_relationship("nausea", "symptom_of", "Migraine")
        self.add_relationship("light sensitivity", "symptom_of", "Migraine")

        self.add_relationship("headache", "symptom_of", "Tension Headache")
        self.add_relationship("fatigue", "symptom_of", "Tension Headache")
        
        # Add a few more complex relationships for demonstration
        self.add_relationship("Influenza", "treated_by", "Antivirals")
        self.add_relationship("Antivirals", "contraindicated_with", "Certain_Heart_Conditions")
        self.add_entity("Antivirals", {"type": "medication"})
        self.add_entity("Certain_Heart_Conditions", {"type": "medical_condition"})


    def add_entity(self, name: str, attributes: Dict[str, Any]):
        self.graph.add_node(name, **attributes)

    def add_relationship(self, source: str, relation_type: str, target: str):
        if source in self.graph and target in self.graph:
            self.graph.add_edge(source, target, type=relation_type)

    def get_related_facts(self, entity: str, relation_type: Optional[str] = None) -> List[str]:
        facts = []
        if entity in self.graph:
            # Outgoing relationships
            for neighbor in self.graph.neighbors(entity):
                edge_data = self.graph.get_edge_data(entity, neighbor)
                if edge_data and (relation_type is None or edge_data['type'] == relation_type):
                    facts.append(f"{entity} {edge_data['type'].replace('_', ' ')} {neighbor}.")
            # Incoming relationships (reverse edges)
            for predecessor in self.graph.predecessors(entity):
                edge_data = self.graph.get_edge_data(predecessor, entity)
                if edge_data and (relation_type is None or edge_data['type'] == relation_type):
                     facts.append(f"{predecessor} {edge_data['type'].replace('_', ' ')} {entity}.")
        return list(set(facts)) # Return unique facts

    def get_all_entities_and_relations_as_text(self) -> List[str]:
        text_facts = []
        for u, v, data in self.graph.edges(data=True):
            text_facts.append(f"{u} {data['type'].replace('_', ' ')} {v}.")
        for node, data in self.graph.nodes(data=True):
            if "description" in data:
                text_facts.append(f"Description of {node}: {data['description']}.")
        return text_facts

# --- 3. NLU & Entity Extraction (spaCy) ---
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'. This may take a moment...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def extract_medical_entities(text: str) -> List[str]:
    doc = nlp(text.lower())
    entities = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "PERSON", "GPE", "PRODUCT"]] # Broad entities
    # Also add noun chunks as potential entities, especially for symptoms
    for chunk in doc.noun_chunks:
        entities.append(chunk.text)
    # Filter out common stop words or very short entities that are unlikely to be medical terms
    entities = [e for e in entities if len(e) > 2 and e not in nlp.Defaults.stop_words]
    return list(set(entities))

# --- 4. Embedding & Retrieval (sentence-transformers) ---
# Load a pre-trained sentence transformer model
try:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    print(f"Error loading SentenceTransformer model: {e}. Please ensure you have internet access or the model is cached.")
    print("Attempting to proceed with a dummy embedding function.")
    embedding_model = None # Fallback for no internet/model issue

def get_embedding(text: str) -> Optional[List[float]]:
    if embedding_model:
        return embedding_model.encode(text).tolist()
    return None

# Simplified in-memory vector store (mock Chroma/Faiss)
class InMemoryVectorStore:
    def __init__(self, kg_facts: List[str]):
        self.documents = kg_facts
        self.embeddings = [get_embedding(fact) for fact in kg_facts if get_embedding(fact) is not None]
        self.indexed_documents = [doc for doc, emb in zip(kg_facts, self.embeddings) if emb is not None]

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        query_embedding = get_embedding(query)
        if not query_embedding or not self.embeddings:
            return []

        query_embedding = np.array(query_embedding)
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            if doc_embedding is not None:
                similarity = np.dot(query_embedding, np.array(doc_embedding)) / \
                             (np.linalg.norm(query_embedding) * np.linalg.norm(np.array(doc_embedding)))
                similarities.append((similarity, self.indexed_documents[i]))
        
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [doc for sim, doc in similarities[:top_k]]

# --- 5. Mock LLM Reasoning Engine (Langchain concept) ---
# We will simulate an LLM's reasoning capabilities directly in Python
# since setting up a real LLM (even a local one) is beyond a self-contained snippet.
# This will represent the 'LLM as Agent' pattern conceptually.

class MockLLM:
    def invoke(self, prompt: str) -> str:
        # Simulate LLM response based on prompt content
        if "diagnose the patient" in prompt.lower() and "symptoms" in prompt.lower():
            if "fever" in prompt.lower() and "headache" in prompt.lower() and "fatigue" in prompt.lower():
                return json.dumps({"diagnoses": [{"condition": "Influenza", "probability": 0.8, "reasoning": "Strong correlation of fever, headache, and fatigue with Influenza based on provided facts.", "evidence": []}], "explanation": "Initial assessment suggests viral infection."})
            elif "headache" in prompt.lower() and "light sensitivity" in prompt.lower():
                 return json.dumps({"diagnoses": [{"condition": "Migraine", "probability": 0.7, "reasoning": "Headache with light sensitivity points towards Migraine based on provided facts.", "evidence": []}], "explanation": "Considering neurological causes."})
            elif "cough" in prompt.lower() and "nasal congestion" in prompt.lower():
                return json.dumps({"diagnoses": [{"condition": "Common Cold", "probability": 0.6, "reasoning": "Respiratory symptoms like cough and congestion are typical for Common Cold.", "evidence": []}], "explanation": "Common respiratory illness suspected."})
            else:
                return json.dumps({"diagnoses": [{"condition": "Undetermined", "probability": 0.3, "reasoning": "More information needed.", "evidence": []}], "explanation": "Cannot determine a clear diagnosis with current information."})
        elif "refine the diagnosis" in prompt.lower():
            # Simulate a refinement step
            return json.dumps({"diagnoses": [{"condition": "Influenza (Confirmed)", "probability": 0.9, "reasoning": "Refined based on additional patient history and symptom duration.", "evidence": []}], "explanation": "Further analysis confirmed initial suspicion."})
        return json.dumps({"diagnoses": [], "explanation": "No clear diagnostic path found."})

mock_llm = MockLLM()

# --- Langchain-like Agent Orchestration (Conceptual) ---
# This function orchestrates the NLU, KG retrieval, and mock LLM reasoning
def run_diagnostic_agent(patient_input: PatientInput, kg: MedicalKnowledgeGraph, vector_store: InMemoryVectorStore) -> DiagnosticOutput:
    # 1. NLU & Entity Extraction
    symptoms_entities = extract_medical_entities(patient_input.symptoms)
    history_entities = extract_medical_entities(patient_input.medical_history)
    all_extracted_entities = list(set(symptoms_entities + history_entities))

    st.sidebar.subheader("Extracted Entities")
    st.sidebar.write(all_extracted_entities)

    # 2. Knowledge Augmentation & Retrieval
    retrieved_facts = []
    for entity in all_extracted_entities:
        # Retrieve direct KG facts
        retrieved_facts.extend(kg.get_related_facts(entity))
        # Retrieve semantically similar facts from vector store
        retrieved_facts.extend(vector_store.retrieve(entity, top_k=3))

    # Deduplicate and limit retrieved facts for prompt efficiency
    retrieved_facts = list(set(retrieved_facts))[:10]
    
    st.sidebar.subheader("Retrieved KG Facts")
    st.sidebar.write(retrieved_facts)

    # 3. Prompt Engineering & LLM Reasoning (Iterative/Agent-like simulation)
    # Initial prompt to LLM
    initial_prompt = f"Given the patient's symptoms: {patient_input.symptoms} and medical history: {patient_input.medical_history}."
    if retrieved_facts:
        initial_prompt += f" Consider the following medical facts from a knowledge graph: {'; '.join(retrieved_facts)}. "
    initial_prompt += "Based on this information, provide a primary diagnosis, its probability, and reasoning with supporting evidence. Also, suggest if further information is needed to refine the diagnosis. Output in JSON format with 'diagnoses' (list of condition, probability, reasoning, evidence) and 'explanation' keys."

    st.sidebar.subheader("LLM Input Prompt")
    st.sidebar.write(initial_prompt)

    llm_response_raw = mock_llm.invoke(initial_prompt)
    
    st.sidebar.subheader("LLM Raw Response")
    st.sidebar.write(llm_response_raw)

    try:
        llm_output = json.loads(llm_response_raw)
    except json.JSONDecodeError:
        st.error(f"Failed to decode LLM response: {llm_response_raw}")
        return DiagnosticOutput(diagnoses=[], explanation="Error: LLM response was not valid JSON.")

    diagnoses_data = llm_output.get("diagnoses", [])
    explanation = llm_output.get("explanation", "")

    # Populate evidence for each diagnosis (simplified: use retrieved facts)
    formatted_diagnoses = []
    for diag_data in diagnoses_data:
        # In a real system, LLM would generate specific evidence pointers.
        # Here, we'll just attach relevant retrieved facts.
        diag_evidence = [Evidence(source="Knowledge Graph", content=fact) for fact in retrieved_facts if any(term in fact.lower() for term in diag_data['condition'].lower().split() + [s.lower() for s in symptoms_entities])]
        formatted_diagnoses.append(Diagnosis(
            condition=diag_data.get("condition", "Unknown"),
            probability=diag_data.get("probability", 0.0),
            reasoning=diag_data.get("reasoning", "No detailed reasoning provided."),
            evidence=diag_evidence if diag_evidence else [Evidence(source="KG Retrieval", content="No specific KG evidence directly linked by mock LLM.")]
        ))

    # 4. Hybrid Pruning Strategy (conceptual) and refinement
    # This part would involve a lightweight model or heuristics to evaluate paths.
    # For this mock, we'll just say the LLM output is the 