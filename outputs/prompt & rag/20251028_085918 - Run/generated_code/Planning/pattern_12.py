"""
clinical_diagnostic_assistant.py

This script implements an AI-powered Clinical Diagnostic Assistant, integrating a Streamlit frontend, a FastAPI backend, and a LangGraph-orchestrated diagnostic engine with various LangChain agents.
It demonstrates the 'Structured AI Problem Solving' pattern by breaking down complex diagnoses into manageable sub-tasks, using multi-step reasoning, and adapting to new information.

Note: This is a simplified, single-file representation. In a production environment, components like the FastAPI server, Streamlit app, and potentially LangGraph agents would run as separate services.
LLM interactions are mocked or simplified for demonstration purposes. Database interactions are also simplified using in-memory or basic implementations.
"""

import streamlit as st
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import threading
import time

# LangChain / LangGraph related imports (mocked or simplified for single-file demo)
# from langchain.agents import AgentExecutor, create_react_agent
# from langchain.tools import tool
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.messages import BaseMessage, HumanMessage
# from langgraph.graph import StateGraph, END

# Database Imports (Simplified for in-memory SQLite for demonstration)
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# ChromaDB (Simplified for in-memory for demonstration)
import chromadb
from chromadb.utils import embedding_functions

### 0. Global Setup and Mock Data/Services ###

# Mock LLM for demonstration
class MockLLM:
    def invoke(self, prompt: str) -> str:
        if "symptoms" in prompt.lower() and "extract" in prompt.lower():
            return "Extracted Symptoms: fever, cough, fatigue"
        elif "differential diagnosis" in prompt.lower():
            return "Differential Diagnoses: Common Cold (90%), Flu (70%), Pneumonia (20%)"
        elif "recommend diagnostic plan" in prompt.lower():
            return "Recommended Plan: Order CBC, Chest X-ray if cough persists, rest and hydration."
        elif "re-evaluate" in prompt.lower():
            return "Updated Diagnoses: Flu (95%), Common Cold (5%)"
        return "LLM processed: " + prompt[:100] + "..."

mock_llm = MockLLM()

# ChromaDB Setup (In-memory for demo)
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
chroma_client = chromadb.Client()

try:
    medical_knowledge_collection = chroma_client.create_collection(
        name="medical_knowledge",
        embedding_function=embedding_function
    )
    # Add some dummy medical knowledge
    medical_knowledge_collection.add(
        documents=[
            "Symptoms of common cold: runny nose, sore throat, cough, congestion.",
            "Symptoms of flu: fever, muscle aches, fatigue, cough, sore throat, headache.",
            "Pneumonia diagnosis often involves chest X-ray and CBC.",
            "Medical guideline for fever: rest, fluids, acetaminophen. Consult doctor if high fever persists.",
            "Drug interaction: Warfarin interacts with many medications, consult pharmacist."
        ],
        metadatas=[
            {"source": "disease_info"},
            {"source": "disease_info"},
            {"source": "diagnosis_protocol"},
            {"source": "guideline"},
            {"source": "drug_interaction"}
        ],
        ids=["doc1", "doc2", "doc3", "doc4", "doc5"]
    )
except Exception as e:
    # Collection might already exist if app is rerun without resetting client
    medical_knowledge_collection = chroma_client.get_collection(
        name="medical_knowledge",
        embedding_function=embedding_function
    )

# SQLAlchemy Setup (In-memory SQLite for demo)
DATABASE_URL = "sqlite:///./medical_data.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
Base = declarative_base()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class PatientRecord(Base):
    __tablename__ = "patient_records"
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, unique=True, index=True)
    symptoms = Column(Text)
    history = Column(Text)
    lab_results = Column(Text)
    current_diagnosis = Column(Text)
    diagnostic_plan = Column(Text)
    status = Column(String, default="initiated") # initiated, analyzing, planning, complete, awaiting_feedback

Base.metadata.create_all(bind=engine)

### 1. Pydantic Models for API ###

class PatientInput(BaseModel):
    patient_id: str
    symptoms: str
    medical_history: Optional[str] = None
    lab_results: Optional[str] = None

class DiagnosisResponse(BaseModel):
    patient_id: str
    status: str
    differential_diagnoses: List[str]
    recommended_plan: str
    rationale: str

class UpdatePatientInfo(BaseModel):
    patient_id: str
    new_info: str # e.g., "CBC results show high white blood cell count"

### 2. LangChain Tools (Simplified/Mocked) ###

def medical_knowledge_retrieval_tool(query: str) -> str:
    """Retrieves relevant medical knowledge from the vector database based on a query."""
    results = medical_knowledge_collection.query(
        query_texts=[query],
        n_results=2
    )
    docs = results['documents'][0] if results and results['documents'] else []
    return "\n".join(docs) if docs else "No relevant medical knowledge found."

def medical_guidelines_tool(condition: str) -> str:
    """Accesses medical protocols and guidelines for a given condition."""
    if "fever" in condition.lower():
        return "Guideline for fever: Rest, fluids, acetaminophen. Monitor for worsening symptoms. Consider antibiotics if bacterial infection confirmed."
    return "No specific guideline found for this condition (mocked)."

def constraint_validation_tool(action: str, patient_info: str) -> str:
    """Checks proposed actions against medical best practices and patient-specific constraints (e.g., drug interactions)."""
    if "warfarin" in patient_info.lower() and "ibuprofen" in action.lower():
        return "Constraint Alert: Ibuprofen may interact with Warfarin. Consider alternative pain relief."
    return "Action passes basic constraint validation (mocked)."

# In a real LangChain setup, these would be decorated with @tool
# from langchain.tools import tool
# @tool
# def medical_knowledge_retrieval_tool(...):
#    ...

### 3. LangGraph Agents (Conceptual/Simplified) ###

# Define the state for our LangGraph
class DiagnosticState(BaseModel):
    patient_id: str
    symptoms: str
    history: str
    lab_results: str
    current_hypotheses: List[str] = []
    differential_diagnoses: List[str] = []
    diagnostic_plan: str = ""
    status: str = "initiated"
    iteration: int = 0
    max_iterations: int = 5

    class Config:
        arbitrary_types_allowed = True

# Each 