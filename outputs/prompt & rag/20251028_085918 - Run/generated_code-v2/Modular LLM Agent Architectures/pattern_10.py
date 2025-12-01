import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain_core.prompts import PromptTemplate
from loguru import logger
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import random

load_dotenv()
logger.add("medical_aid.log", rotation="10 MB", level="INFO")

def get_medical_knowledge(query: str) -> str:
    logger.info(f"Medical Knowledge Base queried with: {query}")
    knowledge_base = {
        "diabetes symptoms": "Common symptoms include increased thirst, frequent urination, blurred vision, and fatigue.",
        "hypertension treatment": "Treatments often involve lifestyle changes (diet, exercise), and medications like ACE inhibitors, diuretics, or beta-blockers.",
        "pneumonia causes": "Pneumonia can be caused by bacteria, viruses, or fungi. Bacterial pneumonia is common.",
        "migraine definition": "A migraine is a type of headache characterized by recurrent attacks of moderate to severe throbbing pain, usually on one side of the head."
    }
    response = knowledge_base.get(query.lower(), "Information not found in medical knowledge base for that specific query.")
    logger.info(f"Medical Knowledge Base response: {response}")
    return response

def run_diagnostic_algorithm(symptoms: str) -> str:
    logger.info(f"Diagnostic Algorithm queried with symptoms: {symptoms}")
    symptoms_lower = symptoms.lower()
    
    possible_diagnoses = []
    if "fever" in symptoms_lower and "cough" in symptoms_lower and "shortness of breath" in symptoms_lower:
        possible_diagnoses.append({"diagnosis": "Pneumonia", "confidence": 0.85})
    if "headache" in symptoms_lower and "nausea" in symptoms_lower and "sensitivity to light" in symptoms_lower:
        possible_diagnoses.append({"diagnosis": "Migraine", "confidence": 0.90})
    if "increased thirst" in symptoms_lower and "frequent urination" in symptoms_lower:
        possible_diagnoses.append({"diagnosis": "Type 2 Diabetes", "confidence": 0.75})
    if "chest pain" in symptoms_lower and "shortness of breath" in symptoms_lower:
        possible_diagnoses.append({"diagnosis": "Angina or Heart Attack (requires immediate medical attention)", "confidence": 0.95})
    
    if not possible_diagnoses:
        return "Based on the provided symptoms, the diagnostic algorithm could not identify a clear diagnosis. Further investigation may be needed."
    
    possible_diagnoses.sort(key=lambda x: x["confidence"], reverse=True)
    response = "Potential Diagnoses:\n"
    for diag in possible_diagnoses[:3]:
        response += f"- {diag['diagnosis']} (Confidence: {diag['confidence']:.0%})\n"
    
    logger.info(f"Diagnostic Algorithm response: {response}")
    return response

def check_drug_interactions(drugs: str) -> str:
    logger.info(f"Drug Interaction Checker queried with drugs: {drugs}")
    drug_list = [d.strip().lower() for d in drugs.split(',')]

    interactions = {
        ("warfarin", "ibuprofen"): "Increased risk of bleeding.",
        ("sildenafil", "nitroglycerin"): "Potentially dangerous drop in blood pressure.",
        ("metformin", "alcohol"): "Increased risk of lactic acidosis.",
        ("simvastatin", "grapefruit juice"): "Increased risk of muscle problems."
    }

    found_interactions = []
    for i in range(len(drug_list)):
        for j in range(i + 1, len(drug_list)):
            pair1 = (drug_list[i], drug_list[j])
            pair2 = (drug_list[j], drug_list[i])
            if pair1 in interactions:
                found_interactions.append(f"Interaction between {drug_list[i].capitalize()} and {drug_list[j].capitalize()}: {interactions[pair1]}")
            elif pair2 in interactions:
                found_interactions.append(f"Interaction between {drug_list[j].capitalize()} and {drug_list[i].capitalize()}: {interactions[pair2]}")
    
    if not found_interactions:
        return "No significant drug interactions found for the provided drugs (simulated data)."
    
    response = "Detected Drug Interactions:\n" + "\n".join(found_interactions)
    logger.info(f"Drug Interaction Checker response: {response}")
    return response

llm = ChatOpenAI(model="gpt-4", temperature=0)

tools = [
    Tool(
        name="MedicalKnowledgeBase",
        func=get_medical_knowledge,
        description="Useful for answering factual questions about diseases, symptoms, and treatments. Input should be a specific medical query.",
    ),
    Tool(
        name="DiagnosticAlgorithm",
        func=run_diagnostic_algorithm,
        description="Useful for suggesting potential diagnoses based on a list of patient symptoms. Input should be a comma-separated string of symptoms.",
    ),
    Tool(
        name="DrugInteractionChecker",
        func=check_drug_interactions,
        description="Useful for checking potential interactions between multiple drugs. Input should be a comma-separated string of drug names.",
    ),
]

prompt_template = PromptTemplate.from_template("""
You are a Medical Inquiry and Diagnostic Aid system. Your goal is to assist healthcare professionals by providing accurate medical information, potential diagnoses, and drug interaction warnings.

When a user asks a medical question or describes symptoms, you should:
1.  **Analyze the query** to determine if factual medical knowledge is needed, if a diagnosis should be considered, or if drug interactions need to be checked.
2.  **Use the appropriate tools** (MedicalKnowledgeBase, DiagnosticAlgorithm, DrugInteractionChecker) to gather information.
3.  **Synthesize the information** from the tools into a comprehensive, clear, and professional response.
4.  If no tools are directly applicable, try to answer based on general medical knowledge if possible, or state that more information is needed.

Ensure your responses are helpful and evidence-based (from the tools).

Query: {input}
{agent_scratchpad}
""")

agent = create_react_agent(llm, tools, prompt_template)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

st.set_page_config(page_title="Medical Inquiry and Diagnostic Aid", layout="centered")

st.title("👨‍⚕️ Medical Inquiry and Diagnostic Aid")
st.markdown("---")
st.markdown("""
This system helps healthcare professionals with initial diagnosis and treatment recommendations by leveraging a Large Language Model (LLM) as a router to various medical tools.
""")

user_query = st.text_area("Enter patient symptoms, a medical query, or drugs for interaction check:", height=150, placeholder="e.g., 'patient has fever, cough, and shortness of breath' or 'what are the causes of pneumonia?' or 'check interactions for Warfarin, Ibuprofen'")

if st.button("Get Aid"):
    if user_query:
        with st.spinner("Analyzing and retrieving medical information..."):
            try:
                response = agent_executor.invoke({"input": user_query})
                st.subheader("System Response:")
                st.write(response["output"])
                logger.info(f"Final System Response for query '{user_query}': {response['output']}")
            except Exception as e:
                st.error(f"An error occurred: {e}")
                logger.error(f"Error processing query '{user_query}': {e}")
    else:
        st.warning("Please enter a query to get medical aid.")

st.markdown("---")
st.caption("Disclaimer: This tool is for informational purposes only and should not replace professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider.")
