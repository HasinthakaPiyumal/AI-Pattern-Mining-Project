import streamlit as st
import openai
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import random

class ChromaDBMock:
    def __init__(self, documents):
        self.documents = documents

    def retrieve(self, query):
        relevant_docs = [doc for doc in self.documents if query.lower() in doc.lower()]
        return relevant_docs if relevant_docs else ["No specific information found, consulting general medical guidelines."]

medical_knowledge_base = [
    "Guidelines for Type 2 Diabetes management: diet, exercise, metformin.",
    "Drug interaction: Warfarin and NSAIDs can increase bleeding risk.",
    "Hypertension treatment: ACE inhibitors, ARBs, diuretics.",
    "Symptoms of Asthma: wheezing, shortness of breath, chest tightness.",
    "Dosage for Acetaminophen: 500-1000mg every 4-6 hours, max 4000mg/day.",
    "Common side effects of statins: muscle pain, liver enzyme elevation."
]

class LLMMedicalAssistant:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")
        self.vector_store = ChromaDBMock(medical_knowledge_base)
        self.output_parser = StrOutputParser()

    def initialize_llm(self):
        pass

    def retrieve_medical_info(self, query: str) -> List[str]:
        return self.vector_store.retrieve(query)

    def generate_initial_plan(self, patient_data: Dict[str, Any], condition: str) -> str:
        rag_info = self.retrieve_medical_info(condition)
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are a medical assistant. Generate a comprehensive initial treatment plan for a patient based on their data and condition. Incorporate general medical guidelines."),
            ("user", "Patient Data: {patient_data}\nCondition: {condition}\nRelevant Medical Info: {rag_info}\nGenerate an initial treatment plan:")
        ])
        chain = prompt_template | self.llm | self.output_parser
        return chain.invoke({"patient_data": patient_data, "condition": condition, "rag_info": rag_info})

    def decompose_task(self, complex_task: str) -> List[str]:
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are a medical assistant. Decompose the following complex medical task into smaller, actionable sub-tasks."),
            ("user", "Complex Task: {complex_task}\nDecompose this task into a list of sub-tasks, one per line:")
        ])
        chain = prompt_template | self.llm | self.output_parser
        response = chain.invoke({"complex_task": complex_task})
        return [task.strip() for task in response.split('\n') if task.strip()]

    def evaluate_and_constrain_plan(self, plan: str, patient_data: Dict[str, Any], medical_context: str) -> str:
        rag_info_plan = self.retrieve_medical_info("drug interactions, contraindications, dosage limits")
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are a medical assistant. Evaluate the given treatment plan for potential conflicts, contraindications, or rule violations based on patient data, general medical context, and specific constraints. Suggest modifications if necessary."),
            ("user", "Current Plan: {plan}\nPatient Data: {patient_data}\nMedical Context: {medical_context}\nRelevant Medical Constraints: {rag_info_plan}\nEvaluate and suggest modifications:")
        ])
        chain = prompt_template | self.llm | self.output_parser
        return chain.invoke({"plan": plan, "patient_data": patient_data, "medical_context": medical_context, "rag_info_plan": rag_info_plan})

    def adapt_plan(self, current_plan: str, feedback: str, new_data: Dict[str, Any]) -> str:
        rag_info_adapt = self.retrieve_medical_info("treatment efficacy, side effects")
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are a medical assistant. Adapt the current treatment plan based on the provided clinician feedback and new patient data. Ensure the plan remains coherent and effective."),
            ("user", "Current Plan: {current_plan}\nClinician Feedback: {feedback}\nNew Patient Data: {new_data}\nRelevant Adaptation Guidelines: {rag_info_adapt}\nAdapt the treatment plan:")
        ])
        chain = prompt_template | self.llm | self.output_parser
        return chain.invoke({"current_plan": current_plan, "feedback": feedback, "new_data": new_data, "rag_info_adapt": rag_info_adapt})

llm_assistant = LLMMedicalAssistant()

st.set_page_config(layout="wide")
st.title("🧠 Intelligent Medical Treatment Plan Generator and Adaptive Assistant")

st.sidebar.header("Patient Information")
patient_name = st.sidebar.text_input("Patient Name", "John Doe")
patient_age = st.sidebar.number_input("Age", 35, min_value=1)
patient_gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
patient_history = st.sidebar.text_area("Medical History", "No significant medical history. Mild hypertension.")
patient_condition = st.sidebar.text_input("Current Condition", "Type 2 Diabetes")

patient_data = {
    "name": patient_name,
    "age": patient_age,
    "gender": patient_gender,
    "history": patient_history,
    "condition": patient_condition
}

st.header("Treatment Plan")

if st.button("Generate Initial Treatment Plan"):
    with st.spinner("Generating initial plan..."):
        initial_plan = llm_assistant.generate_initial_plan(patient_data, patient_condition)
        st.session_state.current_plan = initial_plan
    st.subheader("Initial Plan:")
    st.write(st.session_state.current_plan)

if "current_plan" in st.session_state and st.session_state.current_plan:
    st.subheader("Current Active Plan")
    st.write(st.session_state.current_plan)

    st.header("Plan Evaluation and Adaptation")
    medical_context_input = st.text_area("Additional Medical Context for Evaluation", "", key="medical_context_eval")
    if st.button("Evaluate and Constrain Plan"):
        with st.spinner("Evaluating plan for constraints and conflicts..."):
            evaluated_plan = llm_assistant.evaluate_and_constrain_plan(st.session_state.current_plan, patient_data, medical_context_input)
            st.session_state.current_plan = evaluated_plan
        st.subheader("Evaluated Plan (with suggested modifications if any):")
        st.write(st.session_state.current_plan)

    clinician_feedback = st.text_area("Clinician Feedback or New Observations", "", key="clinician_feedback")
    new_patient_data_input = st.text_area("New Patient Data (e.g., latest lab results, symptom changes)", "", key="new_patient_data")

    if st.button("Adapt Plan based on Feedback/New Data"):
        new_data_dict = {"feedback_text": clinician_feedback, "patient_updates": new_patient_data_input}
        with st.spinner("Adapting plan..."):
            adapted_plan = llm_assistant.adapt_plan(st.session_state.current_plan, clinician_feedback, new_data_dict)
            st.session_state.current_plan = adapted_plan
        st.subheader("Adapted Plan:")
        st.write(st.session_state.current_plan)

    st.header("Task Decomposition")
    task_to_decompose = st.text_input("Enter a complex medical task to decompose", "Manage chronic pain for a diabetic patient")
    if st.button("Decompose Task"):
        with st.spinner("Decomposing task..."):
            sub_tasks = llm_assistant.decompose_task(task_to_decompose)
        st.subheader("Decomposed Sub-tasks:")
        for i, task in enumerate(sub_tasks):
            st.write(f"{i+1}. {task}")
else:
    st.info("Please generate an initial treatment plan first.")
