#!/usr/bin/env python
# -*- coding: utf-8 -*-

import streamlit as st
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory

import os

# --- Pydantic Models for Data Validation ---
class Symptom(BaseModel):
    name: str = Field(..., description="Name of the symptom")
    severity: Optional[str] = Field(None, description="Severity of the symptom (e.g., 'mild', 'moderate', 'severe')")
    duration_days: Optional[int] = Field(None, description="Duration of the symptom in days")

class MedicalHistory(BaseModel):
    conditions: List[str] = Field(default_factory=list, description="List of pre-existing medical conditions")
    medications: List[str] = Field(default_factory=list, description="List of current medications")
    allergies: List[str] = Field(default_factory=list, description="List of known allergies")

class TestResult(BaseModel):
    test_name: str = Field(..., description="Name of the test")
    value: str = Field(..., description="Result value or description")
    unit: Optional[str] = Field(None, description="Unit of the test result")
    reference_range: Optional[str] = Field(None, description="Reference range for the test result")

class PatientCase(BaseModel):
    patient_id: str = Field(..., description="Unique identifier for the patient")
    age: int = Field(..., description="Patient's age in years")
    gender: str = Field(..., description="Patient's gender")
    chief_complaint: str = Field(..., description="The primary reason for the patient's visit")
    symptoms: List[Symptom] = Field(default_factory=list)
    medical_history: MedicalHistory = Field(default_factory=MedicalHistory)
    test_results: List[TestResult] = Field(default_factory=list)
    additional_notes: Optional[str] = Field(None, description="Any additional relevant notes from the healthcare professional")

# --- LangChain Setup ---
# Ensure your OpenAI API key is set as an environment variable or passed directly
# os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"

llm = ChatOpenAI(temperature=0.7, model_name="gpt-4o") # Using gpt-4o for better reasoning

# Prompt template for the diagnostic assistant
# This prompt encourages structured problem-solving, task decomposition, and dynamic adjustment
prompt_template = PromptTemplate.from_template(
    """You are an AI-powered medical diagnostic assistant. Your goal is to help healthcare professionals by analyzing patient data, proposing a differential diagnosis, and explaining your reasoning. "
    "Approach this systematically, breaking down the problem, considering constraints (medical guidelines, patient history), and being prepared to adjust your diagnosis based on new information. "
    "Always provide a concise differential diagnosis and a detailed explanation of your reasoning.

    Patient Case ID: {patient_id}
    Age: {age}
    Gender: {gender}
    Chief Complaint: {chief_complaint}

    Symptoms:
    {symptoms}

    Medical History:
    {medical_history}

    Test Results:
    {test_results}

    Additional Notes:
    {additional_notes}

    Current Conversation History (for dynamic adjustment and backtracking):
    {history}

    Based on the information above, and considering common medical guidelines and the patient's specific context, provide your diagnostic assessment.
    Think step-by-step to arrive at the diagnosis, considering multiple possibilities before narrowing down. If you need more information, state what you would like to know.

    Differential Diagnosis (Top 3-5, with probability/confidence if possible):
    1. 
    2. 

    Reasoning (Detailed breakdown of how you arrived at the diagnosis, considering pros and cons of each possibility, and how constraints were applied):
    """
)

# Using ConversationBufferWindowMemory to simulate dynamic adjustment and backtracking
# The window ensures the LLM focuses on recent interactions while providing context
memory = ConversationBufferWindowMemory(k=5, memory_key="history")

daignosis_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True, memory=memory)

# --- Streamlit UI --- 
st.set_page_config(layout="wide", page_title="AI Medical Diagnostic Assistant")
st.title("AI Medical Diagnostic Assistant")
st.markdown("--- Developed for healthcare professionals to assist in complex disease diagnosis --- ")

st.sidebar.header("Patient Information")

with st.sidebar.form("patient_form"):
    st.subheader("Patient Demographics")
    patient_id = st.text_input("Patient ID", "P001")
    age = st.number_input("Age", min_value=0, max_value=120, value=45)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    chief_complaint = st.text_area("Chief Complaint", "Severe headache and fever for 3 days")

    st.subheader("Symptoms")
    num_symptoms = st.number_input("Number of Symptoms to Add", min_value=0, value=1)
    symptoms_list = []
    for i in range(num_symptoms):
        st.markdown(f"**Symptom {i+1}**")
        s_name = st.text_input(f"Symptom Name {i+1}", key=f"s_name_{i}")
        s_severity = st.selectbox(f"Severity {i+1}", ["", "mild", "moderate", "severe"], key=f"s_severity_{i}")
        s_duration = st.number_input(f"Duration (days) {i+1}", min_value=0, value=3, key=f"s_duration_{i}")
        if s_name:
            symptoms_list.append(Symptom(name=s_name, severity=s_severity if s_severity else None, duration_days=s_duration if s_duration > 0 else None))

    st.subheader("Medical History")
    med_conditions_str = st.text_area("Pre-existing Conditions (comma-separated)", "Hypertension, Diabetes")
    current_meds_str = st.text_area("Current Medications (comma-separated)", "Lisinopril, Metformin")
    allergies_str = st.text_area("Allergies (comma-separated)", "Penicillin")
    
    medical_history = MedicalHistory(
        conditions=[c.strip() for c in med_conditions_str.split(',') if c.strip()],
        medications=[m.strip() for m in current_meds_str.split(',') if m.strip()],
        allergies=[a.strip() for a in allergies_str.split(',') if a.strip()]
    )

    st.subheader("Test Results")
    num_tests = st.number_input("Number of Test Results to Add", min_value=0, value=1)
    test_results_list = []
    for i in range(num_tests):
        st.markdown(f"**Test Result {i+1}**")
        t_name = st.text_input(f"Test Name {i+1}", key=f"t_name_{i}")
        t_value = st.text_input(f"Result Value {i+1}", key=f"t_value_{i}")
        t_unit = st.text_input(f"Unit {i+1}", key=f"t_unit_{i}")
        t_ref_range = st.text_input(f"Reference Range {i+1}", key=f"t_ref_range_{i}")
        if t_name and t_value:
            test_results_list.append(TestResult(test_name=t_name, value=t_value, unit=t_unit if t_unit else None, reference_range=t_ref_range if t_ref_range else None))

    additional_notes = st.text_area("Additional Clinical Notes", "Patient reports recent travel history to Southeast Asia.")

    submitted = st.form_submit_button("Get Diagnosis")

    if submitted:
        try:
            patient_case = PatientCase(
                patient_id=patient_id,
                age=age,
                gender=gender,
                chief_complaint=chief_complaint,
                symptoms=symptoms_list,
                medical_history=medical_history,
                test_results=test_results_list,
                additional_notes=additional_notes
            )
            st.success("Patient data validated successfully!")

            # Prepare inputs for LangChain
            inputs = {
                "patient_id": patient_case.patient_id,
                "age": patient_case.age,
                "gender": patient_case.gender,
                "chief_complaint": patient_case.chief_complaint,
                "symptoms": "\n".join([f"- {s.name} (Severity: {s.severity or 'N/A'}, Duration: {s.duration_days or 'N/A'} days)" for s in patient_case.symptoms]),
                "medical_history": f"Conditions: {', '.join(patient_case.medical_history.conditions)}\nMedications: {', '.join(patient_case.medical_history.medications)}\nAllergies: {', '.join(patient_case.medical_history.allergies)}",
                "test_results": "\n".join([f"- {t.test_name}: {t.value} {t.unit or ''} (Ref: {t.reference_range or 'N/A'})" for t in patient_case.test_results]),
                "additional_notes": patient_case.additional_notes if patient_case.additional_notes else "None",
            }
            
            with st.spinner("Analyzing patient data and generating diagnosis..."):
                response = daignosis_chain.run(**inputs)
            
            st.subheader("AI Diagnostic Assistant Output:")
            st.markdown(response)
            
            # Display conversation history for debugging/understanding dynamic adjustment
            # st.subheader("Conversation History (for debugging):")
            # for i, msg in enumerate(memory.buffer):
            #     st.write(f"**{msg.type.capitalize()}:** {msg.content}")

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.error("Please check your input and ensure all required fields are filled correctly.")


