import streamlit as st
import pandas as pd
import numpy as np
import requests

class AgentState:
    def __init__(self, patient_data, diagnosis=None, treatment_plan=None, reasoning_history=None, tool_results=None):
        self.patient_data = patient_data
        self.diagnosis = diagnosis
        self.treatment_plan = treatment_plan
        self.reasoning_history = reasoning_history if reasoning_history is not None else []
        self.tool_results = tool_results if tool_results is not None else {}

class MockLLM:
    def invoke(self, prompt):
        if "diagnose" in prompt.lower():
            if "fever" in prompt.lower() and "cough" in prompt.lower() and "medical_db_info_respiratory infection symptoms" not in str(prompt):
                return "Reasoning: Symptoms suggest respiratory issues. Need more specific data. \nGenerated Code Tool Call: get_medical_database_info(\"respiratory infection symptoms\")"
            elif "medical_db_info_respiratory infection symptoms" in str(prompt) and "patient has mild symptoms" in str(prompt):
                return "Refined Diagnosis: Common Cold."
            return "Reasoning: Initial assessment. Further data needed. \nGenerated Code Tool Call: get_medical_database_info(\"general symptoms\")"
        
        if "formulate a treatment plan" in prompt.lower() and "Common Cold" in str(prompt):
            if "clinical_guidelines_Common Cold treatment" not in str(prompt):
                return "Reasoning: Diagnosis is Common Cold. Need clinical guidelines. \nGenerated Code Tool Call: search_clinical_guidelines(\"Common Cold treatment\")"
            else:
                return f"Treatment Plan: Based on Common Cold diagnosis and retrieved guidelines: {kb.retrieve_info('Common Cold treatment')}. Suggest rest, fluids, and symptomatic relief."
        
        return f"LLM thought based on: {prompt}"

    def generate_tool_code(self, tool_name, args_dict):
        if tool_name == "get_medical_database_info":
            query = args_dict.get("query", "")
            return f"""
import requests
def get_medical_database_info_impl(query_val):
    print(f"Simulating API call for: {{query_val}}")
    if 'common cold' in query_val.lower():
        return {{'common_cold': 'Symptoms: runny nose, sore throat, cough, congestion. Treatment: symptomatic relief.'}}
    if 'respiratory infection symptoms' in query_val.lower():
        return {{'respiratory_infection_symptoms': 'Common cold, flu, bronchitis, pneumonia symptoms.'}}
    return {{'data': f'Information for {{query_val}}'}}
result = get_medical_database_info_impl("{query}")
"""
        elif tool_name == "simulate_treatment_effectiveness":
            condition = args_dict.get("condition", "")
            treatment = args_dict.get("treatment", "")
            return f"""
import random
def simulate_treatment_effectiveness_impl(condition_val, treatment_val):
    print(f"Simulating treatment effectiveness for {{condition_val}} with {{treatment_val}}")
    return {{'effectiveness': 'high' if 'rest' in treatment_val.lower() else 'medium'}}
result = simulate_treatment_effectiveness_impl("{condition}", "{treatment}")
"""
        elif tool_name == "search_clinical_guidelines":
            query = args_dict.get("query", "")
            return f"""
import requests
def search_clinical_guidelines_impl(query_val):
    print(f"Simulating search for clinical guidelines: {{query_val}}")
    if 'common cold' in query_val.lower():
        return {{'guidelines': 'Adult Common Cold: Symptomatic treatment. No antibiotics.'}}
    return {{'guidelines': f'Guidelines for {{query_val}}'}}
result = search_clinical_guidelines_impl("{query}")
"""
        return "result = {'output': 'No specific tool code generated.'}"

class CodeExecutor:
    def execute(self, code_str):
        local_vars = {}
        try:
            exec(code_str, {}, local_vars)
            return local_vars.get("result", {"error": "No 'result' variable set in executed code."})
        except Exception as e:
            return {"error": str(e), "executed_code": code_str}

class MedicalKnowledgeBase:
    def __init__(self):
        self.data = {
            "fever": ["Infection", "Inflammation"],
            "cough": ["Respiratory infection", "Allergy"],
            "common cold symptoms": "Runny nose, sore throat, cough, congestion.",
            "flu symptoms": "Fever, muscle aches, chills, fatigue, cough.",
            "bronchitis symptoms": "Persistent cough, mucus production, chest discomfort.",
            "common cold treatment": "Rest, fluids, pain relievers, decongestants.",
            "flu treatment": "Antivirals (if caught early), rest, fluids, symptomatic relief.",
            "bronchitis treatment": "Rest, fluids, bronchodilators (if needed)."
        }

    def retrieve_info(self, query):
        results = []
        for key, value in self.data.items():
            if query.lower() in key.lower() or (isinstance(value, str) and query.lower() in value.lower()):
                results.append({key: value})
            elif isinstance(value, list) and any(query.lower() in item.lower() for item in value):
                results.append({key: value})
        return results if results else [{"info": f"No direct info found for '{query}' in knowledge base."}]

class MockGuardrails:
    def validate_diagnosis(self, diagnosis):
        return "Valid" if diagnosis and "diagnosis" in diagnosis.lower() else "Invalid: Missing diagnosis."

    def validate_treatment_plan(self, plan):
        return "Valid" if plan and "treatment" in plan.lower() else "Invalid: Missing treatment plan."

st.title("Medical Diagnostic and Treatment Planning Assistant (ToRA)")

patient_symptoms = st.text_area("Enter Patient Symptoms (e.g., 'fever, cough, fatigue'):")
patient_history = st.text_area("Enter Patient Medical History (e.g., 'smoker, no allergies'):")
submit_button = st.button("Start Diagnosis")

if submit_button and patient_symptoms:
    st.subheader("Assistant's Reasoning Process:")
    llm = MockLLM()
    executor = CodeExecutor()
    kb = MedicalKnowledgeBase()
    guardrails = MockGuardrails()
    patient_data = {"symptoms": patient_symptoms, "history": patient_history}
    agent_state = AgentState(patient_data)

    max_steps = 5
    current_step = 0
    current_phase = "diagnosis"

    def call_llm_for_diagnosis(state):
        prompt = f"Patient Data: {state.patient_data}\nReasoning History: {state.reasoning_history}\nTool Results: {state.tool_results}\nTask: Diagnose the patient. If more information is needed, generate a tool call (e.g., 'get_medical_database_info(\"query\")')."
        llm_response = llm.invoke(prompt)
        state.reasoning_history.append(llm_response)
        st.write(f"**LLM Thought (Diagnosis):** {llm_response}")
        return llm_response

    def call_llm_for_treatment(state):
        prompt = f"Patient Data: {state.patient_data}\nDiagnosis: {state.diagnosis}\nReasoning History: {state.reasoning_history}\nTool Results: {state.tool_results}\nTask: Formulate a treatment plan. If more information is needed, generate a tool call (e.g., 'search_clinical_guidelines(\"query\")')."
        llm_response = llm.invoke(prompt)
        state.reasoning_history.append(llm_response)
        st.write(f"**LLM Thought (Treatment):** {llm_response}")
        return llm_response

    def execute_tool(state, tool_call_str):
        tool_name = ""
        args_dict = {}
        if tool_call_str.startswith("get_medical_database_info("):
            tool_name = "get_medical_database_info"
            query = tool_call_str.replace("get_medical_database_info(", "").replace(")", "").strip("'\" ")
            args_dict = {"query": query}
        elif tool_call_str.startswith("simulate_treatment_effectiveness("):
            tool_name = "simulate_treatment_effectiveness"
            parts = tool_call_str.replace("simulate_treatment_effectiveness(", "").replace(")", "").split(",", 1)
            condition = parts[0].strip("'\" ")
            treatment = parts[1].strip("'\" ") if len(parts) > 1 else ""
            args_dict = {"condition": condition, "treatment": treatment}
        elif tool_call_str.startswith("search_clinical_guidelines("):
            tool_name = "search_clinical_guidelines"
            query = tool_call_str.replace("search_clinical_guidelines(", "").replace(")", "").strip("'\" ")
            args_dict = {"query": query}

        if tool_name:
            generated_tool_code = llm.generate_tool_code(tool_name, args_dict)
            tool_output = executor.execute(generated_tool_code)
            state.tool_results[f"{tool_name}_{'_'.join(args_dict.values())}"] = tool_output
            st.write(f"**Tool Output:** {tool_output}")
        return tool_name, tool_output

    while current_step < max_steps and current_phase != "finished":
        current_step += 1
        st.write(f"--- Step {current_step} ({current_phase.replace('_', ' ').title()}) ---")
        
        if current_phase == "diagnosis":
            llm_response = call_llm_for_diagnosis(agent_state)
            if "Refined Diagnosis:" in llm_response:
                agent_state.diagnosis = llm_response.split("Refined Diagnosis:")[1].strip()
                st.write(f"**Confirmed Diagnosis:** {agent_state.diagnosis}")
                current_phase = "treatment_planning"
            elif "Generated Code Tool Call:" in llm_response:
                tool_call_str = llm_response.split("Generated Code Tool Call:")[1].strip()
                st.write(f"**Tool Call:** `{tool_call_str}`")
                execute_tool(agent_state, tool_call_str)
                current_phase = "diagnosis"

        elif current_phase == "treatment_planning":
            llm_response = call_llm_for_treatment(agent_state)
            if "Treatment Plan:" in llm_response:
                agent_state.treatment_plan = llm_response.split("Treatment Plan:")[1].strip()
                st.write(f"**Proposed Treatment Plan:** {agent_state.treatment_plan}")
            elif "Generated Code Tool Call:" in llm_response:
                tool_call_str = llm_response.split("Generated Code Tool Call:")[1].strip()
                st.write(f"**Tool Call:** `{tool_call_str}`")
                execute_tool(agent_state, tool_call_str)
                current_phase = "treatment_planning"

        diag_validation = guardrails.validate_diagnosis(agent_state.diagnosis)
        plan_validation = guardrails.validate_treatment_plan(agent_state.treatment_plan)
        st.write(f"**Guardrails Check:** Diagnosis: {diag_validation}, Treatment Plan: {plan_validation}")
        
        if agent_state.diagnosis and agent_state.treatment_plan and diag_validation == "Valid" and plan_validation == "Valid":
            st.success("Diagnosis and Treatment Plan finalized and validated!")
            current_phase = "finished"

    st.subheader("Final Output:")
    if agent_state.diagnosis:
        st.success(f"**Final Diagnosis:** {agent_state.diagnosis}")
    else:
        st.warning("Could not finalize diagnosis.")
    
    if agent_state.treatment_plan:
        st.success(f"**Final Treatment Plan:** {agent_state.treatment_plan}")
    else:
        st.warning("Could not finalize treatment plan.")
    
    st.subheader("Complete Reasoning History:")
    for i, step in enumerate(agent_state.reasoning_history):
        st.text(f"Step {i+1}: {step}")