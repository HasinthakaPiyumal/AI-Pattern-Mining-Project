import streamlit as st
import time
import json
import random
from typing import List, Dict, Any

# --- 1. Specialized Medical Tools (Mock Implementations) ---

class MedicalTools:
    @staticmethod
    def search_rare_disease_db(query: str) -> Dict[str, Any]:
        """Mock function for searching rare disease databases (Orphanet, GARD, OMIM)."""
        st.sidebar.info(f"Tool: Searching rare disease databases for '{query}'...")
        time.sleep(1.5)
        if "cystic fibrosis" in query.lower():
            return {
                "tool_name": "RareDiseaseDB",
                "query": query,
                "result": "Cystic Fibrosis (CF) is a genetic disorder that affects mostly the lungs, but also the pancreas, liver, kidneys, and intestine.",
                "prevalence": "~1 in 3,500 newborns in the US",
                "genes": ["CFTR"]
            }
        elif "huntington" in query.lower():
             return {
                "tool_name": "RareDiseaseDB",
                "query": query,
                "result": "Huntington's disease is a progressive neurodegenerative disorder caused by a genetic defect on chromosome 4.",
                "prevalence": "~1 in 10,000 to 20,000 people",
                "genes": ["HTT"]
            }
        else:
            return {
                "tool_name": "RareDiseaseDB",
                "query": query,
                "result": "No specific rare disease information found for this query in mock database."
            }

    @staticmethod
    def search_medical_literature(query: str) -> Dict[str, Any]:
        """Mock function for searching medical literature (PubMed, Medline)."""
        st.sidebar.info(f"Tool: Searching medical literature for '{query}'...")
        time.sleep(2)
        if "CFTR mutation treatment" in query.lower():
            return {
                "tool_name": "PubMed",
                "query": query,
                "result": "Recent studies show promising results for CFTR modulator therapies like Trikafta in patients with specific CFTR mutations.",
                "articles": ["PMID:12345", "PMID:67890"]
            }
        else:
            return {
                "tool_name": "PubMed",
                "query": query,
                "result": "Mock literature search found general information, consider refining query."
            }

    @staticmethod
    def get_ehr_data(patient_id: str) -> Dict[str, Any]:
        """Mock function for retrieving anonymized EHR data (FHIR standard)."""
        st.sidebar.info(f"Tool: Retrieving EHR data for patient '{patient_id}'...")
        time.sleep(1)
        if patient_id == "P1001":
            return {
                "tool_name": "EHR",
                "patient_id": patient_id,
                "symptoms": ["persistent cough", "recurrent lung infections", "poor weight gain"],
                "lab_results": {"sweat_chloride": "elevated (90 mmol/L)", "genetic_test": "CFTR mutation detected"},
                "family_history": "sibling with similar symptoms"
            }
        elif patient_id == "P1002":
            return {
                "tool_name": "EHR",
                "patient_id": patient_id,
                "symptoms": ["involuntary movements", "cognitive decline", "mood swings"],
                "lab_results": {"genetic_test": "HTT gene expansion detected"},
                "family_history": "parent with similar condition"
            }
        else:
            return {
                "tool_name": "EHR",
                "patient_id": patient_id,
                "symptoms": [],
                "lab_results": {}, 
                "family_history": "N/A"
            }

    @staticmethod
    def run_symptom_checker(symptoms: List[str]) -> Dict[str, Any]:
        """Mock diagnostic algorithm based on symptoms."""
        st.sidebar.info(f"Tool: Running symptom checker with {len(symptoms)} symptoms...")
        time.sleep(2.5)
        if "persistent cough" in symptoms and "recurrent lung infections" in symptoms and "poor weight gain" in symptoms:
            return {
                "tool_name": "SymptomChecker",
                "symptoms": symptoms,
                "differential_diagnosis": [{"disease": "Cystic Fibrosis", "probability": 0.9}, {"disease": "Asthma", "probability": 0.1}],
                "recommendation": "Consider sweat chloride test and genetic testing."
            }
        elif "involuntary movements" in symptoms and "cognitive decline" in symptoms:
            return {
                "tool_name": "SymptomChecker",
                "symptoms": symptoms,
                "differential_diagnosis": [{"disease": "Huntington's Disease", "probability": 0.85}, {"disease": "Parkinson's Disease", "probability": 0.1}],
                "recommendation": "Consult neurology for genetic testing and imaging."
            }
        else:
            return {
                "tool_name": "SymptomChecker",
                "symptoms": symptoms,
                "differential_diagnosis": [{"disease": "General Respiratory Infection", "probability": 0.6}, {"disease": "Undetermined Rare Disease", "probability": 0.4}],
                "recommendation": "Further investigation required, consider specialized consultations."
            }

    @staticmethod
    def execute_python_code(code: str) -> Dict[str, Any]:
        """Mock secure Python code interpreter for statistical analysis or data manipulation."""
        st.sidebar.info("Tool: Executing Python code...")
        time.sleep(1)
        try:
            # In a real system, this would be sandboxed and very secure
            # For demonstration, we'll do a very basic eval (DANGEROUS IN PROD!)
            if "1+1" in code:
                result = eval(code)
                return {"tool_name": "CodeInterpreter", "code": code, "output": str(result)}
            elif "mean" in code and "data" in code:
                return {"tool_name": "CodeInterpreter", "code": code, "output": "Mock: Mean calculated as 5.5"}
            else:
                return {"tool_name": "CodeInterpreter", "code": code, "output": "Mock: Code executed, result simulated."}
        except Exception as e:
            return {"tool_name": "CodeInterpreter", "code": code, "output": f"Error executing code: {e}"}

# --- 2. LLM Controller (Mock Implementation) ---

class LLMController:
    def __init__(self):
        self.tools = MedicalTools()
        self.reasoning_trace: List[Dict[str, Any]] = []

    def _orchestrate_tools(self, patient_data: Dict[str, Any]) -> str:
        """Simulates the LLM's decision-making and tool orchestration logic."""
        self.reasoning_trace.append({"step": "Initial Analysis", "thought": "Analyzing patient data to determine diagnostic path."})
        output = []
        patient_id = patient_data.get("patient_id", "N/A")
        initial_symptoms = patient_data.get("symptoms", [])

        # Step 1: Retrieve EHR data if patient_id is provided
        if patient_id != "N/A":
            ehr_result = self.tools.get_ehr_data(patient_id)
            self.reasoning_trace.append({"step": "EHR Retrieval", "tool_output": ehr_result})
            output.append(f"Retrieved EHR data for Patient {patient_id}. Symptoms: {ehr_result.get('symptoms')}, Labs: {ehr_result.get('lab_results')}")
            # Update symptoms with EHR data
            initial_symptoms.extend(ehr_result.get("symptoms", []))
            initial_symptoms = list(set(initial_symptoms)) # Remove duplicates

        # Step 2: Run Symptom Checker
        if initial_symptoms:
            symptom_check_result = self.tools.run_symptom_checker(initial_symptoms)
            self.reasoning_trace.append({"step": "Symptom Check", "tool_output": symptom_check_result})
            output.append(f"Symptom Checker suggests: {symptom_check_result.get('differential_diagnosis')}. Recommendation: {symptom_check_result.get('recommendation')}")
            
            # Focus on the top probable disease for further investigation
            top_diagnosis = symptom_check_result['differential_diagnosis'][0]['disease'] if symptom_check_result['differential_diagnosis'] else None

            if top_diagnosis and top_diagnosis != "Undetermined Rare Disease":
                # Step 3: Search Rare Disease DB based on top diagnosis
                rare_disease_info = self.tools.search_rare_disease_db(top_diagnosis)
                self.reasoning_trace.append({"step": "Rare Disease DB Search", "tool_output": rare_disease_info})
                output.append(f"Rare Disease DB info for '{top_diagnosis}': {rare_disease_info.get('result')}. Genes: {rare_disease_info.get('genes')}")

                # Step 4: Search medical literature for treatment options or further details
                if "genes" in rare_disease_info and rare_disease_info["genes"]:
                    gene = rare_disease_info["genes"][0]
                    literature_query = f"{gene} mutation treatment"
                    literature_result = self.tools.search_medical_literature(literature_query)
                    self.reasoning_trace.append({"step": "Medical Literature Search", "tool_output": literature_result})
                    output.append(f"Medical Literature on '{literature_query}': {literature_result.get('result')}")
                    
                # Step 5: (PAL example) If a specific numerical analysis is needed
                if "elevated" in str(ehr_result.get('lab_results')) and "sweat_chloride" in str(ehr_result.get('lab_results')):
                    code_to_execute = "data = [90, 85, 95]; sum(data)/len(data)" # Mock simple calculation
                    pal_result = self.tools.execute_python_code(code_to_execute)
                    self.reasoning_trace.append({"step": "PAL Execution", "tool_output": pal_result})
                    output.append(f"Executed code for data analysis: {pal_result.get('output')}")

        else:
            output.append("No symptoms provided, limited analysis performed.")

        self.reasoning_trace.append({"step": "Final Synthesis", "thought": "Synthesizing information from all tools for diagnosis."})
        return "\n".join(output)

    def process_patient_query(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Main entry point for processing a patient query through the LLM controller."""
        self.reasoning_trace = [] # Reset trace for new query
        st.write("### LLM Controller in Action")
        
        # Simulate initial LLM thought process
        st.write("LLM thought: User provided patient data. I need to gather information from various medical tools to formulate a diagnosis.")
        
        # Orchestrate tools
        raw_diagnosis_output = self._orchestrate_tools(patient_data)
        
        # Simulate LLM generating a final diagnosis from raw tool outputs
        final_diagnosis = self._synthesize_diagnosis(patient_data, raw_diagnosis_output)
        
        # Hallucination and Guardrails (mock)
        hallucination_check = self._check_hallucinations(final_diagnosis, self.reasoning_trace)
        guardrail_check = self._apply_guardrails(final_diagnosis)

        return {
            "final_diagnosis": final_diagnosis,
            "reasoning_trace": self.reasoning_trace,
            "hallucination_check": hallucination_check,
            "guardrail_check": guardrail_check
        }

    def _synthesize_diagnosis(self, patient_data: Dict[str, Any], raw_output: str) -> str:
        """Mock LLM synthesis of information into a coherent diagnosis."""
        st.write("LLM thought: All relevant tools have been consulted. Now, I will synthesize this information to provide a comprehensive diagnostic insight.")
        time.sleep(1)
        # This is where a real LLM would generate natural language explanation
        symptoms = patient_data.get("symptoms", "No symptoms provided")
        patient_id = patient_data.get("patient_id", "N/A")
        return f"Based on the provided symptoms ({', '.join(symptoms)}) and EHR data for patient {patient_id}, the system has processed information from various medical knowledge bases and diagnostic algorithms. \n\n**Key Findings:**\n{raw_output}\n\n**Conclusion:** \nThis iterative analysis points towards a high likelihood of a rare disease (e.g., Cystic Fibrosis or Huntington's Disease based on mock data). Further clinical evaluation and confirmatory tests are strongly recommended based on these insights. The system's purpose is to augment, not replace, clinical judgment."

    def _check_hallucinations(self, text: str, context: List[Dict[str, Any]]) -> str:
        """Mock AST-based Hallucination Detection. In a real system, this would involve parsing and factual lookup."""
        st.write("LLM thought: Performing a factual consistency check to detect any hallucinations...")
        time.sleep(0.5)
        # For demonstration, we'll assume no hallucinations
        return "No significant factual inconsistencies or hallucinations detected based on available mock data and context."

    def _apply_guardrails(self, text: str) -> str:
        """Mock Guardrails AI application for medical appropriateness and safety."""
        st.write("LLM thought: Applying ethical and safety guardrails to ensure the diagnostic output is appropriate.")
        time.sleep(0.5)
        # For demonstration, assume output is safe and appropriate
        return "Output adheres to mock medical appropriateness and safety guidelines. Always remember, this is a decision support tool, not a definitive diagnosis."

# --- 3. Streamlit User Interface --- 

st.set_page_config(layout="wide", page_title="Rare Disease CDSS")
st.title("🧠 AI-Powered Clinical Decision Support for Rare Disease Diagnosis")
st.markdown("This system leverages advanced AI patterns to assist healthcare professionals in diagnosing rare diseases.")
st.markdown("--- ")

# Sidebar for patient input
st.sidebar.header("Patient Information")
patient_id_input = st.sidebar.text_input("Patient ID (e.g., P1001, P1002)", "P1001")
symptoms_input = st.sidebar.text_area("Enter Symptoms (comma-separated)", "persistent cough, recurrent lung infections, poor weight gain")

# Process button
if st.sidebar.button("Run Diagnosis"): 
    if not symptoms_input and not patient_id_input:
        st.sidebar.warning("Please enter patient symptoms or ID to run diagnosis.")
    else:
        with st.spinner("Running AI Diagnosis..."): 
            patient_data = {
                "patient_id": patient_id_input if patient_id_input else "N/A",
                "symptoms": [s.strip() for s in symptoms_input.split(',')] if symptoms_input else []
            }
            
            controller = LLMController()
            diagnostic_result = controller.process_patient_query(patient_data)
            
            st.success("Diagnosis Complete!")
            st.markdown("--- ")
            
            st.header("1. Final Diagnostic Hypothesis")
            st.write(diagnostic_result["final_diagnosis"])
            st.markdown("--- ")
            
            st.header("2. Reasoning Trace (LLM Tool Orchestration)")
            for i, step in enumerate(diagnostic_result["reasoning_trace"]):
                st.json(step)
            st.markdown("--- ")
            
            st.header("3. Evaluation & Mitigation")
            st.subheader("Hallucination Detection")
            st.info(diagnostic_result["hallucination_check"])
            st.subheader("Guardrails Check")
            st.info(diagnostic_result["guardrail_check"])

st.sidebar.markdown("--- ")
st.sidebar.info("Example Patient IDs: P1001 (Cystic Fibrosis), P1002 (Huntington's Disease)")
