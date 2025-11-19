import pandas as pd
import numpy as np
import json

# --- Mock LLM and External Tools ---

class MockMedicalAPIClient:
    def check_drug_interactions(self, drugs):
        # Simulate checking drug interactions
        if "ibuprofen" in drugs and "warfarin" in drugs:
            return {"interaction": "High risk of bleeding", "severity": "Severe"}
        return {"interaction": "None", "severity": "Low"}

    def get_genetic_risk(self, genetic_markers):
        # Simulate genetic risk assessment
        if "APOE4" in genetic_markers:
            return {"condition": "Alzheimer's Disease", "risk_factor": "High"}
        return {"condition": "General Health", "risk_factor": "Low"}

mock_medical_api_client = MockMedicalAPIClient()

mock_medical_knowledge_base = {
    "fever": {"possible_causes": ["Flu", "Common Cold", "Bacterial Infection"]},
    "cough": {"possible_causes": ["Flu", "Common Cold", "Bronchitis"]},
    "fatigue": {"possible_causes": ["Flu", "Anemia", "Chronic Fatigue Syndrome"]},
    "high_blood_pressure": {"diagnosis": "Hypertension", "treatment": "Lifestyle changes, medication"},
    "diabetes": {"diagnosis": "Diabetes Mellitus", "treatment": "Insulin, diet management"},
}

def mock_llm_generate_code(prompt):
    # This function simulates the LLM generating Python code based on the prompt.
    # In a real scenario, this would be an actual LLM call.
    # For demonstration, we'll return a hardcoded but parameterized script.

    if "lab_results" in prompt and "blood_sugar" in prompt:
        return """
import pandas as pd
import numpy as np
import json

def run_diagnosis(patient_data, medical_api_client, medical_knowledge_base):
    results = {}
    lab_df = pd.DataFrame([patient_data['lab_results']])

    if 'blood_sugar' in lab_df.columns and lab_df['blood_sugar'].iloc[0] > 125:
        results['diabetes_check'] = medical_knowledge_base['diabetes']
    else:
        results['diabetes_check'] = {"diagnosis": "Normal blood sugar", "treatment": "Monitor"}

    if 'medications' in patient_data:
        results['drug_interactions'] = medical_api_client.check_drug_interactions(patient_data['medications'])

    if 'symptoms' in patient_data:
        symptom_causes = set()
        for symptom in patient_data['symptoms']:
            if symptom in medical_knowledge_base:
                symptom_causes.add(tuple(medical_knowledge_base[symptom]['possible_causes']))
        if symptom_causes:
            results['symptom_analysis'] = list(symptom_causes)

    print(json.dumps(results))
"""
    else:
        return """
import json

def run_diagnosis(patient_data, medical_api_client, medical_knowledge_base):
    results = {}
    if 'symptoms' in patient_data:
        symptom_causes = set()
        for symptom in patient_data['symptoms']:
            if symptom in medical_knowledge_base:
                symptom_causes.add(tuple(medical_knowledge_base[symptom]['possible_causes']))
        if symptom_causes:
            results['symptom_analysis'] = list(symptom_causes)

    if 'medications' in patient_data:
        results['drug_interactions'] = medical_api_client.check_drug_interactions(patient_data['medications'])

    print(json.dumps(results))
"""

def mock_llm_synthesize_answer(prompt):
    # This function simulates the LLM synthesizing the final answer and explanation.
    # In a real scenario, this would be an actual LLM call.
    # We'll parse the prompt to extract results and format an answer.
    try:
        prompt_parts = prompt.split("\n\nComputational Results:\n")
        patient_info = prompt_parts[0].replace("Patient Data:\n", "")
        comp_results_str = prompt_parts[1]
        comp_results = json.loads(comp_results_str)

        diagnosis = "Based on the provided information and computational analysis:\n"
        treatment = "\nSuggested Treatment Plan:\n"
        reasoning = "\nReasoning:\n"

        if "diabetes_check" in comp_results:
            diag_info = comp_results["diabetes_check"]
            diagnosis += f"- **Blood Sugar Analysis:** {diag_info['diagnosis']}.\n"
            treatment += f"  - For blood sugar: {diag_info['treatment']}.\n"
            reasoning += "  - The blood sugar level was analyzed against known thresholds, leading to the diabetes assessment.\n"

        if "symptom_analysis" in comp_results and comp_results["symptom_analysis"]:
            flat_causes = [item for sublist in comp_results["symptom_analysis"] for item in sublist]
            common_causes = pd.Series(flat_causes).value_counts().index.tolist()
            diagnosis += f"- **Symptom Analysis:** Possible causes include {', '.join(common_causes[:2])}.\n"
            reasoning += f"  - Symptoms were cross-referenced with medical knowledge base to identify potential causes: {', '.join(flat_causes)}.\n"

        if "drug_interactions" in comp_results:
            interaction_info = comp_results["drug_interactions"]
            diagnosis += f"- **Drug Interactions:** {interaction_info['interaction']} (Severity: {interaction_info['severity']}).\n"
            reasoning += "  - Patient medications were checked against known drug interaction databases.\n"
            if interaction_info['severity'] == "Severe":
                treatment += f"  - **URGENT:** Address {interaction_info['interaction']} (Severity: {interaction_info['severity']}) by consulting a pharmacist or physician immediately.\n"

        diagnosis += "\nFinal Assessment: Please consult with a medical professional for a definitive diagnosis and personalized treatment plan."

        return {"diagnosis": diagnosis.strip(), "treatment": treatment.strip(), "reasoning": reasoning.strip()}
    except Exception as e:
        return {"diagnosis": "Error during synthesis", "treatment": "N/A", "reasoning": f"Could not synthesize: {e}"}

# --- Core Diagnostic Workflow ---

def _generate_code_prompt(patient_data):
    return f"""Generate Python code to diagnose a patient based on the following data. 
Use the provided `patient_data`, `medical_api_client`, and `medical_knowledge_base` objects. 
The code should perform relevant computations and print a JSON string containing the results.

Patient Data:
{json.dumps(patient_data, indent=2)}
"""

def _execute_generated_code(code_string, patient_data):
    # Capture stdout to get the JSON output from the executed code
    import io
    import sys

    old_stdout = sys.stdout
    redirected_output = io.StringIO()
    sys.stdout = redirected_output

    execution_globals = {
        "patient_data": patient_data,
        "medical_api_client": mock_medical_api_client,
        "medical_knowledge_base": mock_medical_knowledge_base,
        "pd": pd, # Make pandas available if the generated code needs it
        "np": np, # Make numpy available if the generated code needs it
        "json": json,
    }

    try:
        # The generated code should define and call a 'run_diagnosis' function
        exec(code_string, execution_globals)
        if "run_diagnosis" in execution_globals and callable(execution_globals["run_diagnosis"]):
            execution_globals["run_diagnosis"](patient_data, mock_medical_api_client, mock_medical_knowledge_base)
        execution_output = redirected_output.getvalue()
    except Exception as e:
        execution_output = json.dumps({"error": str(e), "traceback": "See server logs for full traceback"})
    finally:
        sys.stdout = old_stdout # Restore original stdout

    return execution_output

def _parse_execution_output(output):
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        return {"error": "Failed to parse JSON from executed code output", "raw_output": output}

def _generate_explanation_prompt(patient_data, execution_results):
    return f"""Based on the following patient data and computational analysis results, provide a comprehensive medical diagnosis, a suggested treatment plan, and a clear, explainable reasoning for your conclusions.

Patient Data:
{json.dumps(patient_data, indent=2)}

Computational Results:
{json.dumps(execution_results, indent=2)}
"""

def diagnose_patient(patient_data):
    # Step 1: Generate prompt for code creation
    code_prompt = _generate_code_prompt(patient_data)

    # Step 2: Simulate LLM generating code
    generated_code = mock_llm_generate_code(code_prompt)
    print(f"\n--- Generated Code ---\n{generated_code}\n---")

    # Step 3: Execute the generated code
    execution_output = _execute_generated_code(generated_code, patient_data)
    print(f"\n--- Code Execution Output ---\n{execution_output}\n---")

    # Step 4: Parse the execution output
    execution_results = _parse_execution_output(execution_output)

    if "error" in execution_results:
        return {"diagnosis": "Error in computational step", "treatment": "N/A", "reasoning": execution_results}

    # Step 5: Generate prompt for explanation synthesis
    explanation_prompt = _generate_explanation_prompt(patient_data, execution_results)

    # Step 6: Simulate LLM synthesizing the final answer
    final_answer = mock_llm_synthesize_answer(explanation_prompt)

    return final_answer


if __name__ == "__main__":
    # Example Usage 1: Patient with symptoms and lab results
    patient1_data = {
        "name": "Alice",
        "age": 45,
        "gender": "Female",
        "symptoms": ["fever", "cough", "fatigue"],
        "medical_history": ["hypertension"],
        "medications": ["ibuprofen", "amlodipine"],
        "lab_results": {"blood_sugar": 130, "white_blood_cells": 12000}
    }

    print("\n========== DIAGNOSIS FOR PATIENT ALICE ==========")
    diagnosis1 = diagnose_patient(patient1_data)
    print("\nFinal Diagnosis for Alice:")
    print(f"Diagnosis: {diagnosis1['diagnosis']}")
    print(f"Treatment: {diagnosis1['treatment']}")
    print(f"Reasoning: {diagnosis1['reasoning']}")

    print("\n" + "="*60 + "\n")

    # Example Usage 2: Patient with different symptoms and genetic markers
    patient2_data = {
        "name": "Bob",
        "age": 60,
        "gender": "Male",
        "symptoms": ["fatigue", "memory loss"],
        "medical_history": ["high cholesterol"],
        "medications": ["atorvastatin"],
        "genetic_markers": ["APOE4"]
    }

    # For Bob, we'll manually craft a code string that uses get_genetic_risk
    # In a real PAL system, the LLM would generate this based on genetic_markers in patient_data
    def mock_llm_generate_code_for_bob(prompt):
        if "genetic_markers" in prompt:
            return """
import json

def run_diagnosis(patient_data, medical_api_client, medical_knowledge_base):
    results = {}
    if 'symptoms' in patient_data:
        symptom_causes = set()
        for symptom in patient_data['symptoms']:
            if symptom in medical_knowledge_base:
                symptom_causes.add(tuple(medical_knowledge_base[symptom]['possible_causes']))
        if symptom_causes:
            results['symptom_analysis'] = list(symptom_causes)

    if 'genetic_markers' in patient_data:
        results['genetic_risk_assessment'] = medical_api_client.get_genetic_risk(patient_data['genetic_markers'])

    print(json.dumps(results))
"""
        return mock_llm_generate_code(prompt) # Fallback to default if no specific genetic marker handling

    # Temporarily override the mock_llm_generate_code for this specific test case
    original_mock_llm_generate_code = mock_llm_generate_code
    mock_llm_generate_code = mock_llm_generate_code_for_bob

    print("\n========== DIAGNOSIS FOR PATIENT BOB ==========")
    diagnosis2 = diagnose_patient(patient2_data)
    print("\nFinal Diagnosis for Bob:")
    print(f"Diagnosis: {diagnosis2['diagnosis']}")
    print(f"Treatment: {diagnosis2['treatment']}")
    print(f"Reasoning: {diagnosis2['reasoning']}")

    # Restore original mock_llm_generate_code
    mock_llm_generate_code = original_mock_llm_generate_code

    print("\n" + "="*60 + "\n")

    # Example Usage 3: Simple case with only symptoms
    patient3_data = {
        "name": "Charlie",
        "age": 30,
        "gender": "Male",
        "symptoms": ["cough"],
        "medical_history": [],
        "medications": []
    }

    print("\n========== DIAGNOSIS FOR PATIENT CHARLIE ==========")
    diagnosis3 = diagnose_patient(patient3_data)
    print("\nFinal Diagnosis for Charlie:")
    print(f"Diagnosis: {diagnosis3['diagnosis']}")
    print(f"Treatment: {diagnosis3['treatment']}")
    print(f"Reasoning: {diagnosis3['reasoning']}")

    print("\n" + "="*60 + "\n")
