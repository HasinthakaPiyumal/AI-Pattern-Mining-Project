from pydantic import BaseModel
from typing import List, Dict, Any, Tuple
import random

class PatientData(BaseModel):
    symptoms: List[str]
    medical_history: List[str]
    test_results: Dict[str, Any]

class Diagnosis(BaseModel):
    condition: str
    reasoning: str

class VerificationResult(BaseModel):
    diagnosis: Diagnosis
    confidence_score: float
    predictions: Dict[str, str]

class MockLLM:
    def generate(self, prompt: str) -> str:
        if "diagnose" in prompt.lower():
            if "fever" in prompt.lower() and "cough" in prompt.lower():
                return "Diagnosis: Common Cold\nReasoning: The patient presents with classic symptoms of fever and cough, indicating a likely viral respiratory infection. Rest and hydration are recommended." \
                       "Diagnosis: Flu\nReasoning: Given the fever and cough, influenza is also a possibility, especially if other systemic symptoms like body aches are present. Testing might be required." \
                       "Diagnosis: Bronchitis\nReasoning: Persistent cough could suggest bronchitis, an inflammation of the bronchial tubes, often following a viral infection."
            return "Diagnosis: Unknown Condition\nReasoning: Based on the provided information, further investigation is needed."
        elif "predict the masked information" in prompt.lower():
            if "fever is [MASK]" in prompt.lower() and "Common Cold" in prompt.lower():
                return "predicted_symptom_fever: present"
            if "test_result_white_blood_cell_count is [MASK]" in prompt.lower() and "infection" in prompt.lower():
                return "predicted_test_result_white_blood_cell_count: elevated"
            return "predicted_value: unknown"
        return "LLM Response Placeholder"

def generate_diagnoses(patient_data: PatientData, num_candidates: int, llm: MockLLM) -> List[Diagnosis]:
    diagnoses = []
    prompt = f"As a medical expert, diagnose the patient based on the following: Symptoms: {', '.join(patient_data.symptoms)}. Medical History: {', '.join(patient_data.medical_history)}. Test Results: {patient_data.test_results}. Provide {num_candidates} distinct possible diagnoses with detailed Chain-of-Thought reasoning for each.\n"
    llm_raw_response = llm.generate(prompt)

    # Simulate parsing multiple diagnoses from the LLM response
    raw_diagnoses = [d.strip() for d in llm_raw_response.split('Diagnosis:') if d.strip()]
    for raw_diag in raw_diagnoses:
        parts = raw_diag.split('\nReasoning:', 1)
        if len(parts) == 2:
            condition = parts[0].strip()
            reasoning = parts[1].strip()
            diagnoses.append(Diagnosis(condition=condition, reasoning=reasoning))
        if len(diagnoses) >= num_candidates:
            break
    return diagnoses

def mask_patient_data(patient_data: PatientData) -> Tuple[PatientData, Dict[str, Any], str]:
    masked_data = patient_data.model_copy(deep=True)
    actual_masked_values = {}
    masked_field_name = ""

    maskable_fields = []
    if patient_data.symptoms: maskable_fields.append(("symptoms", random.choice(patient_data.symptoms)))
    if patient_data.medical_history: maskable_fields.append(("medical_history", random.choice(patient_data.medical_history)))
    if patient_data.test_results: maskable_fields.extend([("test_results", k) for k in patient_data.test_results.keys()])

    if not maskable_fields:
        return patient_data, {}, ""

    field_type, item_to_mask = random.choice(maskable_fields)
    masked_field_name = field_type

    if field_type == "symptoms":
        idx = masked_data.symptoms.index(item_to_mask)
        actual_masked_values["symptom"] = masked_data.symptoms[idx]
        masked_data.symptoms[idx] = "[MASK]"
        masked_field_name = f"symptom_{item_to_mask.replace(' ', '_')}"
    elif field_type == "medical_history":
        idx = masked_data.medical_history.index(item_to_mask)
        actual_masked_values["medical_history_item"] = masked_data.medical_history[idx]
        masked_data.medical_history[idx] = "[MASK]"
        masked_field_name = f"medical_history_{item_to_mask.replace(' ', '_')}"
    elif field_type == "test_results":
        actual_masked_values["test_result"] = masked_data.test_results[item_to_mask]
        masked_data.test_results[item_to_mask] = "[MASK]"
        masked_field_name = f"test_result_{item_to_mask.replace(' ', '_')}"

    return masked_data, actual_masked_values, masked_field_name

def verify_diagnosis(diagnosis: Diagnosis, masked_patient_data: PatientData, actual_masked_values: Dict[str, Any], masked_field_name: str, llm: MockLLM) -> VerificationResult:
    prompt = f"Given the patient data: Symptoms: {', '.join(masked_patient_data.symptoms)}. Medical History: {', '.join(masked_patient_data.medical_history)}. Test Results: {masked_patient_data.test_results}. " \
             f"And the proposed diagnosis with reasoning: Condition: {diagnosis.condition}. Reasoning: {diagnosis.reasoning}. " \
             f"Predict the masked information for {masked_field_name}. Provide the prediction in the format: predicted_masked_field_name: predicted_value."

    llm_prediction_raw = llm.generate(prompt)
    predicted_values = {}
    confidence_score = 0.0

    for key, actual_value in actual_masked_values.items():
        expected_prediction_key = f"predicted_{masked_field_name}"
        if expected_prediction_key in llm_prediction_raw:
            predicted_value = llm_prediction_raw.split(f"{expected_prediction_key}: ", 1)[1].split('\n')[0].strip()
            predicted_values[masked_field_name] = predicted_value
            if str(actual_value).lower() == predicted_value.lower():
                confidence_score = 1.0
        else:
            predicted_values[masked_field_name] = "not predicted"

    return VerificationResult(diagnosis=diagnosis, confidence_score=confidence_score, predictions=predicted_values)

def select_best_diagnosis(verification_results: List[VerificationResult]) -> Diagnosis:
    if not verification_results:
        return None
    return max(verification_results, key=lambda res: res.confidence_score).diagnosis

if __name__ == "__main__":
    mock_llm = MockLLM()

    patient_data = PatientData(
        symptoms=["fever", "cough", "fatigue"],
        medical_history=["seasonal allergies"],
        test_results={
            "temperature": "101.5F",
            "white_blood_cell_count": "12000",
            "crp": "elevated"
        }
    )

    print("\n--- Generating Diagnoses ---")
    candidate_diagnoses = generate_diagnoses(patient_data, 3, mock_llm)
    for i, diag in enumerate(candidate_diagnoses):
        print(f"Candidate {i+1}: {diag.condition}")
        print(f"  Reasoning: {diag.reasoning}")

    print("\n--- Performing Self-Verification ---")
    verification_results = []
    for diag in candidate_diagnoses:
        masked_data, actual_masked_values, masked_field_name = mask_patient_data(patient_data)
        print(f"  Masked data for verification (original {masked_field_name} was {actual_masked_values.get('symptom') or actual_masked_values.get('medical_history_item') or actual_masked_values.get('test_result')}): {masked_data}")
        result = verify_diagnosis(diag, masked_data, actual_masked_values, masked_field_name, mock_llm)
        verification_results.append(result)
        print(f"  Verification for '{result.diagnosis.condition}': Confidence Score = {result.confidence_score:.2f}, Predictions = {result.predictions}")

    print("\n--- Selecting Best Diagnosis ---")
    best_diagnosis = select_best_diagnosis(verification_results)

    if best_diagnosis:
        print(f"\nPrimary Recommended Diagnosis: {best_diagnosis.condition}")
        print(f"Reasoning: {best_diagnosis.reasoning}")
    else:
        print("No diagnosis could be selected.")

    print("\n--- All Verified Diagnoses with Scores ---")
    for result in sorted(verification_results, key=lambda r: r.confidence_score, reverse=True):
        print(f"Condition: {result.diagnosis.condition}, Confidence: {result.confidence_score:.2f}")
