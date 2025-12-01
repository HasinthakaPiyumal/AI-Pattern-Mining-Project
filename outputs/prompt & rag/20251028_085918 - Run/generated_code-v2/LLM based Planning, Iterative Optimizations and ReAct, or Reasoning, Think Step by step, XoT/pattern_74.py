from pydantic import BaseModel
from typing import List, Dict, Tuple
import json

# Mocking langchain for demonstration without actual API calls
class MockLLM:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def invoke(self, prompt: str) -> str:
        if "initial diagnosis" in prompt.lower():
            return json.dumps({"diagnosis": "Influenza", "reasoning": "Patient exhibits fever, cough, and body aches, consistent with influenza virus infection."})
        elif "reconstruct the original patient's symptoms" in prompt.lower():
            return json.dumps({"reconstructed_symptoms": ["fever", "cough", "body aches"], "reconstructed_medical_history": ["no chronic conditions"], "reconstructed_test_results": {"flu_test": "positive"}})
        elif "generate structured feedback" in prompt.lower():
            return json.dumps({"inconsistencies": ["Reconstructed test results mentioned 'flu_test: positive' but original patient data did not specify test results."], "suggested_revisions": "Ensure to only reconstruct information explicitly stated or strongly implied by the diagnosis and reasoning."})
        elif "revise its original diagnosis" in prompt.lower():
            return json.dumps({"diagnosis": "Influenza (Confirmed)", "reasoning": "Patient exhibits fever, cough, and body aches, consistent with influenza virus infection. Original diagnosis lacked specific test result. Revised based on reconstructed test result indicating positive flu test, assuming this was implicitly part of the context for the initial diagnosis."})
        return "{}"


class PromptTemplate:
    def __init__(self, template: str, input_variables: List[str]):
        self.template = template
        self.input_variables = input_variables

    def format(self, **kwargs) -> str:
        formatted_template = self.template
        for var in self.input_variables:
            formatted_template = formatted_template.replace(f"{{{var}}}", str(kwargs.get(var, '')))
        return formatted_template


class PatientData(BaseModel):
    symptoms: List[str]
    medical_history: List[str]
    test_results: Dict[str, str]


class Diagnosis(BaseModel):
    diagnosis: str
    reasoning: str


class ReconstructedProblem(BaseModel):
    reconstructed_symptoms: List[str]
    reconstructed_medical_history: List[str]
    reconstructed_test_results: Dict[str, str]


class Feedback(BaseModel):
    inconsistencies: List[str]
    suggested_revisions: str


class MDVSVerifier:
    def __init__(self, llm_model_name: str):
        self.llm = MockLLM(llm_model_name)

    def _initial_diagnosis_llm(self, patient_data: PatientData) -> Diagnosis:
        prompt_template = PromptTemplate(
            template=(
                "You are a medical diagnostic AI. Based on the following patient data, provide a diagnosis and detailed reasoning in JSON format.\n\n"
                "Patient Symptoms: {symptoms}\n"
                "Medical History: {medical_history}\n"
                "Test Results: {test_results}\n\n"
                "JSON Output: {{\"diagnosis\": \"<diagnosis>\", \"reasoning\": \"<reasoning>\"}}"
            ),
            input_variables=["symptoms", "medical_history", "test_results"],
        )
        formatted_prompt = prompt_template.format(
            symptoms=patient_data.symptoms,
            medical_history=patient_data.medical_history,
            test_results=patient_data.test_results,
        )
        response = self.llm.invoke(formatted_prompt)
        return Diagnosis.parse_raw(response)

    def _problem_reconstruction_llm(self, diagnosis: Diagnosis) -> ReconstructedProblem:
        prompt_template = PromptTemplate(
            template=(
                "Based solely on the following diagnosis and reasoning, reconstruct the original patient's symptoms, medical history, and test results in JSON format. Do not invent new information; only infer what is strictly necessary for this diagnosis.\n\n"
                "Diagnosis: {diagnosis}\n"
                "Reasoning: {reasoning}\n\n"
                "JSON Output: {{\"reconstructed_symptoms\": [\"<symptom1>\", ...], \"reconstructed_medical_history\": [\"<history1>\", ...], \"reconstructed_test_results\": {{\"<test_name>\": \"<result>\"}}}}"
            ),
            input_variables=["diagnosis", "reasoning"],
        )
        formatted_prompt = prompt_template.format(
            diagnosis=diagnosis.diagnosis, reasoning=diagnosis.reasoning
        )
        response = self.llm.invoke(formatted_prompt)
        return ReconstructedProblem.parse_raw(response)

    def _compare_data(
        self, original_data: PatientData, reconstructed_data: ReconstructedProblem
    ) -> List[str]:
        inconsistencies = []

        # Compare symptoms
        original_symptoms_set = set(s.lower() for s in original_data.symptoms)
        reconstructed_symptoms_set = set(s.lower() for s in reconstructed_data.reconstructed_symptoms)

        missing_in_reconstructed_symptoms = original_symptoms_set - reconstructed_symptoms_set
        if missing_in_reconstructed_symptoms:
            inconsistencies.append(
                f"Original symptoms missing in reconstructed problem: {', '.join(missing_in_reconstructed_symptoms)}"
            )

        added_in_reconstructed_symptoms = reconstructed_symptoms_set - original_symptoms_set
        if added_in_reconstructed_symptoms:
            inconsistencies.append(
                f"Reconstructed symptoms not present in original problem: {', '.join(added_in_reconstructed_symptoms)}"
            )

        # Compare medical history
        original_history_set = set(h.lower() for h in original_data.medical_history)
        reconstructed_history_set = set(h.lower() for h in reconstructed_data.reconstructed_medical_history)

        missing_in_reconstructed_history = original_history_set - reconstructed_history_set
        if missing_in_reconstructed_history:
            inconsistencies.append(
                f"Original medical history missing in reconstructed problem: {', '.join(missing_in_reconstructed_history)}"
            )
        added_in_reconstructed_history = reconstructed_history_set - original_history_set
        if added_in_reconstructed_history:
            inconsistencies.append(
                f"Reconstructed medical history not present in original problem: {', '.join(added_in_reconstructed_history)}"
            )

        # Compare test results
        for key, value in original_data.test_results.items():
            if key not in reconstructed_data.reconstructed_test_results or reconstructed_data.reconstructed_test_results[key].lower() != value.lower():
                inconsistencies.append(
                    f"Test result '{key}' with value '{value}' from original data is inconsistent with reconstructed data or missing."
                )
        for key, value in reconstructed_data.reconstructed_test_results.items():
            if key not in original_data.test_results:
                inconsistencies.append(
                    f"Reconstructed test result '{key}' with value '{value}' not present in original data."
                )

        return inconsistencies

    def _generate_feedback_llm(
        self, inconsistencies: List[str], original_diagnosis: Diagnosis
    ) -> Feedback:
        prompt_template = PromptTemplate(
            template=(
                "Based on the following factual inconsistencies found by comparing original patient data with reconstructed data from an LLM's diagnosis, generate structured feedback and suggested revisions for the LLM. Focus on how to improve factual consistency.\n\n"
                "Original Diagnosis: {diagnosis}\n"
                "Original Reasoning: {reasoning}\n\n"
                "Inconsistencies: {inconsistencies}\n\n"
                "JSON Output: {{\"inconsistencies\": [\"<inconsistency1>\", ...], \"suggested_revisions\": \"<suggested_revisions>\"}}"
            ),
            input_variables=["diagnosis", "reasoning", "inconsistencies"],
        )
        formatted_prompt = prompt_template.format(
            diagnosis=original_diagnosis.diagnosis,
            reasoning=original_diagnosis.reasoning,
            inconsistencies=inconsistencies,
        )
        response = self.llm.invoke(formatted_prompt)
        return Feedback.parse_raw(response)

    def _revise_diagnosis_llm(
        self, original_diagnosis: Diagnosis, feedback: Feedback
    ) -> Diagnosis:
        prompt_template = PromptTemplate(
            template=(
                "An earlier diagnosis and reasoning had factual inconsistencies. Based on the following feedback, revise the original diagnosis and reasoning to improve factual consistency. Provide the revised diagnosis and reasoning in JSON format.\n\n"
                "Original Diagnosis: {original_diagnosis}\n"
                "Original Reasoning: {original_reasoning}\n"
                "Feedback Inconsistencies: {feedback_inconsistencies}\n"
                "Suggested Revisions: {suggested_revisions}\n\n"
                "JSON Output: {{\"diagnosis\": \"<revised_diagnosis>\", \"reasoning\": \"<revised_reasoning>\"}}"
            ),
            input_variables=[
                "original_diagnosis",
                "original_reasoning",
                "feedback_inconsistencies",
                "suggested_revisions",
            ],
        )
        formatted_prompt = prompt_template.format(
            original_diagnosis=original_diagnosis.diagnosis,
            original_reasoning=original_diagnosis.reasoning,
            feedback_inconsistencies=feedback.inconsistencies,
            suggested_revisions=feedback.suggested_revisions,
        )
        response = self.llm.invoke(formatted_prompt)
        return Diagnosis.parse_raw(response)

    def verify_diagnosis(
        self, patient_data: PatientData
    ) -> Tuple[Diagnosis, Diagnosis, List[str]]:
        print("\n--- Step 1: Initial Diagnosis --- ")
        original_diagnosis = self._initial_diagnosis_llm(patient_data)
        print(f"Original Diagnosis: {original_diagnosis.json()}")

        print("\n--- Step 2: Problem Reconstruction --- ")
        reconstructed_problem = self._problem_reconstruction_llm(original_diagnosis)
        print(f"Reconstructed Problem: {reconstructed_problem.json()}")

        print("\n--- Step 3: Compare Data --- ")
        inconsistencies = self._compare_data(patient_data, reconstructed_problem)
        if inconsistencies:
            print("Inconsistencies found:")
            for inconsistency in inconsistencies:
                print(f"  - {inconsistency}")

            print("\n--- Step 4: Generate Feedback --- ")
            feedback = self._generate_feedback_llm(inconsistencies, original_diagnosis)
            print(f"Generated Feedback: {feedback.json()}")

            print("\n--- Step 5: Revise Diagnosis --- ")
            revised_diagnosis = self._revise_diagnosis_llm(original_diagnosis, feedback)
            print(f"Revised Diagnosis: {revised_diagnosis.json()}")
            return original_diagnosis, revised_diagnosis, inconsistencies
        else:
            print("No inconsistencies found. Diagnosis is factually consistent.")
            return original_diagnosis, original_diagnosis, []


if __name__ == "__main__":
    # Example Usage
    patient_data = PatientData(
        symptoms=["fever", "cough", "body aches"],
        medical_history=["no chronic conditions"],
        test_results={
            "flu_test": "negative",
            "covid_test": "negative"
            }
    )

    verifier = MDVSVerifier(llm_model_name="gpt-4")
    original_diag, final_diag, inconsistencies = verifier.verify_diagnosis(patient_data)

    print("\n--- Verification Summary --- ")
    print(f"Original Diagnosis: {original_diag.diagnosis}")
    print(f"Final Diagnosis: {final_diag.diagnosis}")
    if inconsistencies:
        print(f"Inconsistencies rectified: {len(inconsistencies)}")
    else:
        print("No inconsistencies to rectify.")