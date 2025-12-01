import collections
import re
import random

class MockLLM:
    def __init__(self, model_name="MockGPT-4"):
        self.model_name = model_name
        self.possible_diagnoses = [
            "Common Cold", "Influenza", "Strep Throat", "Bronchitis",
            "Pneumonia", "Allergies", "Sinusitis", "Migraine", "Tension Headache"
        ]
        self.diagnosis_patterns = {
            "Common Cold": "The patient presents with mild upper respiratory symptoms, suggesting a common cold.",
            "Influenza": "Given the sudden onset of fever, body aches, and fatigue, influenza is highly suspected.",
            "Strep Throat": "Pharyngitis with white patches and swollen lymph nodes strongly indicates strep throat.",
            "Bronchitis": "Persistent cough with mucus production and no signs of pneumonia points to bronchitis.",
            "Pneumonia": "Fever, shortness of breath, and chest X-ray findings are consistent with pneumonia.",
            "Allergies": "Itchy eyes, runny nose, and sneezing, especially with seasonal triggers, are typical of allergies.",
            "Sinusitis": "Facial pain, nasal congestion, and headache, particularly after a cold, suggest sinusitis.",
            "Migraine": "Severe, throbbing headache, often unilateral, with light sensitivity, indicates migraine.",
            "Tension Headache": "Bilateral, dull head pain, often described as a tight band around the head, is a tension headache."
        }

    def generate_response(self, prompt, temperature=0.7):
        # Simulate CoT reasoning and diagnosis
        # In a real scenario, this would be an actual LLM API call
        print(f"[MockLLM] Processing prompt (temp={temperature}): {prompt[:50]}...")

        # Determine a simulated outcome based on the prompt or randomly
        # For simplicity, let's randomly pick a diagnosis with some bias for diverse samples
        chosen_diagnosis = random.choices(
            self.possible_diagnoses,
            weights=[0.15, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05], # Biased weights
            k=1
        )[0]

        cot_reasoning = f"The AI analyzes the patient's symptoms and medical history. \nConsidering factors such as {', '.join(random.sample(['fever', 'cough', 'fatigue', 'sore throat', 'headache'], k=random.randint(2,5)))}. \nBased on these observations, the most likely condition is {chosen_diagnosis}.\nReasoning details: {self.diagnosis_patterns[chosen_diagnosis]}"

        return cot_reasoning

    def generate_direct_diagnosis(self, prompt):
        print(f"[MockLLM] Generating direct diagnosis: {prompt[:50]}...")
        # Simulate a direct, confident diagnosis without extensive CoT
        chosen_diagnosis = random.choice(self.possible_diagnoses)
        return f"Based on the patient's information, the most probable diagnosis is: {chosen_diagnosis}."

class PatientDataInputModule:
    @staticmethod
    def format_patient_data_for_llm(patient_data):
        symptoms = ", ".join(patient_data.get("symptoms", []))
        history = patient_data.get("medical_history", "N/A")
        lab_results = patient_data.get("lab_results", "N/A")

        prompt = (
            f"Analyze the following patient data to provide a detailed Chain-of-Thought reasoning and a final diagnosis.\n"
            f"Symptoms: {symptoms}\n"
            f"Medical History: {history}\n"
            f"Lab Results: {lab_results}\n"
            f"Please provide a step-by-step reasoning process ending with a clear diagnosis."
        )
        return prompt

    @staticmethod
    def format_meta_reasoning_prompt(patient_data, conflicting_diagnoses):
        symptoms = ", ".join(patient_data.get("symptoms", []))
        history = patient_data.get("medical_history", "N/A")
        lab_results = patient_data.get("lab_results", "N/A")
        conflicts_str = ", ".join([f"{d[0]} (count: {d[1]})" for d in conflicting_diagnoses.items()])

        prompt = (
            f"Given the patient data below, multiple possible diagnoses have been suggested, including: {conflicts_str}.\n"
            f"Symptoms: {symptoms}\n"
            f"Medical History: {history}\n"
            f"Lab Results: {lab_results}\n"
            f"Considering these conflicting views and all available information, provide the single most probable diagnosis. Focus on the strongest evidence."
        )
        return prompt


class DiagnosisExtractionModule:
    @staticmethod
    def extract_diagnosis(llm_response):
        # Look for phrases like "the most likely condition is X", "final diagnosis: X", etc.
        patterns = [
            r"the most likely condition is ([A-Za-z0-9\s]+)[\.\n]",
            r"final diagnosis: ([A-Za-z0-9\s]+)[\.\n]",
            r"diagnosis is ([A-Za-z0-9\s]+)[\.\n]"
        ]
        for pattern in patterns:
            match = re.search(pattern, llm_response, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return "Unknown Diagnosis"

class CoTReasoningPathSampler:
    def __init__(self, llm_interaction_layer):
        self.llm = llm_interaction_layer

    def sample_paths(self, patient_prompt, num_samples=5, temperature=0.7):
        reasoning_paths = []
        for _ in range(num_samples):
            response = self.llm.generate_response(patient_prompt, temperature=temperature)
            reasoning_paths.append(response)
        return reasoning_paths

class UncertaintyRoutingMechanism:
    def __init__(self, confidence_threshold=0.75):
        self.confidence_threshold = confidence_threshold

    def route_diagnosis(self, extracted_diagnoses, patient_data, llm_interaction_layer):
        diagnosis_counts = collections.Counter(extracted_diagnoses)
        total_samples = len(extracted_diagnoses)

        print(f"\n[Uncertainty Routing] Sampled Diagnoses: {diagnosis_counts}")

        # Check for majority consensus
        for diagnosis, count in diagnosis_counts.items():
            if count / total_samples >= self.confidence_threshold:
                print(f"[Uncertainty Routing] Majority consensus reached for: {diagnosis} ({count/total_samples:.2f} confidence)")
                return diagnosis, "Majority Consensus"

        # If no majority, fall back to greedy sampling / meta-reasoning
        print("[Uncertainty Routing] No majority consensus. Initiating greedy sampling / meta-reasoning fallback.")
        meta_reasoning_prompt = PatientDataInputModule.format_meta_reasoning_prompt(patient_data, diagnosis_counts)
        greedy_response = llm_interaction_layer.generate_direct_diagnosis(meta_reasoning_prompt)
        final_greedy_diagnosis = DiagnosisExtractionModule.extract_diagnosis(greedy_response)

        print(f"[Uncertainty Routing] Greedy sampled diagnosis: {final_greedy_diagnosis}")
        return final_greedy_diagnosis, "Greedy Sampling Fallback"

class MediMindDiagnosticAssistant:
    def __init__(self, num_cot_samples=5, confidence_threshold=0.75):
        self.llm_interaction_layer = MockLLM()
        self.cot_sampler = CoTReasoningPathSampler(self.llm_interaction_layer)
        self.uncertainty_router = UncertaintyRoutingMechanism(confidence_threshold)
        self.num_cot_samples = num_cot_samples

    def diagnose_patient(self, patient_data):
        print(f"\n--- Diagnosing Patient: {patient_data.get('patient_id', 'N/A')} ---")

        # 1. Patient Data Input
        patient_prompt = PatientDataInputModule.format_patient_data_for_llm(patient_data)

        # 2. CoT Reasoning Path Sampler
        cot_responses = self.cot_sampler.sample_paths(patient_prompt, num_samples=self.num_cot_samples)
        print(f"Generated {len(cot_responses)} CoT reasoning paths.")

        # 3. Diagnosis Extraction
        extracted_diagnoses = [DiagnosisExtractionModule.extract_diagnosis(res) for res in cot_responses]
        print(f"Extracted diagnoses from paths: {extracted_diagnoses}")

        # 4. Uncertainty Routing Mechanism
        final_diagnosis, decision_method = self.uncertainty_router.route_diagnosis(
            extracted_diagnoses, patient_data, self.llm_interaction_layer
        )

        return {"final_diagnosis": final_diagnosis, "decision_method": decision_method, "all_sampled_diagnoses": extracted_diagnoses}


# Example Usage:
if __name__ == "__main__":
    assistant = MediMindDiagnosticAssistant(num_cot_samples=7, confidence_threshold=0.6)

    patient_case_1 = {
        "patient_id": "P001",
        "symptoms": ["fever", "cough", "sore throat", "fatigue"],
        "medical_history": "No significant history.",
        "lab_results": "Flu test negative, Strep test pending."
    }

    patient_case_2 = {
        "patient_id": "P002",
        "symptoms": ["severe headache", "light sensitivity", "nausea"],
        "medical_history": "History of migraines.",
        "lab_results": "Normal."
    }

    patient_case_3 = {
        "patient_id": "P003",
        "symptoms": ["runny nose", "sneezing", "itchy eyes"],
        "medical_history": "Seasonal allergies.",
        "lab_results": "N/A"
    }

    patient_case_4 = {
        "patient_id": "P004",
        "symptoms": ["persistent cough", "chest congestion", "mild fever"],
        "medical_history": "Smoker.",
        "lab_results": "Chest X-ray clear of pneumonia."
    }

    result1 = assistant.diagnose_patient(patient_case_1)
    print(f"Final Diagnosis for P001: {result1['final_diagnosis']} (Method: {result1['decision_method']})\n")

    result2 = assistant.diagnose_patient(patient_case_2)
    print(f"Final Diagnosis for P002: {result2['final_diagnosis']} (Method: {result2['decision_method']})\n")

    result3 = assistant.diagnose_patient(patient_case_3)
    print(f"Final Diagnosis for P003: {result3['final_diagnosis']} (Method: {result3['decision_method']})\n")

    result4 = assistant.diagnose_patient(patient_case_4)
    print(f"Final Diagnosis for P004: {result4['final_diagnosis']} (Method: {result4['decision_method']})\n")