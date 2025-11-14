import random

class LLMEmulator:
    """
    Simulates a Large Language Model for generating initial diagnoses and treatment plans.
    In a real application, this would be an actual LLM API call.
    """
    def generate_recommendation(self, patient_data: dict) -> dict:
        symptoms = patient_data.get("symptoms", [])
        age = patient_data.get("age", 30)
        gender = patient_data.get("gender", "unknown").lower()
        ethnicity = patient_data.get("ethnicity", "unknown").lower()

        diagnosis_options = {
            "fever, cough, fatigue": "Viral Infection",
            "chest pain, shortness of breath": "Potential Cardiac Issue",
            "headache, sensitivity to light": "Migraine",
            "joint pain, swelling": "Arthritis",
            "abdominal pain, nausea": "Gastroenteritis",
        }

        treatment_options = {
            "Viral Infection": "Rest, hydration, over-the-counter medication.",
            "Potential Cardiac Issue": "Immediate medical consultation, ECG, blood tests.",
            "Migraine": "Pain relievers, rest in a dark room, lifestyle changes.",
            "Arthritis": "Anti-inflammatory drugs, physical therapy, lifestyle changes.",
            "Gastroenteritis": "Fluid replacement, dietary adjustments, antibiotics (if bacterial).",
        }

        # Simple logic to pick a diagnosis based on symptoms
        initial_diagnosis = "Undetermined Condition"
        for symptom_set, diag in diagnosis_options.items():
            if all(s in symptoms for s in symptom_set.split(", ")):
                initial_diagnosis = diag
                break
        
        initial_treatment = treatment_options.get(initial_diagnosis, "Symptomatic treatment.")

        # Simulate some variations based on demographics for later bias detection
        if initial_diagnosis == "Potential Cardiac Issue" and gender == "female" and age > 50:
            initial_diagnosis = "Atypical Cardiac Presentation (requires further investigation)"
            initial_treatment = "Immediate specialist referral, comprehensive cardiac evaluation. Be aware of potential for misdiagnosis in women."

        return {
            "diagnosis": initial_diagnosis,
            "treatment": initial_treatment,
            "confidence": random.uniform(0.7, 0.95), # Simulated confidence
            "raw_llm_output": f"Based on symptoms: {', '.join(symptoms)}, age: {age}, gender: {gender}, ethnicity: {ethnicity}, the initial assessment points to {initial_diagnosis}. Recommended treatment: {initial_treatment}."
        }

class BiasDetector:
    """
    Detects potential biases in diagnostic and treatment recommendations based on patient demographics.
    This is a simplified simulation of a real bias detection system.
    """
    def detect_bias(self, patient_data: dict, recommendation: dict) -> dict:
        bias_alerts = []
        diagnosis = recommendation.get("diagnosis", "")
        treatment = recommendation.get("treatment", "")
        age = patient_data.get("age", 30)
        gender = patient_data.get("gender", "unknown").lower()
        ethnicity = patient_data.get("ethnicity", "unknown").lower()

        # Rule 1: Gender bias in cardiac issues
        if "cardiac" in diagnosis.lower() and gender == "female" and "atypical" in diagnosis.lower():
            bias_alerts.append({
                "type": "Gender Bias Alert",
                "explanation": "Diagnosis for potential cardiac issues in women can be atypical and is historically prone to misdiagnosis or delayed diagnosis compared to men. Ensure thorough investigation.",
                "severity": "High"
            })
        
        # Rule 2: Age bias in aggressive treatments
        if "aggressive" in treatment.lower() and age > 75:
            bias_alerts.append({
                "type": "Age Bias Alert",
                "explanation": "Consider the patient's overall health and quality of life when recommending aggressive treatments for elderly patients. Discuss prognosis and patient preferences.",
                "severity": "Medium"
            })
        
        # Rule 3: Ethnicity and pain management disparities
        if "pain" in diagnosis.lower() and ethnicity != "white" and "opioid" not in treatment.lower():
            # This is a highly simplified and potentially controversial rule for demonstration.
            # In real systems, this would involve complex fairness metrics.
            bias_alerts.append({
                "type": "Ethnicity Bias Alert (Pain Management)",
                "explanation": "Studies show disparities in pain management across different ethnic groups, with some groups potentially receiving less aggressive pain treatment. Review pain assessment and management plan carefully.",
                "severity": "Medium"
            })

        return {"has_bias": len(bias_alerts) > 0, "alerts": bias_alerts}

class EvidenceAggregator:
    """
    Aggregates medical evidence in a debate-style (pro and con) for a given diagnosis/treatment.
    Simulates searching a medical literature database.
    """
    def __init__(self):
        self.medical_literature = {
            "Viral Infection": {
                "pro": [
                    {"text": "CDC recommends rest and fluids for most viral infections.", "source": "CDC Guidelines, 2023"},
                    {"text": "Antiviral drugs are rarely indicated for common colds.", "source": "Journal of Infectious Diseases, Vol 45, 2022"},
                ],
                "con": [
                    {"text": "Bacterial co-infection should be ruled out if symptoms worsen.", "source": "WHO Clinical Management, 2021"},
                    {"text": "Immunocompromised patients may require more aggressive management.", "source": "New England Journal of Medicine, 2020"},
                ]
            },
            "Potential Cardiac Issue": {
                "pro": [
                    {"text": "Chest pain and shortness of breath are classic symptoms of myocardial infarction.", "source": "American Heart Association, 2023"},
                    {"text": "Early ECG and troponin tests are crucial for timely diagnosis.", "source": "European Society of Cardiology Guidelines, 2022"},
                ],
                "con": [
                    {"text": "Atypical chest pain can be due to musculoskeletal issues or anxiety, especially in younger patients.", "source": "Mayo Clinic Proceedings, 2021"},
                    {"text": "Women often present with non-classic cardiac symptoms like fatigue or nausea.", "source": "Journal of the American College of Cardiology, 2020"},
                ]
            },
            "Migraine": {
                "pro": [
                    {"text": "Triptans are effective acute treatments for moderate to severe migraines.", "source": "Neurology Journal, 22023"},
                    {"text": "Lifestyle modifications (stress management, regular sleep) can reduce migraine frequency.", "source": "Headache: The Journal of Head and Face Pain, 2021"},
                ],
                "con": [
                    {"text": "Overuse of acute medication can lead to medication overuse headache.", "source": "International Headache Society, 2022"},
                    {"text": "Some patients may not respond to standard treatments and require specialized care.", "source": "The Lancet Neurology, 2020"},
                ]
            },
            "Arthritis": {
                "pro": [
                    {"text": "NSAIDs are first-line treatment for symptomatic relief in many forms of arthritis.", "source": "ACR Guidelines, 2023"},
                    {"text": "Physical therapy is essential for maintaining joint function and reducing pain.", "source": "Physical Therapy Journal, 2022"},
                ],
                "con": [
                    {"text": "Long-term NSAID use carries risks of gastrointestinal and cardiovascular side effects.", "source": "British Medical Journal, 2021"},
                    {"text": "Early diagnosis and disease-modifying antirheumatic drugs (DMARDs) are crucial for inflammatory arthritis.", "source": "Annals of the Rheumatic Diseases, 2020"},
                ]
            },
            "Gastroenteritis": {
                "pro": [
                    {"text": "Oral rehydration therapy is the cornerstone of treatment for acute gastroenteritis.", "source": "World Health Organization, 2023"},
                    {"text": "Most viral gastroenteritis resolves spontaneously within a few days.", "source": "UpToDate, 2022"},
                ],
                "con": [
                    {"text": "Persistent or severe symptoms warrant investigation for bacterial, parasitic, or other underlying causes.", "source": "Clinical Infectious Diseases, 2021"},
                    {"text": "Antibiotics are generally not recommended for viral gastroenteritis and can worsen outcomes.", "source": "Cochrane Database of Systematic Reviews, 2020"},
                ]
            }
        }

    def aggregate_evidence(self, diagnosis: str) -> dict:
        evidence = self.medical_literature.get(diagnosis, {"pro": [], "con": []})
        return {
            "pro_evidence": evidence["pro"],
            "con_evidence": evidence["con"],
            "summary": f"Evidence for and against {diagnosis}:"
        }

class HealthcareDiagnosticAssistant:
    """
    The main orchestrator for the AI-powered Healthcare Diagnostic Assistant.
    Integrates LLM emulation, bias detection, and debate-style evidence aggregation.
    """
    def __init__(self):
        self.llm_emulator = LLMEmulator()
        self.bias_detector = BiasDetector()
        self.evidence_aggregator = EvidenceAggregator()

    def get_diagnosis_and_recommendations(self, patient_data: dict) -> dict:
        print("\n--- Step 1: LLM Emulation for Initial Recommendation ---")
        initial_recommendation = self.llm_emulator.generate_recommendation(patient_data)
        print(f"Initial LLM Diagnosis: {initial_recommendation['diagnosis']}")
        print(f"Initial LLM Treatment: {initial_recommendation['treatment']}")
        print(f"LLM Confidence: {initial_recommendation['confidence']:.2f}")

        print("\n--- Step 2: Bias Detection ---")
        bias_check_result = self.bias_detector.detect_bias(patient_data, initial_recommendation)
        if bias_check_result["has_bias"]:
            print("❗ Potential Bias Detected:")
            for alert in bias_check_result["alerts"]:
                print(f"  - Type: {alert['type']} (Severity: {alert['severity']})")
                print(f"    Explanation: {alert['explanation']}")
        else:
            print("✅ No significant biases detected based on current rules.")

        print("\n--- Step 3: Debate-Style Evidence Aggregation ---")
        diagnosis_for_evidence = initial_recommendation["diagnosis"]
        aggregated_evidence = self.evidence_aggregator.aggregate_evidence(diagnosis_for_evidence)
        print(f"Evidence for '{diagnosis_for_evidence}':")
        
        print("\n  --- Pro Evidence ---")
        if aggregated_evidence["pro_evidence"]:
            for i, ev in enumerate(aggregated_evidence["pro_evidence"]):
                print(f"    {i+1}. {ev['text']} (Source: {ev['source']})")
        else:
            print("    No specific 'pro' evidence found in simulated literature for this diagnosis.")

        print("\n  --- Con Evidence ---")
        if aggregated_evidence["con_evidence"]:
            for i, ev in enumerate(aggregated_evidence["con_evidence"]):
                print(f"    {i+1}. {ev['text']} (Source: {ev['source']})")
        else:
            print("    No specific 'con' evidence found in simulated literature for this diagnosis.")

        final_output = {
            "patient_data": patient_data,
            "initial_llm_recommendation": initial_recommendation,
            "bias_detection": bias_check_result,
            "aggregated_evidence": aggregated_evidence
        }
        return final_output

# Main execution block
if __name__ == "__main__":
    assistant = HealthcareDiagnosticAssistant()

    print("--- Scenario 1: Standard Case ---")
    patient_data_1 = {
        "symptoms": ["fever", "cough", "fatigue"],
        "age": 45,
        "gender": "male",
        "ethnicity": "white",
        "medical_history": ["seasonal allergies"]
    }
    result_1 = assistant.get_diagnosis_and_recommendations(patient_data_1)
    # print("\n--- Full Output for Scenario 1 ---")
    # print(result_1)


    print("\n\n--- Scenario 2: Potential Gender Bias Case (Female with Atypical Cardiac Symptoms) ---")
    patient_data_2 = {
        "symptoms": ["chest pain", "shortness of breath", "fatigue", "nausea"], # Adding fatigue and nausea for atypical
        "age": 62,
        "gender": "female",
        "ethnicity": "hispanic",
        "medical_history": ["hypertension", "type 2 diabetes"]
    }
    result_2 = assistant.get_diagnosis_and_recommendations(patient_data_2)
    # print("\n--- Full Output for Scenario 2 ---")
    # print(result_2)

    print("\n\n--- Scenario 3: Potential Ethnicity Bias Case (Pain Management) ---")
    patient_data_3 = {
        "symptoms": ["severe joint pain", "swelling"],
        "age": 55,
        "gender": "female",
        "ethnicity": "black",
        "medical_history": ["osteoarthritis"]
    }
    result_3 = assistant.get_diagnosis_and_recommendations(patient_data_3)
    # print("\n--- Full Output for Scenario 3 ---")
    # print(result_3)