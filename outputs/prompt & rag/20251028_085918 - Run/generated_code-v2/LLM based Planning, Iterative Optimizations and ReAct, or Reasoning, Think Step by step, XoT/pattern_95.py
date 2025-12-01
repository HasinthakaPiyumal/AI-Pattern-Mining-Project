"""Simulates the generation of Chain-of-Thought finetuning data for medical diagnosis."""

def generate_cot_finetuning_data(num_samples: int = 5) -> list[dict]:
    """
    Generates synthetic Chain-of-Thought (CoT) finetuning data for medical diagnosis.
    Each data point includes symptoms, test results, a diagnosis, and a detailed
    CoT explanation referencing medical knowledge.

    Args:
        num_samples: The number of synthetic data samples to generate.

    Returns:
        A list of dictionaries, where each dictionary represents a finetuning data point.
    """
    data = []
    medical_knowledge_base = {
        "pneumonia": {
            "symptoms": ["cough", "fever", "shortness of breath", "chest pain"],
            "tests": {"chest X-ray": "infiltrates", "blood test": "elevated white blood cell count"},
            "treatment": "antibiotics"
        },
        "diabetes_type2": {
            "symptoms": ["frequent urination", "increased thirst", "fatigue", "blurred vision"],
            "tests": {"fasting blood sugar": ">126 mg/dL", "HbA1c": ">6.5%"},
            "treatment": "diet and exercise, medication"
        },
        "common_cold": {
            "symptoms": ["runny nose", "sore throat", "sneezing", "mild cough"],
            "tests": {}, # No specific tests usually needed
            "treatment": "rest, fluids, over-the-counter medication"
        }
    }

    for i in range(num_samples):
        if i % 3 == 0:
            diagnosis = "pneumonia"
            symptoms = ["cough", "fever", "shortness of breath"]
            test_results = {"chest X-ray": "possible infiltrates", "blood test": "WBC 15.0 x10^9/L"}
            cot_explanation = (
                f"Patient presents with symptoms of {", ".join(symptoms)}. "
                f"The chest X-ray result '{test_results['chest X-ray']}' suggests lung inflammation. "
                f"Additionally, the blood test shows '{test_results['blood test']}', indicating an infection. "
                f"Based on these combined findings and consistent with medical guidelines (e.g., CDC guidelines on community-acquired pneumonia), "
                f"a diagnosis of {diagnosis} is probable. Treatment typically involves antibiotics."
            )
        elif i % 3 == 1:
            diagnosis = "diabetes_type2"
            symptoms = ["increased thirst", "frequent urination", "fatigue"]
            test_results = {"fasting blood sugar": "145 mg/dL", "HbA1c": "7.1%"}
            cot_explanation = (
                f"Patient reports {", ".join(symptoms)}. "
                f"Laboratory results show a fasting blood sugar of '{test_results['fasting blood sugar']}' and an HbA1c of '{test_results['HbA1c']}'. "
                f"According to WHO diagnostic criteria for diabetes, a fasting blood sugar above 126 mg/dL and HbA1c above 6.5% "
                f"are indicative of diabetes. Therefore, a diagnosis of {diagnosis} is highly likely. "
                f"Management typically includes lifestyle modifications and oral hypoglycemic agents."
            )
        else:
            diagnosis = "common_cold"
            symptoms = ["runny nose", "sore throat", "sneezing"]
            test_results = {}
            cot_explanation = (
                f"Patient exhibits typical symptoms of {", ".join(symptoms)}. "
                f"These symptoms are generally self-limiting and do not warrant extensive diagnostic tests. "
                f"Consistent with standard primary care practices for viral respiratory infections, "
                f"a diagnosis of {diagnosis} is made. Symptomatic relief with rest and fluids is recommended."
            )

        data.append({
            "patient_symptoms": symptoms,
            "test_results": test_results,
            "diagnosis": diagnosis,
            "chain_of_thought": cot_explanation
        })
    return data

if __name__ == "__main__":
    print("Generating synthetic CoT finetuning data...")
    synthetic_data = generate_cot_finetuning_data(num_samples=3)
    for i, sample in enumerate(synthetic_data):
        print(f"\n--- Sample {i+1} ---")
        print(f"Symptoms: {", ".join(sample['patient_symptoms'])}")
        print(f"Test Results: {sample['test_results']}")
        print(f"Diagnosis: {sample['diagnosis']}")
        print(f"Chain-of-Thought: {sample['chain_of_thought']}")
