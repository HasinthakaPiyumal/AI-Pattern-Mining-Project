import json
import re

class MedicalKnowledgeGraph:
    def __init__(self):
        self.knowledge_base = {
            "fever": [
                "Fever is often a symptom of infection.",
                "Common causes include viral infections (e.g., flu, common cold) and bacterial infections (e.g., strep throat, pneumonia).",
                "High fever can lead to dehydration."
            ],
            "cough": [
                "Cough can be dry or productive.",
                "Causes include respiratory infections, allergies, and asthma.",
                "Persistent cough may indicate bronchitis or pneumonia."
            ],
            "fatigue": [
                "Fatigue is a common symptom in many conditions.",
                "Can be associated with viral infections, anemia, chronic fatigue syndrome, or thyroid issues."
            ],
            "sore throat": [
                "Sore throat is often caused by viral infections.",
                "Streptococcal pharyngitis (strep throat) is a bacterial cause requiring antibiotics."
            ],
            "diabetes": [
                "Diabetes is a metabolic disease causing high blood sugar.",
                "Type 1 diabetes is an autoimmune disease, Type 2 is often lifestyle-related.",
                "Symptoms include frequent urination, increased thirst, and unexplained weight loss."
            ],
            "headache": [
                "Headaches can be tension headaches, migraines, or cluster headaches.",
                "Severe headaches with fever and stiff neck can indicate meningitis."
            ],
            "chest pain": [
                "Chest pain can be musculoskeletal, cardiac, or respiratory in origin.",
                "Acute, crushing chest pain radiating to the arm may indicate a heart attack."
            ]
        }

    def retrieve_knowledge(self, query: str) -> list[str]:
        query = query.lower()
        relevant_facts = []
        for key, facts in self.knowledge_base.items():
            if query in key or any(query in fact.lower() for fact in facts):
                relevant_facts.extend(facts)
        return list(set(relevant_facts)) # Remove duplicates

class LLMClient:
    def __init__(self, model_name="SimulatedLLM"):
        self.model_name = model_name

    def generate_text(self, prompt: str) -> str:
        # This is a simulated LLM response.
        # In a real application, this would involve an API call to a service like OpenAI, Gemini, or a local model.
        print(f"\n--- Simulated LLM Prompt ---\n{prompt}\n----------------------------")

        if "fever and cough" in prompt.lower() and "recent travel" in prompt.lower():
            return json.dumps({
                "diagnosis": "Viral Respiratory Infection (e.g., Influenza or common cold)",
                "justification": "Patient presents with fever, cough, and fatigue, consistent with a viral respiratory infection. The recent travel history also increases the likelihood of exposure. The knowledge base indicates that fever and cough are common symptoms of viral infections. No specific indicators for bacterial infection or more severe conditions were present in the retrieved knowledge or patient history.",
                "suggested_actions": "Rest, hydration, symptomatic treatment (e.g., antipyretics for fever, cough suppressants if needed). If symptoms worsen or persist beyond 7-10 days, consider further evaluation (e.g., flu test, chest X-ray)."
            })
        elif "severe headache" in prompt.lower() and "stiff neck" in prompt.lower() and "fever" in prompt.lower():
            return json.dumps({
                "diagnosis": "Possible Meningitis",
                "justification": "The combination of severe headache, stiff neck, and fever is a classic triad for meningitis, as indicated by the retrieved medical knowledge. This is a medical emergency requiring immediate attention.",
                "suggested_actions": "Immediate emergency medical evaluation, lumbar puncture for CSF analysis, empiric antibiotics/antivirals based on clinical suspicion."
            })
        elif "fatigue" in prompt.lower() and "frequent urination" in prompt.lower() and "increased thirst" in prompt.lower():
            return json.dumps({
                "diagnosis": "Possible Type 2 Diabetes Mellitus",
                "justification": "The patient's symptoms of fatigue, frequent urination, and increased thirst are highly suggestive of diabetes, specifically Type 2 given the typical presentation pattern. The knowledge graph confirms these as key symptoms for diabetes.",
                "suggested_actions": "Blood glucose tests (fasting glucose, HbA1c), lifestyle modification counseling, referral to endocrinologist."
            })
        else:
            return json.dumps({
                "diagnosis": "Uncertain / Requires further investigation",
                "justification": "Based on the provided information and general medical knowledge, the symptoms are non-specific or insufficient to pinpoint a definitive diagnosis. Further information or diagnostic tests are needed for a more accurate assessment.",
                "suggested_actions": "Consider additional diagnostic tests, consult with a specialist, monitor symptom progression."
            })

class KDCoTMedicalAssistant:
    def __init__(self, llm_client: LLMClient, medical_kg: MedicalKnowledgeGraph):
        self.llm_client = llm_client
        self.medical_kg = medical_kg

    def _extract_entities(self, text: str) -> list[str]:
        # A very basic entity extraction for demonstration.
        # In a real system, this would use NLP libraries (e.g., spaCy, NLTK) or a dedicated entity recognizer.
        common_medical_terms = ["fever", "cough", "fatigue", "sore throat", "diabetes", "headache", "chest pain", "stiff neck", "frequent urination", "increased thirst", "recent travel"]
        extracted = [term for term in common_medical_terms if re.search(r'\b' + re.escape(term) + r'\b', text, re.IGNORECASE)]
        return list(set(extracted)) # Remove duplicates

    def diagnose_patient(self, symptoms: str, medical_history: str, test_results: str) -> dict:
        all_patient_info = f"{symptoms} {medical_history} {test_results}"
        extracted_entities = self._extract_entities(all_patient_info)

        retrieved_knowledge = []
        for entity in extracted_entities:
            retrieved_knowledge.extend(self.medical_kg.retrieve_knowledge(entity))
        
        # Deduplicate retrieved knowledge
        retrieved_knowledge = list(set(retrieved_knowledge))

        prompt = f"""
        Patient Symptoms: {symptoms}
        Patient Medical History: {medical_history}
        Patient Test Results: {test_results}

        Relevant Medical Knowledge:
        {'\n'.join([f'- {fact}' for fact in retrieved_knowledge]) if retrieved_knowledge else 'No specific relevant knowledge retrieved.'}

        Based on the provided patient information and the relevant medical knowledge, perform a Chain-of-Thought reasoning process to arrive at a differential diagnosis. Ground your reasoning steps in the provided medical knowledge. Then, formulate the most likely diagnosis, a detailed justification citing the knowledge, and suggested further actions. 
        
        Your output should be a JSON object with the following keys: "diagnosis", "justification", "suggested_actions".
        """

        llm_raw_response = self.llm_client.generate_text(prompt)
        
        try:
            llm_output = json.loads(llm_raw_response)
            return {
                "diagnosis": llm_output.get("diagnosis", "N/A"),
                "justification": llm_output.get("justification", "N/A"),
                "suggested_actions": llm_output.get("suggested_actions", "N/A")
            }
        except json.JSONDecodeError:
            return {
                "diagnosis": "Error parsing LLM response",
                "justification": f"LLM returned invalid JSON: {llm_raw_response}",
                "suggested_actions": "N/A"
            }

if __name__ == "__main__":
    # Initialize components
    medical_kg = MedicalKnowledgeGraph()
    llm_client = LLMClient()
    assistant = KDCoTMedicalAssistant(llm_client, medical_kg)

    # --- Test Case 1: Viral Infection --- #
    print("\n--- Running Test Case 1: Viral Infection ---")
    symptoms1 = "Patient has a fever of 101.5°F, persistent cough, and general fatigue."
    history1 = "No significant medical history. Recently returned from an international trip."
    results1 = "White blood cell count slightly elevated, otherwise normal."
    diagnosis1 = assistant.diagnose_patient(symptoms1, history1, results1)
    print("\n--- Diagnosis for Test Case 1 ---")
    print(json.dumps(diagnosis1, indent=2))

    # --- Test Case 2: Possible Meningitis --- #
    print("\n--- Running Test Case 2: Possible Meningitis ---")
    symptoms2 = "Severe headache, stiff neck, and a fever of 103°F. Patient reports photophobia."
    history2 = "No relevant medical history."
    results2 = "Lumbar puncture pending."
    diagnosis2 = assistant.diagnose_patient(symptoms2, history2, results2)
    print("\n--- Diagnosis for Test Case 2 ---")
    print(json.dumps(diagnosis2, indent=2))

    # --- Test Case 3: Possible Diabetes --- #
    print("\n--- Running Test Case 3: Possible Diabetes ---")
    symptoms3 = "Chronic fatigue, frequent urination throughout the day and night, and increased thirst."
    history3 = "Family history of Type 2 Diabetes. Overweight."
    results3 = "Fasting blood glucose: 135 mg/dL."
    diagnosis3 = assistant.diagnose_patient(symptoms3, history3, results3)
    print("\n--- Diagnosis for Test Case 3 ---")
    print(json.dumps(diagnosis3, indent=2))

    # --- Test Case 4: Non-specific Symptoms --- #
    print("\n--- Running Test Case 4: Non-specific Symptoms ---")
    symptoms4 = "Mild headache and occasional fatigue."
    history4 = "No significant history."
    results4 = "All tests normal."
    diagnosis4 = assistant.diagnose_patient(symptoms4, history4, results4)
    print("\n--- Diagnosis for Test Case 4 ---")
    print(json.dumps(diagnosis4, indent=2))