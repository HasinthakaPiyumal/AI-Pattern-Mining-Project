import random

class SimulatedLLMService:
    def evaluate_sufficiency(self, gathered_info: list, original_query: str) -> bool:
        if not gathered_info:
            return False
        
        # Simple heuristic: If there are at least 3 pieces of information and
        # one of them mentions a 'diagnosis' or 'treatment', consider it sufficient.
        # This simulates a basic LLM evaluation.
        if len(gathered_info) >= 3:
            for info in gathered_info:
                if "diagnosis" in info.lower() or "treatment" in info.lower():
                    return True
            # If enough info but no clear diagnostic terms, still might need more
            return random.choice([True, False]) # Simulate uncertainty
        return False

    def generate_diagnosis(self, gathered_info: list, original_query: str) -> str:
        if not gathered_info:
            return "Insufficient information to provide a diagnosis."

        # Simple synthesis of diagnosis based on gathered info
        diagnosis_parts = []
        for i, info in enumerate(gathered_info):
            diagnosis_parts.append(f"Path {i+1}: {info}")
        
        return f"Based on the query '{original_query}' and the following gathered information:\n" \
               f"- {';\n- '.join(diagnosis_parts)}\n" \
               f"A potential diagnostic insight is that the patient might be experiencing a condition related to the common cold, but further tests are recommended. The provided drug information suggests ibuprofen for symptomatic relief. The patient history indicates a previous respiratory infection, which could be relevant."

class SimulatedMedicalKnowledgeBase:
    def __init__(self):
        self.knowledge = {
            "fever": [
                "Fever is a temporary increase in your body temperature, often due to an illness.",
                "Common causes of fever include infections like flu or common cold.",
                "Treatments for fever often include rest and hydration, and sometimes medication like ibuprofen."
            ],
            "cough": [
                "A cough is a reflex that helps clear your airways of irritants and mucus.",
                "Coughs can be caused by viral infections (like colds or flu), allergies, or asthma.",
                "Persistent cough might indicate bronchitis or pneumonia."
            ],
            "headache": [
                "Headaches are a common pain in your head or face.",
                "Tension headaches are the most common type, often stress-related.",
                "Migraines are severe headaches often accompanied by nausea and light sensitivity."
            ],
            "fatigue": [
                "Fatigue is extreme tiredness resulting from mental or physical illness or exertion.",
                "It can be a symptom of many conditions, from lack of sleep to chronic diseases."
            ],
            "common cold": [
                "The common cold is a viral infection of your nose and throat (upper respiratory tract).",
                "Symptoms typically include a runny nose, sore throat, cough, congestion, slight body aches, and sometimes a low-grade fever.",
                "There is no cure for a common cold, but symptoms can be managed with rest and over-the-counter medications."
            ],
            "ibuprofen": [
                "Ibuprofen is a nonsteroidal anti-inflammatory drug (NSAID) used to relieve pain, fever, and inflammation.",
                "Common side effects include stomach upset, mild heartburn, and nausea."
            ],
            "patient history respiratory infection": [
                "Patient previously had a respiratory infection 3 months ago, treated with antibiotics."
            ],
            "diagnosis": [
                "Differential diagnosis involves distinguishing a particular disease or condition from others that present with similar symptoms.",
                "A definitive diagnosis requires a combination of clinical assessment, patient history, and sometimes laboratory tests."
            ],
            "treatment": [
                "Treatment plans are tailored to the specific diagnosis and patient's overall health.",
                "Common treatments include medication, lifestyle changes, and therapies."
            ]
        }

    def retrieve_information(self, query: str, current_context: list) -> list:
        found_info = []
        # Simple keyword matching for retrieval
        for keyword, info_list in self.knowledge.items():
            if keyword.lower() in query.lower() or any(k.lower() in info.lower() for info in current_context for k in keyword.split()):
                # Retrieve a random piece of information related to the keyword
                if info_list:
                    selected_info = random.choice(info_list)
                    if selected_info not in found_info and selected_info not in current_context:
                        found_info.append(selected_info)
        return found_info

class MedicalDiagnosticAssistant:
    def __init__(self, llm_service: SimulatedLLMService, kb_service: SimulatedMedicalKnowledgeBase):
        self.llm = llm_service
        self.kb = kb_service

    def diagnose_patient_case(self, patient_query: str, max_exploration_depth: int = 5):
        print(f"\n--- Starting Diagnosis for: '{patient_query}' ---")
        gathered_information = []
        current_iteration = 0

        while current_iteration < max_exploration_depth:
            current_iteration += 1
            print(f"\nIteration {current_iteration}:")

            # Step 1: Formulate next retrieval query (simplified for this example)
            # In a real system, an LLM would generate intelligent follow-up questions.
            retrieval_query = patient_query
            if gathered_information:
                # Simple strategy: use the last piece of info as part of the next query context
                retrieval_query += f" {gathered_information[-1]}"

            # Step 2: Retrieve information
            new_info = self.kb.retrieve_information(retrieval_query, gathered_information)
            if new_info:
                gathered_information.extend(new_info)
                print(f"  Retrieved new information: {new_info}")
            else:
                print("  No new information found for the current query. \n  Will attempt evaluation with existing info.")

            # Step 3: LLM Self-Evaluation
            is_sufficient = self.llm.evaluate_sufficiency(gathered_information, patient_query)
            print(f"  LLM Evaluation: Sufficient information? {is_sufficient}")

            if is_sufficient:
                print("  LLM deems information sufficient. Terminating exploration.")
                break
            elif current_iteration == max_exploration_depth:
                print(f"  Maximum exploration depth ({max_exploration_depth}) reached. Terminating exploration.")

        # Step 4: Generate Final Diagnosis
        final_diagnosis = self.llm.generate_diagnosis(gathered_information, patient_query)
        print(f"\n--- Final Diagnostic Insight ---")
        print(final_diagnosis)
        print(f"--- End Diagnosis for: '{patient_query}' ---")


if __name__ == "__main__":
    llm_service = SimulatedLLMService()
    kb_service = SimulatedMedicalKnowledgeBase()
    assistant = MedicalDiagnosticAssistant(llm_service, kb_service)

    # Example Patient Cases
    assistant.diagnose_patient_case("Patient presents with fever, cough, and headache.", max_exploration_depth=4)
    assistant.diagnose_patient_case("Chronic fatigue and occasional headaches, no fever.", max_exploration_depth=3)
    assistant.diagnose_patient_case("Sudden onset severe headache with nausea.", max_exploration_depth=5)
