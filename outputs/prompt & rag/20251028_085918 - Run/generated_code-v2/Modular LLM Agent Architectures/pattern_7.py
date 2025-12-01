class RetrievalModule:
    """Simulates fetching clinical guidelines and patient history from databases."""

    def __init__(self):
        # Mock database for clinical guidelines
        self._mock_clinical_guidelines = {
            "diabetes": "\n--- Clinical Guideline for Diabetes Management ---\n1. Monitor blood glucose levels regularly.\n2. Maintain a balanced diet and regular exercise.\n3. Consider Metformin as a first-line therapy for Type 2 diabetes.\n4. Educate patients on self-management and complications.\n",
            "hypertension": "\n--- Clinical Guideline for Hypertension ---\n1. Lifestyle modifications: reduced sodium intake, increased physical activity.\n2. First-line pharmacotherapy often includes ACE inhibitors, ARBs, CCBs, or thiazide diuretics.\n3. Regular blood pressure monitoring.\n4. Target blood pressure <130/80 mmHg for most adults.\n",
            "asthma": "\n--- Clinical Guideline for Asthma Management ---\n1. Use inhaled corticosteroids for long-term control.\n2. Short-acting beta-agonists for quick relief.\n3. Develop an asthma action plan.\n4. Avoid triggers and consider immunotherapy for allergic asthma.\n"
        }

        # Mock database for patient history
        self._mock_patient_history_db = {
            "patient_001": {
                "name": "Alice Smith",
                "age": 55,
                "conditions": ["Type 2 Diabetes", "Hypertension"],
                "medications": ["Metformin 500mg BID", "Lisinopril 10mg QD"],
                "allergies": ["Penicillin"],
                "last_visit": "2023-10-26"
            },
            "patient_002": {
                "name": "Bob Johnson",
                "age": 30,
                "conditions": ["Asthma"],
                "medications": ["Fluticasone/Salmeterol BID", "Albuterol PRN"],
                "allergies": [],
                "last_visit": "2023-11-15"
            }
        }

    def fetch_clinical_guidelines(self, query: str) -> str:
        """Simulates fetching relevant clinical guidelines based on the query."""
        relevant_guidelines = []
        query_lower = query.lower()
        for keyword, guideline in self._mock_clinical_guidelines.items():
            if keyword in query_lower:
                relevant_guidelines.append(guideline)
        
        if not relevant_guidelines:
            return "No specific clinical guidelines found for the query keywords."
        return "\n".join(relevant_guidelines)

    def fetch_patient_history(self, patient_id: str) -> str:
        """Simulates fetching a patient's medical history."""
        history = self._mock_patient_history_db.get(patient_id)
        if history:
            return (f"\n--- Patient History for {history['name']} (ID: {patient_id}) ---\n"\
                    f"Age: {history['age']}\n"\
                    f"Conditions: {', '.join(history['conditions'])}\n"\
                    f"Medications: {', '.join(history['medications'])}\n"\
                    f"Allergies: {', '.join(history['allergies']) if history['allergies'] else 'None'}\n"\
                    f"Last Visit: {history['last_visit']}\n")
        return f"Patient with ID {patient_id} not found in the database."

class LLMSimulator:
    """Simulates the behavior of a Large Language Model (LLM) for medical queries.
    In a real application, this would be an actual LLM API call.
    """

    def simulate_response(self, user_query: str, context: str) -> str:
        """Generates a simulated LLM response based on the user query and provided context."""
        if not context or "No specific clinical guidelines found" in context:
            context_info = " (without specific external context)"
        else:
            context_info = " (augmented with clinical guidelines and patient history)"

        response_prefix = f"As a Smart Clinical Assistant{context_info}, considering your query about '{user_query}', and based on the provided information:\n\n"
        
        # Simple logic to make the response more dynamic based on content
        if "diagnosis" in user_query.lower() or "condition" in user_query.lower():
            return (f"{response_prefix}"\
                    f"The retrieved context suggests careful consideration of the patient's conditions and relevant guidelines. \n"\
                    f"Potential diagnostic pathways or management strategies should be evaluated in light of the patient's specific history and current clinical standards. \n"\
                    f"Always consult with a qualified medical professional for definitive diagnosis and treatment plans.\n\n"\
                    f"--- Context Used ---\n{context}")
        elif "treatment" in user_query.lower() or "medication" in user_query.lower():
            return (f"{response_prefix}"\
                    f"Based on the contextual data, treatment recommendations would typically align with the outlined clinical guidelines, \n"\
                    f"while strictly adhering to the patient's medication history and known allergies. \n"\
                    f"Personalized treatment plans require direct patient evaluation by a physician.\n\n"\
                    f"--- Context Used ---\n{context}")
        else:
            return (f"{response_prefix}"\
                    f"The information provided indicates various factors to consider. \n"\
                    f"For comprehensive medical advice, please provide more details or consult a healthcare provider. \n"\
                    f"The system has processed your request using the following context:\n\n"\
                    f"--- Context Used ---\n{context}")

if __name__ == "__main__":
    retrieval_module = RetrievalModule()
    llm_simulator = LLMSimulator()

    print("\n--- Smart Clinical Assistant Demo ---\n")

    # Scenario 1: Query for diabetes treatment for patient_001
    patient_id_1 = "patient_001"
    user_query_1 = "What are the recommended treatment options for diabetes for patient_001, considering their history?"
    
    print(f"User Query: {user_query_1}\n")

    # Step 1: Retrieve context
    guidelines_1 = retrieval_module.fetch_clinical_guidelines("diabetes")
    patient_history_1 = retrieval_module.fetch_patient_history(patient_id_1)
    combined_context_1 = f"Clinical Guidelines: {guidelines_1}\nPatient History: {patient_history_1}"

    # Step 2: Augment LLM with context and get response
    llm_response_1 = llm_simulator.simulate_response(user_query_1, combined_context_1)
    print(f"LLM Response:\n{llm_response_1}\n")

    print("\n" + "="*80 + "\n")

    # Scenario 2: General query for asthma management without specific patient
    user_query_2 = "Tell me about general asthma management strategies."
    
    print(f"User Query: {user_query_2}\n")

    # Step 1: Retrieve context (only guidelines here)
    guidelines_2 = retrieval_module.fetch_clinical_guidelines("asthma")
    combined_context_2 = f"Clinical Guidelines: {guidelines_2}"

    # Step 2: Augment LLM with context and get response
    llm_response_2 = llm_simulator.simulate_response(user_query_2, combined_context_2)
    print(f"LLM Response:\n{llm_response_2}\n")

    print("\n" + "="*80 + "\n")

    # Scenario 3: Query for an unknown condition
    patient_id_3 = "patient_002"
    user_query_3 = "What is the best approach for managing exotic tropical fever for patient_002?"
    
    print(f"User Query: {user_query_3}\n")

    # Step 1: Retrieve context
    guidelines_3 = retrieval_module.fetch_clinical_guidelines("exotic tropical fever")
    patient_history_3 = retrieval_module.fetch_patient_history(patient_id_3)
    combined_context_3 = f"Clinical Guidelines: {guidelines_3}\nPatient History: {patient_history_3}"

    # Step 2: Augment LLM with context and get response
    llm_response_3 = llm_simulator.simulate_response(user_query_3, combined_context_3)
    print(f"LLM Response:\n{llm_response_3}\n")