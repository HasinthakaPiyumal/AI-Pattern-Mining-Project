
class MockLLM:
    """A mock Large Language Model to simulate explanation and Q&A."""
    def generate_explanation(self, patient_data, recommendation):
        patient_name = patient_data.get("name", "patient")
        symptoms = ", ".join(patient_data.get("symptoms", []))
        history = ", ".join(patient_data.get("medical_history", []))

        explanation_template = (
            f"Based on {patient_name}'s medical profile, including reported symptoms "
            f"of {symptoms} and a history of {history}, we recommend {recommendation}. "
            f"This treatment is suggested because it directly addresses the identified issues, "
            f"aiming to alleviate symptoms and improve overall health. "
            f"We prioritize treatments with a strong evidence base for similar conditions."
        )
        return explanation_template

    def answer_question(self, patient_data, recommendation, question):
        question = question.lower()
        if "benefits" in question and recommendation:
            return f"The primary benefits of {recommendation} include faster recovery, symptom relief, and prevention of complications. Specific benefits can vary based on individual response."
        elif "risks" in question and recommendation:
            return f"Potential risks of {recommendation} are generally low, but can include mild side effects like fatigue or nausea. Serious risks are rare. Your doctor will discuss specific concerns."
        elif "alternatives" in question and recommendation:
            return f"Alternative treatments to {recommendation} might include [mention a generic alternative, e.g., 'physical therapy' or 'different medication class']. We chose {recommendation} due to its tailored fit for your profile."
        elif "why" in question and recommendation:
            return self.generate_explanation(patient_data, recommendation).replace("We recommend", "The reason we recommend") # Re-use explanation logic
        elif "clarify" in question or "meaning" in question:
            return "Could you please specify which term or concept you'd like me to clarify? I'm here to help you understand all medical jargon."
        else:
            return "I understand your question, but I need a bit more context or specific details to provide a precise answer. Can you rephrase or elaborate?"

def recommend_treatment(patient_data):
    """A simplified rule-based treatment recommendation engine."""
    symptoms = patient_data.get("symptoms", [])
    medical_history = patient_data.get("medical_history", [])

    if "fever" in symptoms and "cough" in symptoms:
        return "Rest, Fluids, and Over-the-Counter Symptom Relief"
    elif "chest pain" in symptoms and "shortness of breath" in symptoms:
        return "Emergency Medical Consultation for Cardiac Evaluation"
    elif "diabetes" in medical_history:
        return "Insulin Management and Dietary Modifications"
    elif "headache" in symptoms and "nausea" in symptoms:
        return "Pain Relief Medication and Rest"
    else:
        return "General Wellness Check and Symptomatic Care"

def main():
    print("Welcome to the Healthcare Treatment Navigator!")
    print("Please provide some patient information for a personalized recommendation.")

    patient_name = input("Patient's Name: ")
    symptoms_input = input("Enter symptoms (comma-separated, e.g., fever, cough, headache): ")
    medical_history_input = input("Enter relevant medical history (comma-separated, e.g., diabetes, asthma): ")

    patient_data = {
        "name": patient_name,
        "symptoms": [s.strip().lower() for s in symptoms_input.split(',') if s.strip()],
        "medical_history": [h.strip().lower() for h in medical_history_input.split(',') if h.strip()]
    }

    llm = MockLLM()

    # 1. Get Recommendation
    recommended_plan = recommend_treatment(patient_data)
    print(f"\n--- Recommended Treatment Plan ---")
    print(f"For {patient_name}, the recommended plan is: {recommended_plan}")

    # 2. Generate Explanation using LLM
    explanation = llm.generate_explanation(patient_data, recommended_plan)
    print(f"\n--- Explanation ---")
    print(explanation)

    # 3. Interactive Q&A with LLM
    print("\n--- Interactive Q&A ---")
    print("You can ask questions about the recommendation (e.g., 'What are the benefits?', 'Any risks?', 'Are there alternatives?'). Type 'exit' to quit.")

    while True:
        user_question = input("Your question: ")
        if user_question.lower() == 'exit':
            break
        
        answer = llm.answer_question(patient_data, recommended_plan, user_question)
        print(f"Navigator: {answer}")

    print("Thank you for using the Healthcare Treatment Navigator!")

if __name__ == "__main__":
    main()
