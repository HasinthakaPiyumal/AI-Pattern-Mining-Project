
class LanguageModel:
    def __init__(self):
        self.responses = {
            "start_interview": "Hello! I'm here to help with your patient intake. Can we start?",
            "generic_greeting": "Welcome to our clinic.",
            "default_response": "I see. Can you provide more details?",
            "What are your symptoms?": "I need to know your symptoms.",
            "How long have you had these symptoms?": "Please tell me the duration.",
            "Have you tried any treatments?": "Any past treatments?"
        }

    def predict(self, prompt):
        # In a real scenario, this would involve complex LLM inference.
        # Here, it's a simple lookup or generic response.
        return self.responses.get(prompt, self.responses["default_response"])

    def fine_tune(self, demonstrations):
        # Simulate fine-tuning by updating the response dictionary
        # In a real LLM, this would be supervised learning on Q&A pairs.
        print("\n--- Simulating Language Model Fine-Tuning --- ")
        for question, expected_answer in demonstrations:
            # The LLM learns to generate the *expected question* given a context
            # For this simplified demo, we're mapping expected answers back to questions
            # to show the model 'learning' to guide the conversation based on expected inputs.
            # A more accurate simulation would be (previous_context, expected_next_question)
            self.responses[expected_answer] = question # This is a simplified learning rule for demo
            self.responses[question] = question # Ensure questions are direct
            print(f"Learned: If patient says '{expected_answer}', respond with '{question}' (or similar context-driven question).")
        print("--- Fine-Tuning Complete ---\n")


class ClinicSystem:
    def __init__(self):
        self.expected_questions = [
            "What are your symptoms?",
            "How long have you had these symptoms?",
            "Have you tried any treatments?",
            "Which areas of your body are affected?",
            "Is there anything else you'd like to mention?"
        ]
        self.valid_answer_patterns = {
            "What are your symptoms?": ["rash", "itch", "pain", "redness", "sore"],
            "How long have you had these symptoms?": ["days", "weeks", "months", "years"],
            "Have you tried any treatments?": ["cream", "ointment", "medication", "nothing"],
            "Which areas of your body are affected?": ["face", "arms", "legs", "torso", "back"],
            "Is there anything else you'd like to mention?": ["yes", "no", "concern"]
        }

    def get_next_question(self, current_step):
        if current_step < len(self.expected_questions):
            return self.expected_questions[current_step]
        return None

    def validate_answer(self, question, answer):
        patterns = self.valid_answer_patterns.get(question, [])
        if not patterns:
            return True  # No specific validation required
        return any(pattern in answer.lower() for pattern in patterns)


class BehaviorCloner:
    def __init__(self, language_model, clinic_system):
        self.language_model = language_model
        self.clinic_system = clinic_system
        self.demonstrations = []

    def generate_demonstrations(self):
        # These are simulated human demonstrations of ideal Q&A flow
        # In a real scenario, this data would be collected from actual interactions.
        self.demonstrations = [
            ("What are your symptoms?", "I have a rash and it itches a lot."),
            ("How long have you had these symptoms?", "About 3 weeks now."),
            ("Have you tried any treatments?", "I've used an over-the-counter cream."),
            ("Which areas of your body are affected?", "Mainly my arms and legs."),
            ("Is there anything else you'd like to mention?", "No, I think that covers it.")
        ]
        print("Generated simulated human demonstrations.")

    def apply_cloning(self):
        print("Applying behavior cloning to the language model...")
        self.language_model.fine_tune(self.demonstrations)
        print("Behavior cloning applied.")


class MedicalChatbot:
    def __init__(self, language_model, clinic_system):
        self.language_model = language_model
        self.clinic_system = clinic_system

    def start_intake(self):
        print("\n--- Starting Patient Intake with Chatbot ---")
        current_step = 0
        patient_data = {}

        initial_greeting = self.language_model.predict("start_interview")
        print(f"Chatbot: {initial_greeting}")
        # Simulate initial patient response
        _ = input("Patient (simulated): ")

        while True:
            question = self.clinic_system.get_next_question(current_step)
            if not question:
                break

            chatbot_question = self.language_model.predict(question)
            print(f"Chatbot: {chatbot_question}")

            patient_response = input("Patient: ")
            patient_data[question] = patient_response

            if not self.clinic_system.validate_answer(question, patient_response):
                print("Chatbot: I'm sorry, I didn't quite understand that. Could you please rephrase or be more specific?")
                # For simplicity, we'll allow an invalid answer to proceed, but in a real system, it would re-prompt.
            else:
                print("Chatbot: Thank you for that information.")

            current_step += 1

        print("--- Patient Intake Complete ---\n")
        print("Collected Patient Data:")
        for q, a in patient_data.items():
            print(f"- {q}: {a}")


if __name__ == "__main__":
    # 1. Initialize ClinicSystem with specific dermatology intake rules.
    clinic_system = ClinicSystem()

    # 2. Initialize LanguageModel with generic conversational capabilities.
    generic_llm = LanguageModel()
    print("\n--- Initial Generic LLM Responses (before cloning) ---")
    print(f"LLM for 'What are your symptoms?': {generic_llm.predict('What are your symptoms?')}")
    print(f"LLM for 'nonexistent_prompt': {generic_llm.predict('nonexistent_prompt')}")

    # 3. Initialize BehaviorCloner with the generic LanguageModel and ClinicSystem.
    cloner = BehaviorCloner(generic_llm, clinic_system)

    # 4. BehaviorCloner generates/loads human demonstrations.
    cloner.generate_demonstrations()

    # 5. BehaviorCloner calls language_model.fine_tune() to apply behavior cloning.
    cloner.apply_cloning()

    # 6. Initialize MedicalChatbot with the now fine-tuned LanguageModel and ClinicSystem.
    #    Note: We're passing the *same* generic_llm instance, which has now been 'fine-tuned'.
    fine_tuned_chatbot = MedicalChatbot(generic_llm, clinic_system)

    print("\n--- Fine-Tuned LLM Responses (after cloning) ---")
    # Demonstrate that the LLM's predict method now reflects the 'learned' behavior.
    # In this simplified model, 'predict' directly outputs the learned question when given an expected patient answer as a prompt.
    # This showcases the *effect* of cloning, where the LLM becomes more aligned with the clinic's flow.
    # For a real LLM, 'predict' would generate the next *appropriate question* given the conversation history.
    print(f"LLM for 'I have a rash and it itches a lot.': {generic_llm.predict('I have a rash and it itches a lot.')}")
    print(f"LLM for 'About 3 weeks now.': {generic_llm.predict('About 3 weeks now.')}")

    # 7. MedicalChatbot.start_intake() is called to simulate a patient interaction.
    fine_tuned_chatbot.start_intake()
