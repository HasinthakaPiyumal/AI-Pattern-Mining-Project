
import collections

class MediBot:
    """
    MediBot: An Enhanced Healthcare AI Assistant demonstrating the Enhanced User Intent Comprehension pattern.
    This class integrates multi-modal input processing with a simplified LLM for intent inference
    and basic personalization.
    """

    def __init__(self):
        self.user_profiles = collections.defaultdict(lambda: {"history": [], "known_conditions": [], "preferences": {}})
        # Mock LLM for intent inference. In a real scenario, this would be a loaded Hugging Face model.
        self.llm = self._initialize_mock_llm()
        print("MediBot initialized with mock components.")

    def _initialize_mock_llm(self):
        """
        Mocks an LLM's capability for intent inference.
        In a real application, this would load a pre-trained and instruction-tuned model
        (e.g., using transformers library).
        """
        def mock_llm_infer(text):
            text_lower = text.lower()
            if "symptom" in text_lower or "feel sick" in text_lower or "pain" in text_lower:
                return {"intent": "symptom_inquiry", "details": text}
            elif "medication" in text_lower or "drug" in text_lower or "prescription" in text_lower:
                return {"intent": "medication_info", "details": text}
            elif "appointment" in text_lower or "doctor" in text_lower or "consult" in text_lower:
                return {"intent": "appointment_booking", "details": text}
            elif "what is" in text_lower or "tell me about" in text_lower:
                return {"intent": "general_medical_info", "details": text}
            else:
                return {"intent": "unclear", "details": text}
        return mock_llm_infer

    def process_speech_input(self, audio_data: bytes) -> str:
        """
        Mocks speech-to-text conversion.
        In a real system, this would use a library like SpeechRecognition.
        """
        print(f"Processing mock speech data of size {len(audio_data)} bytes...")
        # Simulate transcription
        return "User said: I have a persistent cough and fever."

    def process_image_input(self, image_data: bytes) -> str:
        """
        Mocks image analysis to extract relevant information.
        In a real system, this would use libraries like Pillow or models from transformers (e.g., CLIP).
        """
        print(f"Processing mock image data of size {len(image_data)} bytes...")
        # Simulate image content recognition
        if b"skin_rash" in image_data:
            return "Image shows a red skin rash on the arm."
        elif b"medication_bottle" in image_data:
            return "Image shows a medication bottle labeled 'Amoxicillin'."
        return "Image content recognized as: medical related object."

    def translate_text(self, text: str, target_language: str = 'en') -> str:
        """
        Mocks multi-lingual translation.
        In a real system, this would use a translation API or model (e.g., googletrans, transformers).
        """
        if target_language == 'es':
            print(f"Translating '{text}' to Spanish (mock).")
            # Simplified mock translation
            if "cough" in text.lower(): return "El usuario preguntó: tengo tos persistente."
            return "El usuario preguntó: " + text + " (traducido)"
        return text # Return original if target is English or not mocked

    def _get_user_context(self, user_id: str) -> dict:
        """Retrieves personalized context for a user."""
        return self.user_profiles[user_id]

    def _update_user_context(self, user_id: str, new_info: dict):
        """Updates personalized context for a user."""
        self.user_profiles[user_id]['history'].append(new_info)
        if "known_conditions" in new_info:
            self.user_profiles[user_id]["known_conditions"].extend(new_info["known_conditions"])
        print(f"User {user_id} context updated: {self.user_profiles[user_id]}")

    def infer_intent(self, user_id: str, text_input: str) -> dict:
        """
        Infers user intent using the mock LLM, incorporating personalized learning.
        """
        user_context = self._get_user_context(user_id)
        print(f"Inferring intent for user {user_id} with context: {user_context}")

        # Augment input with personalized context for better inference
        context_string = ""
        if user_context["known_conditions"]:
            context_string += f" User has known conditions: {", ".join(user_context["known_conditions"])}."
        # In a real LLM, this context would be part of the prompt.
        augmented_input = text_input + context_string

        intent_result = self.llm(augmented_input)

        # Basic ambiguity clarification logic
        if intent_result["intent"] == "unclear" or "vague" in text_input.lower():
            return self._clarify_ambiguity(intent_result)

        return intent_result

    def _clarify_ambiguity(self, intent_result: dict) -> dict:
        """
        Simulates asking clarifying questions for ambiguous intents.
        """
        print("Intent is unclear, attempting to clarify...")
        if intent_result["intent"] == "unclear":
            return {
                "intent": "clarification_needed",
                "message": "I didn't quite understand. Could you please provide more details or rephrase your request?",
                "original_details": intent_result["details"]
            }
        return intent_result

    def get_response(self, user_id: str, multi_modal_input: dict) -> dict:
        """
        Orchestrates multi-modal input processing, intent inference, and response generation.

        Args:
            user_id: Identifier for the user.
            multi_modal_input: A dictionary containing various input types (e.g., 'text', 'speech_audio', 'image_data', 'language').

        Returns:
            A dictionary containing the inferred intent and a generated response.
        """
        processed_text = ""

        if "speech_audio" in multi_modal_input:
            processed_text += self.process_speech_input(multi_modal_input["speech_audio"]) + ". "
        if "image_data" in multi_modal_input:
            processed_text += self.process_image_input(multi_modal_input["image_data"]) + ". "
        if "text" in multi_modal_input:
            processed_text += multi_modal_input["text"] + ". "

        # Handle multi-lingual input
        if "language" in multi_modal_input and multi_modal_input["language"] != 'en':
            processed_text = self.translate_text(processed_text, multi_modal_input["language"])

        if not processed_text.strip():
            return {"intent": "no_input", "message": "No valid input detected."}

        # Infer intent with personalization
        intent_result = self.infer_intent(user_id, processed_text)

        # Simulate a personalized response based on intent and user context
        response_message = "Thank you for your query."
        user_context = self._get_user_context(user_id)

        if intent_result["intent"] == "symptom_inquiry":
            response_message = f"I understand you are asking about symptoms. Based on your input: \"{intent_result["details"]}\", I recommend consulting a doctor for a proper diagnosis."
            if "fever" in intent_result["details"].lower() and "known_conditions" in user_context and "diabetes" in user_context["known_conditions"]:
                response_message += " Given your known condition of diabetes, it's especially important to monitor your fever and seek medical advice promptly."
        elif intent_result["intent"] == "medication_info":
            response_message = f"You are asking for medication information regarding: \"{intent_result["details"]}\". Please always follow your doctor's instructions and consult your pharmacist for detailed information."
        elif intent_result["intent"] == "appointment_booking":
            response_message = f"You are looking to book an appointment related to: \"{intent_result["details"]}\". I can help you find nearby clinics or provide contact details for your usual doctor."
        elif intent_result["intent"] == "general_medical_info":
            response_message = f"You are seeking general medical information about: \"{intent_result["details"]}\". I can provide educational content, but remember this is not medical advice."
        elif intent_result["intent"] == "clarification_needed":
            response_message = intent_result["message"]

        # Update user history with the current interaction (simplified)
        self._update_user_context(user_id, {"query": processed_text, "response_intent": intent_result["intent"]})

        return {"intent": intent_result["intent"], "message": response_message}

# --- Demonstration --- 
if __name__ == "__main__":
    medibot = MediBot()
    user1 = "patient_alice"
    user2 = "doctor_bob"

    print("\n--- User 1: Alice's Interactions (English) ---\n")

    # 1. Text input - Symptom inquiry
    print("Alice (Text): I have a terrible headache and feel very weak.")
    response = medibot.get_response(user1, {"text": "I have a terrible headache and feel very weak."})
    print(f"MediBot: {response["message"]}\n")

    # 2. Speech input - Medication inquiry (simulated audio)
    mock_audio_data = b"some_audio_data_indicating_medication"
    print("Alice (Speech): Can you tell me about Ibuprofen?")
    response = medibot.get_response(user1, {"speech_audio": mock_audio_data, "text": "Can you tell me about Ibuprofen?"})
    print(f"MediBot: {response["message"]}\n")

    # 3. Multi-modal input (Text + Image) - Skin condition
    mock_image_data_rash = b"skin_rash_image_data"
    print("Alice (Text + Image): This rash appeared on my arm yesterday. What could it be?")
    response = medibot.get_response(user1, {
        "text": "This rash appeared on my arm yesterday. What could it be?",
        "image_data": mock_image_data_rash
    })
    print(f"MediBot: {response["message"]}\n")

    # 4. Ambiguous input
    print("Alice (Text): I need help.")
    response = medibot.get_response(user1, {"text": "I need help."})
    print(f"MediBot: {response["message"]}\n")

    # 5. Personalized response based on previous interactions or known conditions
    # Let's simulate adding a known condition for Alice
    medibot._update_user_context(user1, {"known_conditions": ["diabetes"]})
    print("\nAlice (Text, with known condition): I have a fever and body aches.")
    response = medibot.get_response(user1, {"text": "I have a fever and body aches."})
    print(f"MediBot: {response["message"]}\n")

    print("\n--- User 2: Bob's Interactions (Spanish) ---\n")

    # 1. Text input - Symptom inquiry in Spanish
    print("Bob (Text): Tengo tos persistente.")
    response = medibot.get_response(user2, {"text": "Tengo tos persistente.", "language": 'es'})
    print(f"MediBot: {response["message"]}\n")
