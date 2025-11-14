class SpeechToTextService:
    """Simulates a Speech-to-Text service."""
    def transcribe(self, audio_data: str) -> str:
        # In a real application, this would use an actual STT API/model
        # For demonstration, we'll assume a direct transcription or a mock response.
        print(f"[STT Service] Transcribing audio: '{audio_data[:20]}...' ")
        # Simple mock for audio data, assuming it's a spoken sentence representation
        if "headache and tired" in audio_data.lower():
            return "I have a headache and feel tired."
        elif "fever body aches sore throat" in audio_data.lower():
            return "Yes, I have a slight fever, body aches, and a sore throat."
        elif "rash on my arm" in audio_data.lower():
            return "I have a rash on my arm."
        return audio_data # Fallback for other mock audio

class ImageAnalysisService:
    """Simulates an Image Analysis service."""
    def analyze(self, image_data: str) -> str:
        # In a real application, this would use an actual Image Analysis API/model
        # For demonstration, we'll assume the image_data is a description or a path.
        print(f"[Image Analysis Service] Analyzing image: '{image_data[:20]}...' ")
        if "rash_image.jpg" in image_data:
            return "The image shows a red, slightly raised rash on the skin. Possible allergic reaction or dermatitis."
        elif "xray_lung.png" in image_data:
            return "The image is an X-ray of lungs. Appears to show some consolidation in the lower left lobe."
        return "Unable to provide detailed analysis from the image." # Default mock

class MachineTranslationService:
    """Simulates a Machine Translation service."""
    def translate(self, text: str, target_language: str, source_language: str = "auto") -> str:
        # In a real application, this would use an actual Translation API/model
        # For demonstration, we'll perform a simple mock translation.
        print(f"[Translation Service] Translating '{text[:20]}...' from {source_language} to {target_language}")
        if target_language == "en" and source_language == "es" and "dolor de cabeza" in text.lower():
            return "I have a headache."
        elif target_language == "es" and source_language == "en" and "headache" in text.lower():
            return "Tengo dolor de cabeza."
        # Assume direct passthrough for simplicity if no specific mock translation exists
        return text

class MediBotLLM:
    """Simulates the core LLM orchestration module for MediBot.
    It manages context, intent classification, ambiguity resolution, and personalized learning.
    """
    def __init__(self):
        self.conversation_history = {}
        self.user_profiles = {}
        self.stt_service = SpeechToTextService()
        self.image_service = ImageAnalysisService()
        self.translation_service = MachineTranslationService()

    def _get_user_profile(self, user_id: str) -> dict:
        """Retrieves or initializes a user profile."""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                "past_conditions": [],
                "allergies": [],
                "preferred_language": "en"
            }
        return self.user_profiles[user_id]

    def _update_user_profile(self, user_id: str, new_data: dict):
        """Updates the user profile with new information."""
        profile = self._get_user_profile(user_id)
        for key, value in new_data.items():
            if isinstance(value, list) and key in profile and isinstance(profile[key], list):
                profile[key].extend(item for item in value if item not in profile[key])
            else:
                profile[key] = value
        print(f"[MediBot LLM] Updated profile for {user_id}: {profile}")

    def _generate_response_from_llm(self, user_id: str, combined_input: str) -> dict:
        """Simulates the LLM's processing of the combined input.
        This is where prompt engineering, intent classification, and ambiguity resolution occur.
        """
        history = self.conversation_history.get(user_id, [])
        profile = self._get_user_profile(user_id)

        # Basic prompt engineering structure for the simulated LLM
        prompt_parts = [
            "You are MediBot, a helpful healthcare assistant. Your goal is to understand user intent, provide relevant information, and ask clarifying questions when needed.",
            f"User ID: {user_id}",
            f"User's past context/profile: {profile}",
            "Current conversation history:"]
        
        for entry in history:
            prompt_parts.append(f"  User: {entry['user_query']}")
            prompt_parts.append(f"  MediBot: {entry['medibot_response_text']}")
            
        prompt_parts.append(f"Current User Input: {combined_input}")
        prompt_parts.append("Based on all available information, infer the user's intent, resolve ambiguity, and formulate a precise response. If the input is vague, ask a clarifying question. If the intent is an emergency, advise seeking immediate medical help. Also, extract any new health-related information to update the user's profile. Provide response in JSON format with 'intent', 'response_text', 'clarifying_question' (if any), 'profile_update' (if any).")

        full_prompt = "\n".join(prompt_parts)
        print(f"\n[MediBot LLM] Simulating LLM response for prompt:\n---\n{full_prompt}\n---")

        # --- Mock LLM Logic (Simplified for demonstration) ---
        response = {
            "intent": "informational",
            "response_text": "I understand you're seeking health information.",
            "clarifying_question": None,
            "profile_update": {}
        }

        lower_input = combined_input.lower()

        if "emergency" in lower_input or "severe chest pain" in lower_input or "unconscious" in lower_input:
            response["intent"] = "emergency"
            response["response_text"] = "This sounds like an emergency. Please seek immediate medical attention or call emergency services right away."
        elif "headache" in lower_input and "tired" in lower_input and not any(q in lower_input for q in ["fever", "body aches"]):
            response["intent"] = "symptom_inquiry"
            response["response_text"] = "I understand you have a headache and feel tired."
            response["clarifying_question"] = "Do you also have a fever, body aches, a sore throat, or any other symptoms?"
        elif "fever" in lower_input and "body aches" in lower_input and "sore throat" in lower_input:
            response["intent"] = "symptom_checker"
            response["response_text"] = "Based on your symptoms (headache, tiredness, fever, body aches, sore throat), these could indicate a common cold or flu. Rest, hydrate, and consider over-the-counter pain relievers. If symptoms worsen or persist, please consult a doctor."
            response["profile_update"] = {"past_conditions": ["common cold/flu symptoms"]}
        elif "rash" in lower_input:
            response["intent"] = "symptom_inquiry_image"
            response["response_text"] = "You mentioned a rash. Can you tell me more about its appearance, location, and how long you've had it?"
            if "allergic reaction" in lower_input or "dermatitis" in lower_input:
                response["response_text"] = "The image analysis suggests a rash possibly due to an allergic reaction or dermatitis. Avoid irritants and consider consulting a dermatologist if it persists."
                response["profile_update"] = {"past_conditions": ["rash"], "allergies": ["potential irritant"]}
        elif "flu symptoms" in lower_input:
            response["intent"] = "informational"
            response["response_text"] = "Symptoms of the flu often include fever, body aches, headache, fatigue, cough, and sore throat. It's important to rest and stay hydrated."
        elif "what can I do for a headache" in lower_input: # Direct self-care advice
            response["intent"] = "self_care_advice"
            response["response_text"] = "For a typical headache, try resting in a quiet, dark room, applying a cold compress to your forehead, or taking an over-the-counter pain reliever like ibuprofen or acetaminophen. If headaches are severe or frequent, please consult a doctor."
        elif "allergy" in lower_input and profile.get("allergies"): # Personalized response
            response["intent"] = "informational_personalized"
            response["response_text"] = f"Considering your past interactions, you've mentioned allergies. Are you experiencing a new allergic reaction, or do you have questions about managing your known allergies like {', '.join(profile['allergies'])}?"
            response["clarifying_question"] = "Can you specify what kind of allergy you are asking about?"
        elif "greetings" in lower_input or "hello" in lower_input:
            response["intent"] = "greeting"
            response["response_text"] = "Hello! How can I assist you with your health today?"

        print(f"[MediBot LLM] Simulated LLM Output: {response}")
        return response

    def process_query(self, user_id: str, text_input: str = None, audio_input: str = None, 
                      image_input: str = None, input_language: str = "en") -> dict:
        """Processes a multi-modal user query through the MediBot system."""
        print(f"\n--- Processing Query for User '{user_id}' ---")
        original_query = {
            "text": text_input,
            "audio": audio_input,
            "image": image_input,
            "language": input_language
        }

        processed_text = text_input if text_input else ""
        image_analysis_result = ""

        # 1. Speech-to-Text Processing
        if audio_input:
            stt_result = self.stt_service.transcribe(audio_input)
            if input_language != "en": # If audio is in a different language, translate STT result
                processed_text = self.translation_service.translate(stt_result, target_language="en", source_language=input_language)
            else:
                processed_text = stt_result
            print(f"[MediBot] STT Result: {processed_text}")
        elif input_language != "en" and text_input: # Only translate if text input exists and is not English
            processed_text = self.translation_service.translate(text_input, target_language="en", source_language=input_language)
            print(f"[MediBot] Translated Text: {processed_text}")

        # 2. Image Analysis Processing
        if image_input:
            image_analysis_result = self.image_service.analyze(image_input)
            print(f"[MediBot] Image Analysis Result: {image_analysis_result}")

        # Combine all processed inputs for the LLM
        combined_input_for_llm = f"User Text: {processed_text}. Image Analysis: {image_analysis_result}"
        combined_input_for_llm = combined_input_for_llm.strip('. ')

        # 3. Core LLM Orchestration
        llm_output = self._generate_response_from_llm(user_id, combined_input_for_llm)
        
        # 4. Update Personalized Learning / Profile
        if llm_output.get("profile_update"):
            self._update_user_profile(user_id, llm_output["profile_update"])

        # 5. Translate response back to user's preferred language if necessary
        final_response_text = llm_output["response_text"]
        user_profile = self._get_user_profile(user_id)
        if user_profile["preferred_language"] != "en":
            final_response_text = self.translation_service.translate(
                final_response_text, target_language=user_profile["preferred_language"], source_language="en"
            )
        
        # Store conversation history (original user query and MediBot's final response)
        if user_id not in self.conversation_history:
            self.conversation_history[user_id] = []
        self.conversation_history[user_id].append({
            "user_query": original_query, # Store original multi-modal input
            "processed_llm_input": combined_input_for_llm, # What the LLM actually processed
            "medibot_response_raw_llm_output": llm_output, # Raw LLM output for debugging/analysis
            "medibot_response_text": final_response_text # The final text shown to user
        })

        return {
            "user_id": user_id,
            "original_input": original_query,
            "processed_input_for_llm": combined_input_for_llm,
            "inferred_intent": llm_output["intent"],
            "medibot_response": final_response_text,
            "clarifying_question": llm_output["clarifying_question"],
            "current_profile": self._get_user_profile(user_id)
        }


# --- Demonstration of MediBot in action ---
if __name__ == "__main__":
    medibot = MediBotLLM()

    print("\n--- Scenario 1: Basic Text Query with Ambiguity Resolution ---")
    response1 = medibot.process_query(user_id="user_A", text_input="I don't feel good.")
    print(f"MediBot: {response1['medibot_response']}")
    print(f"Clarifying Question: {response1['clarifying_question']}")
    print(f"User Profile: {response1['current_profile']}")

    print("\n--- Scenario 2: Follow-up with more details (speech input) ---")
    # Simulate audio input that STT would transcribe
    response2 = medibot.process_query(user_id="user_A", audio_input="I have a headache and feel tired.")
    print(f"MediBot: {response2['medibot_response']}")
    print(f"Clarifying Question: {response2['clarifying_question']}")
    print(f"User Profile: {response2['current_profile']}")

    print("\n--- Scenario 3: Providing a full set of symptoms (speech input) ---")
    response3 = medibot.process_query(user_id="user_A", audio_input="Yes, I have a slight fever, body aches, and a sore throat.")
    print(f"MediBot: {response3['medibot_response']}")
    print(f"Clarifying Question: {response3['clarifying_question']}")
    print(f"User Profile: {response3['current_profile']}")

    print("\n--- Scenario 4: Image input for rash diagnosis ---")
    response4 = medibot.process_query(user_id="user_B", image_input="rash_image.jpg", text_input="I have a rash on my arm, what could it be?")
    print(f"MediBot: {response4['medibot_response']}")
    print(f"Clarifying Question: {response4['clarifying_question']}")
    print(f"User Profile: {response4['current_profile']}")

    print("\n--- Scenario 5: Multi-lingual interaction (Spanish input) ---")
    medibot._update_user_profile(user_id="user_C", new_data={"preferred_language": "es"})
    response5 = medibot.process_query(user_id="user_C", text_input="Tengo dolor de cabeza.", input_language="es")
    print(f"MediBot: {response5['medibot_response']}")
    print(f"User Profile: {response5['current_profile']}")

    print("\n--- Scenario 6: Personalized response based on history ---")
    # Simulate previous interaction where allergy was mentioned/inferred
    medibot._update_user_profile(user_id="user_D", new_data={"allergies": ["pollen"]})
    response6 = medibot.process_query(user_id="user_D", text_input="Tell me about allergies.")
    print(f"MediBot: {response6['medibot_response']}")
    print(f"Clarifying Question: {response6['clarifying_question']}")
    print(f"User Profile: {response6['current_profile']}")

    print("\n--- Scenario 7: Emergency Situation ---")
    response7 = medibot.process_query(user_id="user_E", text_input="I have severe chest pain and can't breathe!")
    print(f"MediBot: {response7['medibot_response']}")
    print(f"User Profile: {response7['current_profile']}")

    print("\n--- Scenario 8: Direct self-care advice ---")
    response8 = medibot.process_query(user_id="user_F", text_input="What can I do for a headache?")
    print(f"MediBot: {response8['medibot_response']}")
    print(f"User Profile: {response8['current_profile']}")
