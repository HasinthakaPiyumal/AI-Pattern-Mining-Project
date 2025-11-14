import speech_recognition as sr
from PIL import Image
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import io
import base64

# --- Configuration for Models ---
# For a real application, these would be loaded once or managed by a service.
# Using small models for demonstration purposes.

# Speech-to-Text (Placeholder for a more robust model or API)
# For simplicity, we'll use SpeechRecognition with Google Web Speech API (online)
# or a local model like 'vosk' if offline capabilities are needed.
# For this example, we'll simulate an STT output.

# Image Analysis (Placeholder for a robust image captioning/OCR model)
# For this example, we'll simulate an image analysis output.

# Machine Translation
# Example: English to French translation model
TRANSLATION_MODEL_NAME = "Helsinki-NLP/opus-mt-en-fr"
translation_tokenizer = None
translation_model = None

# Large Language Model (LLM) for Intent Comprehension and Response Generation
# Using a small T5 model for demonstration.
LLM_MODEL_NAME = "google/flan-t5-small"
llm_tokenizer = None
llm_model = None
llm_pipeline = None

def _load_translation_model():
    global translation_tokenizer, translation_model
    if translation_tokenizer is None or translation_model is None:
        print(f"Loading translation model: {TRANSLATION_MODEL_NAME}...")
        translation_tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL_NAME)
        translation_model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATION_MODEL_NAME)
        print("Translation model loaded.")

def _load_llm_model():
    global llm_tokenizer, llm_model, llm_pipeline
    if llm_tokenizer is None or llm_model is None or llm_pipeline is None:
        print(f"Loading LLM model: {LLM_MODEL_NAME}...")
        llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
        llm_model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL_NAME)
        llm_pipeline = pipeline("text2text-generation", model=llm_model, tokenizer=llm_tokenizer)
        print("LLM model loaded.")

class SmartCustomerSupportAgent:
    def __init__(self):
        # Initialize models (if not already loaded globally)
        _load_translation_model()
        _load_llm_model()

        self.r = sr.Recognizer()
        # A simple "user history" for personalized learning simulation
        self.user_histories = {} # user_id: list of past interactions/preferences

    def _speech_to_text(self, audio_data: bytes, language: str = "en-US") -> str:
        """Converts audio data to text."""
        try:
            # For demonstration, let's assume audio_data is a raw WAV byte stream
            # In a real scenario, you'd use a specific audio file format or stream handler
            audio_io = io.BytesIO(audio_data)
            with sr.AudioFile(audio_io) as source:
                audio = self.r.record(source)  # read the entire audio file
            text = self.r.recognize_google(audio, language=language)
            print(f"STT Output: {text}")
            return text
        except sr.UnknownValueError:
            print("Speech Recognition could not understand audio")
            return ""
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
            return ""
        except Exception as e:
            print(f"Error during speech to text: {e}")
            return ""

    def _image_analysis_to_text(self, image_data: bytes) -> str:
        """
        Processes image data to extract relevant textual information.
        This is a placeholder for actual image captioning, OCR, or object detection.
        """
        try:
            image = Image.open(io.BytesIO(image_data))
            # Simulate a very basic image analysis, e.g., describing its size
            width, height = image.size
            # In a real system, you'd use models like CLIP, BLIP, or an OCR engine.
            # Example using a conceptual image captioning pipeline:
            # from transformers import pipeline
            # image_to_text_pipeline = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
            # caption = image_to_text_pipeline(image)
            # return caption[0]["generated_text"]

            print(f"Simulated Image Analysis: Image detected (size: {width}x{height}).")
            return f"User provided an image. Image dimensions: {width}x{height}. " \
                   f"A more advanced model would describe content here (e.g., 'a broken product', 'an invoice')."
        except Exception as e:
            print(f"Error during image analysis: {e}")
            return "Could not analyze image content."

    def _translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translates text from source_lang to target_lang."""
        if not text:
            return ""
        try:
            # Using Hugging Face transformers for translation
            _load_translation_model() # Ensure model is loaded
            if source_lang == target_lang:
                return text # No translation needed if languages are the same

            # The model is trained for a specific language pair (en-fr in this example)
            # For a truly multi-lingual agent, a more generic or multiple models would be needed.
            # Here, we assume "source_lang" means the input text's language, and
            # "target_lang" is the desired internal processing language (e.g., English).
            # If the model is en-fr, and we get "fr" input, we'd translate fr->en first.

            # This is a simplification. For N-to-N translation,
            # you'd need a more complex routing or a universal translator.
            # Let's assume input always comes in, and we translate TO English for LLM.
            # So, if source_lang is not 'en', we translate to 'en'.
            if source_lang.startswith('en') and target_lang.startswith('en'):
                 return text # Already English, no translation needed.

            # Example: translating from target_lang to English (if target_lang is not English)
            # This is a bit tricky with `opus-mt` as models are specific.
            # A general multi-lingual approach would involve a common pivot language or
            # more sophisticated `transformers` models like mBART.

            # For demonstration, let's assume we translate `text` to English if `source_lang` is not English.
            # And our demo model is specifically en-fr. This part needs robust handling in real app.
            # Let's pretend we have a `translate_to_english` model or the `pipeline` can handle it.
            # For this specific `Helsinki-NLP/opus-mt-en-fr` model, direct "fr->en" pipeline might not exist.
            # We'll simulate by checking if source_lang is 'fr' and target 'en'.
            if source_lang.startswith('fr') and target_lang.startswith('en'):
                # Simulate reverse translation if a model was available
                print(f"Simulating translation from French to English for: {text}")
                # Actual translation would look like:
                # inputs = translation_tokenizer(text, return_tensors="pt")
                # outputs = translation_model.generate(**inputs)
                # translated_text = translation_tokenizer.decode(outputs[0], skip_special_tokens=True)
                # For demo, just a placeholder
                return f"Translated from French: '{text}' (Simulated English text)"
            elif source_lang.startswith('en') and target_lang.startswith('fr'):
                print(f"Translating from English to French for: {text}")
                inputs = translation_tokenizer(text, return_tensors="pt")
                outputs = translation_model.generate(**inputs)
                translated_text = translation_tokenizer.decode(outputs[0], skip_special_tokens=True)
                return translated_text
            else:
                print(f"Skipping translation for {source_lang} to {target_lang} with current model config.")
                return text # Return original if not handled by demo model

        except Exception as e:
            print(f"Error during translation: {e}")
            return text # Return original text on error

    def _comprehend_intent(self, processed_input: str, user_id: str) -> dict:
        """
        Uses LLM to comprehend user intent and extract relevant entities.
        Incorporates personalized learning by fetching user history.
        """
        _load_llm_model() # Ensure model is loaded

        user_context = self.user_histories.get(user_id, [])
        context_str = " ".join(user_context[-3:]) # Last 3 interactions as context

        prompt = f"Given the user's past interactions: '{context_str}', and their current input: '{processed_input}'. " \
                 f"What is the user's primary intent? Extract key entities. " \
                 f"Respond in a JSON format with 'intent' (e.g., 'product_inquiry', 'order_status', 'technical_support', 'greeting', 'complaint') and 'entities' (a dict of key-value pairs)."

        print(f"LLM Prompt for Intent Comprehension: {prompt}")

        try:
            # Use the LLM pipeline for text generation
            # For Flan-T5, it's good for instruction-following.
            response = llm_pipeline(prompt, max_new_tokens=100, num_beams=5, early_stopping=True)
            llm_output = response[0]["generated_text"]
            print(f"LLM Raw Output for Intent: {llm_output}")

            # Attempt to parse LLM output as JSON
            try:
                intent_data = eval(llm_output) # Using eval cautiously, assuming LLM outputs valid JSON-like structure
                if not isinstance(intent_data, dict) or "intent" not in intent_data:
                    raise ValueError("LLM output is not a valid intent dictionary.")
            except Exception as json_err:
                print(f"Could not parse LLM output as JSON, attempting fallback: {json_err}")
                # Fallback: Simple keyword extraction if JSON parsing fails
                intent_data = {"intent": "unknown", "entities": {}}
                if "product" in processed_input.lower():
                    intent_data["intent"] = "product_inquiry"
                if "order" in processed_input.lower() or "status" in processed_input.lower():
                    intent_data["intent"] = "order_status"
                if "hello" in processed_input.lower() or "hi" in processed_input.lower():
                    intent_data["intent"] = "greeting"
                intent_data["entities"]["raw_input"] = processed_input

            return intent_data
        except Exception as e:
            print(f"Error during LLM intent comprehension: {e}")
            return {"intent": "error", "entities": {"reason": str(e), "raw_input": processed_input}}

    def _generate_response(self, intent_data: dict, user_id: str) -> str:
        """
        Generates a personalized response based on the inferred intent and entities.
        """
        _load_llm_model() # Ensure model is loaded

        intent = intent_data.get("intent", "unknown")
        entities = intent_data.get("entities", {})
        user_context = self.user_histories.get(user_id, [])
        context_str = " ".join(user_context[-3:])

        prompt = f"Given the user's intent: '{intent}', entities: {entities}, and past context: '{context_str}'. " \
                 f"Generate a helpful and personalized customer support response for an e-commerce platform. " \
                 f"Keep it concise and friendly."

        print(f"LLM Prompt for Response Generation: {prompt}")

        try:
            response = llm_pipeline(prompt, max_new_tokens=150, num_beams=5, early_stopping=True)
            generated_text = response[0]["generated_text"]
            print(f"LLM Raw Output for Response: {generated_text}")
            return generated_text
        except Exception as e:
            print(f"Error during LLM response generation: {e}")
            return "I apologize, I'm currently unable to generate a response. Please try again later."

    def process_query(self, query_data: dict, user_id: str, channel: str, user_lang: str = "en") -> str:
        """
        Main method to process a multi-modal, multi-lingual query.

        Args:
            query_data (dict): Contains query content.
                               e.g., {"type": "text", "content": "Hello, what is my order status?"}
                               e.g., {"type": "voice", "content": b"..." (audio bytes)}
                               e.g., {"type": "image", "content": b"..." (image bytes)}
            user_id (str): Unique identifier for the user to retrieve personalized history.
            channel (str): The communication channel (e.g., "chat", "voice_call", "email").
            user_lang (str): The original language of the user's input (e.g., "en", "fr", "es").

        Returns:
            str: The agent's generated response in the user's language.
        """
        processed_text = ""
        target_processing_lang = "en" # Internal processing language for LLM

        query_type = query_data.get("type")
        content = query_data.get("content")

        if query_type == "text":
            processed_text = content
            print(f"Processing text input: {processed_text}")
        elif query_type == "voice":
            if isinstance(content, str): # Assume base64 encoded for voice via API
                audio_bytes = base64.b64decode(content)
            else: # Assume raw bytes
                audio_bytes = content
            processed_text = self._speech_to_text(audio_bytes, language=user_lang)
            print(f"Processing voice input (STT): {processed_text}")
        elif query_type == "image":
            if isinstance(content, str): # Assume base64 encoded for image via API
                image_bytes = base64.b64decode(content)
            else: # Assume raw bytes
                image_bytes = content
            image_description = self._image_analysis_to_text(image_bytes)
            # Prepend image description to text for LLM
            processed_text = f"User provided an image. Image context: {image_description}. " + (query_data.get("text_complement", "") if query_data.get("text_complement") else "")
            print(f"Processing image input (Description): {processed_text}")
        else:
            return "Sorry, I can only process text, voice, or image inputs."

        # Handle multi-lingual input by translating to internal processing language (English)
        if user_lang.lower() != target_processing_lang:
            original_text = processed_text # Store original for potential re-translation of response
            processed_text = self._translate_text(processed_text, source_lang=user_lang, target_lang=target_processing_lang)
            print(f"Translated input to English: {processed_text}")
            if not processed_text and original_text: # If translation failed, use original text for LLM (might perform poorly)
                 processed_text = original_text
                 print("Translation failed or skipped, proceeding with original text in non-English.")


        # 1. Intent Comprehension
        intent_data = self._comprehend_intent(processed_text, user_id)
        print(f"Inferred Intent: {intent_data}")

        # 2. Personalized Learning (Update user history)
        if user_id not in self.user_histories:
            self.user_histories[user_id] = []
        self.user_histories[user_id].append(processed_text) # Store processed input for history
        # (In a real system, you'd store structured intent/entities or preferences)

        # 3. Response Generation
        agent_response = self._generate_response(intent_data, user_id)
        print(f"Generated Response (English): {agent_response}")

        # 4. Translate response back to user's language if necessary
        final_response = agent_response
        if user_lang.lower() != target_processing_lang:
            final_response = self._translate_text(agent_response, source_lang=target_processing_lang, target_lang=user_lang)
            if not final_response:
                final_response = agent_response # Fallback to English if translation fails
            print(f"Final Response (Translated to {user_lang}): {final_response}")

        return final_response

# Example Usage (assuming you have audio.wav and image.png files)
# To run this, you'll need to install:
# pip install SpeechRecognition Pillow transformers torch sentencepiece
# And for actual audio processing, install pyaudio (e.g., `pip install PyAudio`)
# or make sure you have the necessary system libraries for sound.
# For demo, I'll simulate audio and image bytes.

if __name__ == "__main__":
    agent = SmartCustomerSupportAgent()

    user_id_1 = "user_abc_123"
    user_id_2 = "user_xyz_789"

    print("\n--- Test Case 1: Simple Text Query (English) ---")
    query_text_en = {"type": "text", "content": "Hi, I need help with my recent order."}
    response_en = agent.process_query(query_text_en, user_id_1, channel="chat", user_lang="en")
    print(f"\nAgent's Final Response: {response_en}\n")

    print("\n--- Test Case 2: Text Query (French) ---")
    # For this to work, translation model needs to support fr -> en and en -> fr
    # Our demo model Helsinki-NLP/opus-mt-en-fr only does en -> fr directly.
    # So, we'll simulate the input being translated to English first.
    query_text_fr = {"type": "text", "content": "Bonjour, où est ma commande ?"}
    response_fr = agent.process_query(query_text_fr, user_id_1, channel="chat", user_lang="fr")
    print(f"\nAgent's Final Response: {response_fr}\n")


    print("\n--- Test Case 3: Simulated Voice Query (English) ---")
    # Simulate WAV audio bytes for "Where is my package?"
    # A real implementation would record from microphone or read a file.
    # This is a very basic WAV header + content simulation.
    # For a real STT demo, you'd need an actual .wav file or PyAudio.
    # As SpeechRecognition uses Google API, this will work if internet is available.
    simulated_audio_content = b"RIFF\x00\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00@\x1f\x00\x00\x80>\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
    # To make SpeechRecognition work, we need actual audio that it can parse.
    # Since I cannot create valid audio bytes dynamically that Google STT would recognize easily without actual libraries,
    # I'll provide a more realistic placeholder output for STT.
    # In a real setup, `_speech_to_text` would be called with actual audio.
    # For this __main__ block, let's hardcode a STT result for simplicity.

    class MockSpeechRecognizer: # Mock to avoid actual audio file dependency in main
        def record(self, source): return None
        def recognize_google(self, audio, language): return "Where is my package?"
        def AudioFile(self, audio_io): return self

    agent.r = MockSpeechRecognizer() # Temporarily replace for demo
    query_voice_en = {"type": "voice", "content": simulated_audio_content} # Content here is illustrative
    response_voice_en = agent.process_query(query_voice_en, user_id_2, channel="voice_call", user_lang="en")
    print(f"\nAgent's Final Response: {response_voice_en}\n")
    agent.r = sr.Recognizer() # Reset recognizer

    print("\n--- Test Case 4: Simulated Image Query with Complementary Text ---")
    # Simulate image bytes (a tiny red dot image as a placeholder)
    # A real image would be much larger and contain meaningful data.
    img = Image.new('RGB', (1, 1), color = 'red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    simulated_image_content = img_byte_arr.getvalue()

    query_image_text = {
        "type": "image",
        "content": simulated_image_content,
        "text_complement": "This is a picture of the broken item."
    }
    response_image = agent.process_query(query_image_text, user_id_1, channel="email", user_lang="en")
    print(f"\nAgent's Final Response: {response_image}\n")

    print("\n--- Test Case 5: Further interaction (demonstrating personalization) ---")
    query_follow_up = {"type": "text", "content": "What about a refund?"}
    response_follow_up = agent.process_query(query_follow_up, user_id_1, channel="chat", user_lang="en")
    print(f"\nAgent's Final Response: {response_follow_up}\n")
