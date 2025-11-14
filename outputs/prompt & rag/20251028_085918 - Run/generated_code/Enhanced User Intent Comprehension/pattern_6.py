import speech_recognition as sr
from PIL import Image
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import torch # Added for transformer dependency

class CustomerSupportAssistant:
    def __init__(self, llm_model_path="EleutherAI/gpt-neo-125M"): 
        # Initialize Speech Recognition
        self.recognizer = sr.Recognizer()

        # Initialize Image Analysis (e.g., a simple image captioning or object detection model)
        # For demonstration, we'll use a placeholder or a simple visual question answering model
        # In a real application, consider more robust models like those based on CLIP/BLIP for multimodal understanding.
        self.image_analyser = pipeline("visual-question-answering", model="Salesforce/blip-vqa-base") # Using a BLIP VQA model

        # Initialize Machine Translation (English to Spanish example)
        self.translator_tokenizer_en_es = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
        self.translator_model_en_es = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-es")
        self.translation_pipeline_en_es = pipeline("translation", model=self.translator_model_en_es, tokenizer=self.translator_tokenizer_en_es)

        # Initialize Machine Translation (Spanish to English example)
        self.translator_tokenizer_es_en = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-es-en")
        self.translator_model_es_en = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-es-en")
        self.translation_pipeline_es_en = pipeline("translation", model=self.translator_model_es_en, tokenizer=self.translator_tokenizer_es_en)

        # Initialize LLM for Intent Comprehension (placeholder for actual LLM setup)
        # This would be a more complex setup with instruction tuning and personalized learning
        # For a real application, fine-tune a larger LLM on domain-specific data.
        self.llm_intent_comprehender = pipeline("text-generation", model=llm_model_path, truncation=True) # Using GPT-Neo as a placeholder

        # Placeholder for customer history/profile database
        self.customer_db = {}

    def _get_customer_context(self, customer_id: str) -> dict:
        # Retrieve personalized data for the customer
        return self.customer_db.get(customer_id, {"history": [], "preferences": {}})

    def process_text_input(self, text_query: str, customer_id: str, source_language: str = "en") -> dict:
        processed_text_query = text_query
        if source_language != "en": # Assuming English as the primary processing language
            processed_text_query = self.translate_text(text_query, source_language, "en")
            print(f"Translated text query to English: {processed_text_query}")

        customer_context = self._get_customer_context(customer_id)
        intent_data = self._comprehend_intent(processed_text_query, customer_context)
        response = self._generate_response(intent_data, customer_context, target_language=source_language)
        return {"original_query": text_query, "processed_query": processed_text_query, "intent": intent_data, "response": response}

    def process_speech_input(self, audio_file_path: str, customer_id: str, source_language: str = "en") -> dict:
        try:
            with sr.AudioFile(audio_file_path) as source:
                audio_data = self.recognizer.record(source)
            # Using Google Web Speech API for speech recognition
            text_query = self.recognizer.recognize_google(audio_data, language=source_language)
            print(f"Speech recognized ({source_language}): {text_query}")
            return self.process_text_input(text_query, customer_id, source_language)
        except sr.UnknownValueError:
            return {"error": "Speech Recognition could not understand audio", "original_query": "(audio input)"}
        except sr.RequestError as e:
            return {"error": f"Could not request results from Google Speech Recognition service; {e}", "original_query": "(audio input)"}
        except FileNotFoundError:
            return {"error": "Audio file not found", "original_query": "(audio input)"}
        except Exception as e:
            return {"error": f"An unexpected error occurred during speech processing: {e}", "original_query": "(audio input)"}

    def process_image_input(self, image_file_path: str, customer_id: str, text_context: str = "", source_language: str = "en") -> dict:
        try:
            image = Image.open(image_file_path)
            # Use VQA to get an understanding of the image context
            # For a real system, this would be more sophisticated, e.g., object detection, damage assessment
            vqa_question = f"What is in this image related to {text_context if text_context else 'the product or delivery'}?" # Dynamic question
            image_analysis_result = self.image_analyser(image=image, question=vqa_question)
            print(f"Image analysis result for question \"{vqa_question}\": {image_analysis_result}")
            image_description = image_analysis_result[0]["answer"] if image_analysis_result else "(no clear description)"

            # Combine image context with any provided text context
            combined_query = f"User provided an image. The image shows: {image_description}. Additional context: {text_context}"
            print(f"Combined query from image and text: {combined_query}")
            return self.process_text_input(combined_query, customer_id, source_language)
        except FileNotFoundError:
            return {"error": "Image file not found", "original_query": "(image input)"}
        except Exception as e:
            return {"error": f"Image processing error: {e}", "original_query": "(image input)"}

    def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        if source_lang == target_lang:
            return text
        try:
            if source_lang == "en" and target_lang == "es":
                translated_text = self.translation_pipeline_en_es(text)[0]["translation_text"]
            elif source_lang == "es" and target_lang == "en":
                translated_text = self.translation_pipeline_es_en(text)[0]["translation_text"]
            else:
                print(f"Warning: Translation for {source_lang} to {target_lang} is not explicitly supported by loaded models. Returning original text.")
                return text
            return translated_text
        except Exception as e:
            print(f"Translation error: {e}. Returning original text.")
            return text

    def _comprehend_intent(self, processed_query: str, customer_context: dict) -> dict:
        # This is where the LLM with instruction tuning and personalized learning comes in.
        # The `llm_intent_comprehender` would be invoked here with the processed query.
        # For demonstration, we'll simulate intent detection with some basic keyword matching.
        print(f"Comprehending intent for: \"{processed_query}\" with context: {customer_context.get('history', [])}")

        # Example of how an LLM might process the query
        # In a real scenario, the LLM would be prompted with the query and context
        # For now, a simplified keyword-based simulation.

        intent = "General Inquiry"
        details = {}
        lower_query = processed_query.lower()

        if "return" in lower_query or "damaged" in lower_query or "broken" in lower_query or "wrong item" in lower_query:
            intent = "Return/Refund Request"
            details = {"product_issue": "damaged" if "damaged" in lower_query or "broken" in lower_query else "wrong item" if "wrong item" in lower_query else "return",
                       "context_history": customer_context.get('history', [])}
        elif "order status" in lower_query or "where is my order" in lower_query or "delivery date" in lower_query:
            intent = "Order Status Inquiry"
            details = {"order_id_hint": "extract_from_query_if_present",
                       "context_history": customer_context.get('history', [])}
        elif "delivery issue" in lower_query or "late delivery" in lower_query or "missing package" in lower_query:
            intent = "Delivery Complaint"
            details = {"issue": "late delivery" if "late delivery" in lower_query else "missing package" if "missing package" in lower_query else "delivery issue",
                       "context_history": customer_context.get('history', [])}
        elif "account" in lower_query or "login" in lower_query or "password" in lower_query:
            intent = "Account Related Inquiry"
            details = {"issue": "account access"}
        
        # Further enhance with LLM if a specific model is loaded and fine-tuned
        # try:
        #     llm_output = self.llm_intent_comprehender(f"Given the customer's query: '{processed_query}' and their history: {customer_context.get('history', [])}, classify their intent and extract key details.", max_new_tokens=50)
        #     # Parse llm_output to refine intent and details
        #     # This part would depend heavily on the LLM's fine-tuning and prompt engineering
        # except Exception as e:
        #     print(f"LLM intent comprehension error: {e}. Falling back to keyword matching.")

        return {"intent_type": intent, "details": details}

    def _generate_response(self, intent_data: dict, customer_context: dict, target_language: str = "en") -> str:
        # Based on the intent and customer context, generate a personalized response.
        # This would also involve the LLM.
        print(f"Generating response for intent: {intent_data['intent_type']} with details: {intent_data['details']} for customer: {customer_context.get('name', 'N/A')}")
        response_template = ""
        customer_name = customer_context.get("name", "")

        if intent_data["intent_type"] == "Return/Refund Request":
            reason = intent_data["details"].get("product_issue", "return")
            response_template = f"I understand you have a {reason} request. Please provide your order number and we will assist you with the return process for your item."
        elif intent_data["intent_type"] == "Order Status Inquiry":
            response_template = f"To check the status of your order, please provide your order ID. We will then give you the latest update."
        elif intent_data["intent_type"] == "Delivery Complaint":
            issue = intent_data["details"].get("issue", "delivery issue")
            response_template = f"I apologize for the {issue}. Please provide your order number so I can investigate this further for you."
        elif intent_data["intent_type"] == "Account Related Inquiry":
            response_template = f"For account-related issues, please verify your identity. What specifically do you need help with regarding your account?"
        else:
            response_template = "Hello! How can I assist you further today?"

        # Personalize based on customer context (e.g., call them by name, reference past issues)
        if customer_name:
            response_template = f"Hi {customer_name}, {response_template}"

        if target_language != "en":
            response_template = self.translate_text(response_template, "en", target_language)

        return response_template

# Example Usage:
if __name__ == "__main__":
    # Initialize the assistant with a small placeholder LLM
    assistant = CustomerSupportAssistant(llm_model_path="EleutherAI/gpt-neo-125M") 

    # Simulate adding customers to the DB
    assistant.customer_db['user123'] = {'name': 'Alice', 'history': ['purchased shoes', 'previous return'], 'preferences': {'language': 'en'}}
    assistant.customer_db['user456'] = {'name': 'Bob', 'history': ['purchased electronics'], 'preferences': {'language': 'es'}}
    assistant.customer_db['user789'] = {'name': 'Charlie', 'history': ['recently inquired about delivery'], 'preferences': {'language': 'en'}}

    print("\n--- Text Input (English - Return Request) ---")
    text_result_en = assistant.process_text_input("I want to return the damaged item I received.", "user123", "en")
    print(f"User 123 Original Query: {text_result_en['original_query']}")
    print(f"User 123 Intent: {text_result_en['intent']['intent_type']} (Details: {text_result_en['intent']['details']})")
    print(f"User 123 Response: {text_result_en['response']}")

    print("\n--- Text Input (Spanish - Order Status, translated) ---")
    text_result_es = assistant.process_text_input("Quiero saber el estado de mi pedido.", "user456", "es")
    print(f"User 456 Original Query: {text_result_es['original_query']}")
    print(f"User 456 Processed Query (English): {text_result_es['processed_query']}")
    print(f"User 456 Intent: {text_result_es['intent']['intent_type']} (Details: {text_result_es['intent']['details']})")
    print(f"User 456 Response: {text_result_es['response']}")

    print("\n--- Text Input (English - Delivery Issue) ---")
    text_result_delivery = assistant.process_text_input("My package is late, where is it?", "user789", "en")
    print(f"User 789 Original Query: {text_result_delivery['original_query']}")
    print(f"User 789 Intent: {text_result_delivery['intent']['intent_type']} (Details: {text_result_delivery['intent']['details']})")
    print(f"User 789 Response: {text_result_delivery['response']}")

    # To test speech input, you would need an actual audio file. 
    # For example, create a dummy 'test_audio_en.wav' file (e.g., saying 