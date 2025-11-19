from fastapi import FastAPI, UploadFile, File, Form
from typing import Optional
import speech_recognition as sr
from transformers import pipeline
from PIL import Image
import io

# --- 2. Multimodal Input Processing Layer ---

class ASRModule:
    def __init__(self):
        self.recognizer = sr.Recognizer()

    def transcribe_audio(self, audio_file_path: str) -> str:
        # In a real scenario, this would use a more robust ASR service or model
        with sr.AudioFile(audio_file_path) as source:
            audio_data = self.recognizer.record(source)
            try:
                text = self.recognizer.recognize_google(audio_data)
                return text
            except sr.UnknownValueError:
                return "Could not understand audio"
            except sr.RequestError as e:
                return f"Speech recognition service error: {e}"

class ImageAnalysisModule:
    def __init__(self):
        self.image_captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

    def analyze_image(self, image_bytes: bytes) -> str:
        image = Image.open(io.BytesIO(image_bytes))
        caption = self.image_captioner(image)[0]["generated_text"]
        return f"Image content: {caption}"

class TranslationModule:
    def __init__(self):
        self.translator_en_to_xx = pipeline("translation", model="Helsinki-NLP/opus-mt-en-de") # Example: English to German
        self.translator_xx_to_en = pipeline("translation", model="Helsinki-NLP/opus-mt-de-en") # Example: German to English

    def translate_to_english(self, text: str, src_lang: str = "auto") -> str:
        # For simplicity, assuming src_lang is known or auto-detected in a real app
        if src_lang != "en": # Only translate if not already English
            # This is a simplified example; a real translator would handle multiple languages dynamically
            if self.translator_xx_to_en.model.config.name_or_path == f"Helsinki-NLP/opus-mt-{src_lang}-en":
                 translated_text = self.translator_xx_to_en(text)[0]["translation_text"]
            else: # Fallback for unknown languages or different models
                translated_text = text # Mocking: just return original text if no specific model
            return translated_text
        return text

    def translate_from_english(self, text: str, target_lang: str) -> str:
        if target_lang != "en":
            if self.translator_en_to_xx.model.config.name_or_path == f"Helsinki-NLP/opus-mt-en-{target_lang}":
                translated_text = self.translator_en_to_xx(text)[0]["translation_text"]
            else:
                translated_text = text # Mocking
            return translated_text
        return text

class TextPreprocessingModule:
    def preprocess_text(self, text: str) -> str:
        return text.lower().strip() # Simple preprocessing

# --- 4. Data and Knowledge Base Layer (Mocked) ---

class KnowledgeBase:
    def __init__(self):
        self.customer_profiles = {
            "user123": {"name": "Alice", "email": "alice@example.com", "language": "en"},
            "user456": {"name": "Bob", "email": "bob@example.com", "language": "de"},
        }
        self.order_management_system = {
            "ORD789": {"customer_id": "user123", "status": "shipped", "items": ["Laptop"], "delivery_date": "2023-12-01"},
            "ORD012": {"customer_id": "user456", "status": "processing", "items": ["Mouse"]},
        }
        self.product_catalog = {
            "Laptop": {"price": 1200, "warranty": "1 year"},
            "Mouse": {"price": 25, "warranty": "6 months"},
        }
        self.conversation_history = {}

    def get_customer_info(self, customer_id: str): return self.customer_profiles.get(customer_id)
    def get_order_details(self, order_id: str): return self.order_management_system.get(order_id)
    def get_product_info(self, product_name: str): return self.product_catalog.get(product_name)
    def add_conversation_entry(self, customer_id: str, message: str, response: str):
        if customer_id not in self.conversation_history: self.conversation_history[customer_id] = []
        self.conversation_history[customer_id].append({"message": message, "response": response})
    def get_conversation_history(self, customer_id: str): return self.conversation_history.get(customer_id, [])

# --- 3. Core AI (Foundation Model) Layer ---

class LLMAgent:
    def __init__(self, knowledge_base: KnowledgeBase):
        self.knowledge_base = knowledge_base
        self.llm = pipeline("text-generation", model="gpt2") # Using a small LLM for demonstration

    def _recognize_intent(self, text: str) -> str:
        # Simple keyword-based intent recognition for demonstration
        text_lower = text.lower()
        if "order status" in text_lower or "where is my order" in text_lower:
            return "order_status_inquiry"
        elif "return" in text_lower or "damaged product" in text_lower:
            return "return_request"
        elif "problem" in text_lower or "issue" in text_lower:
            return "general_complaint"
        elif "hello" in text_lower or "hi" in text_lower:
            return "greeting"
        return "unknown_intent"

    def _generate_response(self, intent: str, context: dict) -> str:
        customer_id = context.get("customer_id")
        customer_info = self.knowledge_base.get_customer_info(customer_id) if customer_id else None
        customer_name = customer_info.get("name", "customer") if customer_info else "customer"

        if intent == "greeting":
            return f"Hello {customer_name}! How can I assist you today?"
        elif intent == "order_status_inquiry":
            order_id = context.get("order_id")
            if order_id:
                order_details = self.knowledge_base.get_order_details(order_id)
                if order_details:
                    return f"Your order {order_id} is currently {order_details['status']} and is expected by {order_details.get('delivery_date', 'an unknown date')}."
                else:
                    return f"I couldn't find details for order {order_id}. Can you please double-check the order number?"
            else:
                return "Please provide your order number so I can check its status."
        elif intent == "return_request":
            return "I can help you with a return. Could you please provide your order number and the reason for the return?"
        elif intent == "general_complaint":
            return "I apologize for the inconvenience. Can you please describe your issue in more detail?"
        return "I am not sure how to respond to that. Can you please rephrase your request or provide more details?"

    def process_request(self, text: str, customer_id: Optional[str] = None) -> str:
        # In a real system, LLM would handle more complex dialogue and context
        intent = self._recognize_intent(text)
        context = {"customer_id": customer_id, "raw_input": text}

        # Example of extracting order_id for order status inquiry (simplified)
        if intent == "order_status_inquiry":
            import re
            match = re.search(r"ORD(\d+)", text, re.IGNORECASE)
            if match: context["order_id"] = match.group(0)

        response = self._generate_response(intent, context)
        if customer_id: self.knowledge_base.add_conversation_entry(customer_id, text, response)
        return response

# --- 5. Orchestration and API Layer ---

class WorkflowManager:
    def __init__(self):
        self.asr_module = ASRModule()
        self.image_analysis_module = ImageAnalysisModule()
        self.translation_module = TranslationModule()
        self.text_preprocessing_module = TextPreprocessingModule()
        self.knowledge_base = KnowledgeBase()
        self.llm_agent = LLMAgent(self.knowledge_base)

    async def process_multimodal_input(self, customer_id: str, text_input: Optional[str] = None, audio_file: Optional[UploadFile] = None, image_file: Optional[UploadFile] = None):
        processed_text = ""
        original_language = "en" # Assume English by default, or try to detect from text_input
        customer_info = self.knowledge_base.get_customer_info(customer_id)
        if customer_info: original_language = customer_info.get("language", "en")

        if audio_file:
            # Save audio to a temporary file for SpeechRecognition
            with open("temp_audio.wav", "wb") as f:
                f.write(await audio_file.read())
            asr_text = self.asr_module.transcribe_audio("temp_audio.wav")
            processed_text += f" [AUDIO TRANSCRIPT]: {asr_text}"

        if image_file:
            image_bytes = await image_file.read()
            image_description = self.image_analysis_module.analyze_image(image_bytes)
            processed_text += f" [IMAGE DESCRIPTION]: {image_description}"

        if text_input:
            processed_text += f" [TEXT INPUT]: {text_input}"

        # Preprocess combined text
        preprocessed_text = self.text_preprocessing_module.preprocess_text(processed_text)

        # Translate to English for LLM processing if not already English
        translated_for_llm = self.translation_module.translate_to_english(preprocessed_text, original_language)

        # LLM Agent processes the translated text
        llm_response_en = self.llm_agent.process_request(translated_for_llm, customer_id)

        # Translate response back to original language
        final_response = self.translation_module.translate_from_english(llm_response_en, original_language)

        return {"response": final_response, "processed_text_for_llm": translated_for_llm}

app = FastAPI()
workflow_manager = WorkflowManager()

@app.post("/chat")
async def chat_with_assistant(
    customer_id: str = Form(...),
    text_input: Optional[str] = Form(None),
    audio_file: Optional[UploadFile] = File(None),
    image_file: Optional[UploadFile] = File(None),
):
    if not text_input and not audio_file and not image_file:
        return {"error": "At least one input (text, audio, or image) is required."}

    response_data = await workflow_manager.process_multimodal_input(customer_id, text_input, audio_file, image_file)
    return response_data
