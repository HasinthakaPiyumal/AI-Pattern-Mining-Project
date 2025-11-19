import speech_recognition as sr
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
import io
import re
import os

# --- Module: speech_recognizer.py (Integrated) ---
def recognize_speech_from_audio(audio_file_path: str) -> str:
    r = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file_path) as source:
            audio_data = r.record(source)
            text = r.recognize_google(audio_data)
            return text
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError as e:
        return f"Could not request results from Google Speech Recognition service; {e}"
    except FileNotFoundError:
        return "Audio file not found."

# --- Module: image_analyzer.py (Integrated) ---
class ImageAnalyzer:
    def __init__(self, model_name="Salesforce/blip-image-captioning-base"):
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)

    def analyze_image(self, image_bytes: bytes) -> str:
        try:
            raw_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            inputs = self.processor(raw_image, return_tensors="pt")
            out = self.model.generate(**inputs)
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            return caption
        except Exception as e:
            return f"Error analyzing image: {e}"

# --- Module: translator.py (Integrated) ---
class Translator:
    def __init__(self, src_lang: str = "es", tgt_lang: str = "en"):
        self.model_name_src_to_tgt = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
        self.model_name_tgt_to_src = f"Helsinki-NLP/opus-mt-{tgt_lang}-{src_lang}"
        
        self.translator_src_to_tgt = pipeline("translation", model=self.model_name_src_to_tgt)
        self.translator_tgt_to_src = pipeline("translation", model=self.model_name_tgt_to_src)

    def translate_to_english(self, text: str) -> str:
        try:
            translated_text = self.translator_src_to_tgt(text, max_length=512)[0]["translation_text"]
            return translated_text
        except Exception as e:
            return f"Translation to English failed: {e}"

    def translate_from_english(self, text: str) -> str:
        try:
            translated_text = self.translator_tgt_to_src(text, max_length=512)[0]["translation_text"]
            return translated_text
        except Exception as e:
            return f"Translation from English failed: {e}"

# --- Module: nlu_module.py (Integrated) ---
class NLUModule:
    def __init__(self, llm_model_name="distilbert-base-uncased-mnli"):
        self.intent_classifier = pipeline("zero-shot-classification", model=llm_model_name)
        self.candidate_intents = [
            "check order status", "return item", "report product defect", 
            "ask for product information", "delivery inquiry", "account issue",
            "general greeting", "unknown"
        ]

    def get_intent(self, text: str) -> str:
        if not text.strip():
            return "unknown"
        results = self.intent_classifier(text, self.candidate_intents, multi_label=False)
        return results["labels"][0]

    def extract_entities(self, text: str) -> dict:
        entities = {}
        order_number_match = re.search(r"(?:ORD|order number\s*|#)([A-Za-z0-9\-]+)", text, re.IGNORECASE)
        if order_number_match:
            entities["order_number"] = order_number_match.group(1)

        product_keywords = ["phone", "laptop", "headphone", "shirt", "shoes"]
        for keyword in product_keywords:
            if keyword in text.lower():
                entities["product_name"] = keyword
                break
        return entities

    def handle_clarification(self, intent: str, entities: dict) -> str or None:
        if intent == "check order status" and "order_number" not in entities:
            return "Could you please provide your order number?"
        if intent == "return item" and ("product_name" not in entities and "order_number" not in entities):
            return "What product would you like to return, or do you have an order number?"
        if intent == "report product defect" and "product_name" not in entities:
            return "Which product are you experiencing an issue with?"
        return None

# --- Module: ecommerce_api_client.py (Integrated) ---
class EcommerceAPIClient:
    def __init__(self):
        self.mock_orders = {
            "ORD12345": {"status": "Shipped", "items": ["Laptop X"], "delivery_date": "2023-11-15"},
            "ORD67890": {"status": "Processing", "items": ["Headphones Y"], "delivery_date": "Pending"}
        }
        self.mock_products = {
            "laptop": {"price": "$1200", "warranty": "1 year", "description": "High-performance laptop."},
            "headphones": {"price": "$150", "warranty": "6 months", "description": "Noise-cancelling headphones."}, 
            "phone": {"price": "$800", "warranty": "1 year", "description": "Latest smartphone model."}           
        }

    def get_order_status(self, order_number: str) -> dict or None:
        return self.mock_orders.get(order_number.upper())

    def request_return(self, order_number: str, product_name: str) -> bool:
        if order_number.upper() in self.mock_orders and product_name.lower() in [item.lower() for item in self.mock_orders[order_number.upper()]["items"]]:
            print(f"Simulating return request for {product_name} in order {order_number}.")
            return True
        return False

    def get_product_info(self, product_name: str) -> dict or None:
        return self.mock_products.get(product_name.lower())

    def escalate_to_human_agent(self, reason: str) -> str:
        return f"Escalating to a human agent for: {reason}. Please wait for a moment."

# --- Main Application: app.py ---
class MultimodalCustomerAssistant:
    def __init__(self):
        self.image_analyzer = ImageAnalyzer()
        self.translator_es_en = Translator(src_lang="es", tgt_lang="en")
        self.translator_fr_en = Translator(src_lang="fr", tgt_lang="en")
        self.nlu_module = NLUModule()
        self.ecommerce_client = EcommerceAPIClient()
        self.current_context = {}
        print("Assistant initialized. Supported commands: text, audio <file>, image <file>, quit")

    def process_text_input(self, text: str, original_lang: str = "en") -> str:
        if original_lang != "en":
            if original_lang == "es":
                text = self.translator_es_en.translate_to_english(text)
            elif original_lang == "fr":
                text = self.translator_fr_en.translate_to_english(text)
            print(f"Translated to English: {text}")
            
        intent = self.nlu_module.get_intent(text)
        entities = self.nlu_module.extract_entities(text)

        self.current_context.update(entities)
        print(f"Detected Intent: {intent}, Extracted Entities: {self.current_context}")

        clarification_needed = self.nlu_module.handle_clarification(intent, self.current_context)

        if clarification_needed:
            response = clarification_needed
        elif intent == "check order status":
            order_number = self.current_context.get("order_number")
            if order_number:
                order_info = self.ecommerce_client.get_order_status(order_number)
                if order_info:
                    response = f"Your order {order_number} status is: {order_info["status"]}. Estimated delivery: {order_info["delivery_date"]}."
                else:
                    response = f"I couldn't find order {order_number}. Please double-check the number."
            else:
                response = "Please provide your order number to check its status."
        elif intent == "return item":
            order_number = self.current_context.get("order_number")
            product_name = self.current_context.get("product_name")
            if order_number and product_name:
                if self.ecommerce_client.request_return(order_number, product_name):
                    response = f"Return for {product_name} in order {order_number} has been initiated. You will receive an email with instructions."
                else:
                    response = "I couldn't initiate the return. Please ensure the order and product are correct."
            else:
                response = "To initiate a return, I need the order number and the product name."
        elif intent == "report product defect":
            product_name = self.current_context.get("product_name")
            if product_name:
                response = self.ecommerce_client.escalate_to_human_agent(f"Product defect reported for {product_name}")
            else:
                response = "Which product are you reporting a defect for?"
        elif intent == "ask for product information":
            product_name = self.current_context.get("product_name")
            if product_name:
                product_info = self.ecommerce_client.get_product_info(product_name)
                if product_info:
                    response = f"Information for {product_name.capitalize()}: Price {product_info["price"]}, Warranty {product_info["warranty"]}. Description: {product_info["description"]}"
                else:
                    response = f"I couldn't find information for {product_name}."
            else:
                response = "What product are you interested in?"
        elif intent == "delivery inquiry":
            response = self.ecommerce_client.escalate_to_human_agent("Delivery inquiry")
        elif intent == "general greeting":
            response = "Hello! How can I assist you with your e-commerce needs today?"
        else:
            response = self.ecommerce_client.escalate_to_human_agent("Unrecognized or complex query")

        if original_lang != "en":
            if original_lang == "es":
                response = self.translator_es_en.translate_from_english(response)
            elif original_lang == "fr":
                response = self.translator_fr_en.translate_from_english(response)
        
        return response

    def process_audio_input(self, audio_file_path: str, original_lang: str = "en") -> str:
        print(f"Processing audio file: {audio_file_path}")
        text = recognize_speech_from_audio(audio_file_path)
        if text.startswith("Could not") or text.startswith("Audio file not found"):
            return text
        print(f"Audio Transcribed: {text}")
        return self.process_text_input(text, original_lang)

    def process_image_input(self, image_file_path: str) -> str:
        print(f"Processing image file: {image_file_path}")
        try:
            with open(image_file_path, "rb") as f:
                image_bytes = f.read()
            image_description = self.image_analyzer.analyze_image(image_bytes)
            print(f"Image Described as: {image_description}")
            if image_description.startswith("Error analyzing image"):
                return image_description
            return self.process_text_input(f"The user provided an image showing: {image_description}. How can I help with this?")
        except FileNotFoundError:
            return "Image file not found."
        except Exception as e:
            return f"Error processing image input: {e}"

    def run(self):
        while True:
            user_input = input("\nEnter your query (e.g., 'text What is my order?', 'audio hello.wav', 'image product.png', 'quit'): ")
            if user_input.lower() == "quit":
                print("Exiting assistant. Goodbye!")
                break

            parts = user_input.split(maxsplit=2)
            if len(parts) < 2:
                print("Invalid input format. Use 'text <query>', 'audio <file>', 'image <file>', or 'quit'.")
                continue

            input_type = parts[0].lower()
            content = parts[1] if len(parts) == 2 else parts[2]
            original_lang = "en"
            
            if input_type == "text":
                if content.lower().startswith("hola") or content.lower().startswith("mi pedido"):
                    original_lang = "es"
                elif content.lower().startswith("bonjour") or content.lower().startswith("je voudrais"):
                    original_lang = "fr"
                print(f"User ({original_lang}): {content}")
                response = self.process_text_input(content, original_lang)
            elif input_type == "audio":
                print(f"User (audio file: {content})")
                response = self.process_audio_input(content, original_lang) # Lang detection for audio is complex, defaulting to en for now
            elif input_type == "image":
                print(f"User (image file: {content})")
                response = self.process_image_input(content)
            else:
                response = "Unsupported input type. Please use 'text', 'audio', or 'image'."
            
            print(f"Assistant: {response}")
            self.current_context = {}

if __name__ == "__main__":
    assistant = MultimodalCustomerAssistant()
    assistant.run()