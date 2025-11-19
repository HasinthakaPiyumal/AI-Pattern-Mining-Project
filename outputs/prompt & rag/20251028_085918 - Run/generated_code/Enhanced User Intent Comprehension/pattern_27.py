class SpeechToTextModule:
    def convert_speech_to_text(self, audio_input):
        if "order status" in audio_input.lower():
            return "What is the status of my order?"
        if "product recommendation" in audio_input.lower():
            return "I need a product recommendation for a gift."
        return "Simulated speech input."

class ImageAnalysisModule:
    def analyze_image(self, image_data):
        if "shoes" in image_data.lower():
            return {"detected_objects": ["shoes"], "description": "User uploaded an image of shoes."}
        if "apparel" in image_data.lower():
            return {"detected_objects": ["clothing"], "description": "User uploaded an image of clothing."}
        return {"detected_objects": [], "description": "No specific objects detected in image."}

class MachineTranslationModule:
    def translate_text(self, text, target_language="en", source_language="auto"):
        if "hola" in text.lower():
            return "hello"
        if "gracias" in text.lower():
            return "thank you"
        if "livraison" in text.lower():
            return "delivery"
        return text

class TextPreprocessingModule:
    def preprocess_text(self, text):
        text = text.lower()
        processed_chars = []
        for char in text:
            if char.isalnum() or char.isspace():
                processed_chars.append(char)
        return "".join(processed_chars)

class LLMService:
    def __init__(self):
        self.intents = {
            "order_status": ["order status", "my order", "where is my package", "track shipment"],
            "product_inquiry": ["product information", "tell me about", "details on", "what is this"],
            "return_request": ["return item", "how to return", "exchange product"],
            "recommendation": ["recommend", "suggest a product", "gift idea"],
            "greeting": ["hello", "hi", "hey"],
            "thank_you": ["thank you", "thanks"],
            "clarification_request": ["can you repeat", "clarify", "what do you mean"]
        }
        self.entity_keywords = {
            "order_id": ["order", "number"],
            "product_name": ["shoe", "shirt", "laptop", "book", "watch", "product"],
            "issue_description": ["damaged", "wrong size", "not working"]
        }

    def _match_intent(self, text):
        for intent, keywords in self.intents.items():
            for keyword in keywords:
                if keyword in text:
                    return intent
        return "general_inquiry"

    def _extract_entities(self, text):
        extracted = {}
        words = text.split()

        for i, word in enumerate(words):
            if word == "order" or word == "number":
                if i + 1 < len(words) and words[i+1].isdigit():
                    extracted["order_id"] = words[i+1]
                    break
        
        for keyword in self.entity_keywords["product_name"]:
            if keyword in text:
                extracted["product_name"] = keyword
                break
        
        for keyword in self.entity_keywords["issue_description"]:
            if keyword in text:
                extracted["issue_description"] = keyword
                break

        return extracted

    def process_query(self, processed_text, dialogue_state):
        intent = self._match_intent(processed_text)
        entities = self._extract_entities(processed_text)

        dialogue_state["current_intent"] = intent
        dialogue_state["entities"].update(entities)

        return {
            "intent": intent,
            "entities": entities,
            "clarification_needed": False
        }

    def generate_response(self, intent, entities, dialogue_state):
        if intent == "order_status":
            order_id = entities.get("order_id") or dialogue_state["entities"].get("order_id")
            if order_id:
                return f"Let me check the status for order {order_id}."
            else:
                return "Could you please provide your order ID?"
        elif intent == "product_inquiry":
            product_name = entities.get("product_name") or dialogue_state["entities"].get("product_name")
            if product_name:
                return f"Certainly, what would you like to know about the {product_name}?"
            else:
                return "Which product are you interested in?"
        elif intent == "return_request":
            return "Please provide your order number and the reason for the return."
        elif intent == "recommendation":
            return "I can help with recommendations. What type of product are you looking for?"
        elif intent == "greeting":
            return "Hello! How can I assist you today?"
        elif intent == "thank_you":
            return "You're welcome! Is there anything else?"
        elif intent == "clarification_request":
            return "Could you please rephrase your request or provide more details?"
        else:
            return "I'm sorry, I couldn't fully understand your request. Can you please elaborate?"


class DialogueManager:
    def __init__(self):
        self.session_states = {}

    def get_session_state(self, user_id):
        if user_id not in self.session_states:
            self.session_states[user_id] = {
                "current_intent": "none",
                "entities": {},
                "conversation_history": []
            }
        return self.session_states[user_id]

    def update_session_state(self, user_id, intent, entities, user_message, assistant_response):
        state = self.get_session_state(user_id)
        state["current_intent"] = intent
        state["entities"].update(entities)
        state["conversation_history"].append({"user": user_message, "assistant": assistant_response})

class EcommerceAPIConnector:
    def get_order_details(self, order_id):
        if order_id == "12345":
            return {"status": "shipped", "estimated_delivery": "2024-08-15"}
        return {"status": "not found", "estimated_delivery": "N/A"}

    def get_product_catalog(self, product_name):
        if "shoe" in product_name.lower():
            return {"name": "Running Shoe X", "price": 120.00, "availability": "in stock"}
        elif "shirt" in product_name.lower():
            return {"name": "Blue T-shirt", "price": 25.00, "availability": "in stock", "colors": ["red", "blue", "green"]}
        return {"name": product_name, "price": "N/A", "availability": "unknown"}

class KnowledgeBase:
    def search_faq(self, query):
        if "return policy" in query.lower():
            return "Our return policy allows returns within 30 days of purchase."
        if "shipping fees" in query.lower():
            return "Shipping fees vary based on location and speed."
        return "No relevant FAQ found for your query."

class ClarificationModule:
    def needs_clarification(self, llm_output, session_state):
        if llm_output["intent"] == "order_status" and not (llm_output["entities"].get("order_id") or session_state["entities"].get("order_id")):
            return True, "Could you please provide your order ID?"
        if llm_output["intent"] == "product_inquiry" and not (llm_output["entities"].get("product_name") or session_state["entities"].get("product_name")):
            return True, "Which product are you interested in?"
        return False, None

class TextToSpeechModule:
    def convert_text_to_speech(self, text):
        return f"Simulated audio for: '{text}'"

class MultimodalAssistant:
    def __init__(self):
        self.stt_module = SpeechToTextModule()
        self.image_analysis_module = ImageAnalysisModule()
        self.translation_module = MachineTranslationModule()
        self.preprocessing_module = TextPreprocessingModule()
        self.llm_service = LLMService()
        self.dialogue_manager = DialogueManager()
        self.ecommerce_api = EcommerceAPIConnector()
        self.knowledge_base = KnowledgeBase()
        self.clarification_module = ClarificationModule()
        self.tts_module = TextToSpeechModule()

    def process_input(self, user_id, text_input=None, audio_input=None, image_input=None, lang_code="en"):
        processed_text = ""
        image_analysis_results = {}
        original_user_input = text_input or audio_input or image_input

        if audio_input:
            processed_text = self.stt_module.convert_speech_to_text(audio_input)
        elif text_input:
            processed_text = text_input

        if image_input:
            image_analysis_results = self.image_analysis_module.analyze_image(image_input)
            if image_analysis_results.get("description"):
                processed_text += " " + image_analysis_results["description"]

        if lang_code != "en":
            processed_text = self.translation_module.translate_text(processed_text, target_language="en", source_language=lang_code)

        processed_text_for_llm = self.preprocessing_module.preprocess_text(processed_text)

        session_state = self.dialogue_manager.get_session_state(user_id)
        llm_output = self.llm_service.process_query(processed_text_for_llm, session_state)

        response_text = ""
        clarification_needed, clarification_message = self.clarification_module.needs_clarification(llm_output, session_state)

        if clarification_needed:
            response_text = clarification_message
        else:
            if llm_output["intent"] == "order_status":
                order_id = llm_output["entities"].get("order_id") or session_state["entities"].get("order_id")
                if order_id:
                    order_details = self.ecommerce_api.get_order_details(order_id)
                    response_text = f"Order {order_id} status: {order_details['status']}. Estimated delivery: {order_details['estimated_delivery']}."
                else:
                    response_text = self.llm_service.generate_response(llm_output["intent"], llm_output["entities"], session_state)
            elif llm_output["intent"] == "product_inquiry":
                product_name = llm_output["entities"].get("product_name") or session_state["entities"].get("product_name")
                if product_name:
                    product_info = self.ecommerce_api.get_product_catalog(product_name)
                    response_text = f"The {product_info.get('name', product_name)} is available for ${product_info.get('price', 'N/A')} and is {product_info.get('availability', 'unknown')}."
                    if "colors" in product_info and "what colors" in processed_text_for_llm:
                        response_text += f" It is available in colors: {', '.join(product_info['colors'])}."
                else:
                    response_text = self.llm_service.generate_response(llm_output["intent"], llm_output["entities"], session_state)
            elif llm_output["intent"] in ["return_request", "recommendation", "greeting", "thank_you", "general_inquiry"]:
                response_text = self.llm_service.generate_response(llm_output["intent"], llm_output["entities"], session_state)
            else:
                 faq_response = self.knowledge_base.search_faq(processed_text_for_llm)
                 if "No relevant FAQ" not in faq_response:
                     response_text = faq_response
                 else:
                     response_text = self.llm_service.generate_response(llm_output["intent"], llm_output["entities"], session_state)


        self.dialogue_manager.update_session_state(user_id, llm_output["intent"], llm_output["entities"], original_user_input, response_text)

        if lang_code != "en":
            final_response_lang = self.translation_module.translate_text(response_text, target_language=lang_code, source_language="en")
            response_text = final_response_lang

        audio_response = self.tts_module.convert_text_to_speech(response_text)

        return {"text_response": response_text, "audio_response": audio_response, "session_state": session_state}


if __name__ == "__main__":
    assistant = MultimodalAssistant()
    user_id_1 = "user123"

    print("\n--- Scenario 1: Order Status via Text ---")
    output = assistant.process_input(user_id_1, text_input="What is the status of my order 12345?")
    print(f"Assistant: {output['text_response']}")

    print("\n--- Scenario 2: Product Inquiry via Voice (simulated) ---")
    output = assistant.process_input(user_id_1, audio_input="I need information about a running shoe.")
    print(f"Assistant: {output['text_response']}")

    print("\n--- Scenario 3: Multilingual Greeting ---")
    output = assistant.process_input(user_id_1, text_input="Hola!", lang_code="es")
    print(f"Assistant: {output['text_response']}")

    print("\n--- Scenario 4: Image of apparel (simulated) and a follow-up question ---")
    output = assistant.process_input(user_id_1, image_input="User sees an image of a blue shirt")
    print(f"Assistant (initial image analysis): {output['text_response']}")
    output = assistant.process_input(user_id_1, text_input="What colors is this available in?")
    print(f"Assistant (follow-up on product from image): {output['text_response']}")


    print("\n--- Scenario 5: Ambiguous query, requiring clarification ---")
    output = assistant.process_input(user_id_1, text_input="I want to return something.")
    print(f"Assistant: {output['text_response']}")

    print("\n--- Scenario 6: Follow-up on clarification (providing order ID) ---")
    output = assistant.process_input(user_id_1, text_input="My order number is 987654321 and it's damaged.")
    print(f"Assistant: {output['text_response']}")

    print("\n--- Scenario 7: Knowledge Base Query ---")
    output = assistant.process_input(user_id_1, text_input="What is your return policy?")
    print(f"Assistant: {output['text_response']}")