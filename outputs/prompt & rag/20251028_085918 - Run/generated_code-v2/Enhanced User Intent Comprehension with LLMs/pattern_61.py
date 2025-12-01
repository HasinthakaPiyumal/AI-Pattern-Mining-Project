from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import re
from loguru import logger


logger.remove()
logger.add(lambda msg: print(msg.strip()), level="INFO", format="<level>{message}</level>")

# --- Pydantic Models ---
class Entity(BaseModel):
    type: str
    value: str

class ConversationState(BaseModel):
    user_id: str
    current_intent: Optional[str] = None
    extracted_entities: Dict[str, str] = {}
    conversation_history: List[str] = []
    clarification_needed: bool = False

class UserProfile(BaseModel):
    user_id: str
    name: str = "Guest"
    email: Optional[str] = None
    shipping_address: Optional[str] = None
    past_orders: List[str] = []
    preferences: Dict[str, Any] = {}

# --- NLU Module (nlu.py simulated) ---
class NLU:
    def __init__(self):
        self.intent_patterns = {
            "return_item": [r"return (?:an |my )?(item|product)", r"i want to return"],
            "check_shipping_status": [r"where is my order", r"shipping status", r"track my (?:order|package)"],
            "product_inquiry": [r"tell me about (.*)", r"what is (?:the |this )?(.*)"],
            "order_cancellation": [r"cancel my order", r"i want to cancel"],
            "account_help": [r"my account", r"help with account"],
            "multi_intent": [r"and also", r"as well as"]
        }
        self.entity_patterns = {
            "order_id": r"#?(\d{6,})",
            "product_name": r"(?:about|for|of)\s+(the\s+)?([\w\s-]+?)(?:\?|$)",
            "reason_for_return": r"(?:because of|due to)\s+([\w\s-]+)"
        }

    def predict_intent(self, text: str) -> List[str]:
        detected_intents = []
        lower_text = text.lower()
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, lower_text):
                    if intent == "multi_intent" and len(detected_intents) > 0:
                        detected_intents.append("multi_intent")
                        break
                    elif intent != "multi_intent":
                        detected_intents.append(intent)
        return list(set(detected_intents))

    def extract_entities(self, text: str) -> Dict[str, str]:
        extracted = {}
        lower_text = text.lower()
        for entity_type, pattern in self.entity_patterns.items():
            match = re.search(pattern, lower_text)
            if match:
                if entity_type == "product_name":
                    # Take the last group, which is often the product name
                    extracted[entity_type] = match.groups()[-1].strip()
                else:
                    extracted[entity_type] = match.group(1).strip()
        return extracted

    def process_query(self, query: str) -> Dict[str, Any]:
        intents = self.predict_intent(query)
        entities = self.extract_entities(query)

        is_ambiguous = False
        if len(intents) > 1 and "multi_intent" not in intents:
            is_ambiguous = True
        elif not intents:
            is_ambiguous = True

        return {"intents": intents, "entities": entities, "is_ambiguous": is_ambiguous}


# --- Tool Integration Module (tools.py simulated) ---
class Tools:
    def get_order_details(self, order_id: str) -> str:
        if order_id == "123456":
            return f"Order {order_id}: Status - Shipped, Items - Laptop, Estimated Delivery - Tomorrow."
        return f"Could not find details for order {order_id}."

    def initiate_return_process(self, order_id: str, reason: str) -> str:
        if order_id == "123456":
            return f"Return initiated for order {order_id} due to '{reason}'. A shipping label has been sent to your email."
        return f"Failed to initiate return for order {order_id}. Please check the order ID."

    def get_shipping_information(self, order_id: str) -> str:
        if order_id == "123456":
            return f"Order {order_id} was shipped via FedEx and is expected to arrive by tomorrow, 5 PM."
        return f"Shipping information for order {order_id} is not available."

    def get_product_catalog(self, product_name: str) -> str:
        if "laptop" in product_name.lower():
            return f"We have several laptops available, including the 'UltraBook Pro' and 'Gaming Beast X'."
        return f"Sorry, I couldn't find any products matching '{product_name}'."

    def update_user_address(self, user_id: str, new_address: str) -> str:
        return f"User {user_id}'s address has been updated to {new_address}. (Simulated)"

# --- Personalization Module (personalization.py simulated) ---
class PersonalizationManager:
    def __init__(self):
        self.user_profiles: Dict[str, UserProfile] = {
            "user_123": UserProfile(user_id="user_123", name="Alice", email="alice@example.com", shipping_address="123 Main St"),
            "user_456": UserProfile(user_id="user_456", name="Bob"),
        }

    def get_user_profile(self, user_id: str) -> UserProfile:
        return self.user_profiles.get(user_id, UserProfile(user_id=user_id))

    def update_user_profile(self, user_id: str, **kwargs):
        profile = self.user_profiles.get(user_id)
        if not profile:
            profile = UserProfile(user_id=user_id)
        for key, value in kwargs.items():
            setattr(profile, key, value)
        self.user_profiles[user_id] = profile

# --- Dialogue Management Module (dialogue_manager.py) ---
class DialogueManager:
    def __init__(self, tools: Tools, personalization_manager: PersonalizationManager):
        self.tools = tools
        self.personalization_manager = personalization_manager
        self.active_conversations: Dict[str, ConversationState] = {}

    def get_or_create_state(self, user_id: str) -> ConversationState:
        if user_id not in self.active_conversations:
            self.active_conversations[user_id] = ConversationState(user_id=user_id)
        return self.active_conversations[user_id]

    def update_state(self, user_id: str, new_intents: List[str], new_entities: Dict[str, str], is_ambiguous: bool):
        state = self.get_or_create_state(user_id)
        state.conversation_history.append(f"User: {new_intents} - {new_entities}")

        if is_ambiguous:
            state.clarification_needed = True
            state.current_intent = None
            state.extracted_entities.clear()
            logger.info(f"[{user_id}] Ambiguity detected.")
        elif len(new_intents) == 1:
            state.current_intent = new_intents[0]
            state.extracted_entities.update(new_entities)
            state.clarification_needed = False
            logger.info(f"[{user_id}] Intent: {state.current_intent}, Entities: {state.extracted_entities}")
        elif "multi_intent" in new_intents and len(new_intents) > 1:
            # Handle multi-intent by picking the first specific intent for now, or require clarification
            specific_intents = [i for i in new_intents if i != "multi_intent"]
            if specific_intents:
                state.current_intent = specific_intents[0] # Simplistic handling for multi-intent
            else:
                state.current_intent = None
            state.extracted_entities.update(new_entities)
            state.clarification_needed = True # Still need clarification for multi-intent
            logger.info(f"[{user_id}] Multi-intent detected: {new_intents}, Entities: {state.extracted_entities}")
        else:
            state.current_intent = None # No clear intent
            state.extracted_entities.clear()
            state.clarification_needed = True

    def generate_response(self, user_id: str) -> str:
        state = self.get_or_create_state(user_id)
        user_profile = self.personalization_manager.get_user_profile(user_id)

        if state.clarification_needed:
            if not state.current_intent and not state.extracted_entities:
                return "I'm not sure how to help with that. Could you please rephrase or be more specific?"
            elif state.current_intent == "multi_intent":
                return "I detected multiple requests. Could you please tell me which one you'd like to address first?"
            else:
                 return f"I need a bit more information to help with your request about {state.current_intent}. Can you provide more details?"

        if state.current_intent == "return_item":
            order_id = state.extracted_entities.get("order_id")
            reason = state.extracted_entities.get("reason_for_return", "not specified")
            if order_id:
                return self.tools.initiate_return_process(order_id, reason)
            return "To initiate a return, I need the order ID. Can you provide it?"
        
        elif state.current_intent == "check_shipping_status":
            order_id = state.extracted_entities.get("order_id")
            if order_id:
                return self.tools.get_shipping_information(order_id)
            return "To check shipping status, please provide your order ID."

        elif state.current_intent == "product_inquiry":
            product_name = state.extracted_entities.get("product_name")
            if product_name:
                return self.tools.get_product_catalog(product_name)
            return "What product are you interested in?"

        elif state.current_intent == "order_cancellation":
            order_id = state.extracted_entities.get("order_id")
            if order_id:
                return f"Are you sure you want to cancel order {order_id}? (Simulated: Cancellation would proceed here)"
            return "To cancel an order, please provide the order ID."

        elif state.current_intent == "account_help":
            return f"Hello {user_profile.name}! How can I help with your account today? (Simulated: Offer account specific tools)"
        
        return "I'm sorry, I don't understand that request. Can I help with something else?"

# --- Main Application/Orchestration (main.py) ---
class Chatbot:
    def __init__(self):
        self.nlu = NLU()
        self.tools = Tools()
        self.personalization_manager = PersonalizationManager()
        self.dialogue_manager = DialogueManager(self.tools, self.personalization_manager)
        logger.info("Chatbot initialized.")

    def start_conversation(self, user_id: str = "user_123"):
        logger.info(f"Starting conversation for user: {user_id}")
        print(f"Hello {self.personalization_manager.get_user_profile(user_id).name}! How can I help you today? (Type 'exit' to quit)")
        while True:
            user_input = input("> ").strip()
            if user_input.lower() == 'exit':
                print("Goodbye!")
                break
            
            if not user_input:
                continue

            nlu_result = self.nlu.process_query(user_input)
            intents = nlu_result["intents"]
            entities = nlu_result["entities"]
            is_ambiguous = nlu_result["is_ambiguous"]

            logger.info(f"[{user_id}] User Input: '{user_input}'")
            logger.info(f"[{user_id}] NLU Result: Intents={intents}, Entities={entities}, Ambiguous={is_ambiguous}")

            self.dialogue_manager.update_state(user_id, intents, entities, is_ambiguous)
            response = self.dialogue_manager.generate_response(user_id)
            print(response)

if __name__ == "__main__":
    chatbot = Chatbot()
    chatbot.start_conversation()
