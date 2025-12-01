from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import random

app = FastAPI()

# --- Pydantic Models ---
class UserQuery(BaseModel):
    text: str
    user_id: str

class AgentResponse(BaseModel):
    response: str
    intent: Optional[str] = None
    entities: Optional[Dict[str, Any]] = None
    action_taken: Optional[str] = None
    clarification_needed: bool = False
    conversation_state: Dict[str, Any]

# --- 1. Natural Language Understanding (NLU) Module ---
class NLUService:
    def __init__(self):
        # In a real application, load pre-trained/fine-tuned models here
        self.intents = ["order_status", "return_request", "product_inquiry", "shipping_info", "account_update", "greeting", "thank_you", "unknown"]
        self.entities_map = {
            "order_status": ["order_id"],
            "return_request": ["product_name", "reason"],
            "product_inquiry": ["product_name"],
            "shipping_info": ["order_id", "address"],
            "account_update": ["account_field", "new_value"]
        }

    def recognize_intent(self, text: str) -> Dict[str, Any]:
        # Mock intent recognition with some simple keyword matching or random choice
        text_lower = text.lower()
        if "hello" in text_lower or "hi" in text_lower:
            return {"intent": "greeting", "confidence": 0.95}
        if "thank you" in text_lower or "thanks" in text_lower:
            return {"intent": "thank_you", "confidence": 0.95}
        if "order" in text_lower and ("status" in text_lower or "track" in text_lower):
            return {"intent": "order_status", "confidence": 0.8}
        if "return" in text_lower and ("product" in text_lower or "item" in text_lower):
            return {"intent": "return_request", "confidence": 0.85}
        if "product" in text_lower and ("info" in text_lower or "details" in text_lower or "about" in text_lower):
            return {"intent": "product_inquiry", "confidence": 0.75}
        if "shipping" in text_lower or "delivery" in text_lower:
            return {"intent": "shipping_info", "confidence": 0.7}
        if "account" in text_lower or "profile" in text_lower:
            return {"intent": "account_update", "confidence": 0.65}

        # Fallback to random intent or unknown
        if random.random() < 0.2: # Simulate some unknown queries
            return {"intent": "unknown", "confidence": 0.6}
        return {"intent": random.choice([i for i in self.intents if i not in ["greeting", "thank_you", "unknown"]]), "confidence": random.uniform(0.5, 0.7)}

    def extract_entities(self, text: str, intent: str) -> Dict[str, str]:
        # Mock entity extraction
        entities = {}
        text_lower = text.lower()

        if intent == "order_status" or intent == "shipping_info":
            # Simple regex for a numerical order ID (e.g., #12345, order 12345)
            import re
            match = re.search(r'(?:order(?:_id)?|#)\s*(\d+)', text_lower)
            if match: 
                entities["order_id"] = match.group(1)
            else:
                # Look for a standalone 5-digit number as a potential order_id
                match_standalone = re.search(r'\b(\d{5,8})\b', text_lower)
                if match_standalone: entities["order_id"] = match_standalone.group(1)

        if intent == "return_request" or intent == "product_inquiry":
            # Mock product name extraction (e.g., 'the shirt', 'my phone')
            if "shirt" in text_lower: entities["product_name"] = "shirt"
            elif "phone" in text_lower: entities["product_name"] = "phone"
            elif "laptop" in text_lower: entities["product_name"] = "laptop"

        if intent == "return_request":
            if "damaged" in text_lower: entities["reason"] = "damaged"
            elif "wrong size" in text_lower: entities["reason"] = "wrong size"
            elif "don't like" in text_lower: entities["reason"] = "dislike"

        # Add more sophisticated entity extraction logic here for a real system
        return entities

    def detect_ambiguity(self, intent_result: Dict[str, Any], extracted_entities: Dict[str, str], required_entities: List[str]) -> bool:
        if intent_result["confidence"] < 0.7: # Low confidence intent
            return True
        
        for entity in required_entities:
            if entity not in extracted_entities:
                return True # Missing required entities
        return False

# --- Mock Backend APIs ---
class EcommerceAPIs:
    def get_order_status(self, order_id: str) -> Dict[str, str]:
        if order_id == "12345":
            return {"status": "Shipped", "estimated_delivery": "2023-11-20"}
        return {"status": "Not Found"}

    def initiate_return(self, product_name: str, reason: str) -> Dict[str, str]:
        if product_name and reason:
            return {"status": "Return initiated", "return_id": f"RET-{random.randint(1000, 9999)}"}
        return {"status": "Failed", "message": "Missing product name or reason"}

    def get_product_details(self, product_name: str) -> Dict[str, str]:
        if product_name == "shirt":
            return {"name": "Classic T-Shirt", "price": "$25", "description": "100% cotton"}
        if product_name == "phone":
            return {"name": "Smartphone X", "price": "$799", "description": "Latest model with AI camera"}
        return {"name": "N/A", "description": "Product not found"}

    def update_account_info(self, user_id: str, field: str, new_value: str) -> Dict[str, str]:
        return {"status": "Success", "message": f"User {user_id}'s {field} updated to {new_value}"}

# --- 4. Personalization Module ---
class PersonalizationService:
    def __init__(self):
        self.user_profiles = {}

    def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        return self.user_profiles.get(user_id, {"name": "Customer", "preferred_tone": "neutral", "past_intents": []})

    def update_user_profile(self, user_id: str, updates: Dict[str, Any]):
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {}
        self.user_profiles[user_id].update(updates)

    def adapt_response(self, response_text: str, user_profile: Dict[str, Any]) -> str:
        # Simple adaptation based on preferred_tone
        if user_profile.get("preferred_tone") == "friendly":
            return f"Hey {user_profile.get('name', 'there')}! {response_text.replace('Hello', 'Hi').replace('How can I help you?', 'How may I assist you today?')}"
        return response_text

# --- 2. Dialogue Management Module ---
class DialogueManager:
    def __init__(self, nlu: NLUService, apis: EcommerceAPIs, personalization: PersonalizationService):
        self.nlu = nlu
        self.apis = apis
        self.personalization = personalization
        self.conversation_states: Dict[str, Dict[str, Any]] = {}

    def get_conversation_state(self, user_id: str) -> Dict[str, Any]:
        return self.conversation_states.get(user_id, {"history": [], "current_intent": None, "pending_entities": {}})

    def update_conversation_state(self, user_id: str, updates: Dict[str, Any]):
        if user_id not in self.conversation_states:
            self.conversation_states[user_id] = {"history": [], "current_intent": None, "pending_entities": {}}
        self.conversation_states[user_id].update(updates)

    def process_query(self, user_id: str, query_text: str) -> AgentResponse:
        conv_state = self.get_conversation_state(user_id)
        user_profile = self.personalization.get_user_profile(user_id)

        # NLU
        intent_result = self.nlu.recognize_intent(query_text)
        current_intent = intent_result["intent"]
        confidence = intent_result["confidence"]

        # Update user profile with past intent
        user_profile["past_intents"].append(current_intent)
        self.personalization.update_user_profile(user_id, user_profile)

        # Entity Extraction
        required_entities_for_intent = self.nlu.entities_map.get(current_intent, [])
        extracted_entities = self.nlu.extract_entities(query_text, current_intent)

        # Merge with pending entities from previous turns if applicable
        if conv_state["current_intent"] == current_intent:
            extracted_entities.update(conv_state["pending_entities"])
            conv_state["pending_entities"] = {}

        # Ambiguity Detection
        clarification_needed = self.nlu.detect_ambiguity(intent_result, extracted_entities, required_entities_for_intent)

        response_text = "I'm sorry, I couldn't understand that. Can you please rephrase?"
        action_taken = None

        if clarification_needed:
            missing_entities = [e for e in required_entities_for_intent if e not in extracted_entities]
            if missing_entities:
                response_text = f"To help you with '{current_intent.replace('_', ' ')}', I need the following information: {', '.join(missing_entities)}. Can you provide that?"
                conv_state["pending_entities"] = {entity: None for entity in missing_entities} # Store what's pending
            elif confidence < 0.7: # Low intent confidence
                response_text = f"I'm not entirely sure I understood your request for '{current_intent.replace('_', ' ')}'. Could you please clarify or provide more details?"
            
            self.update_conversation_state(user_id, {"current_intent": current_intent, "pending_entities": conv_state["pending_entities"]})
            return AgentResponse(
                response=self.personalization.adapt_response(response_text, user_profile),
                intent=current_intent,
                entities=extracted_entities,
                clarification_needed=True,
                conversation_state=conv_state
            )

        # Action Fulfillment
        if current_intent == "order_status" and "order_id" in extracted_entities:
            status_data = self.apis.get_order_status(extracted_entities["order_id"])
            if status_data["status"] != "Not Found":
                response_text = f"Your order {extracted_entities['order_id']} is {status_data['status']} with estimated delivery on {status_data['estimated_delivery']}."
                action_taken = f"Checked order status for {extracted_entities['order_id']}"
            else:
                response_text = f"I couldn't find order {extracted_entities['order_id']}. Please double-check the ID."

        elif current_intent == "return_request" and "product_name" in extracted_entities and "reason" in extracted_entities:
            return_data = self.apis.initiate_return(extracted_entities["product_name"], extracted_entities["reason"])
            if return_data["status"] == "Return initiated":
                response_text = f"Return for {extracted_entities['product_name']} due to {extracted_entities['reason']} has been initiated. Your return ID is {return_data['return_id']}."
                action_taken = f"Initiated return for {extracted_entities['product_name']}"
            else:
                response_text = f"I encountered an issue initiating the return: {return_data['message']}"

        elif current_intent == "product_inquiry" and "product_name" in extracted_entities:
            product_data = self.apis.get_product_details(extracted_entities["product_name"])
            if product_data["name"] != "N/A":
                response_text = f"Here are some details for {product_data['name']}: Price - {product_data['price']}, Description - {product_data['description']}."
                action_taken = f"Retrieved details for {extracted_entities['product_name']}"
            else:
                response_text = f"I couldn't find details for a product named '{extracted_entities['product_name']}'."
        
        elif current_intent == "greeting":
            response_text = f"Hello {user_profile.get('name', 'there')}! How can I help you today?"

        elif current_intent == "thank_you":
            response_text = "You're welcome! Is there anything else I can assist you with?"

        elif current_intent == "account_update": # Simplified, would require more entity extraction
            response_text = "I can help with account updates. What specific field would you like to change and to what new value?"
            clarification_needed = True
            self.update_conversation_state(user_id, {"current_intent": current_intent, "pending_entities": {"account_field": None, "new_value": None}})

        # Reset current intent if action was taken successfully or no clarification needed
        if action_taken or not clarification_needed: # For cases where a response is generated without action (e.g., greeting)
            self.update_conversation_state(user_id, {"current_intent": None, "pending_entities": {}})
        else:
             self.update_conversation_state(user_id, {"current_intent": current_intent, "pending_entities": conv_state["pending_entities"]})

        # Final response generation and personalization
        final_response = self.personalization.adapt_response(response_text, user_profile)
        
        # Update conversation history
        conv_state["history"].append({"user": query_text, "agent": final_response})
        self.update_conversation_state(user_id, conv_state)

        return AgentResponse(
            response=final_response,
            intent=current_intent,
            entities=extracted_entities,
            action_taken=action_taken,
            clarification_needed=clarification_needed,
            conversation_state=conv_state
        )

# --- Initialization ---
nlu_service = NLUService()
ecommerce_apis = EcommerceAPIs()
personalization_service = PersonalizationService()
dialogue_manager = DialogueManager(nlu_service, ecommerce_apis, personalization_service)

# --- 5. Deployment & API Layer (FastAPI) ---
@app.post("/chat", response_model=AgentResponse)
async def chat_with_agent(query: UserQuery):
    # Simulate setting a user's preferred tone for demonstration
    if query.user_id == "user123" and not personalization_service.get_user_profile(query.user_id).get("preferred_tone"): # Only set once if not already set
        personalization_service.update_user_profile(query.user_id, {"name": "Alice", "preferred_tone": "friendly"})
    
    response = dialogue_manager.process_query(query.user_id, query.text)
    return response

@app.get("/status")
async def get_status():
    return {"status": "Smart Customer Support Agent is running!"}

# To run this file:
# 1. Save it as `smart_customer_support_agent.py`
# 2. Install dependencies: `pip install fastapi uvicorn pydantic transformers` (transformers is just a placeholder, not actively used here for NLU model loading)
# 3. Run from your terminal: `uvicorn smart_customer_support_agent:app --reload`
# 4. Access the API at http://127.0.0.1:8000/docs for Swagger UI.