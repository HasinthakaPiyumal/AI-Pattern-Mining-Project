from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import re

app = FastAPI()

# --- 1. Natural Language Understanding (NLU) Module ---

class NLU: 
    def process_query(self, query: str):
        query = query.lower()
        intent = "unknown"
        entities = {}

        if "track order" in query or "where is my package" in query or "order status" in query:
            intent = "track_order"
            match = re.search(r"order number (\d+)|order (\d+)|#(\d+)", query)
            if match: 
                entities["order_id"] = match.group(1) or match.group(2) or match.group(3)
        elif "return" in query or "send back" in query or "refund" in query:
            intent = "initiate_return"
            match = re.search(r"item (\w+\s?\w+)", query)
            if match:
                entities["item_name"] = match.group(1)
            if "damaged" in query: 
                entities["reason"] = "damaged"
            elif "wrong item" in query:
                entities["reason"] = "wrong item"
            elif "changed mind" in query:
                entities["reason"] = "changed mind"
        elif "change address" in query or "update shipping" in query:
            intent = "update_address"
            match = re.search(r"customer id (\d+)", query)
            if match:
                entities["customer_id"] = match.group(1)
            match = re.search(r"to (.+)", query) # Simple regex for new address
            if match:
                entities["new_address"] = match.group(1)
        elif "billing inquiry" in query or "charge" in query or "invoice" in query:
            intent = "billing_inquiry"

        return {"intent": intent, "entities": entities}

# --- 2. Dialogue Management Module ---

class SessionData(BaseModel):
    history: list = []
    current_intent: str = None
    required_entities: dict = {}
    collected_entities: dict = {}
    last_response: str = ""

sessions = {}

class DialogueManager:
    def __init__(self):
        self.nlu = NLU()

    def get_next_action(self, session_id: str, user_query: str):
        session = sessions.get(session_id, SessionData())
        session.history.append(f"User: {user_query}")
        
        nlu_output = self.nlu.process_query(user_query)
        intent = nlu_output["intent"]
        entities = nlu_output["entities"]

        response = ""
        tool_call = None

        # Update collected entities for the current session or new intent
        if intent != "unknown" or session.current_intent:
            current_effective_intent = intent if intent != "unknown" else session.current_intent
            session.current_intent = current_effective_intent
            session.collected_entities.update(entities)

            if current_effective_intent == "track_order":
                session.required_entities = {"order_id": "order number"}
            elif current_effective_intent == "initiate_return":
                session.required_entities = {"item_name": "item name", "reason": "reason for return"}
            elif current_effective_intent == "update_address":
                session.required_entities = {"customer_id": "customer ID", "new_address": "new address"}
            elif current_effective_intent == "billing_inquiry":
                session.required_entities = {}

            missing_entities = [entity_key for entity_key, prompt in session.required_entities.items() if entity_key not in session.collected_entities]
            
            if not missing_entities and current_effective_intent != "unknown":
                # All required entities collected, perform tool call
                if current_effective_intent == "track_order":
                    tool_call = {"tool_name": "track_order_api", "parameters": {"order_id": session.collected_entities["order_id"]}}
                elif current_effective_intent == "initiate_return":
                    tool_call = {"tool_name": "initiate_return_api", "parameters": {"order_id": "mock_order_id", "item_name": session.collected_entities["item_name"], "reason": session.collected_entities["reason"]}}
                elif current_effective_intent == "update_address":
                    tool_call = {"tool_name": "update_address_api", "parameters": {"customer_id": session.collected_entities["customer_id"], "new_address": session.collected_entities["new_address"]}}
                elif current_effective_intent == "billing_inquiry":
                    response = "For billing inquiries, please visit our billing section or contact our finance department directly."
                    session.current_intent = None # Reset intent after fulfilling
                    session.required_entities = {}
                    session.collected_entities = {}
            elif current_effective_intent != "unknown":
                # Ask for missing entities
                response = f"I can help with {current_effective_intent.replace('_', ' ')}. Could you please provide the {session.required_entities[missing_entities[0]]}?"
            else:
                response = "I'm not sure how to help with that. Could you please rephrase or tell me more specifically what you need?"
        else:
            response = "I'm not sure how to help with that. Could you please rephrase or tell me more specifically what you need?"

        session.last_response = response
        session.history.append(f"Assistant: {response}")
        sessions[session_id] = session
        return response, tool_call, session

# --- 3. Backend Integration Module (Simulated) ---

class BackendIntegrator:
    def track_order_api(self, order_id: str):
        if order_id == "12345":
            return {"status": "success", "message": f"Order {order_id} is currently in transit and expected by 2023-12-31."}
        else:
            return {"status": "error", "message": f"Order {order_id} not found."}

    def initiate_return_api(self, order_id: str, item_name: str, reason: str):
        return {"status": "success", "message": f"Return for item '{item_name}' (Order: {order_id}) due to '{reason}' initiated. Instructions sent to your email."}

    def update_address_api(self, customer_id: str, new_address: str):
        return {"status": "success", "message": f"Shipping address for customer {customer_id} updated to '{new_address}'."}

    def execute_tool_call(self, tool_call: dict):
        tool_name = tool_call["tool_name"]
        parameters = tool_call["parameters"]

        if hasattr(self, tool_name):
            tool_function = getattr(self, tool_name)
            return tool_function(**parameters)
        else:
            return {"status": "error", "message": f"Unknown tool: {tool_name}"}

# --- API Layer (FastAPI) ---

dialogue_manager = DialogueManager()
backend_integrator = BackendIntegrator()

class ChatRequest(BaseModel):
    user_query: str
    session_id: str

@app.post("/chat")
async def chat(request: ChatRequest):
    assistant_response, tool_call, session = dialogue_manager.get_next_action(request.session_id, request.user_query)
    
    if tool_call:
        tool_result = backend_integrator.execute_tool_call(tool_call)
        if tool_result["status"] == "success":
            assistant_response = tool_result["message"]
        else:
            assistant_response = f"I encountered an issue: {tool_result['message']}"
        
        # Reset session after successful tool call
        session.current_intent = None
        session.required_entities = {}
        session.collected_entities = {}
        sessions[request.session_id] = session

    return {"assistant_response": assistant_response, "session_id": request.session_id}

