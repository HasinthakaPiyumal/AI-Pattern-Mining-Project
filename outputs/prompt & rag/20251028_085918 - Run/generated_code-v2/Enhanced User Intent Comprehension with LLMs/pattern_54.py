import torch
from transformers import pipeline
from fastapi import FastAPI
from pydantic import BaseModel

class IntentClassifier:
    def __init__(self):
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0 if torch.cuda.is_available() else -1)
        self.candidate_intents = [
            "order modification",
            "shipping inquiry",
            "product defect",
            "return request",
            "general inquiry",
            "account issue",
            "payment problem"
        ]

    def predict_intent(self, query: str) -> str:
        result = self.classifier(query, self.candidate_intents)
        predicted_intent = result["labels"][0].replace(" ", "_")
        return predicted_intent

class DialogueManager:
    def __init__(self):
        self.conversation_state = {}

    def handle_query(self, user_id: str, detected_intent: str, query: str) -> str:
        if user_id not in self.conversation_state:
            self.conversation_state[user_id] = {"awaiting_clarification": False}

        if self.conversation_state[user_id]["awaiting_clarification"]:
            self.conversation_state[user_id]["awaiting_clarification"] = False
            return f"Thank you for clarifying. Your request for \'{detected_intent}\' is being processed."
        
        if detected_intent == "general_inquiry" and len(query.split()) < 5:
            self.conversation_state[user_id]["awaiting_clarification"] = True
            return "Your query is a bit general. Could you please provide more details?"

        return f"Acknowledged: \'{detected_intent}\'."

class ActionRouter:
    def __init__(self):
        self.actions = {
            "order_modification": "Initiating order modification process. Please confirm your order ID.",
            "shipping_inquiry": "Let me check the shipping status for your latest order.",
            "product_defect": "Connecting you to a product specialist for your reported defect.",
            "return_request": "To process your return, please provide the item details and reason.",
            "general_inquiry": "I\'m here to help with any general questions you have.",
            "account_issue": "Please verify your account details so I can assist with account issues.",
            "payment_problem": "I can help with payment problems. What seems to be the issue?",
            "default": "I\'m sorry, I couldn\'t fully understand your request. Please try rephrasing."
        }

    def route_action(self, intent: str) -> str:
        return self.actions.get(intent, self.actions["default"])

app = FastAPI(title="E-commerce Customer Support AI Assistant")

intent_classifier = IntentClassifier()
dialogue_manager = DialogueManager()
action_router = ActionRouter()

class ChatRequest(BaseModel):
    user_id: str
    query: str

class ChatResponse(BaseModel):
    response: str
    intent_detected: str = None
    action_taken: str = None

@app.post("/chat", response_model=ChatResponse)
async def chat_with_assistant(request: ChatRequest):
    detected_intent = intent_classifier.predict_intent(request.query)
    
    dialogue_response_part = dialogue_manager.handle_query(request.user_id, detected_intent, request.query)
    
    if "clarify" in dialogue_response_part.lower():
        final_response = dialogue_response_part
        action_response_part = None
    else:
        action_response_part = action_router.route_action(detected_intent)
        final_response = f"{dialogue_response_part} {action_response_part}"

    return ChatResponse(
        response=final_response,
        intent_detected=detected_intent,
        action_taken=action_response_part
    )