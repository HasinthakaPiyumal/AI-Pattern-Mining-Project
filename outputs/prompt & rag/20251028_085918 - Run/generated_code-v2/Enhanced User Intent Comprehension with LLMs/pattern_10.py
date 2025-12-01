from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

# Pydantic Models
class ChatMessage(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    intent: str = "unknown"
    confidence: float = 0.0

# Mock Knowledge Base/Tool Interface
class MockKnowledgeBase:
    def get_response(self, intent: str, user_query: str) -> str:
        responses = {
            "Order Status": f"To check your order status, please provide your order number. (You asked about: {user_query})",
            "Product Information": f"Could you please specify which product you are interested in? (You asked about: {user_query})",
            "Technical Support": f"Please describe your technical issue in more detail. (You asked about: {user_query})",
            "Billing Inquiry": f"For billing inquiries, please confirm your account details. (You asked about: {user_query})",
            "Greeting": "Hello! How can I assist you today?",
            "Goodbye": "Goodbye! Have a great day!",
            "Ambiguous": f"I'm not sure I fully understand. Could you rephrase or provide more details? (Your query: {user_query})"
        }
        return responses.get(intent, responses["Ambiguous"])

# Intent Classifier (NLU Module) - Simplified for demonstration
class IntentClassifier:
    def __init__(self):
        # In a real application, you would load a fine-tuned model here.
        # Example: self.nlp = pipeline("text-classification", model="your-finetuned-model")
        # For this example, we'll simulate intent classification with keywords.
        self.keyword_intents = {
            "order": "Order Status",
            "status": "Order Status",
            "product": "Product Information",
            "info": "Product Information",
            "technical": "Technical Support",
            "issue": "Technical Support",
            "billing": "Billing Inquiry",
            "invoice": "Billing Inquiry",
            "hello": "Greeting",
            "hi": "Greeting",
            "bye": "Goodbye",
            "goodbye": "Goodbye"
        }

    def classify_intent(self, text: str) -> tuple[str, float]:
        text_lower = text.lower()
        for keyword, intent in self.keyword_intents.items():
            if keyword in text_lower:
                return intent, 0.9 # High confidence for keyword match
        
        # Simulate ambiguity or unknown intent
        return "Ambiguous", 0.5

# Dialogue Manager
class DialogueManager:
    def decide_action(self, intent: str, confidence: float) -> str:
        if confidence > 0.7 and intent != "Ambiguous":
            return "direct_response"
        else:
            return "clarification"

# Response Generator
class ResponseGenerator:
    def __init__(self, knowledge_base: MockKnowledgeBase):
        self.kb = knowledge_base

    def generate_response(self, intent: str, user_query: str, action: str) -> str:
        if action == "direct_response":
            return self.kb.get_response(intent, user_query)
        elif action == "clarification":
            # If the intent is already ambiguous from classifier, use that
            # Otherwise, use a generic clarification for low confidence
            clarification_intent = intent if intent == "Ambiguous" else "Ambiguous"
            return self.kb.get_response(clarification_intent, user_query)
        return self.kb.get_response("Ambiguous", user_query) # Fallback

# FastAPI App Initialization
app = FastAPI()

# Component Instances
intent_classifier = IntentClassifier()
mock_knowledge_base = MockKnowledgeBase()
dialogue_manager = DialogueManager()
response_generator = ResponseGenerator(mock_knowledge_base)

@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    user_query = message.message

    # 1. Classify Intent
    intent, confidence = intent_classifier.classify_intent(user_query)

    # 2. Decide Action based on Dialogue Manager
    action = dialogue_manager.decide_action(intent, confidence)

    # 3. Generate Response
    chatbot_response = response_generator.generate_response(intent, user_query, action)

    return ChatResponse(response=chatbot_response, intent=intent, confidence=confidence)