import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

# --- 1. Intent Recognition Module (Simulated) ---
# In a real application, this would load a fine-tuned transformer model
# using libraries like transformers and torch.
class IntentRecognizer:
    def __init__(self):
        # Simulate loading a model (e.g., BERT, RoBERTa)
        self.intents = {
            "order status": "check_order_status",
            "return item": "return_item",
            "product inquiry": "product_inquiry",
            "account help": "account_help",
            "shipping": "check_order_status", # Can be ambiguous, requires clarification
            "hello": "greeting",
            "hi": "greeting",
            "thank you": "farewell",
            "bye": "farewell",
        }

    def predict_intent(self, text: str) -> str:
        text_lower = text.lower()
        for keyword, intent in self.intents.items():
            if keyword in text_lower:
                return intent
        return "unknown"

# --- 4. Personalized Learning Module (Basic In-memory) ---
# In a more robust solution, this would use a database or more sophisticated
# mechanisms for persistence and adaptive learning.
class PersonalizedLearning:
    def __init__(self):
        self.user_preferences = {}

    def get_user_preference(self, user_id: str, key: str, default=None):
        return self.user_preferences.get(user_id, {}).get(key, default)

    def set_user_preference(self, user_id: str, key: str, value):
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = {}
        self.user_preferences[user_id][key] = value


# --- 2. Dialogue Management & 3. Tool/Action Mapping Module (Simulated with LangChain principles) ---
# This class orchestrates the conversation flow and simulates tool calls.
# In a real LangChain setup, this would involve Agents, Chains, and Tools.
class Chatbot:
    def __init__(self, intent_recognizer: IntentRecognizer, personalized_learning: PersonalizedLearning):
        self.intent_recognizer = intent_recognizer
        self.personalized_learning = personalized_learning
        self.conversation_history = {}

    def _call_tool(self, intent: str, user_id: str = None) -> str:
        # Simulate calling internal APIs or tools based on the intent
        if intent == "check_order_status":
            # For demonstration, assume we have an order ID from context or clarification
            # In a real scenario, this would interact with an order management system
            return "Please provide your order ID so I can check its status for you."
        elif intent == "return_item":
            return "To initiate a return, please provide your order ID and the reason for return."
        elif intent == "product_inquiry":
            return "What product are you interested in? I can help you find more information."
        elif intent == "account_help":
            return "I can help with account-related issues. What specific problem are you encountering?"
        elif intent == "greeting":
            preferred_name = self.personalized_learning.get_user_preference(user_id, "preferred_name")
            if preferred_name:
                return f"Hello {preferred_name}! How can I assist you today?"
            return "Hello! How can I help you today?"
        elif intent == "farewell":
            return "You're welcome! Have a great day."
        return "I'm sorry, I don't have a specific tool for that yet."

    def handle_query(self, user_id: str, query: str) -> str:
        # Store current query in history (simple for demonstration)
        if user_id not in self.conversation_history:
            self.conversation_history[user_id] = []
        self.conversation_history[user_id].append(query)

        intent = self.intent_recognizer.predict_intent(query)

        response = ""
        if intent == "unknown":
            # Simulate a clarification dialogue for unknown intent
            response = "I'm not sure I understand. Could you please rephrase your request or provide more details?"
        elif intent == "check_order_status" and "order id" not in query.lower():
            # Simulate a clarification dialogue for incomplete information
            response = self._call_tool(intent, user_id) # Prompts for order ID
        else:
            response = self._call_tool(intent, user_id)

        # Basic personalized learning update (e.g., remembering a greeting)
        if intent == "greeting" and user_id not in self.personalized_learning.user_preferences:
            # In a real scenario, we might ask for their name and store it.
            pass # For this simple demo, we won't ask for name directly in this step

        return response

# --- 5. API Endpoint (FastAPI) ---
app = FastAPI()

# Initialize modules
intent_recognizer_instance = IntentRecognizer()
personalized_learning_instance = PersonalizedLearning()
chatbot_instance = Chatbot(intent_recognizer_instance, personalized_learning_instance)

class QueryRequest(BaseModel):
    user_id: str
    query: str

@app.post("/chat")
async def chat_with_bot(request: QueryRequest):
    response = chatbot_instance.handle_query(request.user_id, request.query)
    return {"response": response}

if __name__ == "__main__":
    # To run this FastAPI application, save it as a .py file (e.g., main.py)
    # and run 'uvicorn main:app --reload' from your terminal.
    # Then you can send POST requests to http://127.0.0.1:8000/chat
    uvicorn.run(app, host="0.0.0.0", port=8000)
