import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import sqlite3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import gradio as gr
import threading
import time

# --- 1. Core NLP Model (Intent & Entity Recognition) ---

# Mocking Hugging Face pipeline for demonstration
# In a real scenario, you would load a fine-tuned model and tokenizer
class MockNLPModel:
    def __init__(self):
        self.intents = {
            "order_status": ["where is my order", "track my package", "order delivery"],
            "reset_password": ["forgot my password", "reset password", "can't log in"],
            "product_info": ["tell me about", "product details", "specifications"],
            "escalate": ["talk to a human", "speak to an agent", "human help"],
            "ambiguous": ["help me", "i have a problem", "query"]
        }
        self.entities = {
            "order": ["order ID", "tracking number", "package"],
            "password": ["password", "account"],
            "product": ["product name", "item"]
        }

    def predict_intent_entities(self, text):
        text_lower = text.lower()
        
        # Simple intent matching
        detected_intent = "ambiguous"
        for intent, keywords in self.intents.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_intent = intent
                break
        
        # Simple entity extraction
        detected_entities = []
        for entity_type, keywords in self.entities.items():
            for keyword in keywords:
                if keyword in text_lower:
                    detected_entities.append((keyword, entity_type))
        
        confidence = 0.8 if detected_intent != "ambiguous" else 0.5
        
        # Simulate lower confidence for ambiguity
        if "help me" in text_lower or "i have a problem" in text_lower:
            detected_intent = "ambiguous"
            confidence = 0.4

        return {"intent": detected_intent, "entities": detected_entities, "confidence": confidence}

# --- 2. Personalization Module ---

class PersonalizationDB:
    def __init__(self, db_name=":memory:"):
        self.conn = sqlite3.connect(db_name, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self._create_table()

    def _create_table(self):
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                history TEXT
            )
        """)
        self.conn.commit()

    def load_user_profile(self, user_id):
        self.cursor.execute("SELECT history FROM user_profiles WHERE user_id = ?", (user_id,))
        result = self.cursor.fetchone()
        if result:
            return eval(result[0]) # Storing/retrieving as string representation of list of tuples
        return []

    def save_user_interaction(self, user_id, interaction):
        history = self.load_user_profile(user_id)
        history.append(interaction)
        self.cursor.execute(
            "INSERT OR REPLACE INTO user_profiles (user_id, history) VALUES (?, ?)", 
            (user_id, str(history))
        )
        self.conn.commit()

    def adapt_intent(self, user_id, initial_intent, current_query):
        history = self.load_user_profile(user_id)
        if not history:
            return initial_intent

        # Simple adaptation: if user frequently asks about order status, lean towards it
        order_status_count = sum(1 for _, resp_intent in history if resp_intent == "order_status")
        if order_status_count > 2 and initial_intent == "ambiguous" and "order" in current_query.lower():
            return "order_status"
        
        return initial_intent

# --- 3. Tool/Action Executor ---

class ToolExecutor:
    def check_order_status(self, entities):
        order_id = next((e[0] for e in entities if e[1] == "order"), "N/A")
        return f"Checking status for order ID {order_id}. It is currently in transit and expected by tomorrow."

    def reset_password(self, entities):
        return "A password reset link has been sent to your registered email address."

    def escalate_to_human(self, entities):
        return "Connecting you to a human agent now. Please wait while we find an available representative."

    def provide_product_info(self, entities):
        product_name = next((e[0] for e in entities if e[1] == "product"), "the requested product")
        return f"I can provide information on {product_name}. Could you please specify which product you are interested in?"

    def default_response(self):
        return "I'm sorry, I couldn't fully understand your request. Could you please rephrase or provide more details?"

# --- 4. Dialogue Manager ---

class DialogueManager:
    def __init__(self, nlp_model, personalization_db, tool_executor):
        self.nlp = nlp_model
        self.personalization_db = personalization_db
        self.tool_executor = tool_executor
        self.conversation_states = {}

    def process_query(self, user_id, query):
        # Load user history
        user_history = self.personalization_db.load_user_profile(user_id)

        # Initial intent and entity recognition
        nlp_result = self.nlp.predict_intent_entities(query)
        initial_intent = nlp_result["intent"]
        entities = nlp_result["entities"]
        confidence = nlp_result["confidence"]

        response_text = ""
        resolved_intent = initial_intent

        # Intent Clarification Logic
        if confidence < 0.6 or initial_intent == "ambiguous":
            if user_id not in self.conversation_states or not self.conversation_states[user_id].get("clarifying_context"):
                clarifying_question = "I'm not entirely sure I understand. Are you asking about an order, password reset, or something else?"
                self.conversation_states[user_id] = {"clarifying_context": True, "initial_query": query}
                response_text = clarifying_question
            else:
                # Attempt to resolve after clarification
                prev_query = self.conversation_states[user_id]["initial_query"]
                combined_query = prev_query + " " + query
                nlp_result_clarified = self.nlp.predict_intent_entities(combined_query)
                resolved_intent = nlp_result_clarified["intent"]
                entities = nlp_result_clarified["entities"]
                confidence = nlp_result_clarified["confidence"]
                
                if confidence < 0.6 or resolved_intent == "ambiguous":
                    response_text = "I'm still having trouble understanding. Could you please be more specific or try asking in a different way?"
                    self.conversation_states[user_id] = {}
                else:
                    self.conversation_states[user_id] = {} # Clear context
        
        # Personalization
        if resolved_intent != "ambiguous":
            resolved_intent = self.personalization_db.adapt_intent(user_id, resolved_intent, query)

        # Tool/Action Execution
        if not response_text: # If not already set by clarification
            if resolved_intent == "order_status":
                response_text = self.tool_executor.check_order_status(entities)
            elif resolved_intent == "reset_password":
                response_text = self.tool_executor.reset_password(entities)
            elif resolved_intent == "escalate":
                response_text = self.tool_executor.escalate_to_human(entities)
            elif resolved_intent == "product_info":
                response_text = self.tool_executor.provide_product_info(entities)
            else:
                response_text = self.tool_executor.default_response()
        
        # Save interaction history
        self.personalization_db.save_user_interaction(user_id, (query, resolved_intent))
        
        return response_text

# Initialize components
nlp_model = MockNLPModel()
personalization_db = PersonalizationDB()
tool_executor = ToolExecutor()
dialogue_manager = DialogueManager(nlp_model, personalization_db, tool_executor)

# --- 5. API Layer (FastAPI) ---

app = FastAPI()

class ChatRequest(BaseModel):
    user_id: str
    query: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        response = dialogue_manager.process_query(request.user_id, request.query)
        return {"user_id": request.user_id, "query": request.query, "response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- 6. User Interface (Gradio) ---

# In-memory history for Gradio chat, will be reset on refresh
gradio_chat_history = {}

def gradio_chat_interface_fn(user_id, message, history):
    global gradio_chat_history

    # Initialize history for new user_id if not present
    if user_id not in gradio_chat_history:
        gradio_chat_history[user_id] = []

    # Process message using the dialogue manager
    response = dialogue_manager.process_query(user_id, message)

    # Update chat history for Gradio display
    gradio_chat_history[user_id].append([message, response])

    # Return the updated history list for Gradio ChatInterface
    return gradio_chat_history[user_id]

# Gradio UI definition
with gr.Blocks() as demo:
    gr.Markdown("# Smart Customer Support Chatbot Demo")
    user_id_input = gr.Textbox(label="Enter User ID", placeholder="e.g., user_123", value="test_user")
    
    chatbot = gr.ChatInterface(
        fn=lambda message, history: gradio_chat_interface_fn(user_id_input.value, message, history),
        chatbot=gr.Chatbot(height=400),
        textbox=gr.Textbox(placeholder="Ask me a question", container=False, scale=7),
        title="Customer Support Chatbot",
        description="Ask questions about your order, password, or products. Try vague queries like 'help me' or 'I have a problem' to see clarification in action!",
        theme="soft",
        submit_btn="Send",
        clear_btn="Clear Chat"
    )

# Function to run FastAPI in a separate thread
def run_fastapi():
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")

if __name__ == "__main__":
    # Start FastAPI in a separate thread
    fastapi_thread = threading.Thread(target=run_fastapi)
    fastapi_thread.daemon = True # Allow the main program to exit even if thread is running
    fastapi_thread.start()
    
    print("FastAPI running on http://127.0.0.1:8000")
    print("Gradio running on http://127.0.0.1:7860 (or similar)")

    # Start Gradio interface
    demo.launch(share=False)

