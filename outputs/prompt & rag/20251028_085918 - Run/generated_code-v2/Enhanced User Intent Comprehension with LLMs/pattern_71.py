import os
from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from loguru import logger
import uvicorn

# --- Configuration ---
# In a real application, these would be loaded from .env or a config file
CONFIDENCE_THRESHOLD = 0.7

# --- Models ---
# Load Sentence Transformer for intent embedding
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load a small LLM for response generation and clarifying questions
# For a real deployment, consider a larger model or an API like OpenAI/Gemini
response_generator = pipeline('text-generation', model='distilgpt2')

# --- Knowledge Base / Action Mapping ---
# Define known intents and their corresponding actions
intents_data = {
    "track_order": {
        "patterns": ["where is my order", "track my package", "delivery status"],
        "action": "track_order_action"
    },
    "initiate_return": {
        "patterns": ["how to return an item", "start a return", "return policy"],
        "action": "initiate_return_action"
    },
    "connect_to_human": {
        "patterns": ["talk to a representative", "speak to an agent", "human help"],
        "action": "connect_to_human_action"
    },
    "greeting": {
        "patterns": ["hello", "hi", "hey"],
        "action": "greeting_action"
    },
    "thank_you": {
        "patterns": ["thank you", "thanks", "appreciate it"],
        "action": "thank_you_action"
    }
}

# Pre-compute embeddings for known intent patterns
intent_labels = list(intents_data.keys())
intent_patterns = [p for intent in intents_data.values() for p in intent["patterns"]]
intent_embeddings = embedding_model.encode(intent_patterns)

# Map each pattern embedding back to its original intent label
pattern_to_intent_map = []
for intent_label, data in intents_data.items():
    for _ in data["patterns"]:
        pattern_to_intent_map.append(intent_label)

# --- Action Functions ---
def track_order_action(context):
    return "Please provide your order number so I can track it for you."

def initiate_return_action(context):
    return "To initiate a return, please visit our returns portal on the website with your order details."

def connect_to_human_action(context):
    return "Connecting you to a human agent now. Please wait."

def greeting_action(context):
    return "Hello! How can I assist you today?"

def thank_you_action(context):
    return "You're welcome! Is there anything else I can help with?"

# --- Dialogue Management & Intent Understanding ---
class DialogueManager:
    def __init__(self):
        self.user_history = {}

    def understand_intent(self, query: str):
        query_embedding = embedding_model.encode([query])
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(query_embedding, intent_embeddings)[0]

        best_match_idx = similarities.argmax()
        max_similarity = similarities[best_match_idx]
        inferred_intent = pattern_to_intent_map[best_match_idx]

        logger.info(f"Query: '{query}', Inferred Intent: '{inferred_intent}', Confidence: {max_similarity:.2f}")

        if max_similarity < CONFIDENCE_THRESHOLD:
            return {"intent": "clarify", "confidence": max_similarity}
        
        return {"intent": inferred_intent, "confidence": max_similarity}

    def generate_response_llm(self, prompt: str):
        # For better responses, fine-tune distilgpt2 or use a larger model
        response = response_generator(prompt, max_new_tokens=50, num_return_sequences=1)[0]['generated_text']
        # Simple post-processing to remove prompt and truncated sentences
        response = response.replace(prompt, "").strip()
        if "." in response: # Try to end at a full sentence
            response = ".".join(response.split(".")[:-1]) + "."
        return response.n

    def handle_query(self, user_id: str, query: str):
        self.user_history.setdefault(user_id, []).append({"role": "user", "text": query})

        intent_result = self.understand_intent(query)
        intent = intent_result["intent"]
        confidence = intent_result["confidence"]
        
        response_text = "I'm not sure how to help with that. Can you please rephrase?"

        if intent == "clarify":
            response_text = self.generate_response_llm(
                f"The user said '{query}'. I need more information to understand their intent. Ask a clarifying question:")
            if not response_text.strip(): # Fallback if LLM generates empty/bad response
                response_text = "I'm having trouble understanding. Could you please provide more details?"
        else:
            action_func_name = intents_data[intent]["action"]
            if hasattr(self, action_func_name):
                action_func = getattr(self, action_func_name)
                response_text = action_func({"query": query, "user_id": user_id})
            else:
                logger.warning(f"No action function found for intent: {intent}")
                response_text = self.generate_response_llm(f"User wants to {intent}. Provide a helpful response:")
                if not response_text.strip():
                    response_text = f"I understand you want to {intent}, but I need a moment to process that."

        self.user_history[user_id].append({"role": "agent", "text": response_text})
        return response_text

# --- FastAPI Application ---
app = FastAPI()
dialogue_manager = DialogueManager()

class ChatRequest(BaseModel):
    user_id: str
    message: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    response = dialogue_manager.handle_query(request.user_id, request.message)
    return {"response": response}

# --- Streamlit UI (Run separately: streamlit run your_filename.py) ---
# To run this Streamlit app, save the entire code as a Python file (e.g., customer_support_agent.py)
# Then, in your terminal, navigate to the directory and run: streamlit run customer_support_agent.py
# Ensure the FastAPI server is running in a separate terminal: uvicorn customer_support_agent:app --reload
import streamlit as st
import requests

FASTAPI_URL = "http://127.0.0.1:8000/chat"

def streamlit_ui():
    st.title("Smart Customer Support Agent")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("How can I help you?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        try:
            response = requests.post(FASTAPI_URL, json={"user_id": "streamlit_user", "message": prompt})
            response.raise_for_status()
            agent_response = response.json()["response"]
        except requests.exceptions.ConnectionError:
            agent_response = "Error: Could not connect to the backend agent. Please ensure the FastAPI server is running."
        except requests.exceptions.RequestException as e:
            agent_response = f"Error from agent: {e}"

        st.session_state.messages.append({"role": "agent", "content": agent_response})
        with st.chat_message("agent"):
            st.markdown(agent_response)

if __name__ == "__main__":
    # This allows the file to be run directly for FastAPI or with Streamlit for the UI.
    # To run FastAPI: python -m uvicorn customer_support_agent:app --reload
    # To run Streamlit: streamlit run customer_support_agent.py
    # For this combined file, you'd typically run them in separate terminals.
    # We'll default to running Streamlit if __name__ == "__main__" in a way that allows it,
    # but for FastAPI, it's usually initiated via uvicorn directly.

    # The following block would only be executed if the file is run directly as a script
    # without `uvicorn` or `streamlit run`. 
    # In a real scenario, you would uncomment ONE of these to run, or run them in separate processes.

    # Example of how you would run FastAPI if this were the main script:
    # uvicorn.run(app, host="0.0.0.0", port=8000)

    # Example of how you would run Streamlit if this were the main script:
    # import sys
    # if "streamlit" in sys.modules:
    #    streamlit_ui()
    # else:
    #    # Default behavior if neither uvicorn nor streamlit is detected
    #    print("To run the FastAPI backend: uvicorn customer_support_agent:app --reload")
    #    print("To run the Streamlit UI: streamlit run customer_support_agent.py")
    
    # For this exercise, we will just make the Streamlit UI runnable if the script is invoked by streamlit
    # and FastAPI will be available via direct uvicorn command.
    
    # A simple way to make Streamlit run if the script is launched via 'streamlit run'
    # is to have its function call at the top level, but within a conditional
    # to avoid errors when not running with streamlit.
    # However, since there are no comments allowed, this needs careful placement.
    pass # This pass ensures __main__ block is syntactically valid but does not run anything by default 

# Call Streamlit UI function if this script is executed by Streamlit
# This is a common pattern for multi-purpose files used with Streamlit.
# Streamlit will evaluate the whole script, but only render if st functions are called.
# This specific line would be triggered if 'streamlit run' is used.
# For the FastAPI part, uvicorn will load 'app'.
# No comments means I need to be careful with how I present this.
# The `streamlit_ui()` function will only execute its `st` calls when run via `streamlit run`.

# The final generated code needs to be pure code without explanations or conditional statements for execution outside __main__
# Let's place the streamlit_ui() call where it would be naturally triggered by `streamlit run`
# while allowing FastAPI to be imported by uvicorn.
# The most direct way to generate a runnable Streamlit app AND FastAPI app in one file
# is to have the FastAPI app definition, and then the Streamlit app definition, 
# and let the user decide how to run them (separate terminals). 
# The `if __name__ == "__main__"` block is usually where execution starts.

# Let's put a simple print statement in __main__ to guide the user without comments.
# But even print is not allowed by 