import streamlit as st
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import requests
import json
import os

# --- Environment Variables (for API keys, etc.) ---
# In a real application, use python-dotenv or similar to load from .env
# os.environ["OPENAI_API_KEY"] = "your_openai_api_key"
# os.environ["CHROMADB_HOST"] = "your_chromadb_host"

# --- Pydantic Models for FastAPI ---
class UserQuery(BaseModel):
    query: str
    user_id: str = "anonymous"

class ChatResponse(BaseModel):
    response: str
    intent: str = "unknown"
    clarifying_questions: List[str] = []
    action_required: Dict[str, Any] = {}

# --- FastAPI Application ---
app = FastAPI(title="E-commerce Chatbot API")

# --- Core AI Logic Placeholders ---

# Placeholder for NLU and Intent Recognition Model
# In a real scenario, load a fine-tuned transformers model here
# from transformers import pipeline
# nlu_pipeline = pipeline("text-classification", model="your-finetuned-intent-model")
# from sentence_transformers import SentenceTransformer
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def recognize_intent_and_entities(text: str) -> Dict[str, Any]:
    # This function would use a fine-tuned LLM (e.g., from transformers)
    # to identify the user's intent and extract relevant entities.
    # Example: "I want to return a shirt." -> intent: return, entity: shirt
    
    # Dummy implementation
    text_lower = text.lower()
    if "return" in text_lower or "refund" in text_lower:
        return {"intent": "return_request", "entities": {"item": "product"}}
    elif "shipping" in text_lower or "delivery" in text_lower:
        return {"intent": "shipping_inquiry", "entities": {}}
    elif "product" in text_lower and ("info" in text_lower or "details" in text_lower):
        return {"intent": "product_information", "entities": {"product_name": ""}}
    elif "hello" in text_lower or "hi" in text_lower:
        return {"intent": "greeting", "entities": {}}
    else:
        return {"intent": "general_inquiry", "entities": {}}

# Placeholder for Knowledge Retrieval (RAG) System
# In a real scenario, this would interact with Chroma/Pinecone and an embedding model
# from langchain.vectorstores import Chroma
# from langchain.embeddings import OpenAIEmbeddings
# vectorstore = Chroma(embedding_function=OpenAIEmbeddings(), persist_directory="./chroma_db")

def retrieve_knowledge(query: str, intent_info: Dict[str, Any]) -> List[str]:
    # This function would query a vector database (Chroma, Pinecone) 
    # based on the user's query and identified intent to fetch relevant documents.
    
    # Dummy implementation
    if intent_info["intent"] == "return_request":
        return ["Our return policy allows returns within 30 days of purchase. Items must be unworn and have original tags."]
    elif intent_info["intent"] == "shipping_inquiry":
        return ["Standard shipping takes 5-7 business days. Express shipping options are available at checkout."]
    elif intent_info["intent"] == "product_information":
        return ["Please specify the product you are interested in. We have detailed descriptions available."]
    return ["I am sorry, I couldn't find specific information for that. Can you rephrase or provide more details?"]

# Placeholder for LLM for Response Generation
# In a real scenario, this would use OpenAI API or a local transformers LLM
# import openai
# openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_llm_response(prompt: str) -> str:
    # This function would send a prompt to an LLM (e.g., GPT-3.5/4 or local Llama 2)
    # to generate a natural language response.
    
    # Dummy implementation using a simple rule-based approach for demonstration
    if "return policy" in prompt.lower():
        return "Our return policy states that items can be returned within 30 days of purchase, provided they are in new condition with all original tags attached."
    elif "shipping time" in prompt.lower():
        return "Standard shipping typically takes 5-7 business days. Expedited options are also available."
    elif "product details" in prompt.lower():
        return "To provide you with accurate product details, please tell me the name or item number of the product you are interested in."
    elif "hello" in prompt.lower() or "hi" in prompt.lower():
        return "Hello! How can I assist you today with your shopping?"
    elif "escalate" in prompt.lower():
        return "I understand. I will connect you with a human agent shortly."
    return "I'm designed to help with e-commerce queries. Could you please clarify your request?"

# Placeholder for Dialogue Management and Ambiguity Resolution
def manage_dialogue(user_query: str, intent_info: Dict[str, Any], context: List[str]) -> Dict[str, Any]:
    # This function orchestrates the conversation flow, handles context, 
    # and detects/resolves ambiguity.
    
    response_data = {"response": "", "intent": intent_info["intent"], "clarifying_questions": [], "action_required": {}}
    
    if intent_info["intent"] == "return_request" and "item" not in intent_info["entities"]:
        response_data["clarifying_questions"].append("Which item would you like to return?")
        response_data["response"] = "To help you with your return, could you please tell me which item you are referring to?"
    elif intent_info["intent"] == "product_information" and "product_name" not in intent_info["entities"] or intent_info["entities"].get("product_name") == "":
        response_data["clarifying_questions"].append("What product are you interested in?")
        response_data["response"] = "I can help with product information. What specific product are you looking for details on?"
    else:
        # Generate a response based on intent and retrieved knowledge
        retrieved_docs = retrieve_knowledge(user_query, intent_info)
        context_for_llm = f"User query: {user_query}. Identified intent: {intent_info['intent']}. Relevant knowledge: {'. '.join(retrieved_docs)}"
        llm_answer = generate_llm_response(context_for_llm)
        response_data["response"] = llm_answer

    return response_data

# Placeholder for User Profile and Conversation Logging
def log_conversation(user_id: str, query: str, response: Dict[str, Any]):
    # In a real app, this would save to a PostgreSQL DB via SQLAlchemy
    print(f"[LOG] User {user_id}: Query='{query}', Response='{response['response']}'")
    # Example: User.update_last_interaction(user_id)
    # ConversationLog.create(user_id, query, response)


# --- FastAPI Endpoints ---
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(user_query: UserQuery):
    try:
        # 1. Intent Recognition & NLU
        intent_info = recognize_intent_and_entities(user_query.query)
        
        # 2. Dialogue Management & Ambiguity Resolution
        # In a full Langchain/Semantic Kernel setup, context would be managed over sessions
        # For simplicity, passing an empty context for now.
        dialogue_output = manage_dialogue(user_query.query, intent_info, context=[])
        
        # 3. Log the conversation
        log_conversation(user_query.user_id, user_query.query, dialogue_output)

        return ChatResponse(
            response=dialogue_output["response"],
            intent=dialogue_output["intent"],
            clarifying_questions=dialogue_output["clarifying_questions"],
            action_required=dialogue_output["action_required"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Streamlit UI Application (for demonstration) ---
def streamlit_ui():
    st.title("🛒 E-commerce Customer Support Chatbot")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Ask me anything about your order or products...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Call FastAPI backend
        try:
            response = requests.post(
                "http://localhost:8000/chat", 
                json={
                    "query": user_input,
                    "user_id": "streamlit_user"
                }
            )
            response.raise_for_status() # Raise an exception for HTTP errors
            chatbot_response = response.json()
            
            assistant_response = chatbot_response["response"]
            if chatbot_response["clarifying_questions"]:
                assistant_response += "\n\n" + " ".join(chatbot_response["clarifying_questions"])
            
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
            with st.chat_message("assistant"):
                st.markdown(assistant_response)
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the FastAPI backend. Make sure it's running on http://localhost:8000.")
        except Exception as e:
            st.error(f"An error occurred: {e}")


# This block allows running either FastAPI or Streamlit separately
# To run FastAPI: uvicorn ecommerce_chatbot:app --reload
# To run Streamlit: streamlit run ecommerce_chatbot.py
if __name__ == "__main__":
    import uvicorn
    # This part will run Streamlit if directly executed, or FastAPI if run via uvicorn
    # For a true single-file setup where both can be run, you'd typically run FastAPI 
    # and then launch Streamlit separately pointing to the FastAPI server.
    # For this example, we'll demonstrate the Streamlit UI's interaction with a *presumed* 
    # running FastAPI server.
    
    # To simplify execution for this demonstration, let's assume the user will run
    # `uvicorn ecommerce_chatbot:app --reload` in one terminal
    # and `streamlit run ecommerce_chatbot.py` in another.
    
    # You can uncomment the uvicorn.run line if you want the script to try and start 
    # FastAPI automatically when run directly, but be aware of port conflicts if you
    # then try to run streamlit in the same process or another process on the same port.
    # uvicorn.run(app, host="0.0.0.0", port=8000)
    
    # For the purpose of this single file output, we'll focus on demonstrating the 
    # Streamlit frontend interacting with a *separately running* FastAPI instance.
    streamlit_ui()