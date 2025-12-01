import os
import requests
import streamlit as st
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any

from langchain.agents import AgentExecutor, initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage
from langchain.tools import Tool
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.memory import VectorStoreRetrieverMemory

class ProductDatabase:
    def __init__(self):
        self.products = {
            "laptop": {"id": "p001", "name": "Laptop Pro X", "price": 1200.00, "category": "Electronics", "description": "High-performance laptop for professionals, 16GB RAM, 512GB SSD."},
            "smartphone": {"id": "p002", "name": "Smartphone Ultra", "price": 800.00, "category": "Electronics", "description": "Latest smartphone with advanced camera features and long battery life."},
            "headphones": {"id": "p003", "name": "Noise-Cancelling Headphones", "price": 250.00, "category": "Audio", "description": "Immersive sound with active noise cancellation for travel and work."},
            "keyboard": {"id": "p004", "name": "Mechanical Keyboard", "price": 150.00, "category": "Accessories", "description": "Durable keyboard with satisfying tactile feedback for gaming and typing."},
            "mouse": {"id": "p005", "name": "Gaming Mouse", "price": 70.00, "category": "Accessories", "description": "Ergonomic design with high precision sensor for competitive gaming."},
            "t-shirt": {"id": "p006", "name": "Cotton T-Shirt", "price": 25.00, "category": "Apparel", "description": "Soft and comfortable everyday t-shirt, available in multiple colors."},
            "jeans": {"id": "p007", "name": "Slim Fit Jeans", "price": 60.00, "category": "Apparel", "description": "Stylish slim fit jeans for a modern look, made from stretch denim."},
        }

    def search_products(self, query: str) -> str:
        query_lower = query.lower()
        results = [
            p for p_key, p in self.products.items()
            if query_lower in p["name"].lower() or query_lower in p["description"].lower() or query_lower in p["category"].lower()
        ]
        if results:
            return "Found the following products:\n" + "\n".join([f"- {r["name"]} ({r["category"]}): ${r["price"]:.2f}" for r in results])
        return "No products found matching your query. Please try a different search term."

    def get_product_details(self, product_name: str) -> str:
        product_name_lower = product_name.lower()
        for p_key, p in self.products.items():
            if product_name_lower in p["name"].lower():
                return f"Details for {p["name"]}:\nCategory: {p["category"]}\nPrice: ${p["price"]:.2f}\nDescription: {p["description"]}"
        return f"Could not find details for product: '{product_name}'. Please specify the exact product name or search again."

def setup_llm_agent_with_memory():
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key=os.getenv("OPENAI_API_KEY"))

    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    memory = VectorStoreRetrieverMemory(retriever=retriever, memory_key="chat_history", input_key="input")

    product_db = ProductDatabase()
    tools = [
        Tool(
            name="ProductSearch",
            func=product_db.search_products,
            description="Useful for searching for products based on keywords like category, name, or description. Input should be a concise query string."
        ),
        Tool(
            name="ProductDetails",
            func=product_db.get_product_details,
            description="Useful for getting detailed information about a specific product. Input should be the exact name of the product."
        )
    ]

    system_message = SystemMessage(
        content="You are an intelligent e-commerce assistant. Your goal is to help users find products and provide recommendations through natural conversation. "
                "You have access to product search and details tools. Use them to assist the user. "
                "Remember to ask clarifying questions if needed and maintain a friendly tone."
    )
    agent_kwargs = {
        "system_message": system_message
    }

    agent_executor = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=False,
        memory=memory,
        handle_parsing_errors=True,
        agent_kwargs=agent_kwargs
    )
    return agent_executor

try:
    llm_agent = setup_llm_agent_with_memory()
except Exception as e:
    print(f"ERROR: Could not initialize LLM agent. Ensure OPENAI_API_KEY is set. Details: {e}")
    llm_agent = None

app = FastAPI()

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default_session"

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if llm_agent is None:
        return ChatResponse(response="The AI assistant is currently unavailable. Please check backend configuration.")
    try:
        result = llm_agent.run(input=request.message)
        return ChatResponse(response=result)
    except Exception as e:
        print(f"Error during agent execution: {e}")
        return ChatResponse(response="An internal error occurred. Please try again.")

if "streamlit" in os.environ.get("STREAMLIT_SERVER_URL", "") or hasattr(st, "_is_running_with_streamlit") and st._is_running_with_streamlit:
    st.title("E-commerce Conversational Recommender")

    if llm_agent is None:
        st.error("The AI assistant backend is not initialized. Please ensure the FastAPI server is running and configured correctly (e.g., `OPENAI_API_KEY` is set).")
    else:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("What are you looking for today?"):
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            try:
                fastapi_url = "http://localhost:8000/chat"
                response = requests.post(fastapi_url, json={"message": prompt})
                response.raise_for_status()
                assistant_response = response.json()["response"]
            except requests.exceptions.ConnectionError:
                assistant_response = "Error: Could not connect to the AI backend. Please ensure the FastAPI server is running (e.g., `uvicorn main:app --reload`)."
            except requests.exceptions.RequestException as e:
                assistant_response = f"An error occurred with the backend: {e}"

            with st.chat_message("assistant"):
                st.markdown(assistant_response)
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})