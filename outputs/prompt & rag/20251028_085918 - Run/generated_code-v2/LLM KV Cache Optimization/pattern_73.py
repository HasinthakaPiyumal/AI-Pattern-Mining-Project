import os
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory


class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    session_id: str
    response: str

PRODUCT_DATABASE = {
    "SKU123": {"name": "Wireless Headphones", "price": "$129.99", "description": "High-fidelity audio with noise cancellation.", "stock": "In Stock"},
    "SKU456": {"name": "Ergonomic Office Chair", "price": "$349.00", "description": "Adjustable lumbar support and breathable mesh.", "stock": "Low Stock"},
    "SKU789": {"name": "Smartwatch Pro", "price": "$249.99", "description": "Fitness tracking, heart rate monitor, notifications.", "stock": "Out of Stock"},
}

def get_product_info(sku: str) -> str:
    product = PRODUCT_DATABASE.get(sku.upper())
    if product:
        return f"Product: {product['name']}, Price: {product['price']}, Description: {product['description']}, Stock: {product['stock']}."
    return "Product not found."

class LLMClient:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo")

    def invoke(self, prompt_messages: List[Any]):
        response = self.llm.invoke(prompt_messages)
        return response.content

app = FastAPI()

session_memories: Dict[str, ConversationBufferMemory] = {}
llm_client = LLMClient()

def get_session_memory(session_id: str) -> ConversationBufferMemory:
    if session_id not in session_memories:
        session_memories[session_id] = ConversationBufferMemory(
            return_messages=True, memory_key="chat_history"
        )
    return session_memories[session_id]

system_prompt = (
    "You are a helpful and friendly e-commerce customer support assistant. "
    "You can answer questions about products, orders, and general inquiries. "
    "If the user asks about a product, try to find its information using the 'product_info' tool. "
    "Current product information available: {product_info}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessage(content="{input}"),
    ]
)

def product_lookup_tool(query: str) -> str:
    if "sku" in query.lower():
        parts = query.split()
        for part in parts:
            if part.startswith("SKU") and len(part) == 6 and part[3:].isdigit():
                return get_product_info(part)
    return ""


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    session_id = request.session_id
    user_message = request.message

    memory = get_session_memory(session_id)

    product_data = product_lookup_tool(user_message)

    current_chat_history = memory.load_memory_variables({})["chat_history"]

    chain_input = {
        "input": user_message,
        "chat_history": current_chat_history,
        "product_info": product_data,
    }

    formatted_messages = prompt.format_messages(**chain_input)

    try:
        ai_response_content = llm_client.invoke(formatted_messages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM inference error: {str(e)}")

    memory.save_context({"input": user_message}, {"output": ai_response_content})

    return ChatResponse(session_id=session_id, response=ai_response_content)

