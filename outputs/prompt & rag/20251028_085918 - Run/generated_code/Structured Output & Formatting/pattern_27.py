from fastapi import FastAPI
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from typing import Literal, Optional
import uvicorn
import os


# 1. Pydantic Models for Structured Output
class ProductInfo(BaseModel):
    product_name: str = Field(description="Name of the product")
    product_id: str = Field(description="Unique identifier for the product")
    description: str = Field(description="Brief description of the product")
    price: float = Field(description="Price of the product")
    availability: bool = Field(description="Whether the product is in stock")
    category: str = Field(description="Category of the product")

class OrderStatus(BaseModel):
    order_id: str = Field(description="Unique identifier for the order")
    status: Literal["pending", "processing", "shipped", "delivered", "cancelled"] = Field(description="Current status of the order")
    estimated_delivery: Optional[str] = Field(None, description="Estimated delivery date if applicable")
    tracking_number: Optional[str] = Field(None, description="Tracking number for the shipment")

class ReturnInfo(BaseModel):
    return_id: str = Field(description="Unique identifier for the return request")
    order_id: str = Field(description="Order ID associated with the return")
    status: Literal["requested", "approved", "denied", "completed"] = Field(description="Current status of the return request")
    instructions: str = Field(description="Instructions for the customer to proceed with the return")

class GeneralResponse(BaseModel):
    message: str = Field(description="A general conversational response to the user's query")

class ChatbotResponse(BaseModel):
    response_type: Literal["product_info", "order_status", "return_info", "general"]
    data: ProductInfo | OrderStatus | ReturnInfo | GeneralResponse # Union type for different response data

# FastAPI App
app = FastAPI("E-commerce Chatbot API")

# Langchain Setup
# Ensure OPENAI_API_KEY is set in your environment variables
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

# Define a system prompt for the LLM
system_prompt_template = (
    "You are an E-commerce customer support chatbot. Your goal is to provide helpful and structured responses "
    "to customer queries. Based on the user's query, identify the intent and provide a structured JSON response "
    "conforming to one of the following Pydantic schemas: ProductInfo, OrderStatus, ReturnInfo, or GeneralResponse. "
    "If the query cannot be categorized or requires a general conversational answer, use the GeneralResponse schema. "
    "Always output a JSON object with a 'response_type' and 'data' field. The 'response_type' must be one of "
    "'product_info', 'order_status', 'return_info', or 'general', and 'data' must be the corresponding Pydantic model's JSON representation."
    "Example for ProductInfo: {'response_type': 'product_info', 'data': {'product_name': 'Laptop', 'product_id': 'P123', ...}}"
    "Example for GeneralResponse: {'response_type': 'general', 'data': {'message': 'Hello! How can I help you today?'}}"
)

# Langchain Runnable for structured output
# This chain will automatically enforce the ChatbotResponse Pydantic schema
structured_llm_chain = llm.with_structured_output(ChatbotResponse)

# Chatbot Query Model
class UserQuery(BaseModel):
    query: str = Field(description="The customer's query")

@app.post("/chat", response_model=ChatbotResponse)
async def chat_with_bot(user_query: UserQuery):
    """Process customer queries and return structured responses."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt_template),
        ("human", "{query}")
    ])
    
    # Invoke the LLM with the structured output enforcer
    response = structured_llm_chain.invoke(prompt.format_messages(query=user_query.query))
    
    return response

if __name__ == "__main__":
    # To run this application:
    # 1. pip install fastapi uvicorn "langchain[all]" openai pydantic
    # 2. Set your OpenAI API key: export OPENAI_API_KEY='your_api_key'
    # 3. Run the app: uvicorn chatbot_app:app --reload
    uvicorn.run(app, host="0.0.0.0", port=8000)