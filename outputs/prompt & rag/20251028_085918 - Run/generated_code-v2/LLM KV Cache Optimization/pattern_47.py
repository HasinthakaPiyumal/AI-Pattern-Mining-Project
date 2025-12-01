import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict

from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# Load environment variables from .env file
load_dotenv()

VLLM_API_BASE = os.getenv("VLLM_API_BASE", "http://localhost:8000/v1")  # Default vLLM OpenAI endpoint
VLLM_MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "llama-2-7b-chat-hf") # Example model, ensure this matches your vLLM deployment

app = FastAPI(
    title="PagedAttention Customer Support Chatbot",
    description="An AI-powered customer support chatbot leveraging vLLM's PagedAttention for efficient KV cache management."
)

# Initialize LangChain ChatOpenAI for vLLM
# vLLM's OpenAI-compatible API usually doesn't require a real API key.
llm = ChatOpenAI(
    openai_api_base=VLLM_API_BASE,
    openai_api_key="sk-dummy",  # A dummy key is sufficient for vLLM
    model_name=VLLM_MODEL_NAME,
    temperature=0.7
)

# Initialize conversational memory
# This memory will store the chat history for each session (though this example uses a single global memory for simplicity)
# In a real-world multi-user application, you would manage memory per user session.
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Define the prompt template for the chatbot
prompt_template = """You are a helpful customer support assistant. Answer the user's questions concisely and professionally.

{chat_history}
Human: {human_input}
AI:"""

prompt = PromptTemplate(
    input_variables=["chat_history", "human_input"],
    template=prompt_template
)

# Create an LLMChain to orchestrate the prompt, LLM, and memory
conversation_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True, # Set to False for production to reduce logging
    memory=memory
)

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat_with_bot(request: ChatRequest):
    """
    Handles customer support queries. Receives a user message, processes it
    with the LLM via LangChain, and returns the AI's response.
    """
    try:
        # Invoke the conversation chain with the new human input
        response = await conversation_chain.ainvoke({"human_input": request.message})
        return ChatResponse(response=response["text"])
    except Exception as e:
        # Log the error for debugging purposes
        print(f"Error processing chat request: {e}")
        # Return a generic error message to the user
        return ChatResponse(response="I'm sorry, I encountered an error. Please try again later.")
