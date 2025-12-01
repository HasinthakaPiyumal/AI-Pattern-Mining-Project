import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from loguru import logger

from chatbot_agent import ChatbotAgent
from config import Config

load_dotenv()

app = FastAPI()

config = Config(
    openai_api_key=os.getenv("OPENAI_API_KEY", "sk-YOUR_OPENAI_API_KEY"),
    chroma_path="./chroma_db"
)

chatbot_agent = ChatbotAgent(config)

class Message(BaseModel):
    text: str

@app.post("/chat")
async def chat(message: Message):
    logger.info(f"Received message: {message.text}")
    response = await chatbot_agent.process_message(message.text)
    logger.info(f"Sending response: {response}")
    return {"response": response}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

