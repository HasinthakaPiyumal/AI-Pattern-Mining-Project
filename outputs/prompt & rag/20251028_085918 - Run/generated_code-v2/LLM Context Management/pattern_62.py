from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import datetime
import os

from sentence_transformers import SentenceTransformer
import chromadb
from langchain_openai import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

class UserEvent(BaseModel):
    user_id: str
    event_type: str
    details: Dict[str, Any]

class RecommendationRequest(BaseModel):
    user_id: str

class ChatRequest(BaseModel):
    user_id: str
    message: str

class LogEventResponse(BaseModel):
    message: str

class RecommendationResponse(BaseModel):
    user_id: str
    recommendations: List[str]

class ChatResponse(BaseModel):
    user_id: str
    response: str

user_history: Dict[str, List[Dict[str, Any]]] = {}
product_catalog: Dict[str, Dict[str, str]] = {
    "prod101": {"name": "Laptop Pro X", "description": "High-performance laptop with 16GB RAM and 512GB SSD. Ideal for professionals.", "category": "Electronics"},
    "prod102": {"name": "Wireless Earbuds", "description": "Noise-cancelling earbuds with 24-hour battery life. Perfect for music lovers.", "category": "Audio"},
    "prod103": {"name": "Smartwatch Series 5", "description": "Fitness tracker and notification hub. Track your steps, heart rate, and more.", "category": "Wearables"},
    "prod104": {"name": "Mechanical Keyboard RGB", "description": "Gaming keyboard with customizable RGB lighting and tactile switches.", "category": "Accessories"},
    "prod105": {"name": "4K UHD Monitor", "description": "27-inch 4K monitor with HDR support. Stunning visuals for work and play.", "category": "Electronics"},
    "prod106": {"name": "Ergonomic Office Chair", "description": "Adjustable office chair designed for comfort during long working hours.", "category": "Office Furniture"},
}

class ContextManager:
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.chroma_client = chromadb.Client()
        self.user_memory_collection = self.chroma_client.get_or_create_collection("user_memory")
        self.product_rag_collection = self.chroma_client.get_or_create_collection("product_rag")
        self.llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-3.5-turbo") # or gpt-4
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
        )
        self._initialize_product_rag()

    def _initialize_product_rag(self):
        for prod_id, prod_data in product_catalog.items():
            text = f"Product ID: {prod_id}, Name: {prod_data['name']}, Description: {prod_data['description']}, Category: {prod_data['category']}"
            embedding = self.embedding_model.encode(text).tolist()
            self.product_rag_collection.add(
                embeddings=[embedding],
                documents=[text],
                metadatas=[{"product_id": prod_id, "name": prod_data["name"]}],
                ids=[prod_id]
            )

    def _update_user_memory(self, user_id: str, event_description: str):
        existing_docs = self.user_memory_collection.get(ids=[user_id], include=['documents'])
        current_user_profile = existing_docs['documents'][0] if existing_docs['documents'] else ""
        
        combined_text = f"{current_user_profile}\nNew event: {event_description}"
        
        summary_prompt_template = """You are an AI assistant that summarizes user activities and preferences to maintain a concise user profile. Summarize the following user history into a compact profile, highlighting key interests, past purchases, and common behaviors. Focus on information relevant for personalization and recommendations. Keep the summary under 200 words.

User History:
{text}

Concise User Profile:"""
        
        summary_prompt = PromptTemplate.from_template(summary_prompt_template)
        summary_chain = load_summarize_chain(self.llm, chain_type="stuff", prompt=summary_prompt)
        
        docs = self.text_splitter.create_documents([combined_text])
        updated_profile = summary_chain.run(docs)
        
        profile_embedding = self.embedding_model.encode(updated_profile).tolist()
        self.user_memory_collection.upsert(
            embeddings=[profile_embedding],
            documents=[updated_profile],
            metadatas=[{"user_id": user_id}],
            ids=[user_id]
        )

    def _retrieve_relevant_history(self, user_id: str, query: str, n_results: int = 3) -> List[str]:
        relevant_docs = []
        
        query_embedding = self.embedding_model.encode(query).tolist()
        results = self.user_memory_collection.query(
            query_embeddings=[query_embedding],
            n_results=1, 
            where={"user_id": user_id},
            include=['documents']
        )
        if results['documents'] and results['documents'][0]:
            relevant_docs.extend(results['documents'][0])

        recent_interactions = user_history.get(user_id, [])
        recent_event_descriptions = [f"Type: {e['event_type']}, Details: {e['details']}" for e in recent_interactions[-5:]]
        
        relevant_docs.extend(recent_event_descriptions)
        return relevant_docs

    def _retrieve_relevant_products(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        query_embedding = self.embedding_model.encode(query).tolist()
        results = self.product_rag_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=['documents', 'metadatas']
        )
        return [{'id': meta['product_id'], 'name': meta['name'], 'description': doc} for meta, doc in zip(results['metadatas'][0], results['documents'][0])]

    def _get_context_for_llm(self, user_id: str, current_input: str, task_type: str, chat_history: List[Dict[str, Any]] = None) -> str:
        context_parts = []
        
        relevant_user_memory = self._retrieve_relevant_history(user_id, current_input)
        if relevant_user_memory:
            context_parts.append(f"User Profile/History: {' '.join(relevant_user_memory)}")

        if task_type == "recommendation":
            relevant_products = self._retrieve_relevant_products(current_input, n_results=5)
            if relevant_products:
                product_info = "\n".join([f"Product {p['name']} (ID: {p['id']}): {p['description']}" for p in relevant_products])
                context_parts.append(f"Available Products:\n{product_info}")
        
        if chat_history:
            chat_history_str = "\n".join([f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}" for msg in chat_history])
            context_parts.append(f"Conversation History:\n{chat_history_str}")

        context_parts.append(f"Current User Input: {current_input}")

        return "\n\n".join(context_parts)

    def generate_recommendations(self, user_id: str) -> List[str]:
        current_user_profile = self._retrieve_relevant_history(user_id, "current user profile")
        user_query_for_products = f"Based on the user's history and profile: {' '.join(current_user_profile)}. What products might they be interested in?"
        context = self._get_context_for_llm(user_id, user_query_for_products, "recommendation")
        
        recommendation_prompt_template = """You are an intelligent e-commerce recommendation system. Based on the provided user context and available products, suggest 3-5 personalized product recommendations. Be concise and list only the product names.

Context:
{context}

Recommendations:"""
        
        recommendation_prompt = PromptTemplate.from_template(recommendation_prompt_template)
        
        response = self.llm.invoke(
            [SystemMessage(content="You are a helpful e-commerce recommendation engine."),
             HumanMessage(content=recommendation_prompt.format(context=context))]
        )
        return [rec.strip() for rec in response.content.split('\n') if rec.strip()]

    def generate_chat_response(self, user_id: str, message: str) -> str:
        chat_history = user_history.get(user_id, [])
        # Simple extraction of past chat messages for context for now
        past_messages = []
        for event in chat_history:
            if event['event_type'] == 'chat_message':
                past_messages.append({'role': 'user', 'content': event['details']['message']})
            elif event['event_type'] == 'chat_response':
                past_messages.append({'role': 'assistant', 'content': event['details']['response']})
        
        context = self._get_context_for_llm(user_id, message, "customer_support", chat_history=past_messages)
        
        chat_prompt_template = """You are an e-commerce customer support AI. Provide helpful and concise responses based on the conversation history and user profile. If you need more information, ask clarifying questions.

Context:
{context}

User: {message}
Assistant:"""

        chat_prompt = PromptTemplate.from_template(chat_prompt_template)

        response = self.llm.invoke(
            [SystemMessage(content="You are a helpful e-commerce customer support assistant."),
             HumanMessage(content=chat_prompt.format(context=context, message=message))]
        )
        return response.content

context_manager = ContextManager()

@app.post("/log_event", response_model=LogEventResponse)
async def log_user_event(event: UserEvent):
    timestamp = datetime.datetime.now().isoformat()
    event_data = event.dict()
    event_data["timestamp"] = timestamp
    
    if event.user_id not in user_history:
        user_history[event.user_id] = []
    user_history[event.user_id].append(event_data)

    event_description = f"User {event.user_id} performed {event.event_type} with details: {event.details}"
    context_manager._update_user_memory(event.user_id, event_description)

    return {"message": "Event logged and user memory updated successfully."}

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    if request.user_id not in user_history:
        raise HTTPException(status_code=404, detail="User not found or no history available.")
    
    recommendations = context_manager.generate_recommendations(request.user_id)
    return {"user_id": request.user_id, "recommendations": recommendations}

@app.post("/chat", response_model=ChatResponse)
async def chat_with_support(request: ChatRequest):
    if request.user_id not in user_history:
        user_history[request.user_id] = []
    
    chat_response = context_manager.generate_chat_response(request.user_id, request.message)

    user_history[request.user_id].append({
        "timestamp": datetime.datetime.now().isoformat(),
        "event_type": "chat_message",
        "details": {"message": request.message}
    })
    user_history[request.user_id].append({
        "timestamp": datetime.datetime.now().isoformat(),
        "event_type": "chat_response",
        "details": {"response": chat_response}
    })
    context_manager._update_user_memory(request.user_id, f"User chatted: {request.message}. Assistant responded: {chat_response}")

    return {"user_id": request.user_id, "response": chat_response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)