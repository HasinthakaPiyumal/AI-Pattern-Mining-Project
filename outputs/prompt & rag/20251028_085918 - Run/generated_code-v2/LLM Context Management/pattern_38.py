import os
import uuid
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY" # Replace with your actual OpenAI API Key

app = FastAPI()

customer_profiles: Dict[str, Dict[str, Any]] = {}
chat_history_db: Dict[str, List[Dict[str, str]]] = {}

embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents([], embeddings)

class ChatRequest(BaseModel):
    customer_id: str
    message: str

class ChatResponse(BaseModel):
    agent_response: str
    context_used: List[str]

llm = ChatOpenAI(temperature=0.7)

system_prompt_template = SystemMessagePromptTemplate.from_template(
    "You are a helpful and empathetic customer support agent. "
    "You have access to the customer's profile and relevant past interactions. "
    "Use this information to provide accurate, personalized, and context-aware responses. "
    "If the customer mentions a past issue, try to retrieve and use that information. "
    "Keep your responses concise and to the point."
)

human_query_prompt_template = HumanMessagePromptTemplate.from_template("{query}")

full_prompt_template = ChatPromptTemplate.from_messages([
    system_prompt_template,
    MessagesPlaceholder(variable_name="chat_history"),
    MessagesPlaceholder(variable_name="retrieved_context"),
    human_query_prompt_template
])

llm_chain = LLMChain(llm=llm, prompt=full_prompt_template)

def add_to_chat_history_db(customer_id: str, speaker: str, message: str):
    if customer_id not in chat_history_db:
        chat_history_db[customer_id] = []
    chat_history_db[customer_id].append({"speaker": speaker, "message": message})

    # Add to vector store for retrieval
    vectorstore.add_documents([Document(page_content=f"{speaker}: {message}", metadata={
        "customer_id": customer_id,
        "timestamp": str(uuid.uuid4()) # Simple unique ID for doc
    })])

def get_customer_profile(customer_id: str) -> Dict[str, Any]:
    return customer_profiles.get(customer_id, {})

def get_recent_chat_history(customer_id: str, k: int = 5) -> List[Dict[str, str]]:
    return chat_history_db.get(customer_id, [])[-k:]

def retrieve_relevant_long_term_context(customer_id: str, query: str, k: int = 3) -> List[str]:
    retrieved_docs = vectorstore.similarity_search_with_score(query, k=k*5) # Retrieve more to filter
    
    context_texts = []
    for doc, score in retrieved_docs:
        if doc.metadata.get("customer_id") == customer_id:
            context_texts.append(doc.page_content)
        if len(context_texts) >= k:
            break
            
    return context_texts

@app.post("/chat", response_model=ChatResponse)
async def chat_with_agent(request: ChatRequest):
    customer_id = request.customer_id
    user_message = request.message

    if customer_id not in customer_profiles:
        customer_profiles[customer_id] = {"name": f"Customer {customer_id}", "preferences": "unknown"}

    add_to_chat_history_db(customer_id, "user", user_message)

    profile = get_customer_profile(customer_id)
    recent_history = get_recent_chat_history(customer_id, k=5)
    long_term_context = retrieve_relevant_long_term_context(customer_id, user_message, k=3)

    formatted_chat_history = []
    for entry in recent_history:
        if entry["speaker"] == "user":
            formatted_chat_history.append(HumanMessage(content=entry["message"]))
        else:
            formatted_chat_history.append(AIMessage(content=entry["message"])) # Agent messages for LLM context
            
    formatted_long_term_context_messages = []
    if long_term_context:
        combined_context = "Relevant historical information:\n" + "\n".join(long_term_context)
        formatted_long_term_context_messages.append(SystemMessage(content=combined_context))

    try:
        response = llm_chain.invoke({
            "query": user_message,
            "chat_history": formatted_chat_history,
            "retrieved_context": formatted_long_term_context_messages
        })
        agent_response = response["text"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM interaction failed: {str(e)}")

    add_to_chat_history_db(customer_id, "agent", agent_response)

    return ChatResponse(agent_response=agent_response, context_used=long_term_context)
