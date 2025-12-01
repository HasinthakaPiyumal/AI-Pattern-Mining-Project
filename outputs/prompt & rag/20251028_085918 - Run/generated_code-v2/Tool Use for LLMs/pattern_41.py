import os
from typing import Type, Optional

import spacy
import pandas as pd
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.tools import BaseTool, tool
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --- 1. User Profile Management & Modeling Layer ---

# Mock Database for User Profiles
class UserProfileDB:
    def __init__(self):
        self.users = {}

    def get_user_profile(self, user_id: str):
        return self.users.get(user_id)

    def update_user_profile(self, user_id: str, profile_data: dict):
        if user_id not in self.users:
            self.users[user_id] = {}
        self.users[user_id].update(profile_data)
        print(f"User {user_id} profile updated: {self.users[user_id]}")

user_profile_db = UserProfileDB()

# Pre-load some mock user data
user_profile_db.update_user_profile("user123", {
    "communication_style": "formal",
    "preferred_resolution": "email",
    "history": [
        {"query": "My order is late.", "resolution": "refund issued"},
        {"query": "How do I reset my password?", "resolution": "sent reset link"}
    ]
})
user_profile_db.update_user_profile("user456", {
    "communication_style": "casual",
    "preferred_resolution": "chat",
    "history": [
        {"query": "Yo, where's my stuff at?", "resolution": "expedited shipping"}
    ]
})

# spaCy for NLP and Sentence-Transformers for embeddings
nlp = spacy.load("en_core_web_sm")
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

class CommunicationStyleAnalyzer:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer()
        self.style_keywords = {
            "formal": ["kindly", "request", "further assistance", "regarding", "sincerely"],
            "casual": ["hey", "yo", "what's up", "gonna", "wanna", "thanks a bunch"]
        }
        self.style_vectors = self.tfidf_vectorizer.fit_transform(self.style_keywords["formal"] + self.style_keywords["casual"])

    def analyze_style(self, text: str) -> str:
        doc = nlp(text.lower())
        tokens = [token.text for token in doc if token.is_alpha]
        text_vector = self.tfidf_vectorizer.transform([" ".join(tokens)])

        formal_score = cosine_similarity(text_vector, self.tfidf_vectorizer.transform(self.style_keywords["formal"]).sum(axis=0).reshape(1, -1))[0][0]
        casual_score = cosine_similarity(text_vector, self.tfidf_vectorizer.transform(self.style_keywords["casual"]).sum(axis=0).reshape(1, -1))[0][0]

        if formal_score > casual_score and formal_score > 0.1:
            return "formal"
        elif casual_score > formal_score and casual_score > 0.1:
            return "casual"
        return "neutral"

    def get_query_embedding(self, query: str):
        return sentence_model.encode(query)

style_analyzer = CommunicationStyleAnalyzer()

# --- 2. Personalized Tool Orchestration Layer ---

# Mock External Systems (for tools)
class MockKnowledgeBase:
    def search(self, query: str) -> str:
        print(f"Searching knowledge base for: {query}")
        if "password" in query.lower():
            return "To reset your password, visit our website and click 'Forgot Password'."
        if "order status" in query.lower() or "delivery" in query.lower():
            return "Please provide your order number to check the status."
        return "No direct answer found, escalating to a human agent."

class MockCRMSysem:
    def escalate_ticket(self, user_id: str, issue: str) -> str:
        print(f"Escalating ticket for user {user_id}: {issue}")
        return f"Ticket escalated for user {user_id}. A human agent will contact you shortly regarding: {issue}"

class MockEmailService:
    def send_email(self, recipient: str, subject: str, body: str, style: str) -> str:
        print(f"Sending {style} email to {recipient} with subject '{subject}' and body: {body}")
        return f"Email sent to {recipient} (style: {style})."

class MockDataRetrievalService:
    def get_order_history(self, user_id: str) -> str:
        print(f"Retrieving order history for user {user_id}")
        if user_id == "user123":
            return "Order #12345 (Laptop, delivered), Order #67890 (Mouse, pending delivery)."
        return "No order history found for this user."

knowledge_base = MockKnowledgeBase()
crm_system = MockCRMSysem()
email_service = MockEmailService()
data_retrieval_service = MockDataRetrievalService()

# Define LangChain Tools
class KnowledgeBaseSearchInput(BaseModel):
    query: str = Field(description="the search query for the knowledge base")

class KnowledgeBaseSearchTool(BaseTool):
    name: str = "knowledge_base_search"
    description: str = "Searches the internal knowledge base for answers to user questions."
    args_schema: Type[BaseModel] = KnowledgeBaseSearchInput

    def _run(self, query: str) -> str:
        return knowledge_base.search(query)

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("Async not implemented for KnowledgeBaseSearchTool")

class TicketEscalationInput(BaseModel):
    user_id: str = Field(description="the ID of the user whose ticket needs escalation")
    issue: str = Field(description="a brief description of the issue to be escalated")

class TicketEscalationTool(BaseTool):
    name: str = "ticket_escalation"
    description: str = "Escalates a complex user issue to a human agent."
    args_schema: Type[BaseModel] = TicketEscalationInput

    def _run(self, user_id: str, issue: str) -> str:
        return crm_system.escalate_ticket(user_id, issue)

    async def _arun(self, user_id: str, issue: str) -> str:
        raise NotImplementedError("Async not implemented for TicketEscalationTool")

class EmailSenderInput(BaseModel):
    recipient: str = Field(description="the email address of the recipient")
    subject: str = Field(description="the subject of the email")
    body: str = Field(description="the body content of the email")
    style: str = Field(description="the communication style for the email (e.g., formal, casual)")

class EmailSenderTool(BaseTool):
    name: str = "email_sender"
    description: str = "Sends a personalized email to a user with a specified style."
    args_schema: Type[BaseModel] = EmailSenderInput

    def _run(self, recipient: str, subject: str, body: str, style: str) -> str:
        return email_service.send_email(recipient, subject, body, style)

    async def _arun(self, recipient: str, subject: str, body: str, style: str) -> str:
        raise NotImplementedError("Async not implemented for EmailSenderTool")

class DataRetrievalInput(BaseModel):
    user_id: str = Field(description="the ID of the user for whom to retrieve data")

class DataRetrievalTool(BaseTool):
    name: str = "data_retrieval"
    description: str = "Retrieves user-specific data like order history."
    args_schema: Type[BaseModel] = DataRetrievalInput

    def _run(self, user_id: str) -> str:
        return data_retrieval_service.get_order_history(user_id)

    async def _arun(self, user_id: str) -> str:
        raise NotImplementedError("Async not implemented for DataRetrievalTool")

# --- 3. Proactive & Learning Layer ---

# Mock interaction logger
class InteractionLogger:
    def log_interaction(self, user_id: str, query: str, response: str, tool_calls: list, feedback: Optional[str] = None):
        print(f"--- Interaction Log for {user_id} ---")
        print(f"Query: {query}")
        print(f"Response: {response}")
        print(f"Tool Calls: {tool_calls}")
        if feedback: print(f"Feedback: {feedback}")
        print("-----------------------------------")

interaction_logger = InteractionLogger()

# --- 4. Large Language Model (LLM) Integration ---

# Initialize LLM (OpenAI as an example)
# Ensure OPENAI_API_KEY environment variable is set
llm = ChatOpenAI(temperature=0, model="gpt-4o") # Using gpt-4o for better tool use

# Define the tools available to the agent
tools = [
    KnowledgeBaseSearchTool(),
    TicketEscalationTool(),
    EmailSenderTool(),
    DataRetrievalTool()
]

# LangChain memory for conversational context
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Personalized Agent Prompt
# The system message now includes dynamic user profile information
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a personalized customer support agent. Adapt your responses and tool usage based on the user's communication style, preferred resolution, and past interactions. Current user profile: {user_profile_info}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# Construct the LangChain agent
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=memory, handle_parsing_errors=True)

# --- 5. API & Deployment ---

app = FastAPI(title="Personalized Customer Support Agent API")

class ChatRequest(BaseModel):
    user_id: str
    message: str
    session_id: Optional[str] = None # For potential future session management

class ChatResponse(BaseModel):
    response: str
    tool_calls: list
    user_profile_updated: bool = False
    detected_style: Optional[str] = None

@app.post("/chat", response_model=ChatResponse)
async def chat_with_agent(request: ChatRequest):
    user_id = request.user_id
    user_message = request.message

    user_profile = user_profile_db.get_user_profile(user_id)
    if not user_profile:
        user_profile = {"communication_style": "neutral", "preferred_resolution": "unknown", "history": []}
        user_profile_db.update_user_profile(user_id, user_profile) # Initialize if new user

    # Analyze user's communication style dynamically
    detected_style = style_analyzer.analyze_style(user_message)
    if detected_style != user_profile.get("communication_style", "neutral"):
        user_profile_db.update_user_profile(user_id, {"communication_style": detected_style})
        user_profile["communication_style"] = detected_style # Update local profile for current turn
        profile_updated = True
    else:
        profile_updated = False

    # Prepare user profile info for the LLM prompt
    user_profile_info = (
        f"Style: {user_profile.get('communication_style', 'neutral')}, "
        f"Preferred Resolution: {user_profile.get('preferred_resolution', 'unknown')}, "
        f"Recent History: {user_profile.get('history', [])[-2:]}"
    )

    try:
        # Invoke the agent with personalized context
        response = await agent_executor.ainvoke({
            "input": user_message,
            "user_profile_info": user_profile_info,
            "chat_history": memory.buffer_as_messages # Pass chat history from memory
        })
        agent_response_content = response["output"]
        # LangChain AgentExecutor doesn't directly return tool calls from 'output' in this setup without custom parsing
        # For simplicity, we'll log them in the _run methods of tools
        tool_calls_made = [] # This would require parsing the agent's thought process if needed precisely

        # Log interaction
        interaction_logger.log_interaction(
            user_id=user_id,
            query=user_message,
            response=agent_response_content,
            tool_calls=tool_calls_made, # Placeholder for now
            feedback=None # Future: integrate user feedback
        )

        return ChatResponse(
            response=agent_response_content,
            tool_calls=tool_calls_made,
            user_profile_updated=profile_updated,
            detected_style=detected_style
        )
    except Exception as e:
        print(f"Error during agent invocation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Example of how to run the FastAPI app locally
    # In a real deployment, you would use 'uvicorn customer_support_agent:app --reload'
    import uvicorn
    print("Starting FastAPI application. Visit http://127.0.0.1:8000/docs for API documentation.")
    uvicorn.run(app, host="127.0.0.1", port=8000)

