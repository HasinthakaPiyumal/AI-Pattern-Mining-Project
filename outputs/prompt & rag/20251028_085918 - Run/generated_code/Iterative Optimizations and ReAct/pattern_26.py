from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import os
from dotenv import load_dotenv

from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import PromptTemplate

load_dotenv()

app = FastAPI()

# --- Langchain Tools ---

@tool
def knowledge_base_search(query: str) -> str:
    """Searches the knowledge base for relevant information based on the query."""
    if "billing issue" in query.lower() or "invoice" in query.lower():
        return "Most billing issues can be resolved by checking your account's 'Billing History' section. You can also update payment methods there."
    elif "password reset" in query.lower():
        return "To reset your password, visit the login page and click 'Forgot Password'. A reset link will be sent to your registered email address."
    elif "product features" in query.lower():
        return "Our product features include real-time analytics, customizable dashboards, and integration with popular CRM systems. More details are available in the 'Features' section of our website."
    return "Could not find an exact match in the knowledge base. Please try rephrasing or provide more details."

@tool
def create_ticket(issue_description: str) -> str:
    """Creates a support ticket for escalation with the provided issue description."""
    ticket_id = f"TICKET-{abs(hash(issue_description)) % (10**6):06d}"
    return f"A support ticket has been created with ID: {ticket_id}. A human agent will review your issue shortly."

class FeedbackSystem:
    def __init__(self):
        self.last_feedback: Dict[str, Any] = {}
        self.resolution_status: bool = False
        self.user_satisfaction: str = ""

    def record_feedback(self, satisfaction: str, resolution: bool, agent_reflection: str = ""):
        self.last_feedback = {
            "satisfaction": satisfaction,
            "resolution": resolution,
            "agent_reflection": agent_reflection
        }
        self.user_satisfaction = satisfaction
        self.resolution_status = resolution

    def get_last_feedback(self) -> Dict[str, Any]:
        return self.last_feedback

feedback_system = FeedbackSystem()

llm = ChatOpenAI(temperature=0, model="gpt-4o")

tools = [
    knowledge_base_search,
    create_ticket,
]

# --- Langchain Agent Setup ---

# Custom prompt template to incorporate feedback
custom_prompt_template = PromptTemplate.from_template(
    """You are an Adaptive Customer Support Agent. Your goal is to resolve customer issues efficiently.

Previous feedback on your last interaction: {last_feedback}

TOOLS:
{tools}

Feel free to use the tools available to you. If a customer is unsatisfied or an issue is not resolved, try a different approach or escalate by creating a ticket if necessary.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

{chat_history}
Question: {input}
Thought:{agent_scratchpad}"""
)

memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=5)

# Create the ReAct agent
agent = create_react_agent(llm, tools, custom_prompt_template)
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True, handle_parsing_errors=True)

# --- FastAPI Models ---

class QueryRequest(BaseModel):
    query: str
    session_id: str

class AgentResponse(BaseModel):
    response: str
    session_id: str
    feedback_prompt: str

class FeedbackRequest(BaseModel):
    session_id: str
    satisfaction: str  # e.g., 