import os
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain.tools import tool
from langchain_core.prompts import PromptTemplate

# --- 1. Knowledge Base Simulation ---
class SimpleKnowledgeBase:
    def __init__(self):
        self.articles = {
            "shipping_policy": "Our standard shipping takes 3-5 business days. Expedited options are available.",
            "return_policy": "Items can be returned within 30 days of purchase with a valid receipt.",
            "password_reset": "To reset your password, visit our login page and click 'Forgot Password'.",
            "product_warranty": "All electronics come with a 1-year manufacturer's warranty.",
            "contact_support": "You can contact support via email at support@example.com or call us at 1-800-555-1234."
        }

    def search(self, query: str) -> str:
        """Searches the knowledge base for relevant information."""
        query_lower = query.lower()
        for key, value in self.articles.items():
            if key in query_lower or any(word in query_lower for word in key.split('_')):
                return value
        return "No direct answer found in the knowledge base. Please try rephrasing or contact support."

# --- 2. Reflexion Agent Implementation ---
class ReflexionCustomerSupportAgent:
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.0):
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        self.knowledge_base = SimpleKnowledgeBase()
        self.reflections: List[str] = []

        # Define tools for the agent
        @tool
        def knowledge_base_search(query: str) -> str:
            """Searches the internal knowledge base for answers to customer queries."""
            return self.knowledge_base.search(query)

        self.tools = [knowledge_base_search]

        # Base prompt for the ReAct agent
        self.base_prompt_template = hub.pull("hwchase17/react")

        # Create the initial agent executor
        self._create_agent_executor()

    def _create_agent_executor(self):
        # Modify the prompt to include reflections
        reflections_context = "\n" + "\n".join([f"Past Reflection: {r}" for r in self.reflections]) if self.reflections else ""
        
        # The ReAct prompt from LangChain Hub already includes `agent_scratchpad`
        # We need to prepend our reflections to the `input` part of the prompt
        # A more robust way would be to create a custom prompt template or chain
        # For simplicity, we'll inject reflections into the primary 'input' string for now

        # LangChain's create_react_agent expects a specific prompt structure. 
        # We'll create a custom prompt to inject reflections directly into the agent's 