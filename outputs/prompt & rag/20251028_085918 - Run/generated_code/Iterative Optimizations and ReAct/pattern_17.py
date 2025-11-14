import os
import requests
from typing import List, Dict, Any, Optional

# LangChain components for building the agent
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_models import ChatOpenAI  # For real LLM integration
from langchain_community.llms import FakeListLLM  # For mocking LLM if API key is not present
from langchain.memory import ConversationBufferWindowMemory

# FastAPI for building the web API
import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel

# Loguru for structured logging
from loguru import logger

# --- Configuration and Environment Variables ---
# Set your OpenAI API key here or as an environment variable (recommended)
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
USE_REAL_LLM = OPENAI_API_KEY is not None

# Configure Loguru
logger.remove()
logger.add("agent.log", rotation="10 MB", level="INFO")
logger.add(os.sys.stderr, level="INFO")

# --- Mock Data and External Services (for demonstration) ---

# Mock Knowledge Base for `knowledge_base_search` tool
MOCK_KNOWLEDGE_BASE = {
    "product_returns": "Our return policy allows returns within 30 days of purchase with a valid receipt. Items must be unused and in original packaging. Refunds are processed within 5-7 business days.",
    "shipping_options": "We offer standard shipping (5-7 business days), express shipping (2-3 business days), and overnight shipping. Costs vary based on destination and speed.",
    "account_reset": "To reset your password, visit our website and click 'Forgot Password' on the login page. Follow the instructions sent to your registered email.",
    "technical_support": "For technical issues, please describe your problem in detail. Our support team is available Monday-Friday, 9 AM - 5 PM EST. You can also visit our troubleshooting guide on our website.",
    "billing_inquiry": "For billing inquiries, please provide your account number and the date of the charge. We can help clarify charges, update payment methods, or assist with invoice requests.",
}

# Mock CRM Database for `get_customer_info` tool
MOCK_CRM_DATABASE = {
    "user_123": {
        "name": "Alice Smith",
        "email": "alice@example.com",
        "order_history": ["ORD-001", "ORD-002"],
        "ticket_status": "open",
        "last_interaction": "2023-10-26"
    },
    "user_456": {
        "name": "Bob Johnson",
        "email": "bob@example.com",
        "order_history": ["ORD-003"],
        "ticket_status": "closed",
        "last_interaction": "2023-10-20"
    }
}

# Mock External API Responses for `track_order_status` and `get_product_details` tools
MOCK_EXTERNAL_APIS = {
    "order_tracking": {
        "ORD-001": {"status": "shipped", "estimated_delivery": "2023-10-28"},
        "ORD-002": {"status": "delivered", "delivered_on": "2023-10-25"},
        "ORD-003": {"status": "processing", "estimated_delivery": "2023-11-01"},
    },
    "product_catalog": {
        "PROD-A": {"name": "Laptop Pro", "price": 1200, "in_stock": True},
        "PROD-B": {"name": "Wireless Mouse", "price": 25, "in_stock": True},
    }
}

# --- Tool Definitions ---

@tool
def knowledge_base_search(query: str) -> str:
    """Searches the internal knowledge base for information relevant to the query.
    Useful for answering FAQs, product information, or troubleshooting steps.
    Input should be a concise search query."""
    logger.info(f"Tool call: knowledge_base_search with query='{query}'")
    query = query.lower()
    for key, content in MOCK_KNOWLEDGE_BASE.items():
        if query in key or query in content.lower():
            logger.info(f"Knowledge Base found: {content}")
            return content
    logger.warning(f"Knowledge Base found no results for query='{query}'")
    return "I couldn't find specific information in our knowledge base for that query. Can you please rephrase or provide more details?"

@tool
def get_customer_info(user_id: str) -> str:
    """Retrieves customer information from the CRM system using a user ID.
    Returns customer details like name, email, order history, and ticket status.
    Input should be a customer identifier (e.g., 'user_123')."""
    logger.info(f"Tool call: get_customer_info with user_id='{user_id}'")
    info = MOCK_CRM_DATABASE.get(user_id)
    if info:
        logger.info(f"CRM info found: {info}")
        return str(info)
    logger.warning(f"CRM found no info for user_id='{user_id}'")
    return f"Could not find customer information for user ID: {user_id}. Please verify the ID."

@tool
def track_order_status(order_id: str) -> str:
    """Checks the status of a specific order using the order ID.
    Returns the current status and estimated delivery or delivery date.
    Input should be an order identifier (e.g., 'ORD-001')."""
    logger.info(f"Tool call: track_order_status with order_id='{order_id}'")
    status = MOCK_EXTERNAL_APIS["order_tracking"].get(order_id)
    if status:
        logger.info(f"Order status found: {status}")
        return str(status)
    logger.warning(f"Order tracking found no status for order_id='{order_id}'")
    return f"Could not find status for order ID: {order_id}. Please verify the order ID."

@tool
def get_product_details(product_id: str) -> str:
    """Retrieves details for a specific product from the product catalog.
    Returns product name, price, and stock availability.
    Input should be a product identifier (e.g., 'PROD-A')."""
    logger.info(f"Tool call: get_product_details with product_id='{product_id}'")
    details = MOCK_EXTERNAL_APIS["product_catalog"].get(product_id)
    if details:
        logger.info(f"Product details found: {details}")
        return str(details)
    logger.warning(f"Product catalog found no details for product_id='{product_id}'")
    return f"Could not find details for product ID: {product_id}. Please verify the product ID."

@tool
def handoff_to_human_agent(reason: str) -> str:
    """Flags the current conversation for human agent intervention.
    Use this when the AI agent cannot resolve the issue or the user explicitly requests human help.
    Input should be a brief reason for the handoff."""
    logger.info(f"Tool call: handoff_to_human_agent with reason='{reason}'")
    return f"Acknowledged. I'm escalating this to a human agent due to: '{reason}'. Please wait while I connect you."

# Collect all tools for the LangChain agent
tools = [
    knowledge_base_search,
    get_customer_info,
    track_order_status,
    get_product_details,
    handoff_to_human_agent,
]

# --- Feedback Mechanism and Self-Correction (Simplified Implementation) ---

class FeedbackMechanism:
    """Simulates capturing and analyzing user feedback for self-correction."""
    def __init__(self):
        self.feedback_history = []

    def capture_feedback(self, user_query: str, agent_response: str, feedback: Optional[str] = None) -> None:
        """Captures user feedback and performs a basic sentiment analysis."""
        sentiment = self._analyze_sentiment(user_query, agent_response, feedback)
        self.feedback_history.append({
            "query": user_query,
            "response": agent_response,
            "feedback": feedback,
            "sentiment": sentiment,
            "timestamp": os.path.getmtime(__file__) # Mock timestamp
        })
        logger.info(f"Feedback captured: Sentiment='{sentiment}', Feedback='{feedback}'")

    def _analyze_sentiment(self, user_query: str, agent_response: str, feedback: Optional[str]) -> str:
        """
        A very basic keyword-based sentiment analysis for demonstration.
        In a real application, you'd use NLTK, spaCy, or a dedicated sentiment model.
        """
        text_to_analyze = f"{user_query} {agent_response} {feedback or ''}".lower()
        if "unhappy" in text_to_analyze or "wrong" in text_to_analyze or "not helpful" in text_to_analyze or "bad" in text_to_analyze:
            return "negative"
        if "thank you" in text_to_analyze or "helpful" in text_to_analyze or "resolved" in text_to_analyze or "good" in text_to_analyze:
            return "positive"
        return "neutral"

    def get_recent_feedback(self, count: int = 5) -> List[Dict[str, Any]]:
        """Returns the most recent feedback entries."""
        return self.feedback_history[-count:]

# --- Adaptive Customer Support Agent Core Logic ---

class AdaptiveCustomerSupportAgent:
    """
    An intelligent customer support agent leveraging LangChain for adaptive reasoning,
    tool integration, and a simplified self-correction loop.
    """
    def __init__(self):
        self.feedback_mechanism = FeedbackMechanism()
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=5  # Keep the last 5 turns of conversation in memory
        )

        # Initialize LLM - use real OpenAI if key is present, otherwise a mock LLM
        if USE_REAL_LLM:
            self.llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
            logger.info("Using real OpenAI LLM (gpt-3.5-turbo or similar based on default).")
        else:
            # Fallback to a mock LLM for local testing without an API key
            self.llm = FakeListLLM(
                responses=[
                    "Hello! How can I help you today?",
                    "I am processing your request...",
                    "Let me check that for you using my tools.",
                    "I found some information regarding your query.",
                    "It seems I need to ask a human for this. Would you like me to connect you?",
                    "Please provide more details so I can assist you better.",
                    "I'm sorry, I couldn't find a definitive answer.",
                ]
            )
            logger.warning("OPENAI_API_KEY not found. Using FakeListLLM for demonstration. Functionality will be limited.")

        # Define the system prompt for the agent
        self.prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=(
                    "You are an intelligent customer support agent designed to assist users with their queries. "
                    "You have access to various tools to help resolve issues. "
                    "Always try to use the most appropriate tool before answering directly. "
                    "If a query is complex, requires personal information you cannot access, or the tools don't provide a clear answer, "
                    "offer to hand off to a human agent using the 'handoff_to_human_agent' tool. "
                    "Be polite, helpful, and concise. "
                    "Leverage your conversation history to maintain context throughout the interaction."
                )),
                MessagesPlaceholder(variable_name="chat_history"), # For conversational memory
                HumanMessage(content="{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"), # For agent's thought process
            ]
        )

        # Create the LangChain ReAct agent
        self.agent = create_react_agent(self.llm, tools, self.prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=tools, verbose=True, memory=self.memory)
        logger.info("Adaptive Customer Support Agent initialized and ready.")

    def process_query(self, query: str) -> str:
        """Processes a user query and returns the agent's response using the LangChain agent."""
        logger.info(f"Processing query: '{query}'")
        try:
            # Invoke the agent executor with the current query
            response = self.agent_executor.invoke({"input": query})
            agent_response = response.get("output", "I'm sorry, I couldn't process that request.")
            logger.info(f"Agent generated response: '{agent_response}'")
            return agent_response
        except Exception as e:
            logger.error(f"Error during query processing: {e}")
            return "An internal error occurred while processing your request. Please try again or ask for human assistance."

    def apply_self_correction(self, user_query: str, agent_response: str, feedback: str) -> None:
        """
        Simulates the self-correction loop based on user feedback.
        In a production system, this would involve more sophisticated logic:
        1.  **Reflection:** The agent (or a separate LLM) might analyze the feedback and its previous actions.
        2.  **Adaptive Prompting:** Dynamically adjust subsequent prompts or agent instructions.
        3.  **Learning:** Store problem-response-feedback triplets for fine-tuning or reinforcement learning.
        4.  **Tool Refinement:** Identify if a tool was misused or if a new tool is needed.
        5.  **Batch Prompting (Efficiency):** If multiple correction steps can run in parallel, execute them efficiently.
        For this example, it primarily logs the feedback and simulates an internal "learning" acknowledgment.
        """
        self.feedback_mechanism.capture_feedback(user_query, agent_response, feedback)
        sentiment = self.feedback_mechanism.feedback_history[-1]["sentiment"]

        if sentiment == "negative":
            logger.warning(f"Negative feedback received for query '{user_query}'. Initiating internal self-correction reflection.")
            # Example of potential adaptive action: If the agent failed to resolve, it might prioritize
            # offering a human handoff in similar future scenarios.
            if "handoff to human" not in agent_response.lower() and "resolved" not in feedback.lower():
                logger.info("Agent notes that a direct resolution was not achieved; considering proactive human handoff for similar future queries.")
            logger.info("Agent acknowledges negative feedback and will strive to improve its responses and tool usage in future interactions.")
        else:
            logger.info("Positive or neutral feedback received. Agent continues normal operation and reinforces successful patterns.")


# --- FastAPI Application (Backend API) ---

app = FastAPI(
    title="Adaptive Customer Support Agent API",
    description="API for an intelligent customer support agent with adaptive reasoning and tool integration. " 
                "Includes mock tools and a simplified self-correction mechanism."
)

# Pydantic models for request and response validation
class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = None  # Optional field to identify the user
    feedback: Optional[str] = None  # Optional feedback from the user after a response

class ChatResponse(BaseModel):
    response: str
    feedback_status: Optional[str] = None

# Global instance of the agent to maintain state across requests
agent_instance: AdaptiveCustomerSupportAgent = None

@app.on_event("startup")
async def startup_event():
    """Initializes the AdaptiveCustomerSupportAgent when the FastAPI application starts."""
    global agent_instance
    agent_instance = AdaptiveCustomerSupportAgent()
    logger.info("FastAPI application started up. Adaptive Customer Support Agent initialized.")

@app.post("/chat", response_model=ChatResponse)
async def chat_with_agent_api(request: ChatRequest):
    """
    Endpoint for users to chat with the intelligent customer support agent.
    Supports optional user_id for context and feedback for self-correction.
    """
    logger.info(f"Received chat request from user '{request.user_id if request.user_id else 'anonymous'}' with message: '{request.message}'")

    # Process the user's query through the agent
    agent_response = agent_instance.process_query(request.message)

    feedback_status = None
    if request.feedback:
        # If feedback is provided, trigger the self-correction mechanism
        agent_instance.apply_self_correction(request.message, agent_response, request.feedback)
        feedback_status = "Feedback processed. Agent is attempting self-correction based on feedback."

    return ChatResponse(response=agent_response, feedback_status=feedback_status)

@app.get("/health")
async def health_check():
    """Health check endpoint to verify the API and agent status."""
    return {"status": "healthy", "agent_initialized": agent_instance is not None}


# --- Streamlit UI (Frontend for demonstration) ---
# To run this Streamlit application:
# 1. Save this file as `app.py`
# 2. Run `streamlit run app.py` in your terminal
# Note: Streamlit runs its own web server, separate from FastAPI.
# For a combined solution, you'd typically have the Streamlit app call the FastAPI backend.

if __name__ == "__main__":
    import streamlit as st

    # Initialize the agent and chat history in Streamlit's session state
    if "agent" not in st.session_state:
        st.session_state.agent = AdaptiveCustomerSupportAgent()
        st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you today?"}]

    st.title("Intelligent Customer Support Chatbot")

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What can I help you with?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."): # Show a spinner while the agent processes
                full_response = st.session_state.agent.process_query(prompt)
                st.markdown(full_response)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        # Simple feedback mechanism in the UI
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("👍 Helpful", key=f"feedback_pos_{len(st.session_state.messages)}", help="Click if the response was helpful."):
                st.session_state.agent.apply_self_correction(
                    user_query=prompt,
                    agent_response=full_response,
                    feedback="Positive: Response was helpful."
                )
                st.toast("Thanks for the positive feedback! Agent is learning.")
        with col2:
            if st.button("👎 Not Helpful", key=f"feedback_neg_{len(st.session_state.messages)}", help="Click if the response was not helpful."):
                st.session_state.agent.apply_self_correction(
                    user_query=prompt,
                    agent_response=full_response,
                    feedback="Negative: Response was not helpful."
                )
                st.toast("Thanks for your feedback. Agent will try to do better!")

    st.sidebar.header("Agent Status & Info")
    st.sidebar.info(f"LLM Provider: {'OpenAI GPT' if USE_REAL_LLM else 'Mock LLM (API Key Missing)'}")
    st.sidebar.info(f"Conversation Turns in Memory: {len(st.session_state.agent.memory.buffer) // 2}") # Each turn has human+AI
    st.sidebar.subheader("How to Use (FastAPI)")
    st.sidebar.markdown(
        "To run the FastAPI backend independently, save this file (e.g., `main.py`) " 
        "and execute: `uvicorn main:app --reload` (ensure `uvicorn` is installed)." 
        "Then interact with `/chat` endpoint using `POST` requests."
    )
    st.sidebar.subheader("How to Use (Streamlit)")
    st.sidebar.markdown(
        "To run this Streamlit frontend, save this file (e.g., `app.py`) " 
        "and execute: `streamlit run app.py` (ensure `streamlit` is installed)." 
        "This will launch a local web UI for interaction."
    )

    st.sidebar.markdown(
        "**Note:** For full functionality, set your `OPENAI_API_KEY` environment variable."
    )

