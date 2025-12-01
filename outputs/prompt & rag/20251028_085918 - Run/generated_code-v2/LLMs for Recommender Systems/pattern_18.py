import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import json

# --- 1. Simulated E-commerce Data/APIs ---

mock_products = {
    "1001": {"id": "1001", "name": "Blue Denim Jeans", "category": "Apparel", "price": 49.99, "description": "Classic blue denim jeans, regular fit.", "in_stock": True},
    "1002": {"id": "1002", "name": "Red T-Shirt", "category": "Apparel", "price": 19.99, "description": "Comfortable cotton t-shirt in vibrant red.", "in_stock": True},
    "1003": {"id": "1003", "name": "Wireless Bluetooth Headphones", "category": "Electronics", "price": 99.99, "description": "High-quality sound with noise cancellation.", "in_stock": False},
    "1004": {"id": "1004", "name": "Leather Wallet", "category": "Accessories", "price": 29.99, "description": "Genuine leather wallet with multiple card slots.", "in_stock": True},
    "1005": {"id": "1005", "name": "Laptop Backpack", "category": "Bags", "price": 59.99, "description": "Durable backpack with laptop compartment.", "in_stock": True},
}

def search_products_api(query: str, category: Optional[str] = None, price_range: Optional[str] = None) -> List[Dict]:
    results = []
    query_lower = query.lower()
    for prod_id, product in mock_products.items():
        if query_lower in product["name"].lower() or query_lower in product["description"].lower():
            if category and product["category"].lower() != category.lower():
                continue
            # Simple price range parsing for demonstration
            if price_range:
                try:
                    min_price, max_price = map(float, price_range.split('-'))
                    if not (min_price <= product["price"] <= max_price):
                        continue
                except ValueError:
                    pass # Ignore invalid price_range
            results.append({"id": product["id"], "name": product["name"], "price": product["price"], "category": product["category"]})
    return results

def get_product_details_api(product_id: str) -> Optional[Dict]:
    return mock_products.get(product_id)

def get_recommendations_api(user_id: str, context_product_id: Optional[str] = None) -> List[Dict]:
    # Simplified recommendations: just return some popular items or related to context
    if context_product_id == "1001": # Blue Denim Jeans
        return [
            {"id": "1002", "name": "Red T-Shirt", "price": 19.99},
            {"id": "1004", "name": "Leather Wallet", "price": 29.99}
        ]
    return [
        {"id": "1001", "name": "Blue Denim Jeans", "price": 49.99},
        {"id": "1005", "name": "Laptop Backpack", "price": 59.99}
    ]

# --- 2. Pydantic Models for Tool Inputs ---

class ProductSearchInput(BaseModel):
    query: str = Field(description="The product name or keywords to search for.")
    category: Optional[str] = Field(None, description="Optional category to filter products by (e.g., 'Apparel', 'Electronics').")
    price_range: Optional[str] = Field(None, description="Optional price range (e.g., '20-50') to filter products by.")

class ProductDetailsInput(BaseModel):
    product_id: str = Field(description="The ID of the product to retrieve details for.")

class RecommendationsInput(BaseModel):
    user_id: str = Field(description="The ID of the user for whom to get recommendations.")
    context_product_id: Optional[str] = Field(None, description="Optional ID of a product to get context-specific recommendations (e.g., similar items).")

# --- 3. Langchain Tools ---

ecommerce_tools = [
    Tool(
        name="search_products",
        func=search_products_api,
        description="Searches for products based on a query, category, and price range.",
        args_schema=ProductSearchInput,
    ),
    Tool(
        name="get_product_details",
        func=get_product_details_api,
        description="Retrieves detailed information about a specific product using its ID.",
        args_schema=ProductDetailsInput,
    ),
    Tool(
        name="get_recommendations",
        func=lambda user_id, context_product_id=None: get_recommendations_api(user_id, context_product_id),
        description="Provides personalized product recommendations based on user history or a specific product context.",
        args_schema=RecommendationsInput,
    ),
]

# --- 4. Memory Module ---

class MemoryModule:
    def __init__(self, user_id: str = "default_user"):
        self.user_id = user_id
        self.user_profile: Dict[str, Any] = {"preferences": [], "past_interactions": []}
        self.conversation_history: List[Dict[str, str]] = []

    def add_message(self, role: str, content: str):
        self.conversation_history.append({"role": role, "content": content})

    def get_history_for_llm(self, limit: int = 5) -> str:
        # Format history for LLM context, showing recent turns
        formatted_history = []
        for msg in self.conversation_history[-limit:]:
            formatted_history.append(f"{msg['role'].capitalize()}: {msg['content']}")
        return "\n".join(formatted_history)

    def update_user_profile(self, facts: Dict):
        if "preferences" in facts and isinstance(facts["preferences"], list):
            for pref in facts["preferences"]:
                if pref not in self.user_profile["preferences"]:
                    self.user_profile["preferences"].append(pref)
        # Add more sophisticated logic for updating other profile aspects
        

    def retrieve_user_context(self) -> str:
        context_parts = []
        if self.user_profile["preferences"]:
            context_parts.append(f"User preferences: {', '.join(self.user_profile['preferences'])}")
        if self.user_profile["past_interactions"]:
             context_parts.append(f"Recent past interactions: {', '.join(self.user_profile['past_interactions'])}")
        
        # In a real system, this would involve more sophisticated retrieval (e.g., vector search)
        return "\n".join(context_parts)

# --- 5. LLM Dialogue Module (Langchain Agent) ---

# Use ChatOpenAI. Ensure OPENAI_API_KEY is set in your environment variables.
# For local testing without a real API key, you could use a mocked LLM or a local model.
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Agent prompt
agent_prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful conversational shopping assistant. Your goal is to understand user preferences, provide personalized product recommendations, and answer product-related queries. You have access to e-commerce tools. When making recommendations, always ask the user about their budget or specific needs if not provided."),
    ("system", "User profile and recent conversation history:\n{user_context}\n{chat_history}"),
    ("placeholder", "{agent_scratchpad}"),
    ("human", "{input}"),
])

# Create the ReAct agent
agent = create_react_agent(llm, ecommerce_tools, agent_prompt_template)
agent_executor = AgentExecutor(agent=agent, tools=ecommerce_tools, verbose=True, handle_parsing_errors=True)

# --- 6. Streamlit UI ---

st.set_page_config(page_title="AI Shopping Assistant")
st.title("🛍️ AI Conversational Shopping Assistant")

# Initialize session state for memory and chat history
if "memory" not in st.session_state:
    st.session_state.memory = MemoryModule(user_id="streamlit_user_123")
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What are you looking for today?"):
    # Add user message to chat history and memory
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.memory.add_message("user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Thinking..."):
        # Retrieve user context from memory
        user_context_str = st.session_state.memory.retrieve_user_context()
        chat_history_str = st.session_state.memory.get_history_for_llm()

        # Invoke the agent
        try:
            response = agent_executor.invoke({
                "input": prompt,
                "user_context": user_context_str,
                "chat_history": chat_history_str,
            })
            assistant_response = response["output"]
        except Exception as e:
            assistant_response = f"An error occurred: {e}. Please try again."
            print(f"Agent execution error: {e}")

        # Add assistant response to chat history and memory
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        st.session_state.memory.add_message("assistant", assistant_response)
        
        # --- Simplified Fact Extraction and Profile Update (for demo) ---
        # In a real system, the LLM itself would be prompted to extract facts
        # or a separate NLU component would do this.
        if "jeans" in prompt.lower() and "blue" in prompt.lower():
            st.session_state.memory.update_user_profile({"preferences": ["blue jeans"]})

        with st.chat_message("assistant"):
            st.markdown(assistant_response)

# Display current user profile for debugging/demonstration
st.sidebar.subheader("User Profile (Memory)")
st.sidebar.json(st.session_state.memory.user_profile)
st.sidebar.subheader("Recent Chat History")
st.sidebar.text(st.session_state.memory.get_history_for_llm(limit=10))