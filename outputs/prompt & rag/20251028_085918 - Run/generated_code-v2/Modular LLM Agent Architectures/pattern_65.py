import os
import json
import random
from typing import List, Dict, Any, Optional

import streamlit as st
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain.tools import tool
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from langchain.memory import ChatMessageHistory

from transformers import pipeline

mock_products_db = {
    "P101": {"name": "Laptop Pro", "price": 1200.00, "stock": 50, "description": "High-performance laptop for professionals."},
    "P102": {"name": "Gaming Mouse X", "price": 75.00, "stock": 200, "description": "Ergonomic gaming mouse with customizable buttons."},
    "P103": {"name": "USB-C Hub", "price": 30.00, "stock": 300, "description": "Multi-port USB-C hub with HDMI and card reader."},
    "P104": {"name": "Wireless Earbuds", "price": 150.00, "stock": 120, "description": "Premium wireless earbuds with noise cancellation."},
}

mock_orders_db = {
    "ORD001": {"customer_id": "C1", "product_id": "P101", "status": "Shipped", "tracking_id": "TRK987654321"},
    "ORD002": {"customer_id": "C2", "product_id": "P102", "status": "Processing", "tracking_id": None},
    "ORD003": {"customer_id": "C1", "product_id": "P104", "status": "Delivered", "tracking_id": "TRK123456789"},
}

mock_customer_profiles_db = {
    "C1": {"name": "Alice", "email": "alice@example.com", "preferences": {"newsletter": True, "product_category": "electronics"}, "purchase_history": ["ORD001", "ORD003"]},
    "C2": {"name": "Bob", "email": "bob@example.com", "preferences": {"newsletter": False, "product_category": "gaming"}, "purchase_history": ["ORD002"]},
}

mock_knowledge_base_documents = [
    "Our return policy allows returns within 30 days of purchase for a full refund. Items must be in original condition.",
    "For technical support, please visit our website's support section or contact us via email.",
    "Shipping usually takes 3-5 business days for standard delivery within the country.",
    "The Laptop Pro comes with a 1-year manufacturer's warranty.",
    "We accept all major credit cards, PayPal, and Apple Pay.",
]

class KnowledgeBaseQueryInput(BaseModel):
    query: str = Field(..., description="The query string to search in the knowledge base.")

class OrderStatusInput(BaseModel):
    order_id: str = Field(..., description="The ID of the order to retrieve status for.")

class ProductDetailsInput(BaseModel):
    product_id: str = Field(..., description="The ID of the product to retrieve details for.")

class CustomerHistoryInput(BaseModel):
    customer_id: str = Field(..., description="The ID of the customer to retrieve purchase history for.")

class CustomerPreferencesInput(BaseModel):
    customer_id: str = Field(..., description="The ID of the customer to retrieve preferences for.")

class UpdateCustomerPreferencesInput(BaseModel):
    customer_id: str = Field(..., description="The ID of the customer whose preferences to update.")
    preference_key: str = Field(..., description="The key of the preference to update (e.g., 'newsletter', 'product_category').")
    preference_value: Any = Field(..., description="The new value for the preference.")

class AnalyzeSentimentInput(BaseModel):
    text: str = Field(..., description="The text to analyze sentiment for.")

class ValidateProductIdInput(BaseModel):
    product_id: str = Field(..., description="The product ID to validate.")

class KnowledgeBaseRetriever:
    def __init__(self):
        self.embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = Chroma.from_texts(
            mock_knowledge_base_documents, self.embeddings, collection_name="ecommerce_kb"
        )
        self.retriever = self.vectorstore.as_retriever()

    @tool("knowledge_base_query", args_schema=KnowledgeBaseQueryInput)
    def query_knowledge_base(self, query: str) -> str:
        docs = self.retriever.invoke(query)
        return "\n".join([doc.page_content for doc in docs])

class ECommerceAPI:
    @tool("get_order_status", args_schema=OrderStatusInput)
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        status = mock_orders_db.get(order_id)
        return status if status else {"error": "Order not found."}

    @tool("get_product_details", args_schema=ProductDetailsInput)
    def get_product_details(self, product_id: str) -> Dict[str, Any]:
        details = mock_products_db.get(product_id)
        return details if details else {"error": "Product not found."}

    @tool("get_customer_purchase_history", args_schema=CustomerHistoryInput)
    def get_customer_purchase_history(self, customer_id: str) -> Dict[str, Any]:
        customer_data = mock_customer_profiles_db.get(customer_id)
        return {"purchase_history": customer_data.get("purchase_history", [])} if customer_data else {"error": "Customer not found."}

class UserProfileManager:
    @tool("get_customer_preferences", args_schema=CustomerPreferencesInput)
    def get_customer_preferences(self, customer_id: str) -> Dict[str, Any]:
        customer_data = mock_customer_profiles_db.get(customer_id)
        return {"preferences": customer_data.get("preferences", {})} if customer_data else {"error": "Customer not found."}

    @tool("update_customer_preferences", args_schema=UpdateCustomerPreferencesInput)
    def update_customer_preferences(self, customer_id: str, preference_key: str, preference_value: Any) -> Dict[str, Any]:
        if customer_id in mock_customer_profiles_db:
            mock_customer_profiles_db[customer_id]["preferences"][preference_key] = preference_value
            return {"status": "success", "message": f"Preference '{preference_key}' updated for customer '{customer_id}'."}
        return {"error": "Customer not found."}

class UtilityTools:
    def __init__(self):
        self.sentiment_analyzer = pipeline("sentiment-analysis")

    @tool("analyze_sentiment", args_schema=AnalyzeSentimentInput)
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        result = self.sentiment_analyzer(text)
        return result[0]

    @tool("validate_product_id", args_schema=ValidateProductIdInput)
    def validate_product_id(self, product_id: str) -> Dict[str, bool]:
        return {"is_valid": product_id in mock_products_db}

def setup_agent():
    llm = ChatOpenAI(model="gpt-4", temperature=0)

    kb_retriever = KnowledgeBaseRetriever()
    ecommerce_api = ECommerceAPI()
    user_profile_manager = UserProfileManager()
    utility_tools = UtilityTools()

    tools = [
        kb_retriever.query_knowledge_base,
        ecommerce_api.get_order_status,
        ecommerce_api.get_product_details,
        ecommerce_api.get_customer_purchase_history,
        user_profile_manager.get_customer_preferences,
        user_profile_manager.update_customer_preferences,
        utility_tools.analyze_sentiment,
        utility_tools.validate_product_id,
    ]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an intelligent e-commerce customer support agent. Answer user questions truthfully and accurately using the available tools. If a tool returns an error or no information, inform the user you cannot find the requested information. Always be polite and helpful."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return agent_executor

def main():
    st.set_page_config(page_title="E-commerce Support Agent", layout="wide")
    st.title("🛍️ Intelligent E-commerce Customer Support Agent")

    if "agent_executor" not in st.session_state:
        st.session_state.agent_executor = setup_agent()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = ChatMessageHistory()

    st.sidebar.header("Agent Controls")
    if st.sidebar.button("Clear Chat History"):
        st.session_state.chat_history = ChatMessageHistory()
        st.rerun()

    st.sidebar.subheader("Mock Databases")
    st.sidebar.json(mock_products_db, expanded=False)
    st.sidebar.json(mock_orders_db, expanded=False)
    st.sidebar.json(mock_customer_profiles_db, expanded=False)

    for message in st.session_state.chat_history.messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)

    user_query = st.chat_input("How can I help you today?")

    if user_query:
        st.chat_message("user").markdown(user_query)
        st.session_state.chat_history.add_user_message(user_query)

        with st.spinner("Thinking..."):
            try:
                response = st.session_state.agent_executor.invoke(
                    {"input": user_query, "chat_history": st.session_state.chat_history.messages}
                )
                agent_response = response["output"]
                st.chat_message("assistant").markdown(agent_response)
                st.session_state.chat_history.add_ai_message(agent_response)
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.chat_message("assistant").markdown("I apologize, but I encountered an error while processing your request. Please try again or rephrase your question.")

if __name__ == "__main__":
    main()
