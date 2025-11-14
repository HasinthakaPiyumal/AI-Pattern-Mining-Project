import os
from langchain_community.llms import FakeListLLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from typing import List, Union, Dict, Any

# --- 1. Core LLM Integration (Mocked LLM) ---
# Using FakeListLLM for a self-contained example without actual API keys.
# In a real application, replace this with a powerful LLM like ChatOpenAI, LlamaCpp, etc.
# For example: from langchain_openai import ChatOpenAI
# llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
llm = FakeListLLM(responses=[
    "I am processing your request. Please wait.", # Generic for RAG path
    "That sounds like an inquiry about our products. Let me check the knowledge base.", # Generic for RAG path
    "Your order is currently in transit.", # For agent path simulation
    "The SuperWidget Pro is available.", # For agent path simulation
    "Sorry, I couldn't find detailed information about that.", # Fallback
    "I am here to help you with any questions about our e-commerce platform."
])

# --- 2. Memory System ---
# Short-Term Memory: Maintains conversational context within a session.
conversation_memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True
)

# Long-Term Memory (Knowledge Base & RAG)
# Embedding Model: Downloads 'all-MiniLM-L6-v2' if not present.
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Sample product/FAQ documents for the knowledge base.
docs = [
    Document(page_content="The 'SuperWidget Pro' features a 12MP camera, 64GB storage, and a 5.5-inch OLED display. Price: $499."),
    Document(page_content="Shipping usually takes 3-5 business days for standard delivery within the US. Express delivery options are available."),
    Document(page_content="To return an item, please visit our returns portal within 30 days of purchase with your order number."),
    Document(page_content="Our customer support is available 24/7 via chat or phone."),
    Document(page_content="The 'EcoBlend Blender' is perfect for smoothies and comes with a 2-year warranty. Price: $89."),
    Document(page_content="You can track your order using the tracking number provided in your shipping confirmation email."),
    Document(page_content="What is the warranty for electronics? Most electronics come with a 1-year manufacturer's warranty."),
    Document(page_content="How do I change my shipping address? You can update your shipping address in your account settings before the order ships.")
]

# ChromaDB as Vector Store: Uses an in-memory database for this example.
# For persistence, provide a path: Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")
vectorstore = Chroma.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()

# RAG Chain:
# 1. Contextualize the user's question based on chat history.
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# 2. Answer the contextualized question using retrieved documents and chat history.
qa_system_prompt = """You are an enthusiastic e-commerce customer support assistant. \
Use the following pieces of retrieved context and the chat history to answer the question. \
If you don't know the answer, just say that you don't know, don't try to make up an answer. \
Keep the answer concise and helpful."""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Combine into a full RAG chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# --- 3. Query Complexity Classifier ---
def classify_query_complexity(query: str) -> str:
    """
    Classifies a query as 'simple' (addressable by RAG) or 'complex' (requiring tools).
    This is a rule-based classifier for demonstration. A real-world application might
    use a fine-tuned classification model.
    """
    query_lower = query.lower()
    # Keywords indicating a complex query that might require tool usage
    complex_keywords = ["order status", "track my order", "return item", "change address", "product availability", "stock"]
    
    if any(keyword in query_lower for keyword in complex_keywords):
        return "complex"
    return "simple"

# --- 4. Adaptive Processing Strategies (Agent-based for complex queries) ---

# Define Agent Tools (simulated interactions with e-commerce systems)
@tool
def order_lookup_tool(order_id: str) -> str:
    """
    Looks up the status of an e-commerce order using its ID.
    Input should be a string representing the order ID (e.g., "12345").
    """
    print(f"[TOOL CALL] order_lookup_tool called with order_id: {order_id}")
    if order_id == "12345":
        return "Order #12345 is currently being processed and is expected to ship within 2 business days."
    elif order_id == "67890":
        return "Order #67890 was delivered on October 26, 2023."
    else:
        return f"Sorry, I couldn't find an order with ID: {order_id}. Please double-check the order number."

@tool
def product_availability_tool(product_name: str) -> str:
    """
    Checks the current availability and stock of a specific product.
    Input should be a string representing the product name (e.g., "SuperWidget Pro").
    """
    print(f"[TOOL CALL] product_availability_tool called with product_name: {product_name}")
    product_name_lower = product_name.lower()
    if "superwidget pro" in product_name_lower:
        return "SuperWidget Pro is currently in stock with over 100 units available."
    elif "ecoblend blender" in product_name_lower:
        return "EcoBlend Blender is low in stock, only 15 units remaining."
    else:
        return f"Availability information for '{product_name}' is not currently in our system. Please check the product page."

tools = [order_lookup_tool, product_availability_tool]

# Agent Prompt: Guides the agent on how to use tools and respond.
agent_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an AI assistant designed to help customers with their e-commerce queries. You have access to tools to look up order status and product availability. Prioritize using tools for specific factual queries like order status or product stock."),
        MessagesPlaceholder("chat_history"), # Incorporate chat history
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"), # For agent's internal thought process
    ]
)

# Create Agent Executor:
# Note: FakeListLLM is not suitable for complex agent reasoning as it lacks true understanding.
# For a functional agent, a more capable LLM is required.
agent_executor = AgentExecutor(
    agent=create_react_agent(llm, tools, agent_prompt), # Uses ReAct agent framework
    tools=tools,
    verbose=True, # Shows agent's thought process
    handle_parsing_errors=True, # Robustness for potentially malformed LLM outputs
    memory=conversation_memory # Share the same conversation memory with RAG
)

# --- 5. System Orchestration ---
class AdaptiveCustomerSupportAssistant:
    def __init__(self, rag_chain, agent_executor, query_classifier, memory):
        self.rag_chain = rag_chain
        self.agent_executor = agent_executor
        self.query_classifier = query_classifier
        self.memory = memory

    def process_query(self, query: str) -> str:
        # Load current chat history from memory for contextualization
        chat_history = self.memory.load_memory_variables({})["chat_history"]

        # Classify the incoming query
        complexity = self.query_classifier(query)
        print(f"\n[DEBUG] Query classified as: {complexity}")

        response = ""
        if complexity == "simple":
            print("[DEBUG] Using RAG chain for simple query (knowledge base retrieval).")
            try:
                # RAG chain takes 'input' and 'chat_history'
                result = self.rag_chain.invoke({"input": query, "chat_history": chat_history})
                response = result["answer"]
            except Exception as e:
                response = f"An error occurred while retrieving information: {e}"
        else:  # complex query
            print("[DEBUG] Using Agent executor for complex query (tool usage).")
            try:
                # Agent executor takes 'input' and automatically uses its linked memory.
                # We explicitly pass chat_history as well to ensure it's available for the agent's prompt.
                result = self.agent_executor.invoke({"input": query, "chat_history": chat_history})
                response = result["output"]
            except Exception as e:
                # This can happen if FakeListLLM generates an unparsable response for the agent.
                response = f"An error occurred while processing your complex query with the agent. This might be due to the mocked LLM's limitations. Please try again or rephrase. Error: {e}"
        
        # LangChain's memory objects usually handle saving context when integrated with chains/agents.
        # However, for explicit demonstration or if not directly integrated, you might manually save:
        # self.memory.save_context({"input": query}, {"output": response})

        return response

# --- Main Execution Loop ---
if __name__ == "__main__":
    assistant = AdaptiveCustomerSupportAssistant(
        rag_chain=rag_chain,
        agent_executor=agent_executor,
        query_classifier=classify_query_complexity,
        memory=conversation_memory
    )

    print("Welcome to the E-commerce Customer Support Assistant! Type 'exit' to quit.")
    print("Try asking about: 'SuperWidget Pro features', 'shipping time', 'my order status for 12345'.")
    while True:
        user_query = input("\nYou: ")
        if user_query.lower() == 'exit':
            print("Assistant: Goodbye!")
            break

        assistant_response = assistant.process_query(user_query)
        print(f"Assistant: {assistant_response}")

        # Optional: Print full chat history after each turn for debugging
        # current_history = conversation_memory.load_memory_variables({})["chat_history"]
        # print("\n--- Current Chat History ---")
        # for msg in current_history:
        #     print(f"{type(msg).__name__}: {msg.content}")
        # print("----------------------------")
