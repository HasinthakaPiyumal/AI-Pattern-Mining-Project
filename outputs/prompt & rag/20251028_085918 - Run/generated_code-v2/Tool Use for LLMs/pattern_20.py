import os
import gradio as gr
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool


# --- 1. Environment Setup ---
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")

# --- 2. External Memory System (ChromaDB) and Embeddings ---
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize ChromaDB collections
# Dummy data for demonstration
customer_data = [
    {"id": "cust123", "name": "Alice Smith", "email": "alice@example.com", "order_history": "Order #98765 (status: delivered, date: 2023-10-20), Order #12345 (status: processing, date: 2023-11-15, items: Laptop, Mouse)"},
    {"id": "cust456", "name": "Bob Johnson", "email": "bob@example.com", "order_history": "Order #54321 (status: shipped, date: 2023-11-01)"},
]
product_data = [
    {"id": "prodA", "name": "Laptop Pro X", "description": "High-performance laptop with 16GB RAM and 512GB SSD.", "price": "$1200"},
    {"id": "prodB", "name": "Wireless Mouse", "description": "Ergonomic wireless mouse with customizable buttons.", "price": "$30"},
]
faqs_data = [
    {"question": "How do I check my order status?", "answer": "You can check your order status by logging into your account and navigating to the 'Order History' section."},
    {"question": "What is your return policy?", "answer": "Our return policy allows returns within 30 days of purchase for a full refund."},
]

# Create ChromaDB instances
customer_vectorstore = Chroma.from_texts(
    [d["order_history"] for d in customer_data],
    embeddings,
    metadatas=[{"id": d["id"], "type": "customer_history"} for d in customer_data],
    collection_name="customer_history"
)
product_vectorstore = Chroma.from_texts(
    [d["description"] for d in product_data],
    embeddings,
    metadatas=[{"id": d["id"], "name": d["name"], "type": "product_info"} for d in product_data],
    collection_name="product_info"
)
faqs_vectorstore = Chroma.from_texts(
    [d["question"] + " " + d["answer"] for d in faqs_data],
    embeddings,
    metadatas=[{"type": "faq"} for d in faqs_data],
    collection_name="faqs"
)

# --- 3. Tooling/Function Calling ---
@tool
def check_order_status(order_id: str) -> str:
    for customer in customer_data:
        if order_id in customer["order_history"]:
            start_index = customer["order_history"].find(f"Order #{order_id}")
            if start_index != -1:
                end_index = customer["order_history"].find(")", start_index)
                return customer["order_history"][start_index:end_index+1]
    return f"Order {order_id} not found or no status available."

@tool
def get_product_info(product_name: str) -> str:
    for product in product_data:
        if product_name.lower() in product["name"].lower():
            return f"Product: {product['name']}, Description: {product['description']}, Price: {product['price']}"
    return f"Product {product_name} not found."

@tool
def get_customer_contact_info(customer_id: str) -> str:
    for customer in customer_data:
        if customer["id"] == customer_id:
            return f"Name: {customer['name']}, Email: {customer['email']}"
    return f"Customer {customer_id} not found."


tools = [check_order_status, get_product_info, get_customer_contact_info]

# --- 4. LLM Integration ---
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# --- 5. Context Management Layer ---
class ContextManager:
    def __init__(self, llm_summarizer: ChatOpenAI, embeddings_model):
        self.llm_summarizer = llm_summarizer
        self.embeddings_model = embeddings_model
        self.conversation_history: List[Any] = []

    def retrieve_information(self, query: str) -> str:
        retrieved_docs = []
        retrieved_docs.extend(customer_vectorstore.similarity_search(query, k=1))
        retrieved_docs.extend(product_vectorstore.similarity_search(query, k=1))
        retrieved_docs.extend(faqs_vectorstore.similarity_search(query, k=1))

        relevant_info = "\n".join([doc.page_content for doc in retrieved_docs])
        return f"Relevant external information:\n{relevant_info}" if relevant_info else "No additional relevant information found."

    def summarize_conversation(self, history: List[Any], max_summary_length: int = 200) -> str:
        if not history or len(str(history)) < max_summary_length * 2:  # Simple heuristic for when to summarize
            return ""

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage("You are a helpful assistant. Summarize the following conversation concisely, focusing on key issues, customer requests, and resolutions. The summary should be used to maintain context in a longer conversation. Keep it brief."),
            MessagesPlaceholder(variable_name="conversation_history")
        ])
        chain = prompt | self.llm_summarizer
        summary = chain.invoke({"conversation_history": history}).content
        return f"Past conversation summary: {summary}"

    def prune_context(self, context_parts: List[str], max_tokens: int = 1500) -> List[str]:
        # A very basic pruning strategy: prioritize recent messages and summaries/retrieved info
        # In a real system, this would be more sophisticated (e.g., token counting, semantic relevance)
        pruned_context = []
        current_length = 0
        
        # Prioritize system messages, summaries, and retrieved info first
        for part in reversed(context_parts):
            if "summary:" in part or "Relevant external information:" in part or isinstance(part, SystemMessage):
                pruned_context.insert(0, part)
                current_length += len(part.split())
                if current_length > max_tokens: break
                
        # Then add recent human/AI messages until token limit is reached
        for part in reversed(context_parts):
            if part not in pruned_context:
                token_count = len(part.split())
                if current_length + token_count <= max_tokens:
                    pruned_context.insert(0, part)
                    current_length += token_count
                else: break
        
        # If still too long, truncate from the oldest non-essential parts
        while current_length > max_tokens and len(pruned_context) > 1:
            oldest_part_len = len(pruned_context[0].split())
            pruned_context.pop(0) # Remove oldest part
            current_length -= oldest_part_len
            
        return pruned_context

context_manager = ContextManager(llm_summarizer=llm, embeddings_model=embeddings)

# --- Chatbot Logic ---
chat_history = []

# Define the agent prompt
agent_prompt = ChatPromptTemplate.from_messages([
    SystemMessage("You are an AI customer support assistant. You have access to tools to help customers. Maintain context by using provided summaries and relevant information. Answer questions directly and use tools when necessary."),
    MessagesPlaceholder(variable_name="chat_history"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
    SystemMessage("If the context is too long, a summary of past interactions and relevant external information will be provided to help you."),
    HumanMessage(content="{input}")
])

agent = create_openai_tools_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

def respond(message, history):
    global chat_history
    
    # Update chat_history with current turn
    current_turn_history = []
    for user_msg, bot_msg in history:
        current_turn_history.append(HumanMessage(content=user_msg))
        if bot_msg: current_turn_history.append(AIMessage(content=bot_msg))
    
    chat_history = current_turn_history # Keep only human/AI messages for summarization
    
    # Retrieve relevant information
    retrieved_info = context_manager.retrieve_information(message)

    # Summarize conversation if it gets too long
    conversation_summary = context_manager.summarize_conversation(chat_history)

    # Construct context for the LLM
    full_context_parts = [
        SystemMessage(content=conversation_summary if conversation_summary else ""),
        SystemMessage(content=retrieved_info if retrieved_info else ""),
    ] + chat_history + [HumanMessage(content=message)]

    # Prune context to fit within LLM window (simplified)
    # For agent execution, the actual pruning needs to happen carefully for the prompt sent to `agent_executor`
    # Here, we'll try to provide the most relevant parts.
    
    # We need to reconstruct the agent's input carefully to include context parts
    # The agent_executor takes messages directly, so we need to ensure the `chat_history` placeholder contains the pruned version.

    # For this simplified demo, we'll pass recent chat_history + summary + retrieved info directly to the agent's history.
    # A more robust solution would involve dynamically managing the prompt messages for the agent itself.
    
    # Let's create a combined history for the agent, prioritizing summary and retrieved info
    agent_combined_history = []
    if conversation_summary: agent_combined_history.append(SystemMessage(content=conversation_summary))
    if retrieved_info: agent_combined_history.append(SystemMessage(content=retrieved_info))
    agent_combined_history.extend(chat_history) # Add actual chat history

    # The actual pruning for `chat_history` sent to agent_executor would need to be more sophisticated
    # For now, we'll let Langchain handle its internal context for the agent_executor, but provide the prepended context.
    
    try:
        # Invoke the agent with the user's message and the combined context history
        result = agent_executor.invoke({"input": message, "chat_history": agent_combined_history})
        response_content = result["output"]
    except Exception as e:
        response_content = f"An error occurred: {e}. Please try again."

    return response_content

# --- Gradio UI ---
if __name__ == "__main__":
    gr.ChatInterface(respond).launch()