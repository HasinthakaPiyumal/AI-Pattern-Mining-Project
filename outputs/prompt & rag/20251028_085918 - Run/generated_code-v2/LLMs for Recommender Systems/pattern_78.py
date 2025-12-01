import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
import gradio as gr

load_dotenv()

# --- 2. E-commerce Tool Integration Module ---

# Dummy E-commerce data for demonstration
PRODUCTS_DB = {
    "1": {"name": "Laptop Pro", "price": 1200, "category": "Electronics", "description": "Powerful laptop for professionals."},
    "2": {"name": "Wireless Mouse", "price": 25, "category": "Electronics", "description": "Ergonomic wireless mouse."},
    "3": {"name": "Coffee Maker", "price": 80, "category": "Home & Kitchen", "description": "Automatic drip coffee maker."},
    "4": {"name": "Running Shoes", "price": 90, "category": "Apparel", "description": "Comfortable shoes for daily runs."},
    "5": {"name": "Smartwatch X", "price": 250, "category": "Wearables", "description": "Track your fitness and notifications."},
    "6": {"name": "Bluetooth Speaker", "price": 70, "category": "Electronics", "description": "Portable speaker with great sound."},
}

@tool
def product_search(query: str) -> str:
    """Searches for products in the e-commerce catalog based on a keyword query."""
    results = []
    for pid, product in PRODUCTS_DB.items():
        if query.lower() in product["name"].lower() or query.lower() in product["description"].lower() or query.lower() in product["category"].lower():
            results.append(f"Product ID: {pid}, Name: {product['name']}, Price: ${product['price']:.2f}, Category: {product['category']}")
    if results:
        return "\n".join(results)
    return f"No products found for '{query}'."

@tool
def product_details(product_id: str) -> str:
    """Retrieves detailed information about a product given its product ID."""
    product = PRODUCTS_DB.get(product_id)
    if product:
        return (
            f"Name: {product['name']}\n"
            f"Price: ${product['price']:.2f}\n"
            f"Category: {product['category']}\n"
            f"Description: {product['description']}"
        )
    return f"Product with ID '{product_id}' not found."

@tool
def recommend_products(category: str = None) -> str:
    """Recommends products based on a given category or general popular items if no category is specified."""
    recommendations = []
    if category:
        for pid, product in PRODUCTS_DB.items():
            if product["category"].lower() == category.lower():
                recommendations.append(f"Product ID: {pid}, Name: {product['name']}, Price: ${product['price']:.2f}")
    else:
        # Simple popular items for demonstration
        popular_items = ["1", "5", "6"]
        for pid in popular_items:
            product = PRODUCTS_DB.get(pid)
            if product:
                recommendations.append(f"Product ID: {pid}, Name: {product['name']}, Price: ${product['price']:.2f}")
    
    if recommendations:
        return f"Here are some recommendations:\n" + "\n".join(recommendations)
    return "Sorry, I don't have recommendations for that category right now."

ecommerce_tools = [product_search, product_details, recommend_products]

# --- 3. Long Conversation Memory Management Module ---

# Short-term memory
short_term_memory = ConversationBufferWindowMemory(
    memory_key="chat_history", 
    return_messages=True, 
    input_key="input",
    k=5 # Keep last 5 turns
)

# Long-term memory (User Profile - using ChromaDB)
embeddings_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_texts(
    texts=["User is interested in electronics.", "User previously looked for running shoes."], 
    embedding=embeddings_model,
    collection_name="user_profiles"
)
retriever = vectorstore.as_retriever()

def get_long_term_context(query: str) -> str:
    """Retrieves relevant facts from long-term memory based on the current query."""
    docs = retriever.invoke(query)
    return "\n".join([doc.page_content for doc in docs]) if docs else ""

# --- 1. LLM-powered Dialogue Module (Core Agent) ---

llm = ChatOpenAI(model="gpt-4o", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful e-commerce shopping assistant. Use the provided tools to help users find products, get details, and receive recommendations."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, ecommerce_tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=ecommerce_tools, memory=short_term_memory, verbose=True)

# --- 4. User Interface (UI) Module ---

def chat_with_assistant(user_message, history):
    long_term_context = get_long_term_context(user_message)
    # Prepend long-term context to the user's message if it exists
    if long_term_context:
        user_message_with_context = f"User message: {user_message}\nRelevant user profile/context: {long_term_context}"
    else:
        user_message_with_context = user_message

    response = agent_executor.invoke({"input": user_message_with_context})
    return response["output"]

if __name__ == "__main__":
    gr.ChatInterface(
        chat_with_assistant,
        chatbot=gr.Chatbot(height=300),
        textbox=gr.Textbox(placeholder="Ask me anything about products!", container=False, scale=7),
        title="E-commerce Shopping Assistant",
        description="Your AI-powered assistant for product discovery and recommendations.",
        theme="soft",
        examples=["Find me a laptop", "Tell me about Product ID: 3", "Recommend some electronics"],
        clear_btn="Clear",
        undo_btn="Undo",
        submit_btn="Send",
    ).launch(share=True)