import os
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.memory import ConversationBufferMemory


# 1. Data Module: Mock Product Data
PRODUCTS = {
    "p101": {"id": "p101", "name": "Wireless Bluetooth Headphones", "description": "High-quality sound with noise cancellation and long battery life.", "category": "Electronics", "price": 79.99},
    "p102": {"id": "p102", "name": "Smartwatch with Heart Rate Monitor", "description": "Track your fitness, notifications, and more on your wrist.", "category": "Electronics", "price": 129.99},
    "p103": {"id": "p103", "name": "Organic Coffee Beans (Dark Roast)", "description": "Rich, bold flavor, ethically sourced organic coffee beans.", "category": "Groceries", "price": 15.50},
    "p104": {"id": "p104", "name": "Ergonomic Office Chair", "description": "Adjustable lumbar support and breathable mesh for maximum comfort.", "category": "Home & Office", "price": 249.00},
    "p105": {"id": "p105", "name": "Portable External SSD 1TB", "description": "Ultra-fast data transfer and durable design for on-the-go storage.", "category": "Electronics", "price": 99.99},
    "p106": {"id": "p106", "name": "Yoga Mat Eco-Friendly", "description": "Non-slip surface and made from sustainable materials for your practice.", "category": "Sports & Outdoors", "price": 29.99},
}

def get_product_details(product_id: str):
    return PRODUCTS.get(product_id)


# 2. Mock Recommender Engine Module
def get_mock_recommendations(user_id: str, num_recommendations: int = 3):
    # In a real system, this would be based on user behavior, preferences, etc.
    # For demonstration, we return a fixed set of popular-like items.
    if user_id == "user_A":
        return ["p101", "p105", "p104"]
    elif user_id == "user_B":
        return ["p103", "p106", "p102"]
    else:
        return ["p101", "p102", "p103"]


# Initialize LLM (Ensure OPENAI_API_KEY is set in your environment variables)
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)


# 3. LLM Explanation Module
def generate_explanation(product_id: str, user_context: str = ""):
    product = get_product_details(product_id)
    if not product:
        return "Product not found."

    explanation_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are an intelligent e-commerce assistant. Your task is to provide a concise and personalized explanation for why a product was recommended to a user."
            "Focus on the product's key features and how they might benefit the user given their context."
        ),
        HumanMessagePromptTemplate.from_template(
            "Product: {product_name} (Category: {product_category}) - Description: {product_description}"
            "User Context: {user_context}"
            "Why was this product recommended?"
        ),
    ])

    explanation_chain = LLMChain(llm=llm, prompt=explanation_template)

    response = explanation_chain.invoke({
        "product_name": product["name"],
        "product_category": product["category"],
        "product_description": product["description"],
        "user_context": user_context
    })
    return response["text"]


# 4. LLM Conversational Assistant Module
def product_lookup_tool(query: str) -> str:
    """Useful for looking up details of a product by its name or partial description."""
    query_lower = query.lower()
    for pid, product in PRODUCTS.items():
        if query_lower in product["name"].lower() or query_lower in product["description"].lower():
            return f"Product ID: {product['id']}, Name: {product['name']}, Category: {product['category']}, Price: ${product['price']:.2f}, Description: {product['description']}"
    return "Could not find product details for your query."


tools = [
    Tool(
        name="Product Lookup",
        func=product_lookup_tool,
        description="useful for when you need to find specific details about a product."
    ),
]

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

conversational_agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
    handle_parsing_errors=True # Added for robustness
)


# 5. Main Application/API Layer
def main():
    print("--- E-commerce Recommender System with LLM Enhancements ---")
    print("Please ensure your OPENAI_API_KEY environment variable is set.")
    print("\n--- Recommendations for user_A ---")
    user_id = "user_A"
    user_context = "The user is interested in technology gadgets and home office improvements."
    recommended_product_ids = get_mock_recommendations(user_id)
    print(f"Recommended Product IDs for {user_id}: {recommended_product_ids}")

    for pid in recommended_product_ids:
        product_details = get_product_details(pid)
        if product_details:
            explanation = generate_explanation(pid, user_context)
            print(f"\nProduct: {product_details['name']} (ID: {pid})")
            print(f"Explanation: {explanation}")

    print("\n--- Conversational Shopping Assistant ---")
    print("Type 'exit' to end the conversation.")
    while True:
        user_query = input("\nUser: ")
        if user_query.lower() == 'exit':
            break
        try:
            response = conversational_agent.invoke({"input": user_query})
            print(f"Assistant: {response['output']}")
        except Exception as e:
            print(f"Assistant Error: {e}")
            print("Please try again or restart the conversation.")

if __name__ == "__main__":
    main()