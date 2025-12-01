from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import tool
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate


class ECommerceTools:
    products_db = {
        "laptop": {"id": "p1", "name": "Gaming Laptop", "price": 1200, "description": "High-performance gaming laptop.", "stock": 5},
        "mouse": {"id": "p2", "name": "Wireless Mouse", "price": 25, "description": "Ergonomic wireless mouse.", "stock": 10},
        "keyboard": {"id": "p3", "name": "Mechanical Keyboard", "price": 75, "description": "RGB mechanical keyboard.", "stock": 8},
        "monitor": {"id": "p4", "name": "4K Monitor", "price": 350, "description": "27-inch 4K monitor.", "stock": 3},
    }
    
    cart_items = {}

    @tool
    def search_products(query: str) -> str:
        """Searches the e-commerce product database for items matching the query. Returns a list of product names and their prices."""
        found_products = [name for name, details in ECommerceTools.products_db.items() if query.lower() in name.lower() or query.lower() in details["description"].lower()]
        if found_products:
            results = [f"{ECommerceTools.products_db[p]["name"]} (${ECommerceTools.products_db[p]["price"]})" for p in found_products]
            return "Found products: " + "; ".join(results)
        return "No products found matching your query."

    @tool
    def get_product_details(product_name: str) -> str:
        """Retrieves detailed information about a specific product, including its description, price, and stock."""
        product_name_lower = product_name.lower()
        for name, details in ECommerceTools.products_db.items():
            if product_name_lower in name.lower():
                return f"Product: {details["name"]}, Price: ${details["price"]}, Description: {details["description"]}, Stock: {details["stock"]}"
        return "Product not found."

    @tool
    def add_to_cart(product_name: str, quantity: int) -> str:
        """Adds a specified quantity of a product to the user's shopping cart. Requires product name and quantity."""
        product_name_lower = product_name.lower()
        for name, details in ECommerceTools.products_db.items():
            if product_name_lower in name.lower():
                if details["stock"] >= quantity:
                    ECommerceTools.cart_items[name] = ECommerceTools.cart_items.get(name, 0) + quantity
                    details["stock"] -= quantity
                    return f"Added {quantity} x {details["name"]} to cart. Current cart: {ECommerceTools.cart_items}"
                else:
                    return f"Not enough stock for {details["name"]}. Only {details["stock"]} left."
        return "Product not found. Cannot add to cart."

    @tool
    def view_cart() -> str:
        """Displays the current items and their quantities in the user's shopping cart."""
        if not ECommerceTools.cart_items:
            return "Your cart is empty."
        cart_summary = [f"{qty} x {name}" for name, qty in ECommerceTools.cart_items.items()]
        return "Items in your cart: " + "; ".join(cart_summary)


def create_agent():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0) # Replace with your actual LLM setup

    tools = [
        ECommerceTools.search_products,
        ECommerceTools.get_product_details,
        ECommerceTools.add_to_cart,
        ECommerceTools.view_cart,
    ]

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    prompt = PromptTemplate.from_template(
        "You are a helpful smart shopping assistant for an e-commerce platform. "
        "You assist users with finding products, checking details, and managing their cart. "
        "You have access to the following tools: {tools}\n\n" # Placeholder for tools
        "Begin!\n\n" # Add the chat history to the prompt
        "Previous conversation:\n{chat_history}\n" 
        "New human input: {input}\n" 
        "{agent_scratchpad}" # Placeholder for agent's scratchpad
    )

    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True, handle_parsing_errors=True)
    
    return agent_executor


if __name__ == "__main__":
    print("Welcome to the Smart Shopping Assistant! Type 'exit' to quit.")
    assistant = create_agent()

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        
        try:
            response = assistant.invoke({"input": user_input})
            print(f"Assistant: {response['output']}")
        except Exception as e:
            print(f"Assistant: An error occurred: {e}")
            print("Please try rephrasing your request.")
