import os
from langchain.agents import AgentType, initialize_agent, Tool
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory

# Set your OpenAI API key as an environment variable or replace os.environ.get with your key directly
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

class ECommerceTools:
    def browse_catalog(self, query: str) -> str:
        # Simulate browsing a product catalog
        products = {
            "laptop": "High-performance laptop with 16GB RAM and 512GB SSD. Price: $1200.",
            "smartphone": "Latest model smartphone with advanced camera. Price: $800.",
            "headphones": "Noise-cancelling over-ear headphones. Price: $250.",
            "keyboard": "Mechanical gaming keyboard. Price: $100.",
            "mouse": "Ergonomic wireless mouse. Price: $50."
        }
        if query.lower() in products:
            return products[query.lower()]
        return f"No direct match found for '{query}'. You might be interested in laptops, smartphones, or headphones."

    def compare_products(self, product_names: str) -> str:
        # Simulate comparing products
        names_list = [name.strip().lower() for name in product_names.split(',')]
        comparison_results = []
        available_products = {
            "laptop": {"price": 1200, "features": "16GB RAM, 512GB SSD"},
            "smartphone": {"price": 800, "features": "Advanced camera, Long battery life"},
            "headphones": {"price": 250, "features": "Noise-cancelling, Comfortable"}
        }
        for name in names_list:
            if name in available_products:
                product_info = available_products[name]
                comparison_results.append(f"{name.capitalize()}: Price ${product_info['price']}, Features: {product_info['features']}")
            else:
                comparison_results.append(f"'{name.capitalize()}' not found in comparison catalog.")
        if comparison_results:
            return "\n".join(comparison_results)
        return "No products provided for comparison."

    def process_order(self, product_name_quantity_customer_details: str) -> str:
        # Simulate processing an order
        try:
            parts = product_name_quantity_customer_details.split(';')
            product_name = parts[0].split(':')[1].strip()
            quantity = int(parts[1].split(':')[1].strip())
            customer_details_str = parts[2].split(':')[1].strip()
            customer_details = eval(customer_details_str) # AVOID IN REAL APPS - for demo only
            
            order_id = f"ORD-{hash(product_name + str(quantity) + str(customer_details)) % 100000}"
            return f"Order for {quantity} x {product_name} processed successfully for {customer_details.get('name', 'customer')}. Your order ID is {order_id}."
        except Exception as e:
            return f"Failed to process order. Please provide product_name, quantity, and customer_details (e.g., 'product_name:laptop;quantity:1;customer_details:{'name':'John Doe','address':'123 Main St'}'). Error: {e}"

    def get_order_status(self, order_id: str) -> str:
        # Simulate checking order status
        if order_id.startswith("ORD-") and len(order_id) == 10:
            # Simulate some order statuses
            if int(order_id.split('-')[1]) % 2 == 0:
                return f"Order {order_id} is currently being shipped and is expected to arrive in 2-3 business days."
            else:
                return f"Order {order_id} has been placed and is awaiting fulfillment."
        return f"Invalid order ID: {order_id}. Please provide a valid order ID."

e_commerce_tools = ECommerceTools()

tools = [
    Tool(
        name="BrowseCatalog",
        func=e_commerce_tools.browse_catalog,
        description="Useful for searching for products in the e-commerce catalog. Input should be a search query (e.g., 'laptop')."
    ),
    Tool(
        name="CompareProducts",
        func=e_commerce_tools.compare_products,
        description="Useful for comparing multiple products. Input should be a comma-separated string of product names (e.g., 'laptop, smartphone')."
    ),
    Tool(
        name="ProcessOrder",
        func=e_commerce_tools.process_order,
        description="Useful for placing an order for a product. Input should be a string in the format 'product_name:product;quantity:num;customer_details:{'name':'Name','address':'Address'}'"
    ),
    Tool(
        name="GetOrderStatus",
        func=e_commerce_tools.get_order_status,
        description="Useful for checking the status of an existing order. Input should be the order ID (e.g., 'ORD-12345')."
    ),
]

llm = ChatOpenAI(temperature=0, model="gpt-4o") # Ensure OPENAI_API_KEY is set in your environment

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent_chain = initialize_agent(
    tools,
    llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
    handle_parsing_errors=True
)

def run_agent_interaction(query: str) -> str:
    return agent_chain.run(input=query)

if __name__ == "__main__":
    print("Autonomous E-commerce Assistant Agent started. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Agent: Goodbye!")
            break
        response = run_agent_interaction(user_input)
        print(f"Agent: {response}")
