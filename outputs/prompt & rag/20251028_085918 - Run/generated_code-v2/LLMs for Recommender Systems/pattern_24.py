
import os
import json
import time
from collections import defaultdict
from typing import Dict, Any, List
from langchain.agents import AgentExecutor, create_react_agent
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.agents import Tool
import random

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")
    ECOMMERCE_PRODUCT_API = "https://api.ecommerce.com/products"
    ECOMMERCE_ORDER_API = "https://api.ecommerce.com/orders"
    ECOMMERCE_INVENTORY_API = "https://api.ecommerce.com/inventory"


def get_product_details(product_id: str) -> dict:
    time.sleep(0.5)
    mock_products = {
        "PROD001": {"name": "Laptop Pro X", "price": 1200.00, "category": "Electronics", "description": "High-performance laptop with 16GB RAM and 512GB SSD."}, 
        "PROD002": {"name": "Wireless Mouse", "price": 25.00, "category": "Accessories", "description": "Ergonomic wireless mouse with adjustable DPI."}, 
        "PROD003": {"name": "Mechanical Keyboard", "price": 80.00, "category": "Accessories", "description": "RGB mechanical keyboard with brown switches."}
    }
    return mock_products.get(product_id, {"error": "Product not found"})

def search_products(query: str, category: str = None, max_price: float = None) -> list:
    time.sleep(1)
    all_products = [
        {"id": "PROD001", "name": "Laptop Pro X", "price": 1200.00, "category": "Electronics", "description": "High-performance laptop with 16GB RAM and 512GB SSD."}, 
        {"id": "PROD002", "name": "Wireless Mouse", "price": 25.00, "category": "Accessories", "description": "Ergonomic wireless mouse with adjustable DPI."}, 
        {"id": "PROD003", "name": "Mechanical Keyboard", "price": 80.00, "category": "Accessories", "description": "RGB mechanical keyboard with brown switches."}, 
        {"id": "PROD004", "name": "4K Monitor", "price": 350.00, "category": "Electronics", "description": "27-inch 4K UHD monitor."}, 
        {"id": "PROD005", "name": "USB-C Hub", "price": 40.00, "category": "Accessories", "description": "Multi-port USB-C adapter."}
    ]
    
    results = []
    for product in all_products:
        match = False
        if query.lower() in product["name"].lower() or query.lower() in product["description"].lower():
            match = True
        if category and product["category"].lower() != category.lower():
            match = False
        if max_price and product["price"] > max_price:
            match = False
        if match:
            results.append(product)
            
    return results

def check_stock(product_id: str) -> dict:
    time.sleep(0.3)
    mock_inventory = {
        "PROD001": {"in_stock": True, "quantity": 15},
        "PROD002": {"in_stock": True, "quantity": 100},
        "PROD003": {"in_stock": False, "quantity": 0},
        "PROD004": {"in_stock": True, "quantity": 50},
        "PROD005": {"in_stock": True, "quantity": 200}
    }
    return mock_inventory.get(product_id, {"error": "Product not found"})

def get_order_status(order_id: str) -> dict:
    time.sleep(0.7)
    mock_orders = {
        "ORDER123": {"status": "Shipped", "delivery_date": "2024-03-15", "items": ["PROD001"]},
        "ORDER456": {"status": "Processing", "delivery_date": "2024-03-20", "items": ["PROD002", "PROD005"]}
    }
    return mock_orders.get(order_id, {"error": "Order not found"})

def process_return(order_id: str, product_id: str) -> dict:
    time.sleep(1.0)
    if order_id == "ORDER123" and product_id == "PROD001":
        return {"status": "Return initiated", "tracking_number": "RETN789"}
    else:
        return {"error": "Could not process return. Invalid order or product."}

def get_langchain_tools():
    return [
        Tool(
            name="GetProductDetails",
            func=get_product_details,
            description="Useful for fetching detailed information about a product by its ID. Input should be a product ID (e.g., \"PROD001\")."
        ),
        Tool(
            name="SearchProducts",
            func=search_products,
            description="Useful for searching products based on a query (e.g., \"laptop\"), optional category (e.g., \"Electronics\"), and optional maximum price (e.g., 500.00). Returns a list of matching products. Always provide a query. Category and max_price are optional."
        ),
        Tool(
            name="CheckStock",
            func=check_stock,
            description="Useful for checking the current stock availability for a product by its ID. Input should be a product ID."
        ),
        Tool(
            name="GetOrderStatus",
            func=get_order_status,
            description="Useful for retrieving the current status of a customer order by its order ID. Input should be an order ID (e.g., \"ORDER123\")."
        ),
        Tool(
            name="ProcessReturn",
            func=process_return,
            description="Useful for initiating a return for a specific product within an order. Input should be a JSON string with \"order_id\" and \"product_id\" keys."
        ),
    ]

class MemoryManager:
    def __init__(self):
        self.user_profiles: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.conversational_facts: Dict[str, list] = defaultdict(list)

    def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        return self.user_profiles[user_id]

    def update_user_profile(self, user_id: str, updates: Dict[str, Any]):
        self.user_profiles[user_id].update(updates)

    def add_conversational_fact(self, user_id: str, fact: str):
        if fact not in self.conversational_facts[user_id]:
            self.conversational_facts[user_id].append(fact)

    def get_relevant_facts(self, user_id: str, query: str, top_k: int = 2) -> list:
        relevant = []
        query_lower = query.lower()
        for fact in self.conversational_facts[user_id]:
            if any(keyword in fact.lower() for keyword in query_lower.split()):
                relevant.append(fact)
                if len(relevant) >= top_k:
                    break
        return relevant
    
    def clear_user_memory(self, user_id: str):
        if user_id in self.user_profiles:
            del self.user_profiles[user_id]
        if user_id in self.conversational_facts:
            del self.conversational_facts[user_id]

class DialogueManager:
    def __init__(self, memory_manager: MemoryManager, llm_model_name: str = "gpt-4-turbo-preview"):
        self.llm = ChatOpenAI(temperature=0, model_name=llm_model_name, openai_api_key=Config.OPENAI_API_KEY)
        self.tools = get_langchain_tools()
        self.memory_manager = memory_manager
        self.agent_executor = self._initialize_agent()
        self.conversation_history = []

    def _initialize_agent(self):
        template = """You are ShopBot, an AI-powered conversational e-commerce assistant.
        Your goal is to help users find products, answer questions about orders, stock, and returns, and provide personalized recommendations.
        You have access to the following tools:

        {tools}

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        Begin!

        Previous conversation history:
        {chat_history}

        User Profile and Relevant Facts:
        {user_context}

        New input: {input}
        Thought:{agent_scratchpad}"""

        prompt = PromptTemplate.from_template(template)

        agent = create_react_agent(self.llm, self.tools, prompt)
        
        return AgentExecutor(agent=agent, tools=self.tools, verbose=True, handle_parsing_errors=True)

    def _get_user_context(self, user_id: str, current_query: str) -> str:
        profile = self.memory_manager.get_user_profile(user_id)
        relevant_facts = self.memory_manager.get_relevant_facts(user_id, current_query)
        
        context_parts = []
        if profile:
            context_parts.append(f"User Profile: {json.dumps(profile)}")
        if relevant_facts:
            context_parts.append(f"Relevant Conversational Facts: {json.dumps(relevant_facts)}")
        
        return "\n".join(context_parts) if context_parts else "No specific user context available."

    def get_response(self, user_id: str, user_input: str) -> str:
        user_context = self._get_user_context(user_id, user_input)

        try:
            chat_history_str = "\n".join([f"{msg.type.capitalize()}: {msg.content}" for msg in self.conversation_history])

            response = self.agent_executor.invoke({
                "input": user_input,
                "chat_history": chat_history_str,
                "user_context": user_context
            })
            
            ai_message = response["output"]

            self.conversation_history.append(HumanMessage(content=user_input))
            self.conversation_history.append(AIMessage(content=ai_message))

            if "recommendation" in ai_message.lower() and user_id not in self.memory_manager.user_profiles:
                self.memory_manager.update_user_profile(user_id, {"has_received_recommendation": True})

            return ai_message
        except Exception as e:
            return "I apologize, but I encountered an error while processing your request. Please try again."

    def _extract_and_store_facts(self, user_id: str, text: str):
        if "laptop" in text.lower() and "interested" in text.lower():
            self.memory_manager.add_conversational_fact(user_id, "User is interested in laptops.")

def generate_synthetic_dialogue_data(num_samples: int = 100) -> list:
    product_types = ["laptop", "smartphone", "headphone", "smartwatch", "camera"]
    brands = ["BrandX", "MegaTech", "InnovateCo", "Elegance"]
    intents = [
        "product_inquiry", "recommendation_request", "stock_check", 
        "order_status_inquiry", "return_request", "general_qa"
    ]

    dialogues = []

    for i in range(num_samples):
        intent = random.choice(intents)
        dialogue = {"id": f"dialogue_{i+1}", "turns": []}

        if intent == "product_inquiry":
            product = random.choice(product_types)
            dialogue["turns"].append({"speaker": "user", "text": f"Tell me about a good {product}."})
            dialogue["turns"].append({"speaker": "bot", "text": f"Are you looking for a specific brand or price range for a {product}?"})
            dialogue["turns"].append({"speaker": "user", "text": f"Something from {random.choice(brands)}."})
            dialogue["turns"].append({"speaker": "bot", "text": f"I found the {random.choice(brands)} {product} Pro. It costs ${random.randint(500, 2000)}. Would you like more details?"})

        elif intent == "recommendation_request":
            preference = random.choice(["gaming", "work", "budget-friendly", "high-performance"])
            dialogue["turns"].append({"speaker": "user", "text": f"Can you recommend a {preference} laptop?"})
            dialogue["turns"].append({"speaker": "bot", "text": f"Certainly! For {preference}, I suggest the {random.choice(brands)} UltraBook, known for its excellent {preference} features."})

        elif intent == "stock_check":
            product_id = f"PROD{random.randint(1, 999):03d}"
            dialogue["turns"].append({"speaker": "user", "text": f"Do you have {product_id} in stock?"})
            dialogue["turns"].append({"speaker": "bot", "text": f"Let me check the stock for product ID {product_id}. (Calling CheckStock tool...)"})
            dialogue["turns"].append({"speaker": "bot", "text": f"Yes, product {product_id} is currently in stock with {random.randint(10, 200)} units." if random.random() > 0.2 else f"Apologies, product {product_id} is currently out of stock."})

        elif intent == "order_status_inquiry":
            order_id = f"ORDER{random.randint(100, 999)}"
            dialogue["turns"].append({"speaker": "user", "text": f"What is the status of my order {order_id}?"})
            dialogue["turns"].append({"speaker": "bot", "text": f"Checking order status for {order_id}. (Calling GetOrderStatus tool...)"})
            dialogue["turns"].append({"speaker": "bot", "text": f"Your order {order_id} is {random.choice(['shipped', 'processing', 'delivered'])}."})

        elif intent == "return_request":
            order_id = f"ORDER{random.randint(100, 999)}"
            product_id = f"PROD{random.randint(1, 999):03d}"
            dialogue["turns"].append({"speaker": "user", "text": f"I want to return product {product_id} from order {order_id}."})
            dialogue["turns"].append({"speaker": "bot", "text": f"Processing return for product {product_id} in order {order_id}. (Calling ProcessReturn tool...)"})
            dialogue["turns"].append({"speaker": "bot", "text": f"Return initiated for product {product_id} from order {order_id}. You will receive a return label shortly."})
        
        elif intent == "general_qa":
            question = random.choice([
                "What are your shipping options?", "Do you offer international shipping?",
                "How do I contact customer support?", "What is your return policy?"
            ])
            answer = random.choice([
                "We offer standard and express shipping.", "Yes, we ship internationally to most countries.",
                "You can contact our support team via email or live chat.", "Our return policy allows returns within 30 days of purchase."
            ])
            dialogue["turns"].append({"speaker": "user", "text": question})
            dialogue["turns"].append({"speaker": "bot", "text": answer})

        dialogues.append(dialogue)
    return dialogues

def main():
    print("Welcome to ShopBot, your AI-powered e-commerce assistant!")
    print("Type \'quit\' or \'exit\' to end the conversation.")

    if not Config.OPENAI_API_KEY or Config.OPENAI_API_KEY == "YOUR_OPENAI_API_KEY":
        print("WARNING: OPENAI_API_KEY not set in config.py or environment variable. Some features may not work.")
        print("Please set the OPENAI_API_KEY environment variable or update config.py.")
    
    memory_manager = MemoryManager()
    dialogue_manager = DialogueManager(memory_manager=memory_manager)

    user_id = input("Please enter your user ID (e.g., user123): ")
    if not user_id:
        user_id = "guest_user"
        print(f"No user ID entered, using default: {user_id}")

    while True:
        user_input = input(f"You ({user_id}): ")
        if user_input.lower() in ["quit", "exit"]:
            print("Thank you for shopping with ShopBot! Goodbye!")
            break

        response = dialogue_manager.get_response(user_id, user_input)
        print(f"ShopBot: {response}")

if __name__ == "__main__":
    main()
