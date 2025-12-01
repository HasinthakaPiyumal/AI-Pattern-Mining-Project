import os
from pydantic import BaseModel, Field
from typing import List, Dict, Any

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

class SearchProductInput(BaseModel):
    query: str = Field(description="The search query for the product.")

class AddToCartInput(BaseModel):
    product_id: str = Field(description="The unique identifier of the product to add.")
    quantity: int = Field(description="The quantity of the product to add.")

class ComparePricesInput(BaseModel):
    product_name: str = Field(description="The name of the product to compare prices for.")

class TrackOrderInput(BaseModel):
    order_id: str = Field(description="The unique identifier of the order to track.")

@tool("search_product", args_schema=SearchProductInput)
def search_product(query: str) -> str:
    if "laptop" in query.lower():
        return """
        [
            {"id": "P001", "name": "Dell XPS 13", "price": 1200.00, "retailer": "TechShop"},
            {"id": "P002", "name": "MacBook Air M2", "price": 1300.00, "retailer": "AppleStore"},
            {"id": "P003", "name": "HP Spectre x360", "price": 1150.00, "retailer": "ElectroWorld"}
        ]
        """
    elif "smartphone" in query.lower():
        return """
        [
            {"id": "P004", "name": "iPhone 15 Pro", "price": 999.00, "retailer": "AppleStore"},
            {"id": "P005", "name": "Samsung Galaxy S24", "price": 899.00, "retailer": "AndroidZone"}
        ]
        """
    else:
        return "[]"

@tool("add_to_cart", args_schema=AddToCartInput)
def add_to_cart(product_id: str, quantity: int) -> str:
    return f"Successfully added {quantity} of product {product_id} to your cart."

@tool("compare_prices", args_schema=ComparePricesInput)
def compare_prices(product_name: str) -> str:
    if "laptop" in product_name.lower():
        return """
        [
            {"retailer": "TechShop", "price": 1200.00, "product_url": "techshop.com/dellxps13"},
            {"retailer": "BestBuy", "price": 1250.00, "product_url": "bestbuy.com/dellxps13"},
            {"retailer": "Amazon", "price": 1180.00, "product_url": "amazon.com/dellxps13"}
        ]
        """
    elif "iphone" in product_name.lower():
        return """
        [
            {"retailer": "AppleStore", "price": 999.00, "product_url": "apple.com/iphone15pro"},
            {"retailer": "Verizon", "price": 1029.00, "product_url": "verizon.com/iphone15pro"}
        ]
        """
    else:
        return "No price comparison data available for this product."

@tool("track_order", args_schema=TrackOrderInput)
def track_order(order_id: str) -> str:
    if order_id == "ORD12345":
        return "Your order ORD12345 is currently 'Shipped' and expected to arrive by 2024-07-20."
    elif order_id == "ORD67890":
        return "Your order ORD67890 is currently 'Processing'."
    else:
        return "Order not found."

tools = [search_product, add_to_cart, compare_prices, track_order]

llm = ChatOpenAI(model="gpt-4o", temperature=0)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an AI e-commerce shopping assistant. Use the available tools to help users with their shopping queries."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

agent = create_react_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

if __name__ == "__main__":
    print("Welcome to the AI E-commerce Shopping Assistant! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        try:
            response = agent_executor.invoke({"input": user_input})
            print(f"Assistant: {response['output']}")
        except Exception as e:
            print(f"Assistant Error: {e}")
            print("Please try again or rephrase your request.")
