import os
from dotenv import load_dotenv
from typing import Type
from pydantic import BaseModel, Field

from langchain_core.tools import BaseTool
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.prompts import PromptTemplate

load_dotenv()

class KnowledgeBaseInput(BaseModel):
    query: str = Field(description="The question or query to search in the knowledge base.")

class KnowledgeBaseTool(BaseTool):
    name = "knowledge_base_search"
    description = "Searches the e-commerce platform's knowledge base for relevant information about products, policies, or common issues."
    args_schema: Type[BaseModel] = KnowledgeBaseInput
    vectorstore: Chroma

    def _run(self, query: str) -> str:
        docs = self.vectorstore.similarity_search(query, k=3)
        return "\n".join([doc.page_content for doc in docs])

class OrderManagementInput(BaseModel):
    order_id: str = Field(description="The ID of the customer's order.")
    action: str = Field(description="The action to perform: 'retrieve' order details, 'cancel' an order, or 'modify' an order. Default is 'retrieve'.")

class OrderManagementTool(BaseTool):
    name = "order_management_system"
    description = "Interacts with the internal order management system to retrieve order details, cancel orders, or modify existing orders."
    args_schema: Type[BaseModel] = OrderManagementInput

    def _run(self, order_id: str, action: str = "retrieve") -> str:
        if action == "retrieve":
            if order_id == "ORD12345":
                return "Order ORD12345: Product A (Qty 1), Status: Shipped, Shipping Address: 123 Main St. Estimated delivery: 2023-11-20."
            elif order_id == "ORD67890":
                return "Order ORD67890: Product B (Qty 2), Status: Processing, Shipping Address: 456 Oak Ave."
            else:
                return f"Order {order_id} not found."
        elif action == "cancel":
            return f"Simulating cancellation for order {order_id}. Confirmation sent."
        elif action == "modify":
            return f"Simulating modification for order {order_id}. Please provide details for modification."
        else:
            return "Invalid action for order management."

class ShippingInput(BaseModel):
    tracking_number: str = Field(description="The tracking number of the package.")
    action: str = Field(description="The action to perform: 'track' package status or 'generate_return_label'. Default is 'track'.")

class ShippingTool(BaseTool):
    name = "shipping_api"
    description = "Interacts with external shipping services to track packages or generate return labels."
    args_schema: Type[BaseModel] = ShippingInput

    def _run(self, tracking_number: str, action: str = "track") -> str:
        if action == "track":
            if tracking_number == "TRK987654":
                return "Tracking TRK987654: Status: In Transit, Last Location: New York, NY. Estimated Delivery: 2023-11-22."
            elif tracking_number == "TRK112233":
                return "Tracking TRK112233: Status: Delivered, Delivered On: 2023-11-15."
            else:
                return f"Tracking number {tracking_number} not found."
        elif action == "generate_return_label":
            return f"Simulating return label generation for tracking number {tracking_number}. Label sent to customer email."
        else:
            return "Invalid action for shipping API."

llm = ChatOpenAI(temperature=0)

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

sample_kb_docs = [
    "Our return policy allows returns within 30 days of purchase for a full refund. Items must be in original condition.",
    "To track an order, please visit our 'Order Status' page and enter your order number.",
    "For technical support, please contact our support team at support@ecommerce.com or call 1-800-TECH-HELP.",
    "Our premium subscription offers free expedited shipping on all orders and exclusive discounts.",
    "How to reset your password: Go to the login page, click 'Forgot Password', and follow the instructions sent to your email.",
    "Shipping usually takes 3-5 business days for standard delivery within the contiguous United States.",
    "We accept Visa, MasterCard, American Express, and PayPal.",
    "Product A is a high-performance gadget with a 1-year warranty.",
    "Product B is a comfortable ergonomic chair designed for long hours of use."
]

vectorstore = Chroma.from_texts(sample_kb_docs, embeddings)

tools = [
    KnowledgeBaseTool(vectorstore=vectorstore),
    OrderManagementTool(),
    ShippingTool()
]

prompt = PromptTemplate.from_template("""
You are an AI-powered intelligent customer support agent for an e-commerce platform.
You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
""")

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

if __name__ == "__main__":
    print("Welcome to the E-commerce Customer Support AI!")
    print("Type 'exit' to quit.")

    while True:
        user_query = input("\nHow can I help you today? ")
        if user_query.lower() == 'exit':
            break

        try:
            response = agent_executor.invoke({"input": user_query})
            print(f"\nAI Response: {response['output']}")
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please try rephrasing your question or contact a human agent.")
