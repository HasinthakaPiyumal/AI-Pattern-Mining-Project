import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr
import random
import time


class CustomerQuery(BaseModel):
    query: str


class KnowledgeBaseTool:
    def search(self, query: str) -> str:
        if "shipping" in query.lower():
            return "Shipping policy: Orders usually arrive within 3-5 business days. Expedited options available."  # noqa: E501
        elif "return" in query.lower():
            return "Return policy: Items can be returned within 30 days of purchase with original receipt."  # noqa: E501
        else:
            return f"No direct knowledge base article found for '{query}'."


class OrderManagementSystemTool:
    def get_order_status(self, customer_id: str, order_id: str) -> str:
        if customer_id == "cust123" and order_id == "order987":
            return "Order order987 for customer cust123: Status - Shipped, Tracking: TRK12345."  # noqa: E501
        else:
            return "Order not found or incorrect customer/order ID."


class MockLLM:
    def generate_response(self, prompt: str, context: list) -> str:
        combined_context = " ".join(context)
        if "shipping" in prompt.lower() or "shipping" in combined_context.lower():
            return "Based on our shipping policy, your order should arrive within 3-5 business days. Would you like to know about expedited shipping?"  # noqa: E501
        elif "return" in prompt.lower() or "return" in combined_context.lower():
            return "Our return policy states that you can return items within 30 days of purchase with the original receipt."  # noqa: E501
        elif "order status" in prompt.lower() or "order status" in combined_context.lower():  # noqa: E501
            return "Please provide your customer ID and order ID so I can check the status for you."  # noqa: E501
        else:
            return f"I'm here to help with your e-commerce queries. You asked about: {prompt}."


class ECommerceAgent:
    def __init__(self):
        self.llm = MockLLM()
        self.kb_tool = KnowledgeBaseTool()
        self.oms_tool = OrderManagementSystemTool()
        self.conversation_history = []

    def process_query(self, query: str) -> str:
        self.conversation_history.append(f"User: {query}")
        context = self.conversation_history[-5:]  # Keep last 5 turns as context

        response = ""
        if "shipping" in query.lower() or "delivery" in query.lower():
            kb_info = self.kb_tool.search("shipping")
            response = self.llm.generate_response(query, context + [kb_info])
        elif "return" in query.lower():
            kb_info = self.kb_tool.search("return")
            response = self.llm.generate_response(query, context + [kb_info])
        elif "order status" in query.lower() or "where is my order" in query.lower():  # noqa: E501
            # In a real scenario, LLM would parse customer/order ID
            # For this mock, we'll simulate a prompt for info
            if "cust123" in query.lower() and "order987" in query.lower():
                order_status = self.oms_tool.get_order_status("cust123", "order987")  # noqa: E501
                response = self.llm.generate_response(query, context + [order_status])  # noqa: E501
            else:
                response = self.llm.generate_response(query, context + ["Need customer and order ID."])  # noqa: E501
        else:
            response = self.llm.generate_response(query, context)

        self.conversation_history.append(f"Agent: {response}")
        return response


class DemonstrationCollector:
    def collect_demonstration(self, task_id: str, actions: list, observations: list) -> dict:
        return {"task_id": task_id, "actions": actions, "observations": observations}


class ComparisonCollector:
    def collect_comparison(self, query: str, response_a: str, response_b: str, preferred: str) -> dict:
        return {"query": query, "response_a": response_a, "response_b": response_b, "preferred": preferred}


class BehaviorCloningTrainer:
    def train(self, demonstrations: list) -> str:
        return f"Trained LLM using {len(demonstrations)} demonstrations for Behavior Cloning."


class RewardModelTrainer:
    def train(self, comparisons: list) -> str:
        return f"Trained Reward Model using {len(comparisons)} comparisons."


class RLHFTrainer:
    def fine_tune(self, bc_model: str, reward_model: str) -> str:
        return f"Fine-tuned {bc_model} with RLHF using {reward_model} for alignment."


# FastAPI Application
app = FastAPI()
e_commerce_agent = ECommerceAgent()


@app.post("/chat")
async def chat_with_agent(customer_query: CustomerQuery):
    response = e_commerce_agent.process_query(customer_query.query)
    return {"response": response}


# Gradio Interface
def gradio_chat(message, history):
    global e_commerce_agent
    # Reset agent history for new Gradio session if it's the first message
    if not history:
        e_commerce_agent = ECommerceAgent() # Re-initialize for a fresh conversation

    response = e_commerce_agent.process_query(message)
    return response


if __name__ == "__main__":
    # Example Data Collection (Mock)
    demo_collector = DemonstrationCollector()
    comp_collector = ComparisonCollector()

    demonstrations_data = [
        demo_collector.collect_demonstration(
            "task1", ["search_kb(\"shipping\")", "draft_response(\"shipping policy\")"], ["kb_result_shipping", "customer_query"]
        ),
        demo_collector.collect_demonstration(
            "task2", ["query_oms(\"cust123\")", "draft_response(\"order status\")"], ["oms_result_shipped", "customer_query"]
        ),
    ]

    comparisons_data = [
        comp_collector.collect_comparison(
            "What is your return policy?", "You can return items in 30 days.", "Our policy allows returns within 30 days of purchase with a receipt.", "response_b"
        ),
        comp_collector.collect_comparison(
            "How fast is shipping?", "Shipping is usually fast.", "Typically 3-5 business days for standard shipping.", "response_b"
        ),
    ]

    print("--- Mock Data Collection Complete ---")
    print(f"Collected {len(demonstrations_data)} demonstrations.")
    print(f"Collected {len(comparisons_data)} comparisons.")

    # Example Model Training (Mock)
    bc_trainer = BehaviorCloningTrainer()
    rm_trainer = RewardModelTrainer()
    rlhf_trainer = RLHFTrainer()

    bc_training_result = bc_trainer.train(demonstrations_data)
    rm_training_result = rm_trainer.train(comparisons_data)
    rlhf_fine_tune_result = rlhf_trainer.fine_tune("BC_LLM_v1", "RewardModel_v1")

    print("\n--- Mock Model Training Complete ---")
    print(bc_training_result)
    print(rm_training_result)
    print(rlhf_fine_tune_result)

    print("\n--- Starting E-commerce Agent (FastAPI & Gradio) ---")
    # Gradio Interface
    iface = gr.ChatInterface(
        gradio_chat,
        title="E-commerce Customer Support Agent (Dual Data Training)",
        description="An AI agent trained with demonstrations and human preferences to assist with e-commerce queries."
    )

    # To run both FastAPI and Gradio, you would typically run them as separate processes.
    # For this demonstration, we'll start Gradio, which can integrate FastAPI if needed, or run FastAPI separately.
    # To run FastAPI: `uvicorn e_commerce_agent:app --reload` in a separate terminal
    # To run Gradio: This script will launch it when __name__ == "__main__"

    print("\nRunning Gradio interface...")
    # This blocks execution, so FastAPI needs to be run separately.
    iface.launch(share=False)

    # To run FastAPI uncomment the following and ensure uvicorn is installed:
    # print("Running FastAPI server on http://127.0.0.1:8000")
    # uvicorn.run(app, host="0.0.0.0", port=8000)
