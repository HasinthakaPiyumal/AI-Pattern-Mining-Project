from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Union, Any
from loguru import logger
import time

# 1. WorkingMemory Module
class WorkingMemory(BaseModel):
    current_query: str = ""
    external_evidence: Dict[str, Any] = {}
    llm_candidate_responses: List[Dict[str, Union[str, float]]] = []
    user_feedback: str = ""
    dialog_history: List[Dict[str, str]] = []

    def update_query(self, query: str):
        self.current_query = query

    def add_evidence(self, key: str, evidence: Any):
        self.external_evidence[key] = evidence

    def add_llm_response(self, response: str, utility_score: float = 0.0):
        self.llm_candidate_responses.append({"response": response, "utility_score": utility_score})

    def add_feedback(self, feedback: str):
        self.user_feedback = feedback

    def add_dialog_turn(self, speaker: str, utterance: str):
        self.dialog_history.append({"speaker": speaker, "utterance": utterance})

    def get_state(self) -> Dict[str, Any]:
        return self.dict()

# 2. ExternalDataConnector Module (Mock)
class ExternalDataConnector:
    def get_order_details(self, order_id: str) -> Dict[str, Any]:
        logger.info(f"Fetching order details for: {order_id}")
        if order_id == "ORD123":
            return {"order_id": order_id, "status": "shipped", "items": [{"product": "Laptop", "qty": 1}], "total": 1200.00}
        return {"order_id": order_id, "status": "not_found"}

    def search_product_catalog(self, query: str) -> List[Dict[str, Any]]:
        logger.info(f"Searching product catalog for: {query}")
        if "laptop" in query.lower():
            return [{"id": "P001", "name": "Gaming Laptop", "price": 1500.00, "description": "High-performance gaming laptop."}]
        if "mouse" in query.lower():
            return [{"id": "P002", "name": "Wireless Mouse", "price": 25.00, "description": "Ergonomic wireless mouse."}]
        return []

    def get_shipping_status(self, tracking_id: str) -> Dict[str, Any]:
        logger.info(f"Fetching shipping status for: {tracking_id}")
        if tracking_id == "TRK789":
            return {"tracking_id": tracking_id, "status": "in_transit", "estimated_delivery": "2023-12-25"}
        return {"tracking_id": tracking_id, "status": "not_found"}

# 3. PromptEngine Module
class PromptEngine:
    def __init__(self, working_memory: WorkingMemory):
        self.memory = working_memory

    def build_prompt(self) -> str:
        prompt_parts = []
        prompt_parts.append("You are an AI customer support agent for an e-commerce platform. Provide helpful and concise responses.")
        prompt_parts.append("\n--- Dialog History ---")
        if not self.memory.dialog_history:
            prompt_parts.append("No prior conversation.")
        else:
            for turn in self.memory.dialog_history:
                prompt_parts.append(f"{turn['speaker']}: {turn['utterance']}")

        prompt_parts.append("\n--- External Evidence ---")
        if not self.memory.external_evidence:
            prompt_parts.append("No external information retrieved.")
        else:
            for key, value in self.memory.external_evidence.items():
                prompt_parts.append(f"{key.replace('_', ' ').title()}: {value}")

        prompt_parts.append("\n--- User Query ---")
        prompt_parts.append(self.memory.current_query)

        prompt_parts.append("\n--- Your Response ---")

        return "\n".join(prompt_parts)

# 4. LLMService Module (Mock)
class LLMService:
    def generate_response(self, prompt: str) -> str:
        logger.info("Simulating LLM response generation...")
        time.sleep(1)  # Simulate API call delay
        # In a real scenario, this would call an actual LLM API (e.g., OpenAI, Gemini)
        if "order status" in prompt.lower() and "shipped" in prompt.lower():
            return "Your order ORD123 has been shipped and is expected to arrive by 2023-12-25."
        elif "product" in prompt.lower() and "laptop" in prompt.lower():
            return "We have a 'Gaming Laptop' for $1500.00. It's a high-performance gaming laptop. Is there anything else you'd like to know?"
        elif "shipping status" in prompt.lower() and "in_transit" in prompt.lower():
            return "Your package with tracking ID TRK789 is currently in transit and is estimated to be delivered by 2023-12-25."
        elif "hello" in prompt.lower() or "hi" in prompt.lower():
            return "Hello! How can I assist you with your e-commerce needs today?"
        return "I understand. How can I help you further?"

# 5. Policy Module
class Policy:
    def decide_action(self, working_memory: WorkingMemory) -> Dict[str, Any]:
        query = working_memory.current_query.lower()

        if "order status" in query or "my order" in query or "where is" in query:
            # Try to extract order ID or tracking ID
            import re
            order_id_match = re.search(r"ord(\\d+)", query)
            tracking_id_match = re.search(r"trk(\\d+)", query)

            if order_id_match:
                return {"action": "GET_ORDER_DETAILS", "param": order_id_match.group(0).upper()}
            elif tracking_id_match:
                return {"action": "GET_SHIPPING_STATUS", "param": tracking_id_match.group(0).upper()}
            else:
                # If no ID, but query is about order status, prompt for it
                if not working_memory.external_evidence.get("order_details") and not working_memory.external_evidence.get("shipping_status"):
                    return {"action": "REQUEST_INFO", "info_needed": "order_id_or_tracking_id", "response": "Could you please provide your order ID or tracking number?"}

        if "product" in query or "item" in query or "looking for" in query or "recommend" in query:
            # Simple extraction for product search
            keywords = ["laptop", "mouse", "keyboard", "monitor"]
            found_keyword = next((k for k in keywords if k in query), None)
            if found_keyword:
                return {"action": "SEARCH_PRODUCT", "param": found_keyword}
            else:
                if not working_memory.external_evidence.get("product_catalog"):
                    return {"action": "REQUEST_INFO", "info_needed": "product_query", "response": "What kind of product are you looking for?"}

        return {"action": "RESPOND"}


# FastAPI Application
app = FastAPI()

# Initialize modules
memory = WorkingMemory()
connector = ExternalDataConnector()
llm_service = LLMService()
policy = Policy()

@app.post("/chat")
async def chat(user_query: Dict[str, str]):
    query_text = user_query.get("query", "")
    logger.info(f"Received user query: {query_text}")

    memory.add_dialog_turn(speaker="User", utterance=query_text)
    memory.update_query(query_text)

    response_to_user = "I'm sorry, I couldn't process your request." # Default fallback

    # Loop for policy-driven actions until a response is generated or escalated
    for _ in range(5): # Max 5 turns for internal reasoning to prevent infinite loops
        action_decision = policy.decide_action(memory)
        action = action_decision.get("action")
        param = action_decision.get("param")
        info_needed = action_decision.get("info_needed")

        logger.info(f"Policy decided action: {action} with param: {param}")

        if action == "GET_ORDER_DETAILS":
            if param:
                order_details = connector.get_order_details(param)
                memory.add_evidence("order_details", order_details)
                # Re-evaluate policy with new evidence
            else:
                response_to_user = action_decision.get("response", "Please provide an order ID.")
                memory.add_dialog_turn(speaker="Agent", utterance=response_to_user)
                break # Exit loop, respond to user

        elif action == "GET_SHIPPING_STATUS":
            if param:
                shipping_status = connector.get_shipping_status(param)
                memory.add_evidence("shipping_status", shipping_status)
                # Re-evaluate policy with new evidence
            else:
                response_to_user = action_decision.get("response", "Please provide a tracking number.")
                memory.add_dialog_turn(speaker="Agent", utterance=response_to_user)
                break # Exit loop, respond to user

        elif action == "SEARCH_PRODUCT":
            if param:
                product_results = connector.search_product_catalog(param)
                memory.add_evidence("product_catalog", product_results)
                # Re-evaluate policy with new evidence
            else:
                response_to_user = action_decision.get("response", "What product are you looking for?")
                memory.add_dialog_turn(speaker="Agent", utterance=response_to_user)
                break # Exit loop, respond to user

        elif action == "REQUEST_INFO":
            response_to_user = action_decision.get("response", "I need more information to help you.")
            memory.add_dialog_turn(speaker="Agent", utterance=response_to_user)
            break

        elif action == "RESPOND":
            prompt_engine = PromptEngine(memory)
            llm_prompt = prompt_engine.build_prompt()
            logger.info(f"LLM Prompt:\n{llm_prompt}")
            llm_response = llm_service.generate_response(llm_prompt)
            memory.add_llm_response(response=llm_response, utility_score=1.0) # Mock utility score
            memory.add_dialog_turn(speaker="Agent", utterance=llm_response)
            response_to_user = llm_response
            break # Exit loop, response generated

        elif action == "ESCALATE":
            response_to_user = "I'm sorry, I cannot fully assist with this request. I will escalate this to a human agent." # Placeholder
            memory.add_dialog_turn(speaker="Agent", utterance=response_to_user)
            break
        
        else:
            logger.warning(f"Unknown action: {action}. Defaulting to respond.")
            prompt_engine = PromptEngine(memory)
            llm_prompt = prompt_engine.build_prompt()
            llm_response = llm_service.generate_response(llm_prompt)
            memory.add_llm_response(response=llm_response, utility_score=1.0)
            memory.add_dialog_turn(speaker="Agent", utterance=llm_response)
            response_to_user = llm_response
            break

    return {"response": response_to_user, "working_memory_state": memory.get_state()}

# To run this application:
# 1. Save the code as ecommerce_agent.py
# 2. Install dependencies: pip install fastapi uvicorn pydantic loguru
# 3. Run from your terminal: uvicorn ecommerce_agent:app --reload
# 4. Access the API at http://127.0.0.1:8000/docs