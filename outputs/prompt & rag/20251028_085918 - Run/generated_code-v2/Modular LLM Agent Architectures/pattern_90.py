import time
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from fastapi import FastAPI
import uvicorn

class WorkingMemoryState(BaseModel):
    current_user_query: Optional[str] = None
    external_evidence: Dict[str, Any] = {}
    llm_candidate_responses: List[Dict[str, Any]] = []
    utility_scores: List[float] = []
    verbalized_feedback: List[str] = []
    dialog_history: List[Dict[str, str]] = []

class WorkingMemory:
    def __init__(self):
        self._state = WorkingMemoryState()

    def update_query(self, query: str):
        self._state.current_user_query = query
        self._state.dialog_history.append({"speaker": "user", "text": query})

    def add_evidence(self, evidence: Dict[str, Any]):
        self._state.external_evidence.update(evidence)

    def add_llm_response(self, response: str, score: float = 0.0):
        self._state.llm_candidate_responses.append({"response": response, "score": score})
        self._state.utility_scores.append(score)

    def add_feedback(self, feedback: str):
        self._state.verbalized_feedback.append(feedback)

    def add_agent_response(self, response: str):
        self._state.dialog_history.append({"speaker": "agent", "text": response})

    def get_context(self) -> Dict[str, Any]:
        return self._state.model_dump()

    def clear_volatile_memory(self):
        self._state.llm_candidate_responses = []
        self._state.utility_scores = []
        self._state.verbalized_feedback = []

class PromptEngine:
    def __init__(self, working_memory: WorkingMemory):
        self.working_memory = working_memory

    def generate_prompt(self) -> str:
        context = self.working_memory.get_context()
        prompt_parts = []

        prompt_parts.append("You are an AI customer support agent for an e-commerce platform.")
        prompt_parts.append("Your goal is to assist users with product information, orders, returns, and general inquiries.")
        prompt_parts.append("Maintain context and be helpful, concise, and professional.")

        if context["dialog_history"]:
            prompt_parts.append("\n--- Dialog History ---")
            for item in context["dialog_history"]:
                prompt_parts.append(f"{item['speaker'].capitalize()}: {item['text']}")

        if context["external_evidence"]:
            prompt_parts.append("\n--- External Evidence ---")
            for key, value in context["external_evidence"].items():
                prompt_parts.append(f"{key}: {value}")

        if context["current_user_query"]:
            prompt_parts.append(f"\n--- Current User Query ---")
            prompt_parts.append(f"User: {context['current_user_query']}")

        prompt_parts.append("\n--- Your Task ---")
        prompt_parts.append("Generate a helpful and relevant response to the user. If you need more information or need to perform an action, indicate it.")
        prompt_parts.append("Consider the entire conversation history and any retrieved evidence.")
        prompt_parts.append("If the query is complex or requires human intervention, suggest escalation.")

        return "\n".join(prompt_parts)

class LLMIntegration:
    def call_llm(self, prompt: str) -> str:
        time.sleep(0.5)
        if "product availability" in prompt.lower() and "laptop" in prompt.lower():
            return "The 'XYZ Laptop' is currently in stock. Would you like to know more about its features?"
        elif "order status" in prompt.lower() and "12345" in prompt.lower():
            return "I found order 12345. It was placed on October 26, 2023, and is currently processing. Estimated delivery is within 3-5 business days."
        elif "return policy" in prompt.lower():
            return "Our return policy allows returns within 30 days of purchase for most items, provided they are in their original condition. Do you have a specific item in mind?"
        elif "complex issue" in prompt.lower() or "escalate" in prompt.lower():
            return "This seems like a complex issue that might require human assistance. Would you like me to connect you with a specialist?"
        elif "hello" in prompt.lower() or "hi" in prompt.lower():
            return "Hello! How can I assist you today?"
        else:
            return "I'm not sure how to respond to that. Could you please rephrase or provide more details?"

class ToolDatabaseIntegration:
    def lookup_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        time.sleep(0.2)
        if order_id == "12345":
            return {
                "order_id": "12345",
                "status": "processing",
                "items": ["XYZ Laptop", "Wireless Mouse"],
                "total": 1200.00,
                "delivery_estimate": "3-5 business days"
            }
        return None

    def get_product_info(self, product_name: str) -> Optional[Dict[str, Any]]:
        time.sleep(0.2)
        if "laptop" in product_name.lower() or "xyz laptop" in product_name.lower():
            return {
                "product_name": "XYZ Laptop",
                "category": "Electronics",
                "price": 1100.00,
                "in_stock": True,
                "description": "Powerful laptop with 16GB RAM and 512GB SSD."
            }
        elif "wireless mouse" in product_name.lower():
            return {
                "product_name": "Wireless Mouse",
                "category": "Accessories",
                "price": 25.00,
                "in_stock": True,
                "description": "Ergonomic wireless mouse."
            }
        return None

class Policy:
    def __init__(
        self,
        working_memory: WorkingMemory,
        prompt_engine: PromptEngine,
        llm_integration: LLMIntegration,
        tool_database_integration: ToolDatabaseIntegration,
    ):
        self.working_memory = working_memory
        self.prompt_engine = prompt_engine
        self.llm_integration = llm_integration
        self.tool_database_integration = tool_database_integration

    def decide_and_act(self) -> str:
        user_query = self.working_memory.get_context()["current_user_query"]
        agent_response = "I'm sorry, I couldn't process your request."

        if user_query and "order status" in user_query.lower():
            order_id = next((word for word in user_query.split() if word.isdigit()), "12345")
            order_info = self.tool_database_integration.lookup_order(order_id)
            if order_info:
                self.working_memory.add_evidence({"order_info": order_info})
                self.working_memory.add_feedback(f"Retrieved order info for {order_id}.")
            else:
                self.working_memory.add_feedback(f"Could not find order {order_id}.")
        elif user_query and ("product info" in user_query.lower() or "tell me about" in user_query.lower()):
            product_name = next((word for word in ["laptop", "wireless mouse"] if word in user_query.lower()), "laptop")
            product_info = self.tool_database_integration.get_product_info(product_name)
            if product_info:
                self.working_memory.add_evidence({"product_info": product_info})
                self.working_memory.add_feedback(f"Retrieved product info for {product_name}.")
            else:
                self.working_memory.add_feedback(f"Could not find product {product_name}.")

        prompt = self.prompt_engine.generate_prompt()
        self.working_memory.add_feedback(f"Generated prompt: {prompt[:100]}...")

        llm_output = self.llm_integration.call_llm(prompt)
        self.working_memory.add_llm_response(llm_output, score=1.0)
        self.working_memory.add_feedback(f"Received LLM output: {llm_output[:100]}...")

        agent_response = llm_output

        self.working_memory.add_agent_response(agent_response)

        self.working_memory.clear_volatile_memory()

        return agent_response

app = FastAPI(title="E-commerce AI Customer Support Agent")

wm = WorkingMemory()
pe = PromptEngine(wm)
li = LLMIntegration()
tdi = ToolDatabaseIntegration()
policy = Policy(wm, pe, li, tdi)

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    agent_response: str
    dialog_history: List[Dict[str, str]]

@app.post("/chat", response_model=ChatResponse)
async def chat_with_agent(request: ChatRequest):
    user_message = request.message
    wm.update_query(user_message)
    agent_response = policy.decide_and_act()
    
    return ChatResponse(agent_response=agent_response, dialog_history=wm.get_context()["dialog_history"])