from pydantic import BaseModel, Field
from typing import List, Dict, Any, TypedDict
from langgraph.graph import StateGraph, END


# 1. WorkingMemory (Pydantic Model)
class WorkingMemory(BaseModel):
    user_query: str = ""
    external_evidence: Dict[str, Any] = Field(default_factory=dict)
    llm_responses: List[str] = Field(default_factory=list)
    utility_scores: List[float] = Field(default_factory=list)
    feedback: List[str] = Field(default_factory=list)
    dialog_history: List[Dict[str, str]] = Field(default_factory=list)
    current_action: str = ""
    agent_response: str = ""


# Define AgentState for Langgraph, mirroring WorkingMemory
class AgentState(TypedDict):
    user_query: str
    external_evidence: Dict[str, Any]
    llm_responses: List[str]
    utility_scores: List[float]
    feedback: List[str]
    dialog_history: List[Dict[str, str]]
    current_action: str
    agent_response: str


# 2. PromptEngine Class
class PromptEngine:
    def generate_prompt(self, working_memory: WorkingMemory, task: str) -> str:
        history_str = "\n".join([f"{entry['role']}: {entry['content']}" for entry in working_memory.dialog_history])
        evidence_str = str(working_memory.external_evidence) if working_memory.external_evidence else "No external evidence."

        if task == "intent_recognition":
            return (
                f"Analyze the following customer query and dialog history to determine the user's primary intent (e.g., 'product_inquiry', 'order_status', 'recommendation', 'greeting', 'other').\n"
                f"Dialog History:\n{history_str}\n"
                f"Customer Query: {working_memory.user_query}\n"
                f"Intent:"
            )
        elif task == "generate_response":
            return (
                f"Based on the following dialog history, customer query, and external evidence, formulate a helpful and concise agent response.\n"
                f"Dialog History:\n{history_str}\n"
                f"Customer Query: {working_memory.user_query}\n"
                f"External Evidence: {evidence_str}\n"
                f"Agent Response:"
            )
        elif task == "product_search":
            return (
                f"Extract keywords for a product search from the user query: '{working_memory.user_query}'.\n"
                f"Keywords:"
            )
        elif task == "order_id_extraction":
            return (
                f"Extract the order ID from the user query: '{working_memory.user_query}'. If no order ID is present, state 'None'.\n"
                f"Order ID:"
            )
        return ""


# 3. LLMService Class (Simulated)
class LLMService:
    def get_response(self, prompt: str) -> str:
        # Simulate LLM response based on prompt keywords for demonstration
        if "intent" in prompt.lower() and "product_inquiry" in prompt.lower():
            return "product_inquiry"
        elif "intent" in prompt.lower() and "order_status" in prompt.lower():
            return "order_status"
        elif "intent" in prompt.lower() and "recommendation" in prompt.lower():
            return "recommendation"
        elif "intent" in prompt.lower() and "greeting" in prompt.lower():
            return "greeting"
        elif "intent" in prompt.lower():
            return "other"
        elif "keywords" in prompt.lower():
            if "laptop" in prompt.lower():
                return "laptop, gaming"
            return ""
        elif "order id" in prompt.lower() and "12345" in prompt.lower():
            return "12345"
        elif "order id" in prompt.lower():
            return "None"
        elif "hello" in prompt.lower() or "hi" in prompt.lower():
            return "Hello! How can I help you today?"
        elif "product details" in prompt.lower() and "laptop" in prompt.lower():
            return "The SuperGaming Laptop features an i9 processor, 32GB RAM, and an RTX 4080 GPU."
        elif "status for order 12345" in prompt.lower():
            return "Your order 12345 is currently being processed and is expected to ship within 2 business days."
        elif "recommend" in prompt.lower():
            return "I recommend checking out our latest line of smartwatches. They are very popular!"
        return "I am sorry, I didn't understand that. Can you please rephrase?"


# 4. EcommerceDataService Class (Simulated)
class EcommerceDataService:
    def get_product_details(self, product_id: str) -> Dict[str, Any]:
        # Simulate database lookup
        if product_id == "SG001": # SuperGaming Laptop
            return {"id": "SG001", "name": "SuperGaming Laptop", "price": 1800, "description": "Powerful gaming laptop with high-end specs.", "in_stock": True}
        return {}

    def search_products(self, query: str) -> List[Dict[str, Any]]:
        # Simulate product search
        if "laptop" in query.lower() or "gaming" in query.lower():
            return [
                {"id": "SG001", "name": "SuperGaming Laptop", "price": 1800},
                {"id": "UL002", "name": "UltraLight Laptop", "price": 1200}
            ]
        return []

    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        # Simulate order status lookup
        if order_id == "12345":
            return {"order_id": "12345", "status": "Processing", "estimated_delivery": "2-3 business days", "items": [{"product_id": "SG001", "quantity": 1}]}
        return {}


# 5. PolicyModule Class
class PolicyModule:
    def decide_action(self, working_memory: WorkingMemory) -> Dict[str, Any]:
        llm_service = LLMService()
        prompt_engine = PromptEngine()

        # First, try to recognize intent
        intent_prompt = prompt_engine.generate_prompt(working_memory, "intent_recognition")
        intent = llm_service.get_response(intent_prompt)

        working_memory.llm_responses.append(intent)
        print(f"[Policy] Detected Intent: {intent}")

        if intent == "product_inquiry":
            search_keywords_prompt = prompt_engine.generate_prompt(working_memory, "product_search")
            keywords = llm_service.get_response(search_keywords_prompt)
            working_memory.llm_responses.append(keywords)
            return {"current_action": "fetch_product_details", "search_query": keywords}
        elif intent == "order_status":
            order_id_prompt = prompt_engine.generate_prompt(working_memory, "order_id_extraction")
            order_id = llm_service.get_response(order_id_prompt)
            working_memory.llm_responses.append(order_id)
            if order_id != "None":
                return {"current_action": "fetch_order_status", "order_id": order_id}
            else:
                return {"current_action": "ask_for_order_id"}
        elif intent == "recommendation":
            return {"current_action": "provide_recommendation"}
        elif intent == "greeting":
            return {"current_action": "respond_greeting"}
        return {"current_action": "unknown_intent"}


# Langgraph Orchestration
def update_memory_with_user_query(state: AgentState) -> AgentState:
    working_memory = WorkingMemory(**state)
    user_query = state["user_query"]
    working_memory.dialog_history.append({"role": "user", "content": user_query})
    print(f"[Node] User query received: {user_query}")
    return working_memory.model_dump()


def decide_action_node(state: AgentState) -> AgentState:
    working_memory = WorkingMemory(**state)
    policy_module = PolicyModule()
    decision = policy_module.decide_action(working_memory)
    working_memory.current_action = decision["current_action"]
    # Pass along any specific data needed for the action
    if "search_query" in decision: working_memory.external_evidence["search_query"] = decision["search_query"]
    if "order_id" in decision: working_memory.external_evidence["order_id"] = decision["order_id"]
    print(f"[Node] Decided action: {working_memory.current_action}")
    return working_memory.model_dump()


def fetch_external_data_node(state: AgentState) -> AgentState:
    working_memory = WorkingMemory(**state)
    ecommerce_service = EcommerceDataService()
    action = working_memory.current_action
    print(f"[Node] Fetching external data for action: {action}")

    if action == "fetch_product_details":
        search_query = working_memory.external_evidence.get("search_query", "")
        products = ecommerce_service.search_products(search_query)
        working_memory.external_evidence["product_search_results"] = products
        if products:
            # For simplicity, just take the first product details
            product_details = ecommerce_service.get_product_details(products[0]["id"])
            working_memory.external_evidence["product_details"] = product_details
            print(f"[Node] Fetched product details for: {search_query}")
        else:
            print(f"[Node] No products found for: {search_query}")
            working_memory.external_evidence["product_details"] = None

    elif action == "fetch_order_status":
        order_id = working_memory.external_evidence.get("order_id", "")
        status = ecommerce_service.get_order_status(order_id)
        working_memory.external_evidence["order_status"] = status
        print(f"[Node] Fetched order status for: {order_id}")

    return working_memory.model_dump()


def generate_llm_response_node(state: AgentState) -> AgentState:
    working_memory = WorkingMemory(**state)
    llm_service = LLMService()
    prompt_engine = PromptEngine()
    action = working_memory.current_action
    agent_response = ""

    print(f"[Node] Generating LLM response for action: {action}")

    if action == "respond_greeting":
        agent_response = llm_service.get_response(prompt_engine.generate_prompt(working_memory, "generate_response"))
    elif action == "fetch_product_details":
        if working_memory.external_evidence.get("product_details"):
            product_details = working_memory.external_evidence["product_details"]
            agent_response = f"I found a product: {product_details['name']} for ${product_details['price']}. It's {product_details['description']}."
        else:
            agent_response = "I couldn't find any products matching your description. Can you be more specific?"
    elif action == "fetch_order_status":
        if working_memory.external_evidence.get("order_status"):
            order_status = working_memory.external_evidence["order_status"]
            agent_response = f"Your order {order_status['order_id']} is currently {order_status['status']} and estimated delivery is {order_status['estimated_delivery']}."
        else:
            agent_response = "I couldn't find information for that order ID. Please double-check it."
    elif action == "provide_recommendation":
        agent_response = llm_service.get_response(prompt_engine.generate_prompt(working_memory, "generate_response"))
    elif action == "ask_for_order_id":
        agent_response = "I need an order ID to check the status. Could you please provide it?"
    else: # unknown_intent or other actions that directly lead to LLM generation
        agent_response = llm_service.get_response(prompt_engine.generate_prompt(working_memory, "generate_response"))

    working_memory.llm_responses.append(agent_response)
    working_memory.agent_response = agent_response
    working_memory.dialog_history.append({"role": "agent", "content": agent_response})
    return working_memory.model_dump()


def format_agent_response_node(state: AgentState) -> AgentState:
    working_memory = WorkingMemory(**state)
    print(f"[Node] Formatting final agent response.")
    # The final response is already in agent_response from the previous node
    return working_memory.model_dump()


def route_actions(state: AgentState):
    action = state["current_action"]
    print(f"[Router] Routing based on action: {action}")
    if action in ["fetch_product_details", "fetch_order_status"]:
        return "fetch_external_data_node"
    elif action in ["respond_greeting", "provide_recommendation", "ask_for_order_id", "unknown_intent"]:
        return "generate_llm_response_node"
    else:
        return "generate_llm_response_node" # Default to generating response


# Build the Langgraph graph
workflow = StateGraph(AgentState)

workflow.add_node("update_memory_with_user_query", update_memory_with_user_query)
workflow.add_node("decide_action_node", decide_action_node)
workflow.add_node("fetch_external_data_node", fetch_external_data_node)
workflow.add_node("generate_llm_response_node", generate_llm_response_node)
workflow.add_node("format_agent_response_node", format_agent_response_node)

workflow.set_entry_point("update_memory_with_user_query")

workflow.add_edge("update_memory_with_user_query", "decide_action_node")
workflow.add_conditional_edges(
    "decide_action_node",
    route_actions,
    {
        "fetch_external_data_node": "fetch_external_data_node",
        "generate_llm_response_node": "generate_llm_response_node",
    },
)
workflow.add_edge("fetch_external_data_node", "generate_llm_response_node")
workflow.add_edge("generate_llm_response_node", "format_agent_response_node")
workflow.add_edge("format_agent_response_node", END)

app = workflow.compile()


# Example Usage
if __name__ == "__main__":
    # Initial state
    initial_state = WorkingMemory().model_dump()

    print("\n--- Turn 1: User greeting ---")
    inputs = {"user_query": "Hello! I need help.", **initial_state}
    for s in app.stream(inputs):
        print(s)
    final_state_turn1 = s[END]
    print(f"Agent: {final_state_turn1['agent_response']}")

    print("\n--- Turn 2: Product Inquiry ---")
    inputs = {"user_query": "I'm looking for a gaming laptop.", **final_state_turn1}
    for s in app.stream(inputs):
        print(s)
    final_state_turn2 = s[END]
    print(f"Agent: {final_state_turn2['agent_response']}")

    print("\n--- Turn 3: Order Status Inquiry ---")
    inputs = {"user_query": "What is the status of my order 12345?", **final_state_turn2}
    for s in app.stream(inputs):
        print(s)
    final_state_turn3 = s[END]
    print(f"Agent: {final_state_turn3['agent_response']}")

    print("\n--- Turn 4: Invalid Order Status Inquiry ---")
    inputs = {"user_query": "Check order 99999.", **final_state_turn3}
    for s in app.stream(inputs):
        print(s)
    final_state_turn4 = s[END]
    print(f"Agent: {final_state_turn4['agent_response']}")

    print("\n--- Turn 5: Recommendation ---")
    inputs = {"user_query": "Can you recommend something cool?", **final_state_turn4}
    for s in app.stream(inputs):
        print(s)
    final_state_turn5 = s[END]
    print(f"Agent: {final_state_turn5['agent_response']}")

    print("\n--- Turn 6: Unrecognized Query ---")
    inputs = {"user_query": "Tell me about quantum physics.", **final_state_turn5}
    for s in app.stream(inputs):
        print(s)
    final_state_turn6 = s[END]
    print(f"Agent: {final_state_turn6['agent_response']}")
