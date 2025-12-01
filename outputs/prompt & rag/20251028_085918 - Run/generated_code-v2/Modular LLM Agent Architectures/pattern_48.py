import random

class WorkingMemory:
    def __init__(self):
        self.user_query = None
        self.external_evidence = []
        self.llm_candidate_responses = []
        self.utility_scores = []
        self.verbalized_feedback = []
        self.dialog_history = []

    def update_query(self, query: str):
        self.user_query = query

    def add_evidence(self, evidence: dict):
        self.external_evidence.append(evidence)

    def add_llm_response(self, response: str, score: float):
        self.llm_candidate_responses.append(response)
        self.utility_scores.append(score)

    def add_feedback(self, feedback: str):
        self.verbalized_feedback.append(feedback)

    def add_to_dialog_history(self, turn: dict):
        self.dialog_history.append(turn)

    def get_state(self) -> dict:
        return {
            "user_query": self.user_query,
            "external_evidence": self.external_evidence,
            "llm_candidate_responses": self.llm_candidate_responses,
            "utility_scores": self.utility_scores,
            "verbalized_feedback": self.verbalized_feedback,
            "dialog_history": self.dialog_history
        }

    def reset(self):
        self.user_query = None
        self.external_evidence = []
        self.llm_candidate_responses = []
        self.utility_scores = []
        self.verbalized_feedback = []
        self.dialog_history = []

def mock_llm_response(query: str, history: list) -> str:
    if "order status" in query.lower():
        return "I can help with your order status. What is your order number?"
    elif "return" in query.lower() and "item" in query.lower():
        return "To initiate a return, please provide the item details and your order number."
    elif "recommend" in query.lower() and "product" in query.lower():
        return "Certainly! To help me recommend a product, what are you looking for? (e.g., type of product, price range, features)"
    elif "issue with delivery" in query.lower():
        return "I'm sorry to hear about the delivery issue. Can you please provide your order number so I can investigate?"
    elif "hello" in query.lower() or "hi" in query.lower():
        return "Hello! How can I assist you today?"
    else:
        return f"I'm not sure how to respond to '{query}'. Can you please rephrase or provide more details?"

def mock_get_external_evidence(query: str) -> dict:
    if "order status" in query.lower() and "12345" in query:
        return {"order_id": "12345", "status": "Shipped", "estimated_delivery": "2023-11-20"}
    elif "return" in query.lower() and "shirt" in query.lower():
        return {"item": "Blue Shirt", "return_policy": "30 days, unworn with tags."}
    elif "laptop" in query.lower() and "gaming" in query.lower():
        return {"category": "laptops", "type": "gaming", "available_models": ["GamingX Pro", "PowerBook Elite"]}
    else:
        return {}

class Policy:
    def decide_action(self, memory_state: dict) -> str:
        user_query = memory_state.get("user_query", "")
        external_evidence = memory_state.get("external_evidence", [])
        llm_candidate_responses = memory_state.get("llm_candidate_responses", [])

        if "escalate" in user_query.lower() or "human agent" in user_query.lower():
            return "ESCALATE_TO_HUMAN"
        if "order status" in user_query.lower() and not external_evidence:
            return "SEARCH_KB"
        if "issue with delivery" in user_query.lower() and not external_evidence:
            return "SEARCH_KB"
        if "return" in user_query.lower() and not external_evidence:
            return "SEARCH_KB"
        if "recommend" in user_query.lower() and "product" in user_query.lower() and not external_evidence:
            return "SEARCH_KB"
        if llm_candidate_responses:
            return "RESPOND_TO_USER"

        return "RESPOND_TO_USER"

class CustomerSupportAgent:
    def __init__(self):
        self.memory = WorkingMemory()
        self.policy = Policy()

    def handle_message(self, user_message: str) -> str:
        self.memory.update_query(user_message)

        # Simulate fetching external evidence
        evidence = mock_get_external_evidence(user_message)
        if evidence:
            self.memory.add_evidence(evidence)

        # Simulate LLM interaction
        llm_response_text = mock_llm_response(user_message, self.memory.dialog_history)
        # Assign a random utility score for demonstration
        llm_score = random.uniform(0.5, 1.0)
        self.memory.add_llm_response(llm_response_text, llm_score)

        current_state = self.memory.get_state()
        action = self.policy.decide_action(current_state)

        agent_response = ""
        if action == "RESPOND_TO_USER":
            if self.memory.llm_candidate_responses:
                # For simplicity, pick the last LLM response
                agent_response = self.memory.llm_candidate_responses[-1]
            else:
                agent_response = "I'm sorry, I couldn't generate a suitable response."
        elif action == "SEARCH_KB":
            agent_response = f"Let me check our knowledge base for '{user_message}'..."
            # In a real scenario, this would trigger a more complex KB search and update memory
            if current_state["external_evidence"]:
                agent_response += f" I found: {current_state['external_evidence']}"
                # Now let LLM try to respond with the new evidence
                llm_response_with_evidence = mock_llm_response(f"{user_message} with evidence: {current_state['external_evidence']}", self.memory.dialog_history)
                self.memory.add_llm_response(llm_response_with_evidence, random.uniform(0.5, 1.0))
                agent_response += f" Agent says: {llm_response_with_evidence}"
        elif action == "ESCALATE_TO_HUMAN":
            agent_response = "I understand this is a complex issue. I'm escalating your query to a human agent who will be able to assist you further."
        else:
            agent_response = "I'm performing an unknown action based on my policy."

        self.memory.add_to_dialog_history({"user": user_message, "agent": agent_response})
        return agent_response


if __name__ == "__main__":
    agent = CustomerSupportAgent()

    print("\n--- Starting Customer Support Chat ---")

    # Test Case 1: Order Status
    user_input = "Hi, what's the status of my order 12345?"
    print(f"User: {user_input}")
    response = agent.handle_message(user_input)
    print(f"Agent: {response}")
    print("Memory after turn 1:", agent.memory.get_state())

    # Test Case 2: Product Recommendation
    user_input = "Can you recommend a gaming laptop?"
    print(f"\nUser: {user_input}")
    response = agent.handle_message(user_input)
    print(f"Agent: {response}")
    print("Memory after turn 2:", agent.memory.get_state())

    # Test Case 3: Return Item
    user_input = "I want to return a blue shirt."
    print(f"\nUser: {user_input}")
    response = agent.handle_message(user_input)
    print(f"Agent: {response}")
    print("Memory after turn 3:", agent.memory.get_state())

    # Test Case 4: Escalation
    user_input = "I need to speak to a human agent, this is urgent!"
    print(f"\nUser: {user_input}")
    response = agent.handle_message(user_input)
    print(f"Agent: {response}")
    print("Memory after turn 4:", agent.memory.get_state())

    # Test Case 5: Complex Delivery Issue (showing memory use)
    user_input = "My delivery for order 12345 is delayed."
    print(f"\nUser: {user_input}")
    response = agent.handle_message(user_input)
    print(f"Agent: {response}")
    print("Memory after turn 5:", agent.memory.get_state())

    # Reset memory and start new session
    agent.memory.reset()
    print("\n--- Memory Reset. Starting New Session ---")
    user_input = "Hello, I have a new question."
    print(f"User: {user_input}")
    response = agent.handle_message(user_input)
    print(f"Agent: {response}")
    print("Memory after new session start:", agent.memory.get_state())

    print("\n--- End of Chat ---")
