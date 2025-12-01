class WorkingMemory:
    def __init__(self):
        self.user_query = None
        self.external_evidence = {}
        self.llm_candidate_responses = []
        self.utility_scores = {}
        self.verbalized_feedback = []
        self.dialog_history = []

    def update_user_query(self, query):
        self.user_query = query
        self.add_to_dialog_history("User", query)

    def add_external_evidence(self, key, value):
        self.external_evidence[key] = value

    def add_llm_candidate_response(self, response, score=None):
        self.llm_candidate_responses.append(response)
        if score is not None:
            self.utility_scores[response] = score

    def add_verbalized_feedback(self, feedback):
        self.verbalized_feedback.append(feedback)

    def add_to_dialog_history(self, speaker, utterance):
        self.dialog_history.append((speaker, utterance))

    def get_state(self):
        return {
            "user_query": self.user_query,
            "external_evidence": self.external_evidence,
            "llm_candidate_responses": self.llm_candidate_responses,
            "utility_scores": self.utility_scores,
            "verbalized_feedback": self.verbalized_feedback,
            "dialog_history": self.dialog_history,
        }

    def clear_session_data(self):
        self.user_query = None
        self.external_evidence = {}
        self.llm_candidate_responses = []
        self.utility_scores = {}
        self.verbalized_feedback = []
        self.dialog_history = []

class PolicyModule:
    def __init__(self, working_memory):
        self.working_memory = working_memory

    def decide_next_action(self):
        state = self.working_memory.get_state()
        # Simplified decision logic for demonstration
        if state["user_query"] and "account" in state["user_query"].lower() and not state["external_evidence"].get("account_details"):
            return {"action": "get_account_details", "payload": state["user_query"]}
        elif state["llm_candidate_responses"]:
            # For simplicity, just pick the first candidate or the one with the highest utility score if available
            if state["utility_scores"]:
                best_response = max(state["utility_scores"], key=state["utility_scores"].get)
                return {"action": "respond", "payload": best_response}
            else:
                return {"action": "respond", "payload": state["llm_candidate_responses"][0]}
        elif state["user_query"]:
            return {"action": "generate_llm_response", "payload": state["user_query"]}
        return {"action": "wait_for_input"}

class PromptEngine:
    def __init__(self, working_memory):
        self.working_memory = working_memory

    def construct_prompt(self):
        state = self.working_memory.get_state()
        prompt_parts = []
        prompt_parts.append("You are a helpful telecommunications customer support agent.")
        if state["dialog_history"]:
            prompt_parts.append("Here is the conversation so far:")
            for speaker, utterance in state["dialog_history"]:
                prompt_parts.append(f"{speaker}: {utterance}")
        if state["external_evidence"]:
            prompt_parts.append("Here is some relevant customer information:")
            for key, value in state["external_evidence"].items():
                prompt_parts.append(f"{key}: {value}")
        if state["user_query"]:
            prompt_parts.append(f"The customer's current query is: {state['user_query']}")
        prompt_parts.append("Please provide a helpful response or ask a clarifying question.")
        return "\n".join(prompt_parts)

class LLMIntegration:
    def __init__(self):
        pass

    def generate_response(self, prompt):
        # This is a mock LLM. In a real scenario, this would call an actual LLM API.
        print(f"[MOCK LLM Request]: {prompt[:100]}...")
        mock_responses = [
            "I understand you're having an issue. Can you please describe it in more detail?",
            "To help you with your account, could you please confirm your account number or registered phone number?",
            "It sounds like a connectivity problem. Have you tried restarting your router?"
        ]
        import random
        selected_response = random.choice(mock_responses)
        return [selected_response], {selected_response: random.uniform(0.7, 0.99)}

class ToolExecutor:
    def __init__(self, working_memory):
        self.working_memory = working_memory

    def execute_tool(self, tool_name, payload=None):
        if tool_name == "get_account_details":
            print(f"[MOCK TOOL]: Retrieving account details for query: {payload}")
            # Simulate fetching data
            account_details = {"account_number": "TELCO12345", "plan": "Premium Unlimited", "balance": "$75.00"}
            self.working_memory.add_external_evidence("account_details", account_details)
            self.working_memory.add_verbalized_feedback("Successfully fetched account details.")
            return True
        print(f"[MOCK TOOL]: Unknown tool: {tool_name}")
        return False

class CustomerSupportAgent:
    def __init__(self):
        self.memory = WorkingMemory()
        self.policy = PolicyModule(self.memory)
        self.prompt_engine = PromptEngine(self.memory)
        self.llm = LLMIntegration()
        self.tool_executor = ToolExecutor(self.memory)

    def converse(self, user_input):
        self.memory.update_user_query(user_input)

        action = self.policy.decide_next_action()

        if action["action"] == "get_account_details":
            self.tool_executor.execute_tool(action["action"], action["payload"])
            # After tool execution, re-evaluate policy
            action = self.policy.decide_next_action()

        if action["action"] == "generate_llm_response":
            prompt = self.prompt_engine.construct_prompt()
            responses, scores = self.llm.generate_response(prompt)
            for res in responses:
                self.memory.add_llm_candidate_response(res, scores.get(res))
            action = self.policy.decide_next_action() # Re-evaluate after LLM response

        if action["action"] == "respond":
            agent_response = action["payload"]
            self.memory.add_to_dialog_history("Agent", agent_response)
            return agent_response
        elif action["action"] == "wait_for_input":
            return "Please tell me how I can assist you today."
        else:
            return "I am processing your request..."

# --- Example Usage ---
if __name__ == "__main__":
    agent = CustomerSupportAgent()

    print("\n--- Turn 1 ---")
    user_input = "I need help with my account, it's not working."
    response = agent.converse(user_input)
    print(f"User: {user_input}")
    print(f"Agent: {response}")
    print("Current Memory State (after Turn 1):")
    print(agent.memory.get_state())

    print("\n--- Turn 2 ---")
    user_input = "My internet is super slow and I can't stream anything."
    response = agent.converse(user_input)
    print(f"User: {user_input}")
    print(f"Agent: {response}")
    print("Current Memory State (after Turn 2):")
    print(agent.memory.get_state())

    print("\n--- Turn 3 ---")
    user_input = "Yes, I've tried restarting my router, still no luck."
    response = agent.converse(user_input)
    print(f"User: {user_input}")
    print(f"Agent: {response}")
    print("Current Memory State (after Turn 3):")
    print(agent.memory.get_state())

    print("\n--- Turn 4 - New session/cleared memory ---")
    agent.memory.clear_session_data()
    user_input = "Hello, I want to upgrade my plan."
    response = agent.converse(user_input)
    print(f"User: {user_input}")
    print(f"Agent: {response}")
    print("Current Memory State (after Turn 4):")
    print(agent.memory.get_state())