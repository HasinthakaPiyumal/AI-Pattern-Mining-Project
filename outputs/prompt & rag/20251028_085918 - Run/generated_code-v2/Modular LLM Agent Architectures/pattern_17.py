class MockLLM:
    def __init__(self):
        pass

    def generate_response(self, prompt, context=""):
        if "order status" in prompt.lower():
            return {"response": "I can help with that. What is the order ID?", "action": {"tool": "get_order_status"}}
        if "create ticket" in prompt.lower():
            return {"response": "What is the issue you are facing?", "action": {"tool": "create_ticket"}}
        if "sentiment" in context.lower() and "negative" in context.lower():
            return {"response": "I understand your frustration. Let me see how I can help.", "action": None}
        return {"response": f"Thank you for your query. How can I assist you further based on: {prompt}", "action": None}

class WorkingMemoryModule:
    def __init__(self):
        self.memory = {}
        self.conversation_history = []

    def embed(self, text):
        return text

    def store_context(self, key, value):
        self.memory[key] = value

    def retrieve_context(self, query):
        relevant_info = [v for k, v in self.memory.items() if query.lower() in str(v).lower() or query.lower() in str(k).lower()]
        return " ".join(relevant_info) if relevant_info else ""

    def add_to_history(self, role, message):
        self.conversation_history.append({"role": role, "message": message})

    def get_history(self, limit=5):
        return "\n".join([f"{item["role"]}: {item["message"]}" for item in self.conversation_history[-limit:]])

class PolicyModule:
    def __init__(self):
        pass

    def analyze_sentiment(self, text):
        if "frustrat" in text.lower() or "unhappy" in text.lower() or "terrible" in text.lower():
            return "negative"
        return "neutral"

    def apply_rules(self, current_context, sentiment):
        if sentiment == "negative" and ("urgent" in current_context.lower() or "now" in current_context.lower()):
            return "escalate_to_human"
        return None

class ActionExecutorModule:
    def __init__(self):
        self.available_tools = {
            "get_order_status": self._get_order_status,
            "create_ticket": self._create_ticket,
        }

    def _get_order_status(self, order_id):
        if order_id == "12345":
            return f"Order {order_id} is currently being shipped."
        return f"Could not find status for order {order_id}."

    def _create_ticket(self, issue_details):
        ticket_id = "TICKET-" + str(hash(issue_details) % 10000)
        return f"A support ticket has been created with ID {ticket_id} for the issue: {issue_details}."

    def execute_action(self, tool_name, *args, **kwargs):
        if tool_name in self.available_tools:
            return self.available_tools[tool_name](*args, **kwargs)
        return f"Error: Tool \'{tool_name}\' not found."

class UtilityModule:
    def __init__(self):
        self.logs = []

    def log_interaction(self, user_query, llm_input, llm_output, final_response, modules_involved):
        log_entry = {
            "user_query": user_query,
            "llm_input": llm_input,
            "llm_output": llm_output,
            "final_response": final_response,
            "modules_involved": modules_involved,
            "timestamp": "MOCK_TIMESTAMP"
        }
        self.logs.append(log_entry)

    def get_logs(self):
        return self.logs

class Orchestrator:
    def __init__(self):
        self.llm = MockLLM()
        self.working_memory = WorkingMemoryModule()
        self.policy = PolicyModule()
        self.action_executor = ActionExecutorModule()
        self.utility = UtilityModule()

        self.working_memory.store_context("product_info", "Our product \'X\' has features A, B, C. Our product \'Y\' has features D, E.")
        self.working_memory.store_context("faq_returns", "Returns are accepted within 30 days with a valid receipt.")

    def process_query(self, user_query):
        self.working_memory.add_to_history("user", user_query)
        modules_involved = []
        final_response = ""

        sentiment = self.policy.analyze_sentiment(user_query)
        modules_involved.append("PolicyModule (Sentiment)")

        retrieved_context = self.working_memory.retrieve_context(user_query)
        modules_involved.append("WorkingMemoryModule (Retrieval)")

        conversation_history = self.working_memory.get_history()

        llm_prompt = f"User query: {user_query}\n"
        if retrieved_context:
            llm_prompt += f"Relevant context: {retrieved_context}\n"
        if conversation_history:
            llm_prompt += f"Conversation history:\n{conversation_history}\n"
        if sentiment == "negative":
            llm_prompt += "The user seems frustrated. Please respond empathetically.\n"

        llm_output = self.llm.generate_response(llm_prompt, context=sentiment)
        modules_involved.append("Blackbox LLM")
        llm_response_text = llm_output["response"]
        llm_suggested_action = llm_output.get("action")

        policy_action = self.policy.apply_rules(user_query, sentiment)
        if policy_action == "escalate_to_human":
            final_response = "I detect an urgent issue and will escalate this to a human agent immediately. Please hold."
            modules_involved.append("PolicyModule (Escalation)")
        elif llm_suggested_action and llm_suggested_action.get("tool"):
            tool_name = llm_suggested_action["tool"]
            if tool_name == "get_order_status":
                order_id_mock = "12345"
                action_result = self.action_executor.execute_action(tool_name, order_id_mock)
            elif tool_name == "create_ticket":
                issue_details_mock = user_query
                action_result = self.action_executor.execute_action(tool_name, issue_details_mock)
            else:
                action_result = f"Unknown tool: {tool_name}"

            final_response = f"{llm_response_text}\n{action_result}"
            modules_involved.append("ActionExecutorModule")
        else:
            final_response = llm_response_text

        self.working_memory.add_to_history("LLM", final_response)
        self.working_memory.store_context("last_llm_response", final_response)
        modules_involved.append("WorkingMemoryModule (Update)")

        self.utility.log_interaction(user_query, llm_prompt, llm_output, final_response, modules_involved)
        modules_involved.append("UtilityModule")

        return final_response

if __name__ == "__main__":
    agent = Orchestrator()

    print("\n--- Scenario 1: Basic Query ---")
    response = agent.process_query("Tell me about product X.")
    print(f"Agent: {response}")

    print("\n--- Scenario 2: Order Status Inquiry ---")
    response = agent.process_query("What is the status of my order?")
    print(f"Agent: {response}")
    response = agent.process_query("My order ID is 12345.")
    print(f"Agent: {response}")

    print("\n--- Scenario 3: Negative Sentiment & Escalation ---")
    response = agent.process_query("I am very unhappy with your service! I need help now.")
    print(f"Agent: {response}")

    print("\n--- Scenario 4: Create Ticket ---")
    response = agent.process_query("I have a problem with my account, it\'s not letting me log in.")
    print(f"Agent: {response}")

    print("\n--- Agent Logs ---")
    for log in agent.utility.get_logs():
        print(f"User Query: {log['user_query']}")
        print(f"Final Response: {log['final_response']}")
        print(f"Modules: {', '.join(log['modules_involved'])}")
        print("--------------------")
