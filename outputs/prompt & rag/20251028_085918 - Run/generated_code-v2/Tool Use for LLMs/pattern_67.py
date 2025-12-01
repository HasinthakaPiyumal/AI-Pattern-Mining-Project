import datetime

class StateManager:
    def __init__(self):
        self.conversation_history = []
        self.current_web_page_context = {
            "url": "",
            "text_content": "",
            "interactive_elements": [],
            "focused_area": ""
        }
        self.agent_past_actions = []
        self.internal_state = {
            "goal": "assist customer",
            "status": "waiting_for_user_input"
        }

    def update_conversation_history(self, speaker, message):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.conversation_history.append(f"[{timestamp}] {speaker}: {message}")

    def update_web_page_context(self, url, text_content, interactive_elements=None, focused_area=""):
        self.current_web_page_context["url"] = url
        self.current_web_page_context["text_content"] = text_content
        self.current_web_page_context["interactive_elements"] = interactive_elements if interactive_elements is not None else []
        self.current_web_page_context["focused_area"] = focused_area

    def add_agent_action(self, action_description):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.agent_past_actions.append(f"[{timestamp}] Agent Action: {action_description}")

    def update_internal_state(self, goal=None, status=None):
        if goal: self.internal_state["goal"] = goal
        if status: self.internal_state["status"] = status

    def get_current_state_summary(self, user_query):
        summary_parts = [
            "### Current State Summary ###",
            f"User Query: {user_query}",
            "\n### Conversation History ###"
        ]
        summary_parts.extend(self.conversation_history)

        summary_parts.append("\n### Current Web Page Context ###")
        summary_parts.append(f"URL: {self.current_web_page_context['url']}")
        summary_parts.append(f"Text Content (excerpt): {self.current_web_page_context['text_content'][:500]}...")
        if self.current_web_page_context['interactive_elements']:
            summary_parts.append("Interactive Elements: " + ", ".join(self.current_web_page_context['interactive_elements']))
        if self.current_web_page_context['focused_area']:
            summary_parts.append(f"Focused Area: {self.current_web_page_context['focused_area']}")

        if self.agent_past_actions:
            summary_parts.append("\n### Agent Past Actions ###")
            summary_parts.extend(self.agent_past_actions)

        summary_parts.append("\n### Agent Internal State ###")
        for key, value in self.internal_state.items():
            summary_parts.append(f"{key.replace('_', ' ').title()}: {value}")

        return "\n".join(summary_parts)

class LLMAgent:
    def __init__(self, name="CustomerSupportLLM"):
        self.name = name

    def get_response_and_action(self, state_summary):
        # In a real application, this would call an actual LLM API
        # For demonstration, we'll simulate a simple response based on keywords

        print("--- LLM Received State Summary ---")
        print(state_summary)
        print("----------------------------------")

        response = "I understand. How can I help you further?"
        action = None

        user_query = state_summary.split("User Query: ", 1)[1].split("\n", 1)[0]

        if "order status" in user_query.lower():
            response = "To check your order status, please provide your order number."
            action = "request_order_number"
        elif "browse products" in user_query.lower() or "look for items" in user_query.lower():
            response = "Certainly! What kind of products are you interested in?"
            action = "ask_product_category"
        elif "hello" in user_query.lower() or "hi" in user_query.lower():
            response = "Hello! How may I assist you with your shopping today?"
            action = None

        return response, action


# --- Example Usage ---
if __name__ == "__main__":
    state_manager = StateManager()
    llm_agent = LLMAgent()

    print("--- Customer Interaction 1 ---")
    user_input_1 = "Hello, I need help with my order status."
    state_manager.update_conversation_history("Customer", user_input_1)
    state_manager.update_web_page_context(
        url="https://example.com/support",
        text_content="Welcome to our support page. Please login to view your orders...",
        interactive_elements=["Login button", "Order lookup form"]
    )

    state_summary_1 = state_manager.get_current_state_summary(user_input_1)
    agent_response_1, agent_action_1 = llm_agent.get_response_and_action(state_summary_1)

    print(f"Agent says: {agent_response_1}")
    print(f"Agent takes action: {agent_action_1}")
    state_manager.update_conversation_history("Agent", agent_response_1)
    if agent_action_1: state_manager.add_agent_action(f"Decided to {agent_action_1}")

    print("\n--- Customer Interaction 2 ---")
    user_input_2 = "My order number is #XYZ123. Can you find it?"
    state_manager.update_conversation_history("Customer", user_input_2)
    state_manager.update_web_page_context(
        url="https://example.com/order-lookup",
        text_content="Enter your order number below.",
        interactive_elements=["Order number input", "Submit button"],
        focused_area="Order number input"
    )
    state_manager.update_internal_state(status="awaiting_order_details")
    state_manager.add_agent_action("Navigated to order lookup page and focused on input field.")

    state_summary_2 = state_manager.get_current_state_summary(user_input_2)
    agent_response_2, agent_action_2 = llm_agent.get_response_and_action(state_summary_2)

    print(f"Agent says: {agent_response_2}")
    print(f"Agent takes action: {agent_action_2}")
    state_manager.update_conversation_history("Agent", agent_response_2)
    if agent_action_2: state_manager.add_agent_action(f"Decided to {agent_action_2}")

    print("\n--- Customer Interaction 3 ---")
    user_input_3 = "I am looking for new running shoes."
    state_manager.update_conversation_history("Customer", user_input_3)
    state_manager.update_web_page_context(
        url="https://example.com/products",
        text_content="Explore our wide range of products.",
        interactive_elements=["Search bar", "Category filters"]
    )
    state_manager.update_internal_state(goal="recommend_products", status="gathering_preferences")

    state_summary_3 = state_manager.get_current_state_summary(user_input_3)
    agent_response_3, agent_action_3 = llm_agent.get_response_and_action(state_summary_3)

    print(f"Agent says: {agent_response_3}")
    print(f"Agent takes action: {agent_action_3}")
    state_manager.update_conversation_history("Agent", agent_response_3)
    if agent_action_3: state_manager.add_agent_action(f"Decided to {agent_action_3}")