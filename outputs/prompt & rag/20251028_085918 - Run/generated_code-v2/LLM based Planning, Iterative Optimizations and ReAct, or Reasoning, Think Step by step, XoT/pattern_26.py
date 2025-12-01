import json

class LLM:
    def __init__(self, model_name="default_model"):
        self.model_name = model_name

    def generate_response(self, prompt, history=None):
        # Simulate an LLM call. In a real application, this would use openai.Completion or similar.
        if "refund issue" in prompt.lower() and "order number" not in prompt.lower():
            return "I understand you have a refund issue. Could you please provide your order number?"
        elif "order number 123" in prompt.lower() and "refund" in prompt.lower():
            return "Thank you for the order number. I'm processing your refund for order 123. It should reflect in your account within 3-5 business days."
        elif "I need help" in prompt.lower():
            return "I can help you with common issues like refunds, order tracking, or product information. What can I assist you with today?"
        elif "reflection_query" in prompt:
            # Simulate reflection model's response
            if "failed to get order number for refund" in prompt:
                return "Reflection Insight: The chatbot failed to ask for the order number early enough when a refund issue was detected. Future strategy: Prioritize asking for the order number when a refund or return query is initiated."
            return "Reflection Insight: The agent could improve by clarifying user intent more explicitly or by asking for specific details relevant to the initial problem statement."
        return "I am an AI assistant. How can I help you today?"


class Chatbot:
    def __init__(self, name="SupportBot"):
        self.name = name
        self.llm = LLM(model_name="chatbot_brain")
        self.conversation_history = []
        self.reflection_insights = []

    def add_message(self, speaker, message):
        self.conversation_history.append({"speaker": speaker, "message": message})

    def _format_history_for_llm(self):
        formatted_history = ""
        if self.reflection_insights:
            formatted_history += "\n".join(self.reflection_insights) + "\n\n"
        for entry in self.conversation_history:
            formatted_history += f"{entry['speaker']}: {entry['message']}\n"
        return formatted_history.strip()

    def respond(self, user_input):
        self.add_message("User", user_input)
        current_prompt = self._format_history_for_llm() + f"\nChatbot: "
        response = self.llm.generate_response(current_prompt, history=self.conversation_history)
        self.add_message("Chatbot", response)
        return response

    def detect_failure(self, user_feedback):
        # Simple failure detection based on keywords in user feedback
        if "unhelpful" in user_feedback.lower() or "didn't understand" in user_feedback.lower() or "still have the problem" in user_feedback.lower():
            return True
        return False

    def incorporate_insights(self, insights):
        self.reflection_insights.append(insights)
        print(f"[{self.name}] Incorporated new insights: {insights}")


class ReflectionModel:
    def __init__(self):
        self.llm = LLM(model_name="reflection_brain")

    def reflect_on_trajectory(self, trajectory):
        # Analyze the trajectory and identify points of failure or inefficiency
        trajectory_str = json.dumps(trajectory)
        reflection_prompt = f"reflection_query: Analyze the following conversation trajectory to identify the chatbot's failure points or suboptimal responses. Provide a high-level insight for improvement. Conversation:\n{trajectory_str}"
        
        # Simulate detailed analysis for specific scenarios
        if "refund issue" in trajectory_str and "order number" not in trajectory_str:
            return self.llm.generate_response("reflection_query: failed to get order number for refund")
        
        return self.llm.generate_response(reflection_prompt)


# --- Main Application Logic --- 
if __name__ == "__main__":
    customer_chatbot = Chatbot()
    reflector = ReflectionModel()

    print("--- Initial Interaction ---")
    print("User: I need help with a refund issue.")
    bot_response = customer_chatbot.respond("I need help with a refund issue.")
    print(f"Chatbot: {bot_response}")

    print("\nUser: It's still not resolved, your bot didn't ask for my order number!")
    user_feedback = "It's still not resolved, your bot didn't ask for my order number!"
    if customer_chatbot.detect_failure(user_feedback):
        print("\n--- Failure Detected! Initiating Reflection ---")
        insights = reflector.reflect_on_trajectory(customer_chatbot.conversation_history)
        customer_chatbot.incorporate_insights(insights)
        # Clear conversation history for a fresh start with new insights, or continue building
        customer_chatbot.conversation_history = [] 

    print("\n--- Subsequent Interaction (with reflection insights) ---")
    print("User: I need a refund.")
    bot_response = customer_chatbot.respond("I need a refund.")
    print(f"Chatbot: {bot_response}")

    print("\nUser: My order number is 123.")
    bot_response = customer_chatbot.respond("My order number is 123.")
    print(f"Chatbot: {bot_response}")

    print("\n--- Another scenario ---")
    customer_chatbot_2 = Chatbot("NewBot")
    reflector_2 = ReflectionModel()
    print("\n--- Initial Interaction (NewBot) ---")
    print("User: I need help.")
    bot_response_2 = customer_chatbot_2.respond("I need help.")
    print(f"NewBot: {bot_response_2}")
    print("\nUser: This is unhelpful.")
    user_feedback_2 = "This is unhelpful."
    if customer_chatbot_2.detect_failure(user_feedback_2):
        print("\n--- Failure Detected! Initiating Reflection (NewBot) ---")
        insights_2 = reflector_2.reflect_on_trajectory(customer_chatbot_2.conversation_history)
        customer_chatbot_2.incorporate_insights(insights_2)
        customer_chatbot_2.conversation_history = []

    print("\n--- Subsequent Interaction (NewBot with reflection insights) ---")
    print("User: Can you assist me?")
    bot_response_2 = customer_chatbot_2.respond("Can you assist me?")
    print(f"NewBot: {bot_response_2}")
