class WorkingMemory:
    def __init__(self):
        self.user_query = ""
        self.external_evidence = []
        self.llm_candidate_responses = []
        self.utility_scores = []
        self.verbalized_feedback = ""
        self.dialog_history = []

    def update_memory(self, user_input=None, external_data=None, llm_response=None, utility=None, feedback=None):
        if user_input:
            self.user_query = user_input
            self.dialog_history.append(("user", user_input))
        if external_data:
            self.external_evidence.append(external_data)
        if llm_response:
            self.llm_candidate_responses.append(llm_response)
        if utility is not None:
            self.utility_scores.append(utility)
        if feedback:
            self.verbalized_feedback = feedback

    def get_context(self):
        context = {
            "user_query": self.user_query,
            "external_evidence": self.external_evidence,
            "dialog_history": self.dialog_history
        }
        return context

class PromptEngine:
    def generate_prompt(self, working_memory_context):
        user_query = working_memory_context["user_query"]
        external_evidence = working_memory_context["external_evidence"]
        dialog_history = working_memory_context["dialog_history"]

        prompt_parts = []
        if dialog_history:
            prompt_parts.append("### Conversation History:")
            for speaker, text in dialog_history:
                prompt_parts.append(f"{speaker.capitalize()}: {text}")

        if external_evidence:
            prompt_parts.append("\n### Relevant Information:")
            for evidence in external_evidence:
                prompt_parts.append(f"- {evidence}")

        prompt_parts.append(f"\n### User Query: {user_query}")
        prompt_parts.append("\nBased on the above, please provide a helpful response.")

        return "\n".join(prompt_parts)

class SimulatedLLM:
    def generate_response(self, prompt):
        if "network issues" in prompt.lower() or "wifi not working" in prompt.lower():
            return "It sounds like you're having trouble with your network. Have you tried restarting your router and modem?"
        elif "restarted router" in prompt.lower():
            return "Okay, if restarting didn't work, let's check your connection settings. Are you connected via Wi-Fi or Ethernet?"
        elif "ethernet" in prompt.lower():
            return "Please ensure your Ethernet cable is securely plugged into both your computer and the router. Also, check if the network adapter drivers are up to date."
        elif "wifi" in prompt.lower():
            return "Could you please check if your Wi-Fi is enabled on your device and if you are connected to the correct network? What error messages are you seeing, if any?"
        else:
            return "I'm sorry, I'm having trouble understanding. Could you please rephrase your question or provide more details about your network issue?"

class PolicyModule:
    def decide_action(self, working_memory):
        user_query = working_memory.user_query.lower()
        llm_responses = working_memory.llm_candidate_responses

        if not llm_responses and user_query:
            return "generate_llm_response"
        elif "escalate" in user_query or "talk to human" in user_query:
            return "escalate_to_human"
        elif llm_responses and "restart" in llm_responses[-1].lower() and "tried that" in user_query:
            return "generate_llm_response"
        elif llm_responses and "connection settings" in llm_responses[-1].lower() and ("ethernet" in user_query or "wifi" in user_query):
            return "generate_llm_response"
        elif "thank you" in user_query or "problem solved" in user_query:
            return "end_conversation"
        else:
            return "generate_llm_response"

class CustomerSupportAgent:
    def __init__(self):
        self.working_memory = WorkingMemory()
        self.prompt_engine = PromptEngine()
        self.policy_module = PolicyModule()
        self.simulated_llm = SimulatedLLM()

    def process_user_input(self, user_input):
        self.working_memory.update_memory(user_input=user_input)
        print(f"User: {user_input}")

        action = self.policy_module.decide_action(self.working_memory)

        if action == "generate_llm_response":
            prompt = self.prompt_engine.generate_prompt(self.working_memory.get_context())
            llm_response = self.simulated_llm.generate_response(prompt)
            self.working_memory.update_memory(llm_response=llm_response)
            self.working_memory.dialog_history.append(("agent", llm_response))
            print(f"Agent: {llm_response}")
            return llm_response
        elif action == "escalate_to_human":
            response = "I understand this is frustrating. Let me connect you with a human agent who can provide more in-depth assistance."
            self.working_memory.dialog_history.append(("agent", response))
            print(f"Agent: {response}")
            return response
        elif action == "end_conversation":
            response = "Great! I'm glad I could help resolve your issue. Have a great day!"
            self.working_memory.dialog_history.append(("agent", response))
            print(f"Agent: {response}")
            return response
        else:
            response = "I'm not sure how to proceed. Could you please provide more information?"
            self.working_memory.dialog_history.append(("agent", response))
            print(f"Agent: {response}")
            return response

if __name__ == "__main__":
    agent = CustomerSupportAgent()

    print("Welcome to Network Troubleshooter Support. How can I help you today?")

    agent.process_user_input("My internet isn't working.")
    agent.process_user_input("I tried restarting the router already, it didn't help.")
    agent.process_user_input("I am using an ethernet connection.")
    agent.process_user_input("Okay, I will try that. Thank you!")
