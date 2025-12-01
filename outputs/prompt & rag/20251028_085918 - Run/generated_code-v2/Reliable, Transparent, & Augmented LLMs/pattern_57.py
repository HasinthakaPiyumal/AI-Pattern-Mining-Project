import random
from collections import Counter
import gradio as gr

class LLMWrapper:
    def __init__(self, model_name="mock_llm"):
        self.model_name = model_name

    def generate_response(self, prompt: str) -> str:
        if "billing inquiry" in prompt.lower():
            responses = [
                "Your billing statement can be found in your account settings.",
                "Please navigate to the billing section of your profile.",
                "I suggest checking the 'My Payments' tab for your statement."
            ]
            return random.choice(responses)
        elif "technical issue" in prompt.lower():
            responses = [
                "Have you tried restarting your device and router?",
                "A quick troubleshoot might involve clearing your browser cache.",
                "Please describe the technical issue in more detail."
            ]
            return random.choice(responses)
        elif "product refund" in prompt.lower():
            responses = [
                "Refunds are processed within 3-5 business days after approval.",
                "You can request a refund through your order history page.",
                "Our refund policy states items must be returned within 30 days."
            ]
            return random.choice(responses)
        else:
            responses = [
                "I'm not sure how to answer that. Could you rephrase?",
                "Please provide more context for your request.",
                "I can only assist with common customer support topics."
            ]
            return random.choice(responses)

class ExemplarManager:
    def __init__(self, exemplars: list[dict]):
        self.exemplars = exemplars

    def get_random_exemplar_subset(self, num_exemplars: int) -> list[dict]:
        if num_exemplars >= len(self.exemplars):
            return self.exemplars
        return random.sample(self.exemplars, num_exemplars)

class DenseChatbot:
    def __init__(self, llm_wrapper: LLMWrapper, exemplar_manager: ExemplarManager):
        self.llm_wrapper = llm_wrapper
        self.exemplar_manager = exemplar_manager

    def _construct_prompt(self, user_query: str, exemplars: list[dict]) -> str:
        prompt_parts = ["Instruction: Provide a helpful customer support response."]
        for ex in exemplars:
            prompt_parts.append(f"Customer: {ex['query']}\nAgent: {ex['response']}")
        prompt_parts.append(f"Customer: {user_query}\nAgent:")
        return "\n\n".join(prompt_parts)

    def _aggregate_responses(self, responses: list[str]) -> str:
        if not responses:
            return "No response generated."

        response_counts = Counter(responses)
        most_common_response, count = response_counts.most_common(1)[0]

        # If the most common response appears in more than half of the total responses
        if count > len(responses) / 2:
            return most_common_response
        else:
            # If no clear majority, provide a summary or list of distinct responses
            distinct_responses = sorted(list(set(responses)))
            if len(distinct_responses) == 1:
                return distinct_responses[0]
            else:
                return "Aggregated responses (multiple perspectives):\n" + "\n".join([f"- {res}" for res in distinct_responses])

    def answer_query(self, user_query: str, num_prompts: int = 3, exemplars_per_prompt: int = 2) -> str:
        individual_responses = []
        for _ in range(num_prompts):
            exemplar_subset = self.exemplar_manager.get_random_exemplar_subset(exemplars_per_prompt)
            prompt = self._construct_prompt(user_query, exemplar_subset)
            response = self.llm_wrapper.generate_response(prompt)
            individual_responses.append(response)
        
        final_answer = self._aggregate_responses(individual_responses)
        return final_answer

# --- Demonstration Setup ---

# 1. Sample Exemplars (Training Set)
exemplar_data = [
    {"query": "My bill seems too high this month.", "response": "Let me check your recent usage and subscription details for any discrepancies."},
    {"query": "How do I reset my password?", "response": "You can reset your password by clicking 'Forgot Password' on the login page."},
    {"query": "I can't log in to my account.", "response": "Please ensure you are using the correct username and password. If the issue persists, try resetting your password."},
    {"query": "My internet is not working.", "response": "Have you tried restarting your router? Often, a simple restart can resolve connectivity issues."},
    {"query": "I want to know about my refund status.", "response": "Refunds are typically processed within 3-5 business days. Could you provide your order number?"},
    {"query": "How do I upgrade my service plan?", "response": "You can upgrade your service plan through your account dashboard under 'Manage Subscription'."},
    {"query": "I have a question about my last payment.", "response": "I can look into your payment history. What is your account number?"}
]

# 2. Initialize Components
llm = LLMWrapper()
exemplar_manager = ExemplarManager(exemplar_data)
dense_chatbot = DenseChatbot(llm, exemplar_manager)

# 3. Gradio Interface
def chatbot_interface(query):
    return dense_chatbot.answer_query(query, num_prompts=5, exemplars_per_prompt=3)

iface = gr.Interface(
    fn=chatbot_interface,
    inputs=gr.Textbox(lines=2, placeholder="Enter your customer support query here..."),
    outputs="text",
    title="DENSE Chatbot Demo",
    description="A customer support chatbot using Demonstration Ensembling (DENSE) for more robust responses."
)

if __name__ == "__main__":
    iface.launch()