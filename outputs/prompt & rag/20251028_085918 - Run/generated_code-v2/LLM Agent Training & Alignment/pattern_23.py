from typing import List

# 1. The "Constitution" Module
CONSTITUTIONAL_PRINCIPLES = [
    "Ensure all information provided is factually accurate.",
    "Avoid any biased or discriminatory language.",
    "Prioritize user safety and avoid generating harmful content.",
    "Maintain user privacy and do not ask for or reveal sensitive personal information.",
    "Be helpful and polite in all interactions."
]

# 2. Language Model (LLM) Core (Mock Implementation)
class MockLLM:
    def generate(self, prompt: str) -> str:
        # Simulate an LLM generating an initial response
        if "product details for 'wireless headphones'" in prompt.lower():
            return "The wireless headphones offer noise cancellation, 20-hour battery life, and are compatible with all Bluetooth devices. However, some users have reported occasional connectivity issues with older phone models. They are made by CompanyX, a known brand with some controversies in the past."
        elif "return policy" in prompt.lower():
            return "Our return policy states that items can be returned within 30 days if unopened. Used items are generally not eligible. Some customers found this unfair."
        elif "troubleshoot 'smartwatch'" in prompt.lower():
            return "If your smartwatch isn't connecting, try restarting it. If that doesn't work, it's likely a hardware issue and you should buy a new one."
        else:
            return f"I'm an initial LLM response for: {prompt}."

# 3. Critique and Revision Module (Constitutional AI Core - Mock Implementation)
class ConstitutionalAICore:
    def __init__(self, constitution: List[str]):
        self.constitution = constitution

    def critique_and_revise(self, initial_response: str, query: str) -> str:
        revised_response = initial_response

        # Simulate AI-driven critique and revision based on principles
        if "CompanyX, a known brand with some controversies in the past" in revised_response:
            revised_response = revised_response.replace("CompanyX, a known brand with some controversies in the past.", "CompanyX, a reputable brand in electronics.")
            revised_response = "Critique: Removed potentially biased negative sentiment about the brand. " + revised_response

        if "some users have reported occasional connectivity issues with older phone models" in revised_response:
            revised_response = revised_response.replace("However, some users have reported occasional connectivity issues with older phone models.", "For optimal performance, ensure your device's Bluetooth is up to date.")
            revised_response = "Critique: Reworded potential issue to be more constructive and helpful. " + revised_response

        if "Some customers found this unfair" in revised_response:
            revised_response = revised_response.replace("Some customers found this unfair.", "Please refer to our full return policy details on our website for specific conditions.")
            revised_response = "Critique: Removed unhelpful subjective opinion and guided user to official source. " + revised_response

        if "it's likely a hardware issue and you should buy a new one" in revised_response:
            revised_response = revised_response.replace("If that doesn't work, it's likely a hardware issue and you should buy a new one.", "If that doesn't resolve the issue, please contact our technical support for further assistance or to explore warranty options.")
            revised_response = "Critique: Provided a more helpful and less pushy solution than immediately recommending a new purchase. " + revised_response

        # Add a general adherence statement for demonstration
        if "Critique:" not in revised_response:
            revised_response = f"Critique: Response reviewed against constitutional principles. " + revised_response

        return revised_response

# Main Customer Support Assistant Function
def run_customer_support_assistant(query: str) -> str:
    llm_core = MockLLM()
    constitutional_ai_core = ConstitutionalAICore(CONSTITUTIONAL_PRINCIPLES)

    # 1. Generate initial response
    initial_response = llm_core.generate(query)

    # 2. Critique and revise the response based on the Constitution
    final_response = constitutional_ai_core.critique_and_revise(initial_response, query)

    return final_response

# Example Usage:
if __name__ == "__main__":
    print("\n--- User Query 1 ---")
    user_query_1 = "Tell me about the product details for 'wireless headphones'."
    print(f"User: {user_query_1}")
    response_1 = run_customer_support_assistant(user_query_1)
    print(f"Assistant: {response_1}")

    print("\n--- User Query 2 ---")
    user_query_2 = "What is your return policy for a used item?"
    print(f"User: {user_query_2}")
    response_2 = run_customer_support_assistant(user_query_2)
    print(f"Assistant: {response_2}")

    print("\n--- User Query 3 ---")
    user_query_3 = "My new smartwatch isn't connecting to my phone, how can I troubleshoot it?"
    print(f"User: {user_query_3}")
    response_3 = run_customer_support_assistant(user_query_3)
    print(f"Assistant: {response_3}")

    print("\n--- User Query 4 ---")
    user_query_4 = "General query."
    print(f"User: {user_query_4}")
    response_4 = run_customer_support_assistant(user_query_4)
    print(f"Assistant: {response_4}")