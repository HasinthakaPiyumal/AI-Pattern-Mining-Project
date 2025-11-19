def mock_llm(prompt: str) -> str:
    if "query as a Frustrated User" in prompt:
        return "I've been waiting forever! Why isn't my order here yet?! Order ID: XYZ789"
    elif "query as a Detailed Inquirer" in prompt:
        return "Could you please provide a step-by-step guide on how to reset my password, including any security verification methods?"
    elif "query as a Newbie User" in prompt:
        return "How do I even start using this app? I'm totally lost."
    elif "query as a Technical User" in prompt:
        return "I'm experiencing an API authentication failure with error code 401 on endpoint /api/v1/users. Can you confirm current rate limits and any recent changes to OAuth scopes?"
    elif "critique the response as an Efficient Problem Solver" in prompt:
        return "The chatbot efficiently addressed the core issue, providing a clear solution without unnecessary pleasantries. Score: 9/10"
    elif "critique the response as an Empathetic Listener" in prompt:
        return "The chatbot's response lacked empathy and failed to acknowledge the user's frustration. It sounded robotic. Score: 4/10"
    elif "critique the response as a Policy Adherence Checker" in prompt:
        return "The chatbot's response was compliant with company policy regarding returns. It clearly stated the next steps. Score: 8/10"
    else:
        return f"LLM response to: {prompt[:50]}..."

PERSONAS = {
    "Frustrated User": {
        "description": "A user who is annoyed and impatient, expecting quick resolution.",
        "role": "customer"
    },
    "Detailed Inquirer": {
        "description": "A user who seeks comprehensive, step-by-step instructions and all relevant details.",
        "role": "customer"
    },
    "Newbie User": {
        "description": "A user completely new to the system, needing basic guidance and reassurance.",
        "role": "customer"
    },
    "Technical User": {
        "description": "A user with technical background, expecting precise, technical information and solutions.",
        "role": "customer"
    },
    "Efficient Problem Solver": {
        "description": "Evaluates responses based on clarity, directness, and effectiveness in solving the problem.",
        "role": "evaluator"
    },
    "Empathetic Listener": {
        "description": "Evaluates responses based on their ability to show understanding, empathy, and positive tone.",
        "role": "evaluator"
    },
    "Policy Adherence Checker": {
        "description": "Evaluates responses for compliance with predefined company policies and guidelines.",
        "role": "evaluator"
    }
}

class CustomerSupportChatbot:
    def get_response(self, query: str) -> str:
        query_lower = query.lower()
        if "order id: xyz789" in query_lower and "where is my order" in query_lower:
            return "I understand your frustration regarding order XYZ789. Let me check the status for you. It appears your order is currently in transit and expected to arrive within 1-2 business days."
        elif "reset password" in query_lower:
            return "To reset your password, please visit our website, click 'Forgot Password', and follow the instructions sent to your registered email address."
        elif "start using this app" in query_lower:
            return "Welcome! To get started with our app, you can explore the 'Dashboard' for an overview or visit the 'Help' section for tutorials."
        elif "api authentication failure" in query_lower or "error code 401" in query_lower:
            return "For API authentication failures (401), please ensure your API key is correctly configured and not expired. Refer to our API documentation on authorization for more details. Current rate limits are 1000 requests/minute."
        else:
            return "I'm sorry, I don't have enough information to answer that. Could you please rephrase or provide more details?"

class PersonaAgent:
    def __init__(self, name: str, description: str, role: str):
        self.name = name
        self.description = description
        self.role = role

    def generate_query(self, scenario: str) -> str:
        if self.role != "customer":
            raise ValueError("Only customer personas can generate queries.")
        prompt = f"Given the scenario: '{scenario}', generate a customer query as a {self.name} whose description is: {self.description}."
        return mock_llm(prompt)

    def evaluate_response(self, customer_query: str, chatbot_response: str) -> str:
        if self.role != "evaluator":
            raise ValueError("Only evaluator personas can evaluate responses.")
        prompt = (
            f"As an {self.name} ({self.description}), critique the following chatbot response.\n"
            f"Customer Query: '{customer_query}'\n"
            f"Chatbot Response: '{chatbot_response}'\n"
            "Provide a qualitative assessment and a score (out of 10) based on your persona's criteria."
        )
        return mock_llm(prompt)

def run_evaluation_scenario(scenario: str, customer_persona_name: str, evaluator_persona_names: list[str]):
    print(f"--- Running Evaluation Scenario ---")
    print(f"Scenario: {scenario}")
    print(f"Customer Persona: {customer_persona_name}")
    print(f"Evaluator Personas: {', '.join(evaluator_persona_names)}\n")

    chatbot = CustomerSupportChatbot()

    customer_persona_data = PERSONAS.get(customer_persona_name)
    if not customer_persona_data or customer_persona_data["role"] != "customer":
        print(f"Error: '{customer_persona_name}' is not a valid customer persona.")
        return
    customer_agent = PersonaAgent(
        customer_persona_name,
        customer_persona_data["description"],
        customer_persona_data["role"]
    )

    customer_query = customer_agent.generate_query(scenario)
    print(f"Customer ({customer_agent.name}) Query: {customer_query}")

    chatbot_response = chatbot.get_response(customer_query)
    print(f"Chatbot Response: {chatbot_response}\n")

    for eval_persona_name in evaluator_persona_names:
        eval_persona_data = PERSONAS.get(eval_persona_name)
        if not eval_persona_data or eval_persona_data["role"] != "evaluator":
            print(f"Error: '{eval_persona_name}' is not a valid evaluator persona. Skipping.\n")
            continue

        evaluator_agent = PersonaAgent(
            eval_persona_name,
            eval_persona_data["description"],
            eval_persona_data["role"]
        )
        evaluation_feedback = evaluator_agent.evaluate_response(customer_query, chatbot_response)
        print(f"Evaluation by {evaluator_agent.name} ({evaluator_agent.description}):\n{evaluation_feedback}\n")

if __name__ == "__main__":
    # Example Usage 1: Frustrated User with multiple evaluators
    run_evaluation_scenario(
        scenario="User's online order (ID: XYZ789) is significantly delayed, and they are upset.",
        customer_persona_name="Frustrated User",
        evaluator_persona_names=["Efficient Problem Solver", "Empathetic Listener", "Policy Adherence Checker"]
    )

    print("\n" + "="*80 + "\n")

    # Example Usage 2: Technical User with a specific evaluator
    run_evaluation_scenario(
        scenario="User is reporting an API error during integration.",
        customer_persona_name="Technical User",
        evaluator_persona_names=["Efficient Problem Solver"]
    )

    print("\n" + "="*80 + "\n")

    # Example Usage 3: Newbie User with a specific evaluator
    run_evaluation_scenario(
        scenario="User is new to the application and doesn't know where to start.",
        customer_persona_name="Newbie User",
        evaluator_persona_names=["Empathetic Listener"]
    )
