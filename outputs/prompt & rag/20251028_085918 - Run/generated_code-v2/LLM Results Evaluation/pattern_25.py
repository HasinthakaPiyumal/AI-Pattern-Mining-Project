import gradio as gr
from langchain_core.prompts import ChatPromptTemplate

# Mock LLM for generating responses and agent evaluations
class MockLLM:
    def __init__(self, name="MockLLM"):
        self.name = name

    def generate_response(self, prompt):
        # Simulate LLM generating a customer support response
        if "customer query" in prompt.lower():
            return "Thank you for contacting us. I understand you're having an issue. Please provide more details so I can assist you further." \
                   if "issue" in prompt.lower() else \
                   "Hello! How can I help you today?"
        return f"[Mock LLM generated response based on: {prompt[:50]}...]"

    def evaluate_response(self, role, query, response):
        if role == "Customer Advocate":
            if "issue" in query.lower() and "further details" in response.lower():
                return "Satisfied. The agent acknowledged the issue and asked for necessary clarification, showing empathy."
            elif "hello" in response.lower():
                return "Neutral. The response is generic and lacks specific problem-solving direction."
            else:
                return "Needs improvement. The response could be more direct or empathetic."
        elif role == "Support Agent Manager":
            if len(response) > 50 and "further details" in response.lower():
                return "Good. The agent followed protocol by gathering more information and maintaining a polite tone."
            elif "hello" in response.lower():
                return "Acceptable. Standard opening, but could be more proactive."
            else:
                return "Fair. Response is brief, might lack efficiency in resolving the issue quickly."
        elif role == "Company Policy Expert":
            if "policy violation" in query.lower():
                return "Warning: Potential policy violation related to [specific policy]. Needs review."
            elif "refund" in query.lower() and "further details" in response.lower():
                return "Compliant. The agent is gathering information before promising any action, which aligns with policy."
            else:
                return "Compliant. No obvious policy violations detected."
        return f"[{role} Evaluation for query: '{query}' and response: '{response[:50]}...']"

mock_llm = MockLLM()

class Agent:
    def __init__(self, name, role_description, llm):
        self.name = name
        self.role_description = role_description
        self.llm = llm
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are a {role}. Your task is to evaluate a customer support response."),
            ("user", "Customer Query: {query}\nLLM Response: {response}\nBased on your role, provide a concise evaluation focusing on {focus_areas}.")
        ])

    def evaluate(self, customer_query, llm_response):
        # In a real scenario, this would call the LLM with the prompt template
        # For this mock, we use the simplified MockLLM evaluate_response
        focus_areas = "" # This would be derived from role_description
        if "Customer Advocate" in self.name:
            focus_areas = "empathy, clarity, and problem resolution"
        elif "Support Agent Manager" in self.name:
            focus_areas = "efficiency, adherence to best practices, and agent performance"
        elif "Company Policy Expert" in self.name:
            focus_areas = "compliance with company policies, legal requirements, and product knowledge"

        return self.llm.evaluate_response(self.name, customer_query, llm_response)

class DebateOrchestrator:
    def __init__(self, agents):
        self.agents = agents

    def conduct_evaluation_debate(self, customer_query, llm_response):
        evaluations = {}
        for agent in self.agents:
            evaluations[agent.name] = agent.evaluate(customer_query, llm_response)
        return evaluations

    def synthesize_report(self, evaluations):
        report = "Comprehensive LLM Response Evaluation Report:\n\n"
        for agent_name, evaluation in evaluations.items():
            report += f"- {agent_name}: {evaluation}\n"

        # Simple synthesis logic
        if any("Needs improvement" in e for e in evaluations.values()):
            report += "\nOverall Recommendation: Needs significant revision due to identified shortcomings."
        elif any("Neutral" in e for e in evaluations.values()) or any("Fair" in e for e in evaluations.values()):
            report += "\nOverall Recommendation: Acceptable, but could benefit from refinement."
        elif any("Warning" in e for e in evaluations.values()):
            report += "\nOverall Recommendation: Critical review required due to potential policy issues."
        else:
            report += "\nOverall Recommendation: Good to excellent, well-rounded response."
        return report

# Initialize agents
customer_advocate = Agent(
    "Customer Advocate",
    "Evaluates the response from the perspective of a frustrated or satisfied customer, focusing on empathy, clarity, and problem resolution.",
    mock_llm
)
support_manager = Agent(
    "Support Agent Manager",
    "Assesses the response from a managerial perspective, focusing on efficiency, adherence to best practices, and agent performance.",
    mock_llm
)
policy_expert = Agent(
    "Company Policy Expert",
    "Verifies if the response complies with company policies, legal requirements, and product knowledge.",
    mock_llm
)

# Initialize orchestrator
orchestrator = DebateOrchestrator([customer_advocate, support_manager, policy_expert])

def evaluate_customer_support_response(customer_query):
    # 1. LLM-generated Customer Support Response Module
    llm_response = mock_llm.generate_response(customer_query)

    # 2 & 3. Multi-Agent Debate Framework & Debate Orchestration
    evaluations = orchestrator.conduct_evaluation_debate(customer_query, llm_response)

    # 4. Evaluation Synthesis and Reporting
    evaluation_report = orchestrator.synthesize_report(evaluations)

    # 5. Data Storage (simple in-memory for this demo)
    # In a real app, you'd store customer_query, llm_response, evaluations, and report in a database.
    return llm_response, evaluation_report

# Gradio Interface
if __name__ == "__main__":
    interface = gr.Interface(
        fn=evaluate_customer_support_response,
        inputs=gr.Textbox(lines=2, label="Customer Query", placeholder="e.g., I have an issue with my recent order, it never arrived."),
        outputs=[
            gr.Textbox(label="LLM Generated Response", lines=5),
            gr.Textbox(label="ChatEval Framework Evaluation Report", lines=15)
        ],
        title="AI Customer Support Response Evaluator (ChatEval Framework)",
        description="Enter a customer query to see how an LLM-generated response is evaluated by multiple AI agents in a simulated debate."
    )
    interface.launch()