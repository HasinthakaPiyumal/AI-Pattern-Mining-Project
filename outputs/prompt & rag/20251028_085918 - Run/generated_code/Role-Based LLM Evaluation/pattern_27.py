
import os
from typing import Dict, Any, List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class PersonaEvaluator:
    """A class to represent an LLM agent with a specific persona for evaluation."""

    def __init__(self, name: str, description: str, evaluation_criteria: str, llm: ChatOpenAI):
        self.name = name
        self.description = description
        self.llm = llm
        self.evaluation_criteria = evaluation_criteria

        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", f"You are a {self.name}. {self.description}. Your task is to evaluate a customer support chatbot's response based on the following criteria: {self.evaluation_criteria}\n\nProvide a concise evaluation, a score from 1 to 5 (1 being very poor, 5 being excellent), and detailed feedback explaining your score. Your output should be in the format:\nScore: [1-5]\nFeedback: [Your detailed feedback]"),
                ("human", "Chatbot Response: {chatbot_response}\nCustomer Query: {customer_query}"),
            ]
        )
        self.chain = self.prompt_template | self.llm | StrOutputParser()

    def evaluate(self, chatbot_response: str, customer_query: str) -> Dict[str, Any]:
        """Evaluates a chatbot's response based on the persona's criteria."""
        print(f"\n--- {self.name} is evaluating ---")
        raw_evaluation = self.chain.invoke({"chatbot_response": chatbot_response, "customer_query": customer_query})
        print(f"Raw evaluation from {self.name}:\n{raw_evaluation}")

        score = 0
        feedback = ""
        try:
            lines = raw_evaluation.split('\n')
            for line in lines:
                if line.startswith("Score:"):
                    score = int(line.split(":")[1].strip())
                elif line.startswith("Feedback:"):
                    feedback = line.split(":", 1)[1].strip()
            if not (1 <= score <= 5):
                score = 0 # Mark as invalid if outside range
        except Exception as e:
            print(f"Error parsing evaluation from {self.name}: {e}")
            feedback = f"Parsing error: {raw_evaluation}"

        return {
            "persona": self.name,
            "score": score,
            "feedback": feedback
        }

class CustomerSupportChatbot:
    """A simulated customer support chatbot for demonstration."""
    def get_response(self, query: str) -> str:
        # In a real system, this would call an actual chatbot API
        if "reset password" in query.lower():
            return "To reset your password, please visit our website's 'Forgot Password' link and follow the instructions. A temporary password will be sent to your registered email address."
        elif "billing issue" in query.lower():
            return "We apologize for the billing issue. Please provide your account number and a brief description of the problem, and our billing department will review it within 24-48 hours."
        elif "product warranty" in query.lower():
            return "Our standard product warranty covers manufacturing defects for one year from the purchase date. Please refer to your product manual for specific terms and conditions."
        else:
            return "I am sorry, I need more information to assist you. Can you please elaborate on your request?"

def aggregate_evaluations(evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregates scores and feedback from all evaluators."""
    total_score = 0
    valid_scores = 0
    all_feedback = []

    for eval_data in evaluations:
        if 1 <= eval_data["score"] <= 5:
            total_score += eval_data["score"]
            valid_scores += 1
        all_feedback.append(f"**{eval_data['persona']} (Score: {eval_data['score']}/5):** {eval_data['feedback']}")

    average_score = total_score / valid_scores if valid_scores > 0 else 0

    return {
        "average_score": f"{average_score:.2f}",
        "detailed_feedback": "\n\n".join(all_feedback)
    }

if __name__ == "__main__":
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set. Please set it to your OpenAI API key.")

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7, api_key=OPENAI_API_KEY)

    # Initialize persona evaluators
    evaluators = [
        PersonaEvaluator(
            name="Frustrated Customer",
            description="You are a customer who is frustrated and looking for a quick, empathetic, and clear resolution.",
            evaluation_criteria="Empathy, clarity of solution, speed of resolution, addressing emotional tone.",
            llm=llm
        ),
        PersonaEvaluator(
            name="Technical Expert",
            description="You are a subject matter expert, highly knowledgeable about product specifics and technical details.",
            evaluation_criteria="Accuracy of information, technical correctness, completeness of solution, potential for follow-up issues.",
            llm=llm
        ),
        PersonaEvaluator(
            name="Business Policy Officer",
            description="You ensure all responses adhere strictly to company policies, legal guidelines, and ethical standards.",
            evaluation_criteria="Adherence to company policy, legal compliance, ethical considerations, risk assessment.",
            llm=llm
        ),
        PersonaEvaluator(
            name="Efficiency Analyst",
            description="You focus on the efficiency and clarity of communication, ensuring responses are concise, easy to understand, and actionable.",
            evaluation_criteria="Conciseness, clarity, readability, actionability, avoidance of jargon.",
            llm=llm
        ),
    ]

    # Initialize simulated chatbot
    chatbot = CustomerSupportChatbot()

    # --- Scenario 1: Password Reset ---
    print("\n===== EVALUATING SCENARIO 1: PASSWORD RESET =====")
    customer_query_1 = "I forgot my password and can't log in. I need help urgently!"
    chatbot_response_1 = chatbot.get_response(customer_query_1)
    print(f"\nCustomer Query: {customer_query_1}")
    print(f"Chatbot Response: {chatbot_response_1}")

    scenario_1_evals = []
    for evaluator in evaluators:
        scenario_1_evals.append(evaluator.evaluate(chatbot_response_1, customer_query_1))

    aggregated_results_1 = aggregate_evaluations(scenario_1_evals)
    print("\n--- Aggregated Evaluation Results for Scenario 1 ---")
    print(f"Average Score: {aggregated_results_1['average_score']}")
    print("Detailed Feedback:\n" + aggregated_results_1['detailed_feedback'])

    # --- Scenario 2: Billing Issue ---
    print("\n===== EVALUATING SCENARIO 2: BILLING ISSUE =====")
    customer_query_2 = "My latest bill is completely wrong! I was overcharged for something I didn't even use."
    chatbot_response_2 = chatbot.get_response(customer_query_2)
    print(f"\nCustomer Query: {customer_query_2}")
    print(f"Chatbot Response: {chatbot_response_2}")

    scenario_2_evals = []
    for evaluator in evaluators:
        scenario_2_evals.append(evaluator.evaluate(chatbot_response_2, customer_query_2))

    aggregated_results_2 = aggregate_evaluations(scenario_2_evals)
    print("\n--- Aggregated Evaluation Results for Scenario 2 ---")
    print(f"Average Score: {aggregated_results_2['average_score']}")
    print("Detailed Feedback:\n" + aggregated_results_2['detailed_feedback'])
