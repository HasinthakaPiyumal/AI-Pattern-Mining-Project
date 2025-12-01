from pydantic import BaseModel
import random

class ConfidenceEvaluation(BaseModel):
    is_correct: bool
    confidence_score: float
    reasoning: str

class InitialLLMResponseGenerator:
    def generate_response(self, query: str) -> str:
        if "shipping" in query.lower():
            return "Shipping usually takes between 3-5 business days for standard delivery. Expedited options are also available."
        elif "return policy" in query.lower():
            return "Our return policy allows returns within 30 days of purchase, provided the item is in its original condition with a receipt."
        elif "account access" in query.lower():
            return "I cannot directly assist with account access issues for security reasons. Please visit our 'Forgot Password' link or contact live support."
        elif "product warranty" in query.lower():
            return "Most products come with a one-year manufacturer's warranty. Specific details can be found on the product page or by contacting support."
        else:
            return "I am not sure I fully understand your request. Can you please rephrase or provide more details?"

class SelfCalibrationLLMEvaluator:
    def evaluate_response(self, query: str, response: str) -> ConfidenceEvaluation:
        if "not sure I fully understand" in response.lower() or "cannot directly assist" in response.lower():
            return ConfidenceEvaluation(
                is_correct=False,
                confidence_score=random.uniform(0.1, 0.4),
                reasoning="The initial response indicated uncertainty or inability to directly assist, suggesting low confidence."
            )
        elif "standard delivery" in response.lower() and "shipping" in query.lower():
             return ConfidenceEvaluation(
                is_correct=True,
                confidence_score=random.uniform(0.8, 0.95),
                reasoning="The response directly addresses shipping information with relevant details."
            )
        elif "30 days of purchase" in response.lower() and "return policy" in query.lower():
             return ConfidenceEvaluation(
                is_correct=True,
                confidence_score=random.uniform(0.85, 0.98),
                reasoning="The response accurately describes the return policy within the typical timeframe."
            )
        else:
            return ConfidenceEvaluation(
                is_correct=True,
                confidence_score=random.uniform(0.5, 0.75),
                reasoning="The response seems generally correct but lacks specific keywords for high confidence, or is a default good response."
            )

class DecisionLogicModule:
    def decide_action(self, evaluation: ConfidenceEvaluation) -> str:
        if evaluation.confidence_score > 0.75 and evaluation.is_correct:
            return "Send Directly"
        elif evaluation.confidence_score < 0.4 or not evaluation.is_correct:
            return "Regenerate"
        else:
            return "Flag for Human Review"

def main_customer_support_system(customer_query: str):
    initial_llm = InitialLLMResponseGenerator()
    evaluator_llm = SelfCalibrationLLMEvaluator()
    decision_logic = DecisionLogicModule()

    print(f"Customer Query: {customer_query}")

    initial_response = initial_llm.generate_response(customer_query)
    print(f"Initial LLM Response: {initial_response}")

    evaluation = evaluator_llm.evaluate_response(customer_query, initial_response)
    print(f"Self-Calibration Evaluation: Correctness={evaluation.is_correct}, Confidence={evaluation.confidence_score:.2f}, Reasoning='{evaluation.reasoning}'")

    action = decision_logic.decide_action(evaluation)
    print(f"System Decision: {action}")

    if action == "Send Directly":
        print(f"Action Taken: Response sent to customer: '{initial_response}'")
    elif action == "Flag for Human Review":
        print(f"Action Taken: Response flagged for human review. Reason: '{evaluation.reasoning}'")
    elif action == "Regenerate":
        print(f"Action Taken: Attempting to regenerate response (simulated). Initial response was not confident/correct enough.")

if __name__ == "__main__":
    print("--- Test Case 1: Clear Shipping Query ---")
    main_customer_support_system("How long does shipping take?")
    print("\n--- Test Case 2: Ambiguous Query ---")
    main_customer_support_system("I have a problem.")
    print("\n--- Test Case 3: Account Access Query ---")
    main_customer_support_system("I can't log into my account, help me.")
    print("\n--- Test Case 4: Return Policy Query ---")
    main_customer_support_system("What is your return policy?")
    print("\n--- Test Case 5: Product Warranty Query ---")
    main_customer_support_system("Is there a warranty for this product?")
