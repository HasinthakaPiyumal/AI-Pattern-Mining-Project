
import os
from typing import Any, Dict

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel as LCBaseModel
from langchain_core.pydantic_v1 import Field as LCField
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import create_structured_output_runnable

# 1. chatbot_simulator.py
def simulate_chatbot_response(customer_query: str) -> str:
    customer_query = customer_query.lower()
    if "account" in customer_query and "balance" in customer_query:
        return "Your current account balance is $1,250.75. Is there anything else I can help you with रिगार्डिंग योर अकाउंट?"
    elif "reset password" in customer_query or "forgot password" in customer_query:
        return "To reset your password, please visit our website and click on the 'Forgot Password' link. A reset link will be sent to your registered email address."
    elif "shipping status" in customer_query or "order status" in customer_query:
        return "I can help with that! Please provide your order number and I'll check the shipping status for you."
    elif "technical issue" in customer_query:
        return "I understand you're experiencing a technical issue. Could you please describe the problem in more detail, and I'll connect you with a specialist if needed?"
    elif "complaint" in customer_query:
        return "I'm truly sorry to hear you're unhappy. Please tell me more about your complaint so I can assist you better."
    else:
        return "Thank you for contacting customer support. How can I assist you today?"

# 2. evaluation_criteria.py
class EvaluationCriteria(LCBaseModel):
    correctness: str = LCField(description="Is the information provided accurate and factually correct?")
    helpfulness: str = LCField(description="Does the response adequately address the customer's problem or question?")
    empathy: str = LCField(description="Does the response show understanding and empathy towards the customer's situation?")
    policy_adherence: str = LCField(description="Does the response comply with company policies and guidelines?")

# 3. llm_evaluator.py
class EvaluationResult(LCBaseModel):
    score_correctness: int = LCField(description="Score for correctness (1-5, 5 being best)", ge=1, le=5)
    explanation_correctness: str = LCField(description="Explanation for correctness score")
    score_helpfulness: int = LCField(description="Score for helpfulness (1-5, 5 being best)", ge=1, le=5)
    explanation_helpfulness: str = LCField(description="Explanation for helpfulness score")
    score_empathy: int = LCField(description="Score for empathy (1-5, 5 being best)", ge=1, le=5)
    explanation_empathy: str = LCField(description="Explanation for empathy score")
    score_policy_adherence: int = LCField(description="Score for policy adherence (1-5, 5 being best)", ge=1, le=5)
    explanation_policy_adherence: str = LCField(description="Explanation for policy adherence score")
    overall_recommendation: str = LCField(description="Overall recommendation for improving the chatbot response.")

class LLMEvaluator:
    def __init__(self, llm_model_name: str, temperature: float = 0.0):
        if "gpt" in llm_model_name.lower():
            self.llm = ChatOpenAI(model=llm_model_name, temperature=temperature)
        elif "gemini" in llm_model_name.lower():
            self.llm = ChatGoogleGenerativeAI(model=llm_model_name, temperature=temperature)
        else:
            raise ValueError(f"Unsupported LLM model: {llm_model_name}. Use 'gpt' or 'gemini' models.")

        self.parser = create_structured_output_runnable(EvaluationResult, self.llm)

        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are an expert customer support chatbot response quality evaluator. Evaluate the provided chatbot response based on the given criteria. Provide a score from 1 to 5 for each criterion (5 being the best), along with a detailed explanation. Also provide an overall recommendation for improvement. Output your evaluation in JSON format."),
            ("human", "Customer Query: {customer_query}\n\nChatbot Response: {chatbot_response}\n\nEvaluation Criteria:\n- Correctness: {correctness}\n- Helpfulness: {helpfulness}\n- Empathy: {empathy}\n- Policy Adherence: {policy_adherence}")
        ])

    def evaluate_response(self, customer_query: str, chatbot_response: str, criteria: EvaluationCriteria) -> Dict[str, Any]:
        try:
            evaluation_input = self.prompt_template.invoke({
                "customer_query": customer_query,
                "chatbot_response": chatbot_response,
                "correctness": criteria.correctness,
                "helpfulness": criteria.helpfulness,
                "empathy": criteria.empathy,
                "policy_adherence": criteria.policy_adherence,
            })
            result = self.parser.invoke(evaluation_input)
            return result.dict()
        except Exception as e:
            return {"error": str(e), "message": "Failed to get evaluation from LLM."}

# 4. main.py
def main():
    load_dotenv()

    # Choose your LLM model
    # For OpenAI models:
    # llm_model = "gpt-4o-mini" 
    # For Google Gemini models:
    llm_model = "gemini-1.5-flash"

    # Ensure API key is set for the chosen model
    if "gpt" in llm_model.lower() and not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables. Please set it.")
        return
    if "gemini" in llm_model.lower() and not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY not found in environment variables. Please set it.")
        return

    try:
        evaluator = LLMEvaluator(llm_model_name=llm_model)
    except ValueError as e:
        print(f"Initialization error: {e}")
        return

    sample_queries = [
        "What is my account balance?",
        "I forgot my password, how do I reset it?",
        "Where is my order #12345?",
        "I'm really upset with the service I received, I want to file a complaint.",
        "My internet is not working."
    ]

    criteria = EvaluationCriteria(
        correctness="Is the information provided accurate and factually correct? Does it make sense in the context of the query?",
        helpfulness="Does the response fully address the customer's problem or question? Is it actionable?",
        empathy="Does the response acknowledge the customer's feelings or situation? Is it polite and understanding?",
        policy_adherence="Does the response follow company guidelines regarding privacy, tone, and information disclosure? Does it avoid giving personal advice?"
    )

    print(f"\n--- Starting LLM-based Chatbot Response Evaluation using {llm_model} ---\n")

    for i, customer_query in enumerate(sample_queries):
        print(f"\n===== Scenario {i+1} =====")
        chatbot_response = simulate_chatbot_response(customer_query)

        print(f"Customer Query: {customer_query}")
        print(f"Chatbot Response: {chatbot_response}")

        evaluation_results = evaluator.evaluate_response(customer_query, chatbot_response, criteria)

        if "error" in evaluation_results:
            print(f"Evaluation Error: {evaluation_results['error']}")
            print(f"Message: {evaluation_results['message']}")
        else:
            print("\n--- Evaluation Results ---")
            for key, value in evaluation_results.items():
                print(f"{key.replace('_', ' ').capitalize()}: {value}")
        print("========================\n")

if __name__ == "__main__":
    main()
