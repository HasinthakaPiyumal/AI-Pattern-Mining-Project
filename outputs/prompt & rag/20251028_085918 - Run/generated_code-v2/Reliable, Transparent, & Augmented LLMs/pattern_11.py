import os
import json
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage

class LLMGuidelineGenerator:
    def __init__(self, model_name="gpt-4o", temperature=0.7):
        try:
            self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        except Exception as e:
            print(f"Error initializing LLM: {e}. Make sure OPENAI_API_KEY is set.")
            self.llm = None

    def generate_guidelines(self, expert_annotations: list = None) -> str:
        if not self.llm:
            return "LLM not initialized. Cannot generate guidelines."

        system_template = (
            "You are an expert evaluator assistant for AI customer support agents. "
            "Your task is to generate clear, comprehensive, and step-by-step evaluation guidelines "
            "for assessing the quality and effectiveness of AI customer support responses. "
            "The guidelines should cover aspects such as accuracy, helpfulness, empathy, conciseness, grammar, and adherence to company policies. "
            "Provide specific criteria and a suggested scoring mechanism (e.g., 1-5 scale with descriptions for each level)."
        )

        user_prompt_content = (
            "Please generate detailed evaluation guidelines for AI customer support agent responses. "
            "Focus on criteria that ensure the AI provides excellent customer service. "
            "For each criterion, provide a description and a suggested scoring rubric (e.g., 1-5 scale). "
            "Ensure the guidelines are actionable and easy to follow for an automated evaluation system or human reviewer."
        )

        if expert_annotations:
            user_prompt_content += (
                "\n\nConsider the following examples of expert-annotated customer interactions when formulating your guidelines:"
            )
            for i, annotation in enumerate(expert_annotations):
                user_prompt_content += f"\n\nExample {i+1}:\nCustomer Query: {annotation["query"]}\nAI Agent Response: {annotation["response"]}\nExpert Feedback: {annotation["feedback"]}"

        messages = [
            SystemMessage(content=system_template),
            HumanMessage(content=user_prompt_content),
        ]

        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            return f"Error during guideline generation: {e}"


class EvaluationEngine:
    def __init__(self, model_name="gpt-4o", temperature=0.2):
        try:
            self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        except Exception as e:
            print(f"Error initializing LLM: {e}. Make sure OPENAI_API_KEY is set.")
            self.llm = None

    def evaluate_response(self, customer_query: str, agent_response: str, generated_guidelines: str) -> dict:
        if not self.llm:
            return {"error": "LLM not initialized. Cannot perform evaluation.", "overall_score": None, "justification": None}

        system_template = (
            "You are an impartial and meticulous AI evaluator. Your task is to evaluate an AI customer support agent's response "
            "based on the provided comprehensive evaluation guidelines. "
            "Adhere strictly to the guidelines and provide an overall score (e.g., 1-5) and a detailed justification for your evaluation. "
            "Your output should be a JSON object with 'overall_score' (integer) and 'justification' (string) fields."
        )

        user_prompt_content = (
            f"Customer Query: {customer_query}\n\n"
            f"AI Agent Response: {agent_response}\n\n"
            f"--- Evaluation Guidelines ---\n{generated_guidelines}\n\n"
            "Please evaluate the AI Agent's response using the guidelines above. "
            "Provide a single overall score (e.g., 1-5, where 5 is excellent) and a clear justification "
            "explaining how the response meets or fails the criteria specified in the guidelines. "
            "Output your response as a JSON object with 'overall_score' and 'justification' keys."
        )

        messages = [
            SystemMessage(content=system_template),
            HumanMessage(content=user_prompt_content),
        ]

        try:
            response = self.llm.invoke(messages)
            try:
                evaluation_result = json.loads(response.content)
                if "overall_score" not in evaluation_result or "justification" not in evaluation_result:
                    raise ValueError("JSON response missing 'overall_score' or 'justification' keys.")
                return evaluation_result
            except json.JSONDecodeError:
                print(f"Warning: LLM did not return a valid JSON. Raw response: {response.content}")
                return {"overall_score": None, "justification": response.content, "error": "Invalid JSON from LLM"}
        except Exception as e:
            return {"error": f"Error during evaluation: {e}", "overall_score": None, "justification": None}


if __name__ == '__main__':
    # Ensure OPENAI_API_KEY is set as an environment variable
    if "OPENAI_API_KEY" not in os.environ:
        print("Error: OPENAI_API_KEY environment variable is not set. Please set it to run the example.")
    else:
        print("--- Starting AI Customer Support Agent Evaluator Example ---")

        # 1. Initialize Guideline Generator
        guideline_generator = LLMGuidelineGenerator()

        # Dummy expert annotations for guideline generation
        dummy_expert_annotations = [
            {
                "query": "My order #12345 hasn't shipped yet. Can you help?",
                "response": "I apologize for the delay. Order #12345 is currently being processed and is expected to ship within 2 business days. You will receive a tracking number via email once it ships.",
                "feedback": "Excellent clarity and provided a realistic timeframe. Empathetic tone."
            },
            {
                "query": "How do I reset my password?",
                "response": "To reset your password, go to our website, click 'Login', then 'Forgot Password', and follow the instructions. A reset link will be sent to your registered email.",
                "feedback": "Accurate and concise steps. Easy to follow."
            }
        ]

        print("\n--- Generating evaluation guidelines with expert annotations ---")
        generated_guidelines = guideline_generator.generate_guidelines(expert_annotations=dummy_expert_annotations)
        print(generated_guidelines)

        # 2. Initialize Evaluation Engine
        evaluation_engine = EvaluationEngine()

        # Example customer queries and AI agent responses
        customer_query_1 = "My internet is not working. What should I do?"
        agent_response_1 = "Please check if your router is plugged in and if all cables are securely connected. If the issue persists, try restarting your router. If it still doesn't work, contact our technical support at 1-800-XXX-XXXX."

        customer_query_2 = "I want to return an item I bought last week. It's a t-shirt."
        agent_response_2 = "Returns are accepted within 30 days of purchase with a receipt. The item must be unworn and unwashed. Please bring it to any of our store locations for a refund or exchange."

        customer_query_3 = "Can I get a discount for my next purchase?"
        agent_response_3 = "We do not offer discounts."

        print("\n--- Evaluating AI Agent Responses ---")

        print("\nEvaluating Response 1:")
        evaluation_1 = evaluation_engine.evaluate_response(customer_query_1, agent_response_1, generated_guidelines)
        print(f"Customer Query: {customer_query_1}")
        print(f"Agent Response: {agent_response_1}")
        print(f"Evaluation Result: {json.dumps(evaluation_1, indent=2)}")

        print("\nEvaluating Response 2:")
        evaluation_2 = evaluation_engine.evaluate_response(customer_query_2, agent_response_2, generated_guidelines)
        print(f"Customer Query: {customer_query_2}")
        print(f"Agent Response: {agent_response_2}")
        print(f"Evaluation Result: {json.dumps(evaluation_2, indent=2)}")

        print("\nEvaluating Response 3:")
        evaluation_3 = evaluation_engine.evaluate_response(customer_query_3, agent_response_3, generated_guidelines)
        print(f"Customer Query: {customer_query_3}")
        print(f"Agent Response: {agent_response_3}")
        print(f"Evaluation Result: {json.dumps(evaluation_3, indent=2)}")

        print("\n--- AI Customer Support Agent Evaluator Example Finished ---")