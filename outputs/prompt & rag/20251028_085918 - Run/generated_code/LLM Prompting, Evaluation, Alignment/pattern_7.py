import json
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.messages import SystemMessage # Used conceptually for role-based prompting

class MockLLM:
    """A mock Large Language Model to simulate response generation and evaluation."""
    def invoke(self, prompt: str) -> str:
        prompt_lower = prompt.lower()

        if "evaluate the following customer support response" in prompt_lower:
            # Simulate LLM-based Autorating by parsing the prompt and returning structured JSON
            query_start = prompt_lower.find("customer query:") + len("customer query:")
            query_end = prompt_lower.find("\nagent response:")
            customer_query = prompt[query_start:query_end].strip()

            response_start = prompt_lower.find("agent response:") + len("agent response:")
            response_end = prompt_lower.find("\n\nrate the response")
            agent_response = prompt[response_start:response_end].strip()

            # Simple heuristic for mock evaluation based on keywords
            accuracy = 4
            relevance = 4
            helpfulness = 4
            reasoning = f"Response appears to address the query. General quality assessment based on mock logic."
            biases = False
            inconsistencies = False

            if "new product line" in customer_query.lower() and "thank you for your query" in agent_response.lower():
                relevance = 2
                helpfulness = 2
                reasoning = "The response was too generic and did not provide specific information about the new product line."
            elif "troubleshoot" in customer_query.lower() and "power supply" in agent_response.lower():
                accuracy = 5
                relevance = 5
                helpfulness = 5
                reasoning = "Provided a highly relevant and accurate troubleshooting step."
            elif "reset password" in customer_query.lower() and "login page" in agent_response.lower():
                accuracy = 5
                relevance = 5
                helpfulness = 5
                reasoning = "Gave clear and accurate instructions for password reset."

            eval_output = {
                "accuracy_rating": accuracy,
                "relevance_rating": relevance,
                "helpfulness_rating": helpfulness,
                "reasoning": reasoning,
                "biases_found": biases,
                "inconsistencies_found": inconsistencies
            }
            return json.dumps(eval_output)

        # Simulate content generation for various query types
        elif "troubleshooting" in prompt_lower or "expert troubleshooter" in prompt_lower:
            return "Please check the power supply and network connection. If the issue persists, try restarting the device. For further diagnosis, could you provide any error codes?"
        elif "account balance" in prompt_lower:
            return "Your current account balance is $1,234.56. Would you like to see your recent transactions? If so, please confirm your identity."
        elif "return policy" in prompt_lower:
            return "Our return policy allows returns within 30 days of purchase with a valid receipt. Items must be in their original condition and packaging. Customized items may have different policies."
        elif "reset password" in prompt_lower or "forgot password" in prompt_lower:
             return "To reset your password, navigate to the login page and click 'Forgot Password'. You will receive an email with instructions to set a new password."
        else:
            # Default generic response
            return f"Thank you for your query. I'm processing your request regarding: '{prompt_lower}'. How else can I assist you?"

class StructuredResponse(BaseModel):
    """Pydantic model for structured output of agent responses."""
    category: str = Field(description="The category of the customer's query.")
    solution: str = Field(description="The proposed solution or answer to the query.")
    confidence: float = Field(description="A confidence score for the solution (0.0 to 1.0).")

class IntelligentCustomerSupportAgent:
    """An Intelligent Customer Support Agent leveraging Generative AI and evaluation frameworks."""
    def __init__(self, llm_model: MockLLM, evaluator_llm: MockLLM):
        self.llm = llm_model
        self.evaluator_llm = evaluator_llm

        # 1. Prompt Engineering Layer (LangChain-inspired structures)
        self.standard_template = PromptTemplate(
            input_variables=["query"],
            template="Customer query: {query}\nAgent response:"
        )

        # Few-shot examples for specific scenarios
        self.few_shot_examples = [
            {"query": "My internet is not working.", "answer": "Please check if your modem and router are powered on and connected correctly."},
            {"query": "How do I reset my password?", "answer": "You can reset your password by clicking on 'Forgot Password' on the login page and following the instructions."}
        ]
        self.few_shot_template = FewShotPromptTemplate(
            examples=self.few_shot_examples,
            example_prompt=PromptTemplate(
                input_variables=["query", "answer"],
                template="Query: {query}\nResponse: {answer}"
            ),
            prefix="Here are some examples of how to respond to customer queries:\n\n",
            suffix="\nNow, please respond to the following query:\nQuery: {query}\nResponse:",
            input_variables=["query"]
        )

    def _dynamic_prompt_selector(self, query: str) -> str:
        """Dynamically selects the appropriate prompt strategy based on the query."""
        query_lower = query.lower()
        if "reset password" in query_lower or "login issue" in query_lower or "forgot password" in query_lower:
            return "few_shot"
        elif "troubleshoot" in query_lower or "problem with" in query_lower or "not working" in query_lower:
            return "role_based_troubleshooter"
        elif "balance" in query_lower or "account" in query_lower or "policy" in query_lower:
            return "standard"
        else:
            return "standard"

    def _generate_response(self, query: str, prompt_strategy: str) -> str:
        """Generates a response using the LLM based on the selected prompt strategy."""
        if prompt_strategy == "standard":
            prompt = self.standard_template.format(query=query)
        elif prompt_strategy == "few_shot":
            prompt = self.few_shot_template.format(query=query)
        elif prompt_strategy == "role_based_troubleshooter":
            # For a real LLM, this would involve SystemMessage in a ChatPromptTemplate
            # For MockLLM, we inject the role directly into the prompt string
            prompt = f"As an expert troubleshooter, respond to this customer query: {query}"
        else:
            prompt = self.standard_template.format(query=query) # Fallback to standard

        raw_response = self.llm.invoke(prompt)
        return self._post_process_response(raw_response)

    def _post_process_response(self, raw_response: str) -> str:
        """Applies basic post-processing to the LLM's raw output."""
        processed_response = raw_response.strip()
        # Ensure the response starts with a capital letter and ends with punctuation
        if processed_response and processed_response[0].islower():
            processed_response = processed_response[0].upper() + processed_response[1:]
        if processed_response and not processed_response.endswith(('.', '!', '?')):
            processed_response += '.'
        return processed_response

    def _evaluate_response(self, query: str, generated_response: str) -> dict:
        """Evaluates the generated response using an LLM-based autorating framework."""
        eval_prompt = PromptTemplate(
            input_variables=["query", "generated_response"],
            template=(
                "Please evaluate the following customer support response:\n"
                "Customer Query: {query}\n"
                "Agent Response: {generated_response}\n\n"
                "Rate the response on a scale of 1-5 for Accuracy, Relevance, and Helpfulness. "
                "Provide a short reasoning for each rating. Also, identify if there are any biases or factual inconsistencies." 
                "\nFormat your output as JSON: {{'accuracy_rating': int, 'relevance_rating': int, 'helpfulness_rating': int, 'reasoning': str, 'biases_found': bool, 'inconsistencies_found': bool}}"
            )
        )
        full_eval_prompt = eval_prompt.format(query=query, generated_response=generated_response)
        eval_output_raw = self.evaluator_llm.invoke(full_eval_prompt)

        try:
            evaluation_results = json.loads(eval_output_raw)
            # Ensure boolean types if they were parsed as strings
            evaluation_results['biases_found'] = bool(evaluation_results.get('biases_found'))
            evaluation_results['inconsistencies_found'] = bool(evaluation_results.get('inconsistencies_found'))
        except json.JSONDecodeError:
            # Fallback if the evaluator LLM doesn't return perfect JSON
            evaluation_results = {
                "accuracy_rating": 3,
                "relevance_rating": 3,
                "helpfulness_rating": 3,
                "reasoning": f"Could not parse evaluator LLM output. Raw output: {eval_output_raw[:100]}... Defaulting to average ratings.",
                "biases_found": False,
                "inconsistencies_found": False
            }
        return evaluation_results

    def handle_query(self, query: str) -> dict:
        """Processes a customer query, generates a response, and evaluates it."""
        prompt_strategy = self._dynamic_prompt_selector(query)
        generated_response = self._generate_response(query, prompt_strategy)

        # Simulate Pydantic structured output based on the generated response
        # In a real scenario, the LLM might be prompted to directly output JSON.
        try:
            # Try to parse as JSON if the mock LLM somehow returned it
            if "{" in generated_response and "}" in generated_response:
                json_part = generated_response[generated_response.find("{"):generated_response.rfind("}")+1]
                structured_data = json.loads(json_part)
                structured_output = StructuredResponse(**structured_data)
            else:
                # Default structured response for demonstration
                category = "General Support"
                if "password" in query.lower():
                    category = "Account Management"
                elif "troubleshoot" in query.lower() or "not working" in query.lower():
                    category = "Technical Support"
                structured_output = StructuredResponse(
                    category=category,
                    solution=generated_response,
                    confidence=0.85 # Placeholder confidence
                )
        except json.JSONDecodeError:
            # Fallback for structured output if parsing fails
            structured_output = StructuredResponse(
                category="General Support",
                solution=generated_response,
                confidence=0.80
            )

        evaluation = self._evaluate_response(query, generated_response)

        return {
            "query": query,
            "prompt_strategy_used": prompt_strategy,
            "generated_response": generated_response,
            "structured_output": structured_output.dict(),
            "evaluation_results": evaluation
        }

if __name__ == "__main__":
    # Initialize Mock LLMs for the main agent and the evaluator
    main_llm = MockLLM()
    evaluator_llm = MockLLM()

    # Create the Intelligent Customer Support Agent instance
    icsa = IntelligentCustomerSupportAgent(llm_model=main_llm, evaluator_llm=evaluator_llm)

    # Example customer queries to demonstrate the agent's capabilities
    queries = [
        "I need to know my account balance.",
        "My internet is not working, can you help troubleshoot?",
        "How do I reset my password?",
        "What is your return policy for electronics?",
        "Tell me about your new product line."
    ]

    print("\n--- Intelligent Customer Support Agent Simulation ---\n")
    for i, q in enumerate(queries):
        print(f"\n[{i+1}/{len(queries)}] Processing Query: {q}")
        result = icsa.handle_query(q)

        print(f"  Prompt Strategy Used: {result['prompt_strategy_used']}")
        print(f"  Agent Generated Response: {result['generated_response']}")
        print(f"  Structured Output: {json.dumps(result['structured_output'], indent=2)}")
        print(f"  Evaluation Results: {json.dumps(result['evaluation_results'], indent=2)}")
        print("--------------------------------------------------")
