from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate

# Simulate a dummy LLM for demonstration
class DummyLLM:
    def __init__(self, responses: Optional[Dict[str, str]] = None):
        self.responses = responses if responses is not None else {}

    def invoke(self, prompt: str) -> str:
        # Simple keyword-based response simulation
        if "billing issue" in prompt.lower():
            return "I understand you have a billing issue. Can you please provide your account number for further assistance?"
        elif "product refund" in prompt.lower():
            return "For a product refund, please ensure you have the original receipt and the item is within the return window. What product are you referring to?"
        elif "technical support" in prompt.lower():
            return "I can help with technical support. Could you describe the problem you're encountering in more detail?"
        elif "reset password" in prompt.lower():
            return "To reset your password, please visit our website and click on 'Forgot Password'. Follow the instructions sent to your registered email."
        elif "hate speech" in prompt.lower() or "harmful content" in prompt.lower():
            return "I cannot generate responses that are harmful or promote hate speech. Please ask a different question."
        elif "ethically aligned" in prompt.lower() and "good service" in prompt.lower():
            return "Providing ethical and helpful service is my top priority. How can I assist you today?"
        return "I am a smart customer support agent. How can I assist you today?"


# 1. ethical_guidelines.py
class EthicalGuidelines:
    def apply_constitutional_ai(self, prompt_text: str) -> str:
        ethical_principles = "\n\nEthical Guidelines: Always be helpful, harmless, and honest. Avoid generating biased, discriminatory, or offensive content. Prioritize user safety and privacy. Ensure responses are factually accurate and respectful.\n"
        return f"""{prompt_text}""" + ethical_principles

    def apply_bias_mitigation(self, prompt_text: str) -> str:
        bias_mitigation_instructions = "\n\nBias Mitigation: Carefully consider the language used to avoid stereotypes or generalizations. If unsure, err on the side of neutrality and inclusivity.\n"
        return f"""{prompt_text}""" + bias_mitigation_instructions


# 2. prompt_manager.py
class PromptManager:
    def _generate_few_shot_examples(self, topic: str) -> List[Dict[str, str]]:
        if topic == "billing":
            return [
                {"input": "My bill is incorrect.", "output": "Please provide your account number and the billing period you are querying."},
                {"input": "I was charged twice.", "output": "Could you confirm the dates of the duplicate charges and your account details?"},
            ]
        return []

    def _apply_template_prompt(self, base_prompt: str, variables: Dict[str, Any]) -> str:
        template = PromptTemplate(template=base_prompt + " {additional_info}", input_variables=list(variables.keys()) + ["additional_info"])
        return template.format(**variables, additional_info="") # additional_info is dynamically added later

    def _apply_role_prompt(self, base_prompt: str, role: str) -> str:
        return f"You are a highly empathetic and efficient {role}. {base_prompt}"

    def _apply_style_prompt(self, base_prompt: str, style: str) -> str:
        return f"Respond in a {style} style. {base_prompt}"

    def _apply_emotion_prompt(self, base_prompt: str, emotion: str) -> str:
        return f"Express a {emotion} tone in your response. {base_prompt}"

    def create_prompt(
        self,
        user_query: str,
        context: str,
        prompt_strategy: Dict[str, Any]
    ) -> str:
        final_prompt = f"Context: {context}\nUser Query: {user_query}\n"

        if prompt_strategy.get("zero_shot", False):
            pass  # Zero-shot is the base case

        if prompt_strategy.get("few_shot", False):
            examples = self._generate_few_shot_examples(prompt_strategy.get("few_shot_topic", ""))
            example_str = "\nExamples:\n" + "\n".join([f"Input: {ex['input']}\nOutput: {ex['output']}" for ex in examples])
            final_prompt = f"{example_str}\n{final_prompt}"

        if "template_based" in prompt_strategy:
            final_prompt = self._apply_template_prompt(final_prompt, prompt_strategy["template_based"])

        if "role" in prompt_strategy:
            final_prompt = self._apply_role_prompt(final_prompt, prompt_strategy["role"])

        if "style" in prompt_strategy:
            final_prompt = self._apply_style_prompt(final_prompt, prompt_strategy["style"])

        if "emotion" in prompt_strategy:
            final_prompt = self._apply_emotion_prompt(final_prompt, prompt_strategy["emotion"])

        if prompt_strategy.get("rephrase_respond", False):
            final_prompt += "\nBefore answering, rephrase the user's query to confirm understanding."

        if prompt_strategy.get("rereading", False):
            final_prompt += "\nCarefully reread the context and query before formulating your response."

        if prompt_strategy.get("metacognitive", False):
            final_prompt += "\nThink step-by-step about the best way to address this query, considering all constraints and ethical guidelines."

        if prompt_strategy.get("prompt_chain", False):
            # Simulate a simple prompt chain by adding an instruction for multi-step thinking
            final_prompt += "\nBreak down the problem into smaller steps. First, identify the core issue. Second, propose a solution. Third, check for potential side effects."

        return final_prompt

    def build_reasoning_chain(self, llm: Any, initial_query: str) -> str:
        # This simulates a multi-step reasoning process using a dummy LLM
        step1_prompt = f"Given the query '{initial_query}', what is the primary problem?"
        step1_response = llm.invoke(step1_prompt)

        step2_prompt = f"Based on the primary problem: '{step1_response}', what is a direct solution?"
        step2_response = llm.invoke(step2_prompt)

        return f"Reasoning Chain Summary:\n1. Problem Identification: {step1_response}\n2. Proposed Solution: {step2_response}\nFinal Response derived from chain: {step2_response}"


# 3. response_validator.py
class Truthfulness(BaseModel):
    is_truthful: bool = Field(description="True if the response is factually accurate, false otherwise.")
    reason: Optional[str] = Field(description="Reason for the truthfulness assessment.", default=None)

class Helpfulness(BaseModel):
    is_helpful: bool = Field(description="True if the response directly addresses the user's query and provides useful information.")
    reason: Optional[str] = Field(description="Reason for the helpfulness assessment.", default=None)

class Bias(BaseModel):
    contains_bias: bool = Field(description="True if the response contains any biased or discriminatory language, false otherwise.")
    reason: Optional[str] = Field(description="Reason for the bias assessment.", default=None)

class Safety(BaseModel):
    is_safe: bool = Field(description="True if the response is safe and does not promote harmful content, false otherwise.")
    reason: Optional[str] = Field(description="Reason for the safety assessment.", default=None)

class ValidationResult(BaseModel):
    truthfulness: Truthfulness
    helpfulness: Helpfulness
    bias: Bias
    safety: Safety
    overall_status: str

class ResponseValidator:
    def _llm_based_evaluation(self, llm: Any, query: str, response: str) -> Dict[str, Any]:
        # Simulate LLM-based evaluation using Pydantic for structure
        # In a real scenario, guardrails-ai would parse LLM output into these models.
        eval_prompt = f"Evaluate the following response for a customer query.\nQuery: {query}\nResponse: {response}\n" \
                      "Is the response truthful, helpful, free of bias, and safe? Provide a brief reason."
        eval_output = llm.invoke(eval_prompt).lower()

        truthful_status = "accurate" in eval_output or "truthful" in eval_output
        helpful_status = "helpful" in eval_output or "useful" in eval_output
        bias_status = "bias" in eval_output or "discriminatory" in eval_output
        safe_status = "safe" in eval_output or "harmful" not in eval_output

        return {
            "truthfulness": Truthfulness(is_truthful=truthful_status, reason="Simulated assessment based on keywords in eval output.").dict(),
            "helpfulness": Helpfulness(is_helpful=helpful_status, reason="Simulated assessment based on keywords in eval output.").dict(),
            "bias": Bias(contains_bias=bias_status, reason="Simulated assessment based on keywords in eval output.").dict(),
            "safety": Safety(is_safe=safe_status, reason="Simulated assessment based on keywords in eval output.").dict(),
        }

    def _round_trip_consistency(self, llm: Any, original_query: str, generated_response: str) -> bool:
        # Simulate rephrasing and checking consistency
        rephrase_prompt = f"Rephrase the following customer query: '{original_query}'"
        rephrased_query = llm.invoke(rephrase_prompt)

        # A very simple consistency check: if the original response is still relevant to the rephrased query
        # In a real system, this would involve embedding similarity or another LLM call.
        return generated_response is not None and len(generated_response) > 10 # Placeholder for actual logic

    def _adversarial_evaluation(self, llm: Any, query: str, response: str) -> bool:
        # Simulate checking for adversarial responses (e.g., trying to trigger harmful content)
        adversarial_check_prompt = f"Given the query '{query}' and response '{response}', does the response contain any adversarial or harmful content, or promote unethical behavior? Answer with 'Yes' or 'No'."
        check_result = llm.invoke(adversarial_check_prompt).strip().lower()
        return check_result == "no"

    def validate(self, llm: Any, query: str, response: str) -> ValidationResult:
        llm_eval_results = self._llm_based_evaluation(llm, query, response)
        round_trip_ok = self._round_trip_consistency(llm, query, response)
        adversarial_ok = self._adversarial_evaluation(llm, query, response)

        overall_status = "Pass" if all([llm_eval_results["truthfulness"]["is_truthful"], 
                                         llm_eval_results["helpfulness"]["is_helpful"], 
                                         not llm_eval_results["bias"]["contains_bias"], 
                                         llm_eval_results["safety"]["is_safe"],
                                         round_trip_ok, adversarial_ok]) else "Fail"

        return ValidationResult(
            truthfulness=Truthfulness(**llm_eval_results["truthfulness"]),
            helpfulness=Helpfulness(**llm_eval_results["helpfulness"]),
            bias=Bias(**llm_eval_results["bias"]),
            safety=Safety(**llm_eval_results["safety"]),
            overall_status=overall_status
        )


# 4. customer_support_agent.py
class CustomerSupportAgent:
    def __init__(self, llm: Any):
        self.llm = llm
        self.prompt_manager = PromptManager()
        self.response_validator = ResponseValidator()
        self.ethical_guidelines = EthicalGuidelines()

    def _simulate_llm_response(self, prompt: str) -> str:
        return self.llm.invoke(prompt)

    def process_query(
        self,
        user_query: str,
        context: str,
        prompt_strategy: Dict[str, Any] = None,
        apply_ethical_guidelines: bool = True,
        apply_bias_mitigation: bool = True,
        enable_reasoning_chain: bool = False
    ) -> Dict[str, Any]:

        if prompt_strategy is None:
            prompt_strategy = {"zero_shot": True}

        final_prompt = self.prompt_manager.create_prompt(user_query, context, prompt_strategy)

        if apply_ethical_guidelines:
            final_prompt = self.ethical_guidelines.apply_constitutional_ai(final_prompt)
        if apply_bias_mitigation:
            final_prompt = self.ethical_guidelines.apply_bias_mitigation(final_prompt)

        print(f"\n--- Generated Prompt ---\n{final_prompt}\n------------------------")

        if enable_reasoning_chain:
            llm_response = self.prompt_manager.build_reasoning_chain(self.llm, user_query)
        else:
            llm_response = self._simulate_llm_response(final_prompt)

        print(f"\n--- LLM Raw Response ---\n{llm_response}\n------------------------")

        validation_result = self.response_validator.validate(self.llm, user_query, llm_response)

        return {
            "query": user_query,
            "context": context,
            "generated_response": llm_response,
            "validation_result": validation_result.dict()
        }


# 5. main.py (integrated)
if __name__ == "__main__":
    dummy_llm = DummyLLM()
    agent = CustomerSupportAgent(llm=dummy_llm)

    print("\n===== Scenario 1: Basic Billing Inquiry (Zero-Shot) =====")
    result1 = agent.process_query(
        user_query="My bill for last month seems too high. Can you check?",
        context="Customer has been with us for 3 years, recent service upgrade.",
        prompt_strategy={"zero_shot": True}
    )
    print(f"Final Response: {result1['generated_response']}")
    print(f"Validation Status: {result1['validation_result']['overall_status']}")

    print("\n===== Scenario 2: Product Refund (Few-Shot, Empathetic Role) =====")
    result2 = agent.process_query(
        user_query="I want a refund for the product I bought last week. It's defective.",
        context="Product XYZ, purchased on 2023-10-20, standard 30-day return policy.",
        prompt_strategy={
            "few_shot": True,
            "few_shot_topic": "billing", # Using billing examples for simplicity, but ideally product refund examples
            "role": "Customer Service Specialist",
            "emotion": "empathetic"
        }
    )
    print(f"Final Response: {result2['generated_response']}")
    print(f"Validation Status: {result2['validation_result']['overall_status']}")

    print("\n===== Scenario 3: Technical Issue (Prompt Chain, Metacognitive) =====")
    result3 = agent.process_query(
        user_query="My internet is constantly disconnecting. What should I do?",
        context="Customer's router model is RT-AX88U, plan is 1Gbps fiber. Basic troubleshooting already performed (restart router).",
        prompt_strategy={
            "metacognitive": True,
            "prompt_chain": True,
            "style": "technical expert"
        },
        enable_reasoning_chain=True
    )
    print(f"Final Response: {result3['generated_response']}")
    print(f"Validation Status: {result3['validation_result']['overall_status']}")

    print("\n===== Scenario 4: Query with potential for bias (Ethical Guidelines) =====")
    result4 = agent.process_query(
        user_query="Tell me about typical users for a budget smartphone.",
        context="Marketing query to understand target audience.",
        prompt_strategy={
            "zero_shot": True,
            "rephrase_respond": True
        },
        apply_ethical_guidelines=True,
        apply_bias_mitigation=True
    )
    print(f"Final Response: {result4['generated_response']}")
    print(f"Validation Status: {result4['validation_result']['overall_status']}")

    print("\n===== Scenario 5: Simple Password Reset (Template-Based, Rereading) =====")
    result5 = agent.process_query(
        user_query="How do I reset my account password?",
        context="User logged out, cannot remember password.",
        prompt_strategy={
            "template_based": {"product": "account"},
            "rereading": True,
            "style": "concise"
        },
        apply_ethical_guidelines=False
    )
    print(f"Final Response: {result5['generated_response']}")
    print(f"Validation Status: {result5['validation_result']['overall_status']}")
