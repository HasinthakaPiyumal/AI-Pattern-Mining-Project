import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from guardrails import Guard
import gradio as gr

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    LLM_MODEL_NAME = "gpt-3.5-turbo"

llm = ChatOpenAI(api_key=Config.OPENAI_API_KEY, model=Config.LLM_MODEL_NAME)

def apply_zero_shot_prompt(query: str) -> str:
    return query

def apply_few_shot_prompt(query: str, examples: list[dict]) -> str:
    example_prompts = "\n".join([f"Customer: {ex['input']}\nAgent: {ex['output']}" for ex in examples])
    return f"{example_prompts}\nCustomer: {query}\nAgent:"

def apply_template_based_prompt(query: str, template: str) -> str:
    return template.format(query=query)

def apply_role_prompting(query: str, role: str = "expert customer service agent") -> str:
    return f"You are an {role}. {query}"

def apply_style_emotion_prompting(query: str, style: str = "professional", emotion: str = "empathetic") -> str:
    return f"Respond in a {style} and {emotion} tone. {query}"

def apply_reasoning_prompts(query: str, method: str = "rephrase_and_respond") -> str:
    if method == "rephrase_and_respond":
        return f"First, rephrase the following query to confirm understanding, then provide a detailed response: {query}"
    return query

def build_prompt_chain(initial_query: str, steps: list[str]) -> list[str]:
    return [initial_query] + steps

def get_llm_response(prompt_content: str) -> str:
    messages = [HumanMessage(content=prompt_content)]
    response = llm.invoke(messages)
    return response.content

class LLMEvaluation(BaseModel):
    relevance: str = Field(description="How relevant is the response to the original query? (e.g., 'Highly relevant', 'Partially relevant', 'Irrelevant')")
    accuracy: str = Field(description="How accurate is the information in the response? (e.g., 'Accurate', 'Minor inaccuracies', 'Major inaccuracies')")
    completeness: str = Field(description="How complete is the response? (e.g., 'Comprehensive', 'Adequate', 'Incomplete')")
    safety: bool = Field(description="Is the response safe and harmless? True or False.")
    overall_rating: int = Field(description="Overall rating of the response from 1 to 5, where 5 is excellent.", ge=1, le=5)
    feedback: str = Field(description="Constructive feedback for improving the response.")

def llm_autorater_evaluation(original_query: str, llm_response: str) -> dict:
    guard = Guard.from_pydantic(output_class=LLMEvaluation)
    evaluation_prompt = f"Evaluate the following LLM response based on the original query. Provide a structured evaluation using the provided schema.\n\nOriginal Query: {original_query}\nLLM Response: {llm_response}"
    
    llm_evaluation_raw = get_llm_response(evaluation_prompt)
    
    try:
        validated_evaluation = guard.parse(llm_evaluation_raw, llm=llm)
        return validated_evaluation.model_dump()
    except Exception as e:
        return {"error": f"Guardrails validation failed: {str(e)}", "raw_llm_evaluation": llm_evaluation_raw}

def round_trip_consistency_check(original_query: str, llm_response: str) -> bool:
    check_prompt = f"Given the original question: '{original_query}' and the response: '{llm_response}'. If you were to ask a new question based on the information in the response, would it clearly lead back to the original question's intent? Answer 'yes' or 'no'."
    check_response = get_llm_response(check_prompt).strip().lower()
    return "yes" in check_response

def adversarial_evaluation(original_query: str, llm_response: str) -> bool:
    adversarial_prompt = f"Given the customer query: '{original_query}' and the LLM's response: '{llm_response}'. Please critically analyze the response and identify any potential contradictions, logical fallacies, factual inaccuracies, or areas where the response could be misleading. If you find any issues, describe them. If the response seems robust and correct, state 'No issues found'."
    eval_response = get_llm_response(adversarial_prompt).strip()
    return "no issues found" in eval_response.lower()

def apply_constitutional_principles(prompt: str) -> str:
    principles = [
        "Ensure the response is harmless, helpful, and honest.",
        "Avoid generating content that is illegal, unethical, or biased.",
        "Prioritize user safety and privacy.",
        "Do not spread misinformation or hate speech."
    ]
    return f"{prompt}\n\nStrictly adhere to these ethical guidelines: {'; '.join(principles)}."

def apply_bias_mitigation(prompt: str) -> str:
    mitigation_instructions = [
        "Use neutral, inclusive language and avoid stereotypes.",
        "Ensure fairness and representativeness in your response.",
        "Do not make assumptions about individuals' demographics, beliefs, or circumstances."
    ]
    return f"{prompt}\n\nActively mitigate bias by: {'; '.join(mitigation_instructions)}."

def orchestrate_customer_support(
    customer_query: str,
    prompt_strategy: str,
    validation_strategy: str,
    ethical_strategy: str
) -> str:
    final_prompt = customer_query
    orchestration_log = []

    orchestration_log.append(f"Customer Query: {customer_query}")
    orchestration_log.append(f"Selected Prompt Strategy: {prompt_strategy}")
    orchestration_log.append(f"Selected Ethical Strategy: {ethical_strategy}")
    orchestration_log.append(f"Selected Validation Strategy: {validation_strategy}")

    if prompt_strategy == "Zero-Shot":
        final_prompt = apply_zero_shot_prompt(customer_query)
        orchestration_log.append("Applied Zero-Shot Prompting.")
    elif prompt_strategy == "Few-Shot":
        examples = [
            {"input": "How do I reset my password?", "output": "You can reset your password by clicking 'Forgot Password' on the login page."},
            {"input": "What are your operating hours?", "output": "Our customer support is available 24/7."}
        ]
        final_prompt = apply_few_shot_prompt(customer_query, examples)
        orchestration_log.append("Applied Few-Shot Prompting with dummy examples.")
    elif prompt_strategy == "Template-Based":
        template = "Regarding the query: '{query}', please provide a concise and helpful answer."
        final_prompt = apply_template_based_prompt(customer_query, template)
        orchestration_log.append("Applied Template-Based Prompting.")
    elif prompt_strategy == "Role-Based":
        final_prompt = apply_role_prompting(customer_query, role="polite and efficient customer support specialist")
        orchestration_log.append("Applied Role-Based Prompting.")
    elif prompt_strategy == "Style/Emotion-Based":
        final_prompt = apply_style_emotion_prompting(customer_query, style="friendly", emotion="supportive")
        orchestration_log.append("Applied Style/Emotion-Based Prompting.")
    elif prompt_strategy == "Rephrase and Respond":
        final_prompt = apply_reasoning_prompts(customer_query, method="rephrase_and_respond")
        orchestration_log.append("Applied Rephrase and Respond Reasoning Prompt.")

    if "Constitutional AI" in ethical_strategy:
        final_prompt = apply_constitutional_principles(final_prompt)
        orchestration_log.append("Applied Constitutional AI principles.")
    if "Bias Mitigation" in ethical_strategy:
        final_prompt = apply_bias_mitigation(final_prompt)
        orchestration_log.append("Applied Bias Mitigation instructions.")

    orchestration_log.append(f"Prompt sent to LLM: \n---\n{final_prompt}\n---")

    raw_llm_response = get_llm_response(final_prompt)
    orchestration_log.append(f"Raw LLM Response: \n---\n{raw_llm_response}\n---")

    validation_status = "Not performed"
    validation_details = "N/A"
    
    if validation_strategy == "LLM Autorater":
        validated_output = llm_autorater_evaluation(customer_query, raw_llm_response)
        if "error" in validated_output:
            validation_status = "LLM Autorater Failed"
            validation_details = validated_output["error"]
            orchestration_log.append(f"Validation Error: {validated_output['error']}")
            orchestration_log.append(f"Raw LLM Evaluation for Autorater: {validated_output.get('raw_llm_evaluation', 'N/A')}")
            return f"Validation Failed: {validation_details}\n\n--- Orchestration Log ---\n" + "\n".join(orchestration_log)
        else:
            validation_status = "LLM Autorated"
            validation_details = validated_output
            orchestration_log.append(f"LLM Autorater Result: {validation_details}")
            if not validated_output.get("safety", False):
                return f"Validation Failed (Unsafe response identified by Autorater).\n\n--- Orchestration Log ---\n" + "\n".join(orchestration_log)

    elif validation_strategy == "Round-Trip Consistency":
        is_consistent = round_trip_consistency_check(customer_query, raw_llm_response)
        validation_status = "Round-Trip Checked"
        validation_details = f"Consistent: {is_consistent}"
        orchestration_log.append(f"Round-Trip Consistency Check: {is_consistent}")
        if not is_consistent:
            return f"Validation Failed (Response is inconsistent with query intent).\n\n--- Orchestration Log ---\n" + "\n".join(orchestration_log)

    elif validation_strategy == "Adversarial Evaluation":
        no_adversarial_issues = adversarial_evaluation(customer_query, raw_llm_response)
        validation_status = "Adversarially Evaluated"
        validation_details = f"No Adversarial Issues: {no_adversarial_issues}"
        orchestration_log.append(f"Adversarial Evaluation: No Issues Found: {no_adversarial_issues}")
        if not no_adversarial_issues:
            return f"Validation Failed (Adversarial issues found in response).\n\n--- Orchestration Log ---\n" + "\n".join(orchestration_log)

    final_response_content = raw_llm_response
    orchestration_log.append(f"Final Orchestrated Response:\n---\n{final_response_content}\n---")
    orchestration_log.append(f"Validation Status: {validation_status}")
    orchestration_log.append(f"Validation Details: {validation_details}")

    return f"Orchestrated Response:\n{final_response_content}\n\n--- Orchestration Log ---\n" + "\n".join(orchestration_log)


prompt_strategies = [
    "Zero-Shot",
    "Few-Shot",
    "Template-Based",
    "Role-Based",
    "Style/Emotion-Based",
    "Rephrase and Respond"
]
validation_strategies = [
    "None",
    "LLM Autorater",
    "Round-Trip Consistency",
    "Adversarial Evaluation"
]
ethical_strategies = [
    "None",
    "Constitutional AI",
    "Bias Mitigation",
    "Constitutional AI & Bias Mitigation"
]

iface = gr.Interface(
    fn=orchestrate_customer_support,
    inputs=[
        gr.Textbox(label="Customer Query", placeholder="How can I get a refund for my recent purchase?"),
        gr.Dropdown(prompt_strategies, label="Prompt Engineering Strategy", value="Zero-Shot"),
        gr.Dropdown(validation_strategies, label="Validation Strategy", value="None"),
        gr.Dropdown(ethical_strategies, label="Ethical Alignment Strategy", value="None")
    ],
    outputs="textbox",
    title="Intelligent Customer Support Orchestrator",
    description="This application demonstrates advanced Generative AI orchestration. It applies various prompt engineering techniques, integrates ethical alignment, and validates LLM outputs before presenting a final response. Please ensure you have an OPENAI_API_KEY set in your environment variables (.env file)."
)

if __name__ == "__main__":
    iface.launch()