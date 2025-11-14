
import os
from openai import OpenAI
from pydantic import BaseModel, Field
from guardrails.hub import CompetitorCheck
from guardrails import Guard

# --- 1. LLM Integration Layer ---
class LLMIntegrationLayer:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def query_llm(self, prompt: str, model: str = "gpt-4o-mini", temperature: float = 0.7, max_tokens: int = 500) -> str:
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error querying LLM: {e}")
            return ""

# --- 2. Prompt Management Module ---
class PromptManager:
    def __init__(self):
        pass

    def _create_few_shot_prompt(self, instruction: str, examples: list[dict], query: str) -> str:
        prompt_parts = [instruction]
        for example in examples:
            prompt_parts.append(f"\nInput: {example['input']}\nOutput: {example['output']}")
        prompt_parts.append(f"\nInput: {query}\nOutput:")
        return "\n".join(prompt_parts)

    def _create_zero_shot_prompt(self, instruction: str, query: str) -> str:
        return f"{instruction}\n\n{query}"

    def _create_role_based_prompt(self, role: str, instruction: str, query: str) -> str:
        return f"You are a {role}. {instruction}\n\n{query}"

    def _create_template_driven_prompt(self, template: str, data: dict) -> str:
        return template.format(**data)

    def generate_prompt(self, strategy: str, **kwargs) -> str:
        if strategy == "few_shot":
            return self._create_few_shot_prompt(kwargs['instruction'], kwargs['examples'], kwargs['query'])
        elif strategy == "zero_shot":
            return self._create_zero_shot_prompt(kwargs['instruction'], kwargs['query'])
        elif strategy == "role_based":
            return self._create_role_based_prompt(kwargs['role'], kwargs['instruction'], kwargs['query'])
        elif strategy == "template_driven":
            return self._create_template_driven_prompt(kwargs['template'], kwargs['data'])
        else:
            raise ValueError(f"Unknown prompt strategy: {strategy}")

# --- 3. Content Generation Module ---
class ContentGenerator:
    def __init__(self, llm_integration_layer: LLMIntegrationLayer, prompt_manager: PromptManager):
        self.llm = llm_integration_layer
        self.prompt_manager = prompt_manager

    def generate_content(self, content_type: str, medical_data: str, strategy: str = "zero_shot", **kwargs) -> str:
        instruction_map = {
            "patient_education": "Generate clear and concise patient education material.",
            "research_summary": "Summarize the following medical research paper, highlighting key findings and implications.",
            "clinical_note_draft": "Draft a clinical note based on the provided patient information. Focus on conciseness and accuracy."
        }
        instruction = instruction_map.get(content_type, "Generate medical content based on the following information.")

        prompt_kwargs = {"instruction": instruction, "query": medical_data}
        if strategy == "few_shot":
            prompt_kwargs["examples"] = kwargs.get("examples", [])
        elif strategy == "role_based":
            prompt_kwargs["role"] = kwargs.get("role", "medical professional")
        elif strategy == "template_driven":
            prompt_kwargs["template"] = kwargs.get("template", "")
            prompt_kwargs["data"] = kwargs.get("data", {})

        prompt = self.prompt_manager.generate_prompt(strategy, **prompt_kwargs)
        print(f"\n--- Generated Prompt ({strategy}) ---\n{prompt[:500]}...") # Show first 500 chars
        return self.llm.query_llm(prompt)

# --- 4. Evaluation and Quality Assurance Module ---
class EvaluationAndQualityAssurance:
    def __init__(self, llm_integration_layer: LLMIntegrationLayer):
        self.llm = llm_integration_layer

    def llm_autorate(self, generated_content: str, rubric: str) -> str:
        rating_prompt = f"Rate the following medical content based on the given rubric. Provide a score out of 5 and a brief justification.\n\nRubric: {rubric}\n\nContent: {generated_content}"
        return self.llm.query_llm(rating_prompt, max_tokens=200)

    def round_trip_consistency_check(self, original_data: str, generated_content: str) -> str:
        summary_prompt = f"Summarize the key information from the following medical content:\n\n{generated_content}"
        summarized_content = self.llm.query_llm(summary_prompt, max_tokens=300)

        comparison_prompt = f"Compare the following original medical data with its summary. Indicate if the summary accurately reflects the original data and if any crucial information is missing or misrepresented. Respond with 'Consistent', 'Partially Consistent', or 'Inconsistent'.\n\nOriginal Data: {original_data}\n\nSummarized Content: {summarized_content}"
        return self.llm.query_llm(comparison_prompt, max_tokens=100)

    def adversarial_evaluation(self, generated_content: str, bias_check_prompt: str) -> str:
        adv_prompt = f"{bias_check_prompt}\n\nContent: {generated_content}"
        return self.llm.query_llm(adv_prompt, max_tokens=200)

    def constitutional_ai_check(self, generated_content: str, ethical_guidelines: list[str]) -> dict:
        class MedicalContentEthics(BaseModel):
            is_ethical: bool = Field(description="True if the content adheres to all ethical guidelines, false otherwise.")
            violations: list[str] = Field(description="List of specific ethical guidelines violated, if any.")
            suggestions: str = Field(description="Suggestions for improvement to meet ethical standards.", default="No suggestions.")

        # Example of using Guardrails for ethical review
        # For more complex checks, you'd integrate the ethical_guidelines more deeply
        # Here, we demonstrate a simple structured output and an optional `CompetitorCheck`
        
        # The `CompetitorCheck` is an example of a guardrail, but its direct applicability
        # to 