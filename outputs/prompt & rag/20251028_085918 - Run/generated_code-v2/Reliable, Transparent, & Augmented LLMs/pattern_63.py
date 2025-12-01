import pandas as pd
from typing import List, Dict, Any

# Simulated imports for demonstration
class FewShotPromptTemplate:
    def __init__(self, examples, example_prompt, suffix, input_variables):
        self.examples = examples
        self.example_prompt = example_prompt
        self.suffix = suffix
        self.input_variables = input_variables

    def format(self, **kwargs):
        formatted_examples = "\n\n".join([
            self.example_prompt.format(text=ex["text"])
            for ex in self.examples
        ])
        return f"{formatted_examples}\n\n{self.suffix.format(**kwargs)}"

class PromptTemplate:
    def __init__(self, template, input_variables):
        self.template = template
        self.input_variables = input_variables

    def format(self, **kwargs):
        return self.template.format(**kwargs)


class DemonstrationManager:
    def __init__(self, demonstrations: List[Dict[str, Any]]):
        self.df = pd.DataFrame(demonstrations)

    def get_demonstrations(self, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        if filters is None:
            return self.df.to_dict(orient="records")
        filtered_df = self.df.copy()
        for key, value in filters.items():
            if key == "bias_tags":
                filtered_df = filtered_df[filtered_df["bias_tags"].apply(lambda tags: value in tags)]
            else:
                filtered_df = filtered_df[filtered_df[key] == value]
        return filtered_df.to_dict(orient="records")


class BiasMitigator:
    def __init__(self, demonstration_manager: DemonstrationManager):
        self.demonstration_manager = demonstration_manager

    def select_balanced_demonstrations(self, user_requirements: Dict[str, Any], num_demonstrations: int = 3) -> List[Dict[str, Any]]:
        # Simple balancing logic: prioritize gender-neutral examples
        # and then add others if needed, ensuring diversity if possible.
        all_demonstrations = self.demonstration_manager.get_demonstrations()
        
        relevant_demonstrations = []
        for demo in all_demonstrations:
            # Basic relevance check (can be expanded)
            if user_requirements.get("industry") in demo.get("industry", "") or \
               user_requirements.get("role") in demo.get("role", ""):
                relevant_demonstrations.append(demo)

        gender_neutral_demos = [d for d in relevant_demonstrations if "gender-neutral" in d.get("bias_tags", [])]
        other_demos = [d for d in relevant_demonstrations if "gender-neutral" not in d.get("bias_tags", [])]

        selected_demos = []
        
        # Try to fill with gender-neutral examples first
        for i in range(min(num_demonstrations, len(gender_neutral_demos))):
            selected_demos.append(gender_neutral_demos[i])
        
        # Fill remaining spots with other diverse examples
        remaining_slots = num_demonstrations - len(selected_demos)
        if remaining_slots > 0:
            for i in range(min(remaining_slots, len(other_demos))):
                selected_demos.append(other_demos[i])
                
        return selected_demos[:num_demonstrations]


class JobDescriptionGenerator:
    def __init__(self, llm_model: Any):
        self.llm_model = llm_model  # This would be a real LLM client in a real app

    def _format_prompt(self, demonstrations: List[Dict[str, Any]], user_requirements: Dict[str, Any]) -> str:
        example_prompt = PromptTemplate(
            template="Role: {role}\nIndustry: {industry}\nJob Description: {text}",
            input_variables=["role", "industry", "text"],
        )

        few_shot_prompt = FewShotPromptTemplate(
            examples=demonstrations,
            example_prompt=example_prompt,
            suffix="Role: {role}\nIndustry: {industry}\nJob Description:",
            input_variables=["role", "industry"],
        )
        return few_shot_prompt.format(role=user_requirements["role"], industry=user_requirements["industry"])

    def generate_job_description(self, demonstrations: List[Dict[str, Any]], user_requirements: Dict[str, Any]) -> str:
        prompt = self._format_prompt(demonstrations, user_requirements)
        # Simulate LLM call
        print(f"\n--- Simulating LLM Call with Prompt ---\n{prompt}\n---")
        simulated_response = f"Generated Job Description for {user_requirements['role']} in {user_requirements['industry']}:\n\n" \
                             f"We are seeking a highly motivated and skilled individual to join our team as a {user_requirements['role']}. " \
                             f"The ideal candidate will have a strong background in relevant technologies and a passion for innovation. " \
                             f"This role offers an exciting opportunity to contribute to challenging projects and grow professionally within the {user_requirements['industry']} sector. " \
                             f"We value diversity and encourage all qualified applicants to apply, regardless of background or identity."
        return simulated_response


# Main execution
if __name__ == "__main__":
    # 1. Data Layer: Balanced Demonstration Dataset
    sample_demonstrations = [
        {"id": 1, "text": "We are looking for a dynamic leader to drive strategic initiatives.", "bias_tags": ["gender-neutral", "leadership"], "industry": "Tech", "role": "Project Manager"},
        {"id": 2, "text": "Seeking a detail-oriented administrator to manage office operations.", "bias_tags": ["gender-neutral", "administrative"], "industry": "Finance", "role": "Office Administrator"},
        {"id": 3, "text": "A skilled craftsman is needed for intricate design work.", "bias_tags": ["masculine-leaning", "technical"], "industry": "Manufacturing", "role": "Craftsman"},
        {"id": 4, "text": "We require a nurturing caregiver for our elderly clients.", "bias_tags": ["feminine-leaning", "healthcare"], "industry": "Healthcare", "role": "Caregiver"},
        {"id": 5, "text": "Join our innovative team as a Software Engineer, shaping the future of technology.", "bias_tags": ["gender-neutral", "tech"], "industry": "Tech", "role": "Software Engineer"},
        {"id": 6, "text": "Seeking a brilliant architect to design groundbreaking solutions.", "bias_tags": ["gender-neutral", "design"], "industry": "Construction", "role": "Architect"},
    ]

    demonstration_manager = DemonstrationManager(sample_demonstrations)
    bias_mitigator = BiasMitigator(demonstration_manager)

    # Simulate a dummy LLM (e.g., a simple function that returns a string)
    def simulated_llm(prompt: str) -> str:
        return f"[LLM Response based on prompt: {prompt[:50]}...]"

    job_description_generator = JobDescriptionGenerator(simulated_llm)

    # User Request 1: Software Engineer
    user_requirements_1 = {"role": "Software Engineer", "industry": "Tech"}
    print(f"\n--- Generating Job Description for {user_requirements_1['role']} ---")
    selected_demos_1 = bias_mitigator.select_balanced_demonstrations(user_requirements_1, num_demonstrations=2)
    print(f"Selected demonstrations for {user_requirements_1['role']}: {[d['id'] for d in selected_demos_1]}")
    generated_jd_1 = job_description_generator.generate_job_description(selected_demos_1, user_requirements_1)
    print(f"\nGenerated Job Description:\n{generated_jd_1}")

    # User Request 2: Project Manager (seeking a gender-neutral description)
    user_requirements_2 = {"role": "Project Manager", "industry": "Tech"}
    print(f"\n--- Generating Job Description for {user_requirements_2['role']} ---")
    selected_demos_2 = bias_mitigator.select_balanced_demonstrations(user_requirements_2, num_demonstrations=3)
    print(f"Selected demonstrations for {user_requirements_2['role']}: {[d['id'] for d in selected_demos_2]}")
    generated_jd_2 = job_description_generator.generate_job_description(selected_demos_2, user_requirements_2)
    print(f"\nGenerated Job Description:\n{generated_jd_2}")

    # User Request 3: Caregiver (to show how selection might work with less neutral data)
    user_requirements_3 = {"role": "Caregiver", "industry": "Healthcare"}
    print(f"\n--- Generating Job Description for {user_requirements_3['role']} ---")
    selected_demos_3 = bias_mitigator.select_balanced_demonstrations(user_requirements_3, num_demonstrations=2)
    print(f"Selected demonstrations for {user_requirements_3['role']}: {[d['id'] for d in selected_demos_3]}")
    generated_jd_3 = job_description_generator.generate_job_description(selected_demos_3, user_requirements_3)
    print(f"\nGenerated Job Description:\n{generated_jd_3}")
