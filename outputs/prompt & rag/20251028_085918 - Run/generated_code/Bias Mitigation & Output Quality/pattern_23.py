import gradio as gr
import random
import collections

class LLMService:
    def __init__(self, model_name="mock_llm"):
        self.model_name = model_name

    def generate(self, prompt):
        if "diagnostic suggestion" in prompt.lower():
            return random.choice([
                "Initial diagnosis: Viral Infection. Consider supportive care and monitor symptoms.",
                "Initial diagnosis: Bacterial Pneumonia. Recommend broad-spectrum antibiotics and chest X-ray.",
                "Initial diagnosis: Allergic Reaction. Advise antihistamines and identify allergen."
            ])
        elif "pro/con evidence" in prompt.lower():
            return f"""Pro: Evidence suggests high likelihood due to {random.choice(['symptom A', 'lab result B'])}.
Con: However, {random.choice(['symptom C is absent', 'another condition D presents similarly'])}.
"""
        elif "culturally adapted" in prompt.lower():
            return f"Culturally adapted response based on your input: The common cold, known as '风寒' (fēnghán) in traditional Chinese medicine, often responds well to rest and warm ginger tea. "
        elif "synthetic data" in prompt.lower():
            return f"Synthetic patient data: Patient exhibits {random.choice(['mild', 'severe'])} fever, {random.choice(['persistent', 'intermittent'])} cough, and {random.choice(['fatigue', 'headache'])}. Demographics: Age {random.randint(20, 70)}, Gender {random.choice(['Male', 'Female'])}."
        return f"LLM response for prompt: {prompt[:50]}..."

class ExemplarDatabase:
    def __init__(self):
        self.exemplars = [
            {"id": 1, "demographics": {"age_group": "adult", "gender": "male", "ethnicity": "caucasian"}, "case": "Case 1: 45 y.o. male with chest pain..."},
            {"id": 2, "demographics": {"age_group": "adult", "gender": "female", "ethnicity": "african"}, "case": "Case 2: 30 y.o. female with fatigue..."},
            {"id": 3, "demographics": {"age_group": "senior", "gender": "male", "ethnicity": "asian"}, "case": "Case 3: 70 y.o. male with shortness of breath..."},
            {"id": 4, "demographics": {"age_group": "adult", "gender": "female", "ethnicity": "hispanic"}, "case": "Case 4: 28 y.o. female with abdominal pain..."},
            {"id": 5, "demographics": {"age_group": "adult", "gender": "male", "ethnicity": "african"}, "case": "Case 5: 50 y.o. male with persistent cough..."},
            {"id": 6, "demographics": {"age_group": "senior", "gender": "female", "ethnicity": "caucasian"}, "case": "Case 6: 65 y.o. female with joint pain..."},
        ]

    def get_exemplars(self, num_examples, patient_demographics):
        # Simple balancing: try to include diverse demographics if possible
        selected = []
        available_exemplars = list(self.exemplars)
        random.shuffle(available_exemplars)

        # Prioritize matching age/gender if possible, then diversify
        for demo_key, demo_val in patient_demographics.items():
            for ex in available_exemplars:
                if ex["demographics"].get(demo_key) == demo_val and ex not in selected:
                    selected.append(ex)
                    if len(selected) == num_examples: return selected

        # Fill up with remaining diverse examples
        for ex in available_exemplars:
            if ex not in selected:
                selected.append(ex)
                if len(selected) == num_examples: return selected
        return selected

class CulturalContextDatabase:
    def __init__(self):
        self.contexts = {
            "chinese": {
                "greeting": "你好，医生 (Nǐ hǎo, yīshēng)",
                "common_ailments": "风寒 (fēnghán - common cold), 上火 (shànghuǒ - internal heat)",
                "advice_phrases": "建议多喝热水 (Jiànyì duō hē rè shuǐ - It is recommended to drink more hot water)"
            },
            "spanish": {
                "greeting": "Hola Doctor/a",
                "common_ailments": "gripe (flu), dolor de cabeza (headache)",
                "advice_phrases": "Descanse y beba muchos líquidos (Rest and drink plenty of fluids)"
            }
        }

    def get_context(self, culture):
        return self.contexts.get(culture.lower(), {})

class PromptEngineer:
    def __init__(self, llm_service, exemplar_db, cultural_db):
        self.llm_service = llm_service
        self.exemplar_db = exemplar_db
        self.cultural_db = cultural_db

    def generate_dense_prompts(self, patient_info, exemplars, num_variants=3):
        base_prompt = f"Patient presents with: {patient_info}. Based on the following examples, provide a diagnostic suggestion and rationale.\n"
        prompts = []
        for i in range(num_variants):
            # Varying subsets of exemplars
            current_exemplars = random.sample(exemplars, min(len(exemplars), max(1, len(exemplars) - i)))
            exemplar_text = "\n".join([ex["case"] for ex in current_exemplars])
            prompts.append(base_prompt + f"Examples:\n{exemplar_text}\nDiagnostic Suggestion:")
        return prompts

    def aggregate_responses(self, responses):
        # Simple majority vote for diagnostic suggestion
        suggestions = [r.split('Initial diagnosis: ')[-1].split('. ')[0] for r in responses if 'Initial diagnosis:' in r]
        if suggestions:
            most_common = collections.Counter(suggestions).most_common(1)
            return f"Aggregated diagnostic suggestion: {most_common[0][0]}. Rationale derived from ensemble responses."
        return "Could not aggregate a clear diagnostic suggestion."

    def get_balanced_demonstrations(self, patient_demographics, num_examples=3):
        return self.exemplar_db.get_exemplars(num_examples, patient_demographics)

    def apply_cultural_awareness(self, prompt, culture):
        cultural_context = self.cultural_db.get_context(culture)
        if cultural_context:
            return f"Considering {culture} cultural context ({cultural_context.get('advice_phrases', '')}), {prompt}"
        return prompt

    def generate_attr_prompt_data(self, base_symptoms, attributes_to_vary):
        generated_data = []
        for attr, values in attributes_to_vary.items():
            for val in values:
                prompt = f"Generate synthetic patient data. Base symptoms: {base_symptoms}. Vary attribute {attr} to {val}. "
                generated_data.append(self.llm_service.generate(prompt))
        return generated_data

class BiasMitigation:
    def detect_bias(self, predictions, patient_demographics):
        # Placeholder for actual bias detection logic (e.g., using AIF360, Fairlearn)
        # For demonstration, we'll just indicate if a specific demographic is present
        if patient_demographics.get("ethnicity") == "african" and random.random() < 0.2: # Simulate a small chance of detected bias
            return "Potential bias detected for African demographic in this batch."
        return "No significant bias detected in current predictions."

    def apply_intervention(self, original_suggestion, intervention_strategy="re-weighting"):
        # Placeholder for intervention strategies
        if "bias" in original_suggestion.lower():
            return f"Applying {intervention_strategy} to refine diagnosis: {original_suggestion.replace('Initial diagnosis:', 'Refined diagnosis (post-intervention):')}"
        return original_suggestion

class EvidenceAggregator:
    def __init__(self, llm_service):
        self.llm_service = llm_service

    def generate_debate_style_evidence(self, diagnosis_claim):
        prompt = f"Provide pro/con evidence for the diagnosis: {diagnosis_claim}"
        response = self.llm_service.generate(prompt)
        pro_evidence = "No pro evidence generated."  # Default
        con_evidence = "No con evidence generated."  # Default

        if "Pro:" in response:
            pro_part = response.split("Pro:")[1].split("Con:")[0].strip()
            pro_evidence = f"Pro: {pro_part}"
        if "Con:" in response:
            con_part = response.split("Con:")[1].strip()
            con_evidence = f"Con: {con_part}"

        return pro_evidence, con_evidence

    def synthesize_rationale(self, pro_evidence, con_evidence):
        return f"\n--- Diagnostic Rationale (Debate Style) ---\n{pro_evidence}\n{con_evidence}\n--- End Rationale ---"

class DiagnosticAssistant:
    def __init__(self):
        self.llm_service = LLMService()
        self.exemplar_db = ExemplarDatabase()
        self.cultural_db = CulturalContextDatabase()
        self.prompt_engineer = PromptEngineer(self.llm_service, self.exemplar_db, self.cultural_db)
        self.bias_mitigation = BiasMitigation()
        self.evidence_aggregator = EvidenceAggregator(self.llm_service)

    def diagnose(self, patient_symptoms, patient_age_group, patient_gender, patient_ethnicity, cultural_context):
        patient_info = f"Symptoms: {patient_symptoms}"
        patient_demographics = {"age_group": patient_age_group.lower(), "gender": patient_gender.lower(), "ethnicity": patient_ethnicity.lower()}

        # 1. Select Balanced Demonstrations
        balanced_exemplars = self.prompt_engineer.get_balanced_demonstrations(patient_demographics)
        exemplar_cases = [ex["case"] for ex in balanced_exemplars]

        # 2. Demonstration Ensembling (DENSE)
        dense_prompts = self.prompt_engineer.generate_dense_prompts(patient_info, balanced_exemplars)
        llm_responses = [self.llm_service.generate(p) for p in dense_prompts]
        diagnostic_suggestion = self.prompt_engineer.aggregate_responses(llm_responses)

        # 3. Cultural Awareness
        culturally_adapted_suggestion = self.prompt_engineer.apply_cultural_awareness(diagnostic_suggestion, cultural_context)

        # 4. AttrPrompt (conceptual for runtime demonstration, usually for data gen/fine-tuning)
        # We'll just generate some synthetic data examples based on base symptoms for illustration
        synthetic_data_prompts = self.prompt_engineer.generate_attr_prompt_data(patient_symptoms, {"severity": ["mild", "severe"]})
        synthetic_data_output = "\n".join(synthetic_data_prompts)

        # 5. Bias-Aware Design & Mitigation
        bias_detection_result = self.bias_mitigation.detect_bias([diagnostic_suggestion], patient_demographics)
        final_suggestion_after_bias_mitigation = self.bias_mitigation.apply_intervention(culturally_adapted_suggestion, "re-weighting")

        # 6. Debate-Style Evidence Aggregation
        pro_evidence, con_evidence = self.evidence_aggregator.generate_debate_style_evidence(final_suggestion_after_bias_mitigation)
        rationale = self.evidence_aggregator.synthesize_rationale(pro_evidence, con_evidence)

        full_output = f"Patient Info: {patient_info}\n"
        full_output += f"Patient Demographics: {patient_demographics}\n"
        full_output += f"\nSelected Balanced Demonstrations:\n{exemplar_cases}\n"
        full_output += f"\nDemonstration Ensembling Result: {diagnostic_suggestion}\n"
        full_output += f"\nCulturally Adapted Suggestion: {culturally_adapted_suggestion}\n"
        full_output += f"\nAttrPrompt (Synthetic Data Examples):\n{synthetic_data_output}\n"
        full_output += f"\nBias Detection Result: {bias_detection_result}\n"
        full_output += f"\nFinal Suggestion (Post Bias Mitigation): {final_suggestion_after_bias_mitigation}\n"
        full_output += rationale

        return full_output

# Gradio Interface Setup
assistant = DiagnosticAssistant()

iface = gr.Interface(
    fn=assistant.diagnose,
    inputs=[
        gr.Textbox(label="Patient Symptoms (e.g., fever, cough, fatigue)"),
        gr.Dropdown(["Adult", "Senior", "Child"], label="Patient Age Group"),
        gr.Dropdown(["Male", "Female", "Other"], label="Patient Gender"),
        gr.Dropdown(["Caucasian", "African", "Asian", "Hispanic", "Other"], label="Patient Ethnicity"),
        gr.Dropdown(["None", "Chinese", "Spanish"], label="Cultural Context for Explanation"),
    ],
    outputs=gr.Textbox(label="Diagnostic Assistant Output"),
    title="AI-Powered Diagnostic Assistant",
    description="Provides diagnostic support leveraging advanced prompt engineering and bias mitigation for accuracy, fairness, and cultural sensitivity."
)

if __name__ == "__main__":
    iface.launch()