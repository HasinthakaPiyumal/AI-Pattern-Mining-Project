import random
import pandas as pd

class CoreLLMService:
    def __init__(self, model_name="simulated_llm"):
        self.model_name = model_name

    def generate_response(self, prompt, max_tokens=100):
        if "culturally adapted for Japan" in prompt:
            return "どういたしまして！お客様のご質問にお答えします。ご要望があれば何でもお申し付けください。(You're welcome! I will answer your questions. Please let me know if you have any requests.)"
        if "pro arguments" in prompt and "con arguments" in prompt:
             return f"Arguments For: This is beneficial because X. It also helps with Y.\nArguments Against: However, Z is a concern. There might be issues with W."
        if "customer age: 25, product category: electronics" in prompt and "generate a query" in prompt:
            return "My new smartphone is not charging. What should I do? (age:25, electronics)"
        if "customer age: 60, product category: apparel" in prompt and "generate a query" in prompt:
            return "I need to return a dress that doesn't fit. How can I do that? (age:60, apparel)"
        return f"Simulated LLM response to: '{prompt[:50]}...' (truncated for brevity)"

class BalancedDemonstrationsSelector:
    def __init__(self, exemplars_df):
        self.exemplars_df = exemplars_df

    def select_balanced_subset(self, k, sensitive_attributes=None):
        if sensitive_attributes is None:
            return self.exemplars_df.sample(k).to_dict("records")

        balanced_exemplars = []
        for attr in sensitive_attributes:
            unique_values = self.exemplars_df[attr].unique()
            if not unique_values.size:
                continue

            num_per_value = max(1, k // len(unique_values))
            for value in unique_values:
                subset = self.exemplars_df[self.exemplars_df[attr] == value]
                if not subset.empty:
                    balanced_exemplars.extend(subset.sample(min(len(subset), num_per_value)).to_dict("records"))

        if len(balanced_exemplars) < k:
            remaining_needed = k - len(balanced_exemplars)
            selected_ids = {tuple(ex.items()) for ex in balanced_exemplars}
            available_for_random = self.exemplars_df[~self.exemplars_df.apply(lambda x: tuple(x.items()) in selected_ids, axis=1)]
            if not available_for_random.empty:
                balanced_exemplars.extend(available_for_random.sample(min(len(available_for_random), remaining_needed)).to_dict("records"))
        
        final_exemplars = []
        seen = set()
        for ex in balanced_exemplars:
            ex_tuple = tuple(sorted(ex.items()))
            if ex_tuple not in seen:
                final_exemplars.append(ex)
                seen.add(ex_tuple)
            if len(final_exemplars) >= k:
                break
        
        return final_exemplars[:k]


class CulturalAwarenessAdapter:
    def __init__(self):
        self.cultural_instructions = {
            "japan": "respond politely and formally, as is customary in Japan.",
            "usa": "be friendly and direct, as is common in the USA.",
            "india": "use respectful and elaborate language, common in India.",
            "default": "be helpful and professional."
        }
        self.vocabulary_maps = {
            "japan": {"hello": "konnichiwa", "thank you": "arigato"},
            "usa": {"hello": "hi", "thank you": "thanks"},
            "india": {"hello": "namaste", "thank you": "dhanyawad"},
        }

    def adapt_prompt(self, base_prompt, cultural_context):
        instruction = self.cultural_instructions.get(cultural_context.lower(), self.cultural_instructions["default"])
        adapted_prompt = f"{base_prompt} Please {instruction}"

        vocab_map = self.vocabulary_maps.get(cultural_context.lower(), {})
        for english_word, local_word in vocab_map.items():
            adapted_prompt = adapted_prompt.replace(english_word, local_word)

        return adapted_prompt

class DENSEEnsembler:
    def __init__(self, llm_service):
        self.llm_service = llm_service

    def ensemble_responses(self, base_prompt, exemplars_pool, n_prompts=3, k_exemplars_per_prompt=2):
        responses = []
        for _ in range(n_prompts):
            if len(exemplars_pool) < k_exemplars_per_prompt:
                selected_exemplars = exemplars_pool
            else:
                selected_exemplars = random.sample(exemplars_pool, k_exemplars_per_prompt)

            exemplar_text = "\n".join([f"Example Input: {ex['input']}\nExample Output: {ex['output']}" for ex in selected_exemplars])
            full_prompt = f"{exemplar_text}\n\n{base_prompt}"
            responses.append(self.llm_service.generate_response(full_prompt))

        return responses[0]

class AttrPromptGenerator:
    def __init__(self, llm_service):
        self.llm_service = llm_service

    def generate_synthetic_data(self, base_scenario, attributes_to_vary, num_samples=1):
        generated_data = []
        for _ in range(num_samples):
            variations = []
            for attr, values in attributes_to_vary.items():
                if values:
                    variations.append(f"{attr}: {random.choice(values)}")
            
            variation_str = ", ".join(variations)
            prompt = f"Generate a customer query based on the following scenario: '{base_scenario}'. Vary attributes: {variation_str}. The query should be realistic."
            
            synthetic_query = self.llm_service.generate_response(prompt)
            generated_data.append({"query": synthetic_query, "attributes": variation_str})
        return generated_data

class BiasAwareDesignMitigation:
    def __init__(self):
        self.log_data = []
        self.sensitive_attributes = ["demographic", "issue_type"]
        self.df_log = pd.DataFrame()

    def log_interaction(self, query, response, metadata):
        new_entry = {"query": query, "response": response, "timestamp": pd.Timestamp.now()}
        new_entry.update(metadata) # Add metadata directly to the entry
        self.log_data.append(new_entry)
        self.df_log = pd.DataFrame(self.log_data)

    def detect_bias(self, metric="response_quality_sentiment"):
        if self.df_log.empty:
            return "No data to detect bias."

        bias_reports = []
        for attr in self.sensitive_attributes:
            if attr in self.df_log.columns:
                bias_report = f"Bias report for attribute '{attr}':\n"
                if metric == "response_quality_sentiment":
                    grouped = self.df_log.groupby(attr).size()
                    bias_report += f"  Response counts by {attr}:\n{grouped.to_string()}\n"
                bias_reports.append(bias_report)
        
        return "\n".join(bias_reports) if bias_reports else "No bias detected for specified sensitive attributes."

    def suggest_mitigation(self, bias_report):
        if "No bias" in bias_report:
            return "No specific mitigation suggested as no bias was detected."
        suggestions = []
        if "demographic" in bias_report:
            suggestions.append("Adjust Balanced Demonstrations to ensure better representation across demographics.")
        if "issue_type" in bias_report:
            suggestions.append("Generate more synthetic data using AttrPrompt for underrepresented issue types.")
        suggestions.append("Consider fine-tuning the base LLM with debiased datasets.")
        return "Mitigation Suggestions:\n" + "\n".join(suggestions)


class DebateEvidenceAggregator:
    def __init__(self, llm_service):
        self.llm_service = llm_service

    def aggregate_evidence(self, claim):
        pro_prompt = f"Provide strong arguments *for* the following claim: \"{claim}\". List at least 3 points."
        con_prompt = f"Provide strong arguments *against* the following claim: \"{claim}\". List at least 3 points."

        pro_arguments = self.llm_service.generate_response(pro_prompt)
        con_arguments = self.llm_service.generate_response(con_prompt)

        return {
            "claim": claim,
            "for": pro_arguments,
            "against": con_arguments
        }

class CustomerSupportAI:
    def __init__(self, exemplars_data, sensitive_attributes=None):
        self.llm_service = CoreLLMService()
        self.exemplars_df = pd.DataFrame(exemplars_data)
        self.balanced_selector = BalancedDemonstrationsSelector(self.exemplars_df)
        self.cultural_adapter = CulturalAwarenessAdapter()
        self.dense_ensembler = DENSEEnsembler(self.llm_service)
        self.attr_prompt_gen = AttrPromptGenerator(self.llm_service)
        self.bias_mitigation = BiasAwareDesignMitigation()
        self.debate_aggregator = DebateEvidenceAggregator(self.llm_service)
        self.sensitive_attributes = sensitive_attributes if sensitive_attributes else ["demographic"]

    def handle_customer_query(self, query, customer_metadata=None):
        if customer_metadata is None:
            customer_metadata = {"demographic": "unknown", "issue_type": "general", "culture": "default"}

        balanced_exemplars = self.balanced_selector.select_balanced_subset(
            k=5, sensitive_attributes=self.sensitive_attributes
        )

        adapted_query = self.cultural_adapter.adapt_prompt(query, customer_metadata.get("culture", "default"))

        base_prompt = f"Customer Query: {adapted_query}\nAI Assistant:"
        response = self.dense_ensembler.ensemble_responses(
            base_prompt, balanced_exemplars, n_prompts=3, k_exemplars_per_prompt=2
        )

        self.bias_mitigation.log_interaction(query, response, customer_metadata)

        return response

    def generate_synthetic_training_data(self, base_scenario, attributes_to_vary, num_samples):
        return self.attr_prompt_gen.generate_synthetic_data(base_scenario, attributes_to_vary, num_samples)

    def get_bias_report(self):
        return self.bias_mitigation.detect_bias()

    def get_mitigation_suggestions(self, bias_report):
        return self.bias_mitigation.suggest_mitigation(bias_report)

    def get_debated_evidence(self, claim):
        return self.debate_aggregator.aggregate_evidence(claim)

sample_exemplars = [
    {"input": "My order is late.", "output": "Please provide your order number to track it.", "demographic": "young_adult", "issue_type": "shipping", "sentiment": "negative"},
    {"input": "How do I return a product?", "output": "You can initiate a return through your order history.", "demographic": "adult", "issue_type": "returns", "sentiment": "neutral"},
    {"input": "My item arrived broken.", "output": "We apologize. Please contact support for a replacement.", "demographic": "senior", "issue_type": "damage", "sentiment": "very_negative"},
    {"input": "What is your privacy policy?", "output": "Our privacy policy can be found on our website's footer.", "demographic": "young_adult", "issue_type": "policy", "sentiment": "neutral"},
    {"input": "I want to change my shipping address.", "output": "You can update your address in your account settings before dispatch.", "demographic": "adult", "issue_type": "shipping", "sentiment": "positive"},
    {"input": "Where is my refund?", "output": "Refunds typically process within 3-5 business days.", "demographic": "senior", "issue_type": "returns", "sentiment": "negative"},
    {"input": "Can I cancel my subscription?", "output": "Yes, you can cancel in your account settings.", "demographic": "young_adult", "issue_type": "subscription", "sentiment": "neutral"},
    {"input": "The color of the shirt is wrong.", "output": "We can arrange an exchange. Please visit our returns page.", "demographic": "adult", "issue_type": "wrong_item", "sentiment": "negative"},
    {"input": "How to use my discount code?", "output": "Apply the code at checkout.", "demographic": "young_adult", "issue_type": "discount", "sentiment": "positive"},
    {"input": "Need help with product assembly.", "output": "Please refer to the instruction manual or find assembly videos online.", "demographic": "senior", "issue_type": "product_help", "sentiment": "neutral"},
]

customer_ai = CustomerSupportAI(exemplars_data=sample_exemplars, sensitive_attributes=["demographic", "issue_type"])

query1 = "My recent purchase has not arrived yet."
customer_meta1 = {"demographic": "adult", "issue_type": "shipping", "culture": "usa"}
response1 = customer_ai.handle_customer_query(query1, customer_meta1)

query2 = "私の注文がまだ届いていません。(My order has not arrived yet.)"
customer_meta2 = {"demographic": "senior", "issue_type": "shipping", "culture": "japan"}
response2 = customer_ai.handle_customer_query(query2, customer_meta2)
