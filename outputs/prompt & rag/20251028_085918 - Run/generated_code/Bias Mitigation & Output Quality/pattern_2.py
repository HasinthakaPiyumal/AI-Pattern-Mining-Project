import random
from collections import Counter

# --- Mock LLM and Data Structures ---

class MockLLM:
    """A mock LLM to simulate responses."""
    def generate(self, prompt):
        print(f"\n--- Mock LLM Processing Prompt ---\n{prompt[:300]}...\n---")
        # Simulate different responses based on prompt keywords
        if "cultural_context: Japan" in prompt:
            return "Konnichiwa! Thank you for your inquiry. In Japan, customer service highly values politeness and efficiency. How may I assist you?"
        elif "balanced_demonstrations: true" in prompt and "debit card issue" in prompt:
            return "I understand you're having an issue with your debit card. It's important to consider both fraud protection measures and user accessibility. Let's look into the specific transaction details."
        elif "evidence_pro_con: true" in prompt:
            return (
                "Claim: Product X is the best for feature Y.\n"
                "Evidence FOR: Several user reviews highlight its superior performance in feature Y due to Z technology. Independent testing showed a 95% satisfaction rate.\n"
                "Evidence AGAINST: Some users reported compatibility issues with older systems. Competitor A offers similar performance at a lower price point, though with less robust support.\n"
                "Conclusion: While Product X excels in feature Y, users should consider their system compatibility and budget."
            )
        elif "prompt_variant" in prompt:
            return f"This is a response from prompt variant {prompt.split('prompt_variant ')[1].split(':')[0]}. The core query was: {prompt.split('Query: ')[1].split('\n')[0]}."
        else:
            return "Thank you for contacting customer support. I will do my best to assist you with your query."

class KnowledgeBase:
    """A mock knowledge base for evidence retrieval."""
    def __init__(self):
        self.data = {
            "debit card issue": {
                "pro": ["Faster transactions", "Wider acceptance", "Online shopping ease"],
                "con": ["Risk of fraud if lost", "Direct link to bank account", "Can be overspent if not monitored"]
            },
            "product X feature Y": {
                "pro": ["High performance", "Advanced technology Z", "Positive user reviews"],
                "con": ["Compatibility issues", "Higher cost", "Competitors offer alternatives"]
            },
            "general inquiry": {
                "pro": ["quick resolution", "24/7 availability"],
                "con": ["may not understand complex nuances", "impersonal feeling"]
            }
        }

    def retrieve(self, topic):
        return self.data.get(topic.lower(), self.data["general inquiry"])

# --- Prompt Engineering Classes ---

class PromptGenerator:
    """Handles various prompt engineering patterns like DENSE, Balanced Demonstrations, and Cultural Awareness."""

    def __init__(self, all_exemplars):
        self.all_exemplars = all_exemplars # A list of {'input': ..., 'output': ..., 'attributes': [...]} dicts

    def _select_balanced_demonstrations(self, query_attributes, num_examples=3):
        """Selects exemplars ensuring balance across relevant attributes related to the query."""
        if not self.all_exemplars:
            return []

        selected_examples = []
        attribute_counts = Counter()

        # Prioritize examples matching query attributes, then balance
        candidates = [ex for ex in self.all_exemplars if any(attr in query_attributes for attr in ex.get('attributes', []))]
        if not candidates:
            candidates = self.all_exemplars # Fallback to all if no direct attribute match

        # Simple greedy approach to balance
        random.shuffle(candidates) # Shuffle to add randomness for selection
        for ex in candidates:
            if len(selected_examples) >= num_examples: # Stop if we have enough examples
                break

            # Check if adding this example improves balance (or doesn't worsen it significantly)
            temp_counts = attribute_counts.copy()
            for attr in ex.get('attributes', []):
                temp_counts[attr] += 1

            # A simplistic heuristic: add if it's not over-representing an already dominant attribute
            # This can be made more sophisticated.
            if not attribute_counts or (max(temp_counts.values()) - min(temp_counts.values())) <= (max(attribute_counts.values()) - min(attribute_counts.values())) + 1:
                selected_examples.append(ex)
                attribute_counts = temp_counts
            elif random.random() < 0.3: # Small chance to pick even if it slightly imbalances for diversity
                 selected_examples.append(ex)
                 attribute_counts = temp_counts

        # Ensure we have num_examples, if possible
        while len(selected_examples) < num_examples and len(selected_examples) < len(self.all_exemplars):
            remaining = [ex for ex in self.all_exemplars if ex not in selected_examples]
            if not remaining: break
            selected_examples.append(random.choice(remaining))

        return selected_examples[:num_examples]

    def _add_cultural_context(self, prompt, cultural_preference=None):
        """Injects cultural sensitivity into the prompt."""
        if cultural_preference:
            return f"Please respond with cultural awareness for {cultural_preference}. {prompt}"
        return prompt

    def generate_dense_prompts(self, query, num_variants=3, query_attributes=None, cultural_preference=None):
        """Generates multiple prompts with distinct exemplar subsets for DENSE pattern."""
        prompts = []
        for i in range(num_variants):
            # Each variant gets a distinct (and balanced) subset of demonstrations
            # In a real scenario, subsets would be significantly distinct. Here we re-select.
            demonstrations = self._select_balanced_demonstrations(query_attributes or [], num_examples=2)
            
            exemplar_section = ""
            if demonstrations:
                exemplar_section = "\nHere are some examples:\n"
                for j, ex in enumerate(demonstrations):
                    exemplar_section += f"Example {j+1}:\nInput: {ex['input']}\nOutput: {ex['output']}\n"

            base_prompt = f"prompt_variant {i+1}: Please act as a helpful customer support assistant.\nQuery: {query}{exemplar_section}\nYour Answer:"
            
            # Apply cultural context per variant (or once to the base query)
            final_prompt = self._add_cultural_context(base_prompt, cultural_preference)
            prompts.append(final_prompt)
        return prompts


class EvidenceAggregator:
    """Handles Debate-Style Evidence Aggregation."""

    def __init__(self, knowledge_base):
        self.knowledge_base = knowledge_base
        self.llm = MockLLM() # Re-using MockLLM for internal synthesis

    def debate_and_aggregate(self, claim_or_query):
        """Retrieves evidence for and against a claim and synthesizes a balanced response."""
        # Simulate retrieval of pro/con arguments from knowledge base
        evidence = self.knowledge_base.retrieve(claim_or_query)
        pro_args = evidence.get('pro', [])
        con_args = evidence.get('con', [])

        if not pro_args and not con_args:
            return "No specific evidence found for this query. I will provide a general response."

        # Construct a prompt for the LLM to synthesize a debate-style response
        debate_prompt = (
            f"Given the claim/query: '{claim_or_query}', analyze the following points and provide a balanced argument, presenting both supporting and opposing evidence.\n"
            f"Evidence FOR: {'; '.join(pro_args)}\n"
            f"Evidence AGAINST: {'; '.join(con_args)}\n"
            f"Based on this evidence, provide a comprehensive and unbiased response (evidence_pro_con: true):"
        )

        # Use LLM to synthesize the debate
        return self.llm.generate(debate_prompt)


# --- Main Customer Support Assistant ---

class CustomerSupportAssistant:
    """Intelligent customer support assistant using various AI design patterns."""
    def __init__(self):
        self.llm = MockLLM()
        
        # Sample exemplars for demonstration. In a real system, these would be rich and diverse.
        self.exemplars = [
            {'input': 'My internet is down.', 'output': 'Please restart your router and modem.', 'attributes': ['technical', 'connectivity']},
            {'input': 'How do I pay my bill?', 'output': 'You can pay online via our portal or through our mobile app.', 'attributes': ['billing', 'payment']},
            {'input': 'I need to update my address.', 'output': 'You can update your address in your account settings or by calling our service line.', 'attributes': ['account_management', 'personal_info']},
            {'input': 'My debit card was declined.', 'output': 'Please check your balance or contact your bank for more details.', 'attributes': ['technical', 'payment', 'debit_card']},
            {'input': 'I want to know about Product X.', 'output': 'Product X is our latest innovation with features A, B, and C.', 'attributes': ['product_info']},
            {'input': 'Why is my debit card blocked?', 'output': 'For security reasons, cards can be blocked. Please contact us directly.', 'attributes': ['technical', 'security', 'debit_card']},
            {'input': 'What are the benefits of Product X?', 'output': 'Product X offers enhanced performance and efficiency.', 'attributes': ['product_info', 'benefits']}
        ]

        self.prompt_generator = PromptGenerator(self.exemplars)
        self.knowledge_base = KnowledgeBase()
        self.evidence_aggregator = EvidenceAggregator(self.knowledge_base)

    def process_query(self, user_query, cultural_preference=None, use_debate_style=False):
        """Processes a user query using DENSE, Balanced Demonstrations, and Cultural Awareness. """
        
        # Step 1: Bias-Aware Design & Mitigation (conceptual - in real-world, this is continuous monitoring/refinement)
        # For example, we might log query attributes to detect under-represented categories or biased responses.
        print(f"\n[System] Processing query: '{user_query}' with cultural preference: {cultural_preference}")

        # Determine attributes from the query to help with balanced demonstration selection
        query_attributes = []
        if "debit card" in user_query.lower():
            query_attributes.append('debit_card')
            query_attributes.append('payment')
        if "internet" in user_query.lower() or "router" in user_query.lower():
            query_attributes.append('connectivity')
            query_attributes.append('technical')
        if "product x" in user_query.lower():
            query_attributes.append('product_info')

        # Step 2: Generate DENSE prompts with Balanced Demonstrations and Cultural Awareness
        dense_prompts = self.prompt_generator.generate_dense_prompts(
            query=user_query,
            num_variants=3, # Use 3 distinct prompt variants for ensembling
            query_attributes=query_attributes,
            cultural_preference=cultural_preference
        )

        # Step 3: Get responses from each prompt variant
        llm_responses = [self.llm.generate(prompt) for prompt in dense_prompts]

        # Step 4: Aggregate responses (DENSE). A simple aggregation here is to pick the first or a majority.
        # In a real system, this would involve more sophisticated techniques like voting, confidence scoring, etc.
        aggregated_response = llm_responses[0] # Simplistic aggregation for demonstration
        print(f"\n[System] Aggregated response from DENSE variants (showing first variant's output):\n{aggregated_response}")

        # Step 5: If debate-style evidence is required, use the EvidenceAggregator
        if use_debate_style:
            print("\n[System] Initiating Debate-Style Evidence Aggregation...")
            debate_response = self.evidence_aggregator.debate_and_aggregate(user_query)
            return f"Based on your query and balanced evidence: {debate_response}\n\nInitial aggregated response: {aggregated_response}"

        return aggregated_response

# --- Example Usage ---
if __name__ == "__main__":
    assistant = CustomerSupportAssistant()

    print("\n--- Scenario 1: Basic Query with DENSE & Balanced Demonstrations ---")
    response1 = assistant.process_query("My debit card is not working.")
    print(f"Final Assistant Response 1: {response1}")

    print("\n--- Scenario 2: Query with Cultural Awareness ---")
    response2 = assistant.process_query("What are the payment options?", cultural_preference="Japan")
    print(f"Final Assistant Response 2: {response2}")

    print("\n--- Scenario 3: Complex Query with Debate-Style Evidence Aggregation ---")
    response3 = assistant.process_query("Is Product X truly the best solution for feature Y?", use_debate_style=True)
    print(f"Final Assistant Response 3: {response3}")

    print("\n--- Scenario 4: Another basic query ---")
    response4 = assistant.process_query("How to reset my internet connection?")
    print(f"Final Assistant Response 4: {response4}")

    print("\n[Bias-Aware Design & Mitigation Note]: In a real application, continuous monitoring of LLM outputs for biases (e.g., disproportionate responses to certain demographics, unfair language) would be implemented. Refinement of training data and objectives, along with documentation of limitations, are crucial ongoing processes.")
