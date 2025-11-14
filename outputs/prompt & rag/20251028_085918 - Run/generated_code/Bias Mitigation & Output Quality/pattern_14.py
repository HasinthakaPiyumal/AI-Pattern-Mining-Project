import random

class DummyLLM:
    """
    A dummy LLM to simulate response generation without loading an actual large model.
    In a real application, this would be replaced by an actual LLM client (e.g., from
    Hugging Face Transformers, OpenAI API, Gemini API, etc.).
    """
    def generate(self, prompt: str) -> str:
        # Simulate LLM thinking and generating a response
        responses = [
            f"Acknowledged: '{prompt[:70]}...' - Here is a helpful answer based on your request.",
            f"Processing your inquiry about '{prompt.splitlines()[-1].strip()}'...",
            f"Based on your query, a possible solution involves checking our FAQs or contacting a specialist.",
            f"I understand your question regarding '{prompt.splitlines()[-1].strip()}'. Let me provide some insights and steps.",
            f"Here's a detailed response to: '{prompt.splitlines()[-1].strip()}'."
        ]
        return random.choice(responses) + "\n(This is a simulated LLM response for demonstration purposes.)"

class IntelligentChatbot:
    """
    An intelligent customer support chatbot integrating various AI design patterns
    to enhance performance, mitigate biases, and improve output quality.
    """
    def __init__(self, llm_model=None):
        self.llm = llm_model if llm_model else DummyLLM()
        self.base_instructions = (
            "You are an intelligent, empathetic, and accurate customer support assistant. "
            "Your goal is to provide helpful, culturally sensitive, and unbiased answers. "
            "Please be concise and direct."
        )

        # Pattern: Selecting Balanced Demonstrations
        # These demonstrations are curated to represent different scenarios and avoid biases.
        # In a real system, these would be carefully selected from a diverse dataset.
        self.balanced_demonstrations = {
            "account_management": [
                {"input": "How do I update my profile?", "output": "To update your profile, log in to your account and navigate to 'Settings' or 'My Profile'."},
                {"input": "I need to change my password.", "output": "You can change your password by clicking 'Forgot Password' on the login page or in your account settings."},
                {"input": "What information can I see in my account?", "output": "Your account typically shows billing history, service details, and personal information."}
            ],
            "technical_support": [
                {"input": "My service is down.", "output": "Please try restarting your device/router. If the issue persists, check our status page or contact technical support directly."},
                {"input": "How do I troubleshoot connection issues?", "output": "Common troubleshooting steps include checking cables, restarting hardware, and verifying network settings."},
                {"input": "I cannot access a specific feature.", "output": "Ensure your software is updated and you have the necessary permissions. Refer to our user manual for feature-specific guidance."}
            ],
            "billing_inquiries": [
                {"input": "Explain my last bill.", "output": "Your last bill details charges for the period, including service fees and any usage-based charges. You can view a detailed breakdown in your online portal."},
                {"input": "I was overcharged.", "output": "Please provide the charge details and your account number so we can investigate the discrepancy."},
                {"input": "When is my next payment due?", "output": "Your next payment due date is typically displayed on your current bill or in your account dashboard."}
            ]
        }

        # Pattern: Cultural Awareness
        # Stores culturally specific greetings or phrasing adjustments.
        self.cultural_adaptations = {
            "en": {"greeting": "Hello! How may I assist you today?", "tone": "formal but friendly"},
            "es": {"greeting": "¡Hola! ¿En qué puedo ayudarte hoy?", "tone": "friendly"},
            "fr": {"greeting": "Bonjour! Comment puis-je vous aider aujourd'hui?", "tone": "polite"},
            "de": {"greeting": "Guten Tag! Wie kann ich Ihnen behilflich sein?", "tone": "formal"}
            # Add more cultural contexts and their specific adaptations.
        }

        # Pattern: Bias-Aware Design & Mitigation (for post-processing)
        # A very simplistic list of keywords to flag potential bias.
        # In a real system, this would involve advanced NLP models for bias detection.
        self.bias_flag_keywords = [
            "always best for men", "women should", "only rich people can",
            "poor people cannot", "certain race", "religious duty",
            "gender-specific role"
        ]

    def _get_demonstrations(self, topic: str = None, num_examples: int = 2) -> str:
        """
        Selects balanced demonstrations for few-shot prompting.
        If a topic is provided, it tries to select examples relevant to that topic.
        Otherwise, it samples from all available demonstrations.
        """
        selected_examples = []
        if topic and topic in self.balanced_demonstrations:
            topic_examples = self.balanced_demonstrations[topic]
            selected_examples.extend(random.sample(topic_examples, min(num_examples, len(topic_examples))))
        else:
            # If no specific topic or not enough examples, sample broadly
            all_available_examples = []
            for t in self.balanced_demonstrations:
                all_available_examples.extend(self.balanced_demonstrations[t])
            selected_examples.extend(random.sample(all_available_examples, min(num_examples, len(all_available_examples))))

        demonstrations_str = ""
        for ex in selected_examples:
            demonstrations_str += f"Customer: {ex['input']}\nAgent: {ex['output']}\n\n"
        return demonstrations_str

    def _apply_cultural_awareness_to_prompt(self, prompt_body: str, culture: str = "en") -> str:
        """
        Injects cultural sensitivity into the prompt.
        This could involve adjusting greetings, tone, or specific phrasing.
        """
        adaptations = self.cultural_adaptations.get(culture, self.cultural_adaptations["en"])
        greeting = adaptations["greeting"]
        # In a more advanced system, 'tone' could guide LLM generation directly.
        
        # Prepend the cultural greeting to the prompt instructions.
        return f"{greeting}\n\n{self.base_instructions}\n\n{prompt_body}"

    def _generate_dense_prompts(self, user_query: str, num_variations: int = 3) -> list[str]:
        """
        Pattern: Demonstration Ensembling (DENSE).
        Generates multiple prompts, each with a distinct subset of demonstrations,
        to reduce variance and improve accuracy.
        """
        dense_prompts = []
        all_topics = list(self.balanced_demonstrations.keys())

        for i in range(num_variations):
            # Each prompt variation gets a different set of examples.
            # We randomly pick 1 or 2 topics to draw examples from for diversity.
            chosen_topics = random.sample(all_topics, min(random.randint(1, 2), len(all_topics)))
            
            demonstrations = ""
            for topic in chosen_topics:
                demonstrations += self._get_demonstrations(topic=topic, num_examples=1) # Fewer examples per prompt in DENSE variations

            # Construct the prompt for this variation
            prompt_for_variation = (
                f"{demonstrations}"
                f"Customer Query: {user_query}\n"
                f"Agent:"
            )
            dense_prompts.append(prompt_for_variation)
        return dense_prompts

    def _aggregate_responses(self, responses: list[str]) -> str:
        """
        Aggregates outputs from multiple LLM responses (e.g., from DENSE).
        In a real application, this might involve:
        - Majority voting for classification tasks.
        - Summarization by another LLM.
        - Identifying common themes or the most comprehensive answer.
        For this simulation, it provides a combined view.
        """
        if not responses:
            return "No valid responses were generated."

        # A very basic aggregation: combine and indicate multiple perspectives.
        # This demonstrates the concept, not a sophisticated aggregation algorithm.
        return "Here are insights from multiple perspectives:\n\n" + "\n---\n".join(responses)

    def _check_and_mitigate_bias(self, text: str) -> str:
        """
        Pattern: Bias-Aware Design & Mitigation (post-processing).
        Checks the generated text for potentially biased language and flags it.
        In a production system, this would also include rephrasing or escalating.
        """
        text_lower = text.lower()
        for keyword in self.bias_flag_keywords:
            if keyword in text_lower:
                # Flag the response. More advanced mitigation would attempt to rephrase.
                return f"{text}\n\n[WARNING: This response might contain biased language based on internal checks. Please review for fairness and inclusivity.]"
        return text

    def _generate_debate_style_response(self, query: str) -> str:
        """
        Pattern: Debate-Style Evidence Aggregation (simplified).
        For complex or controversial queries, generate arguments for and against a point.
        This helps in providing a more balanced and robust answer, avoiding 'cherry-picking'.
        """
        # Prompts to elicit different perspectives
        pro_prompt = (
            f"{self.base_instructions}\n\n"
            f"Present a strong argument *supporting* the following claim/question in the context of customer support:\n"
            f"CLAIM: '{query}'\nAgent:"
        )
        con_prompt = (
            f"{self.base_instructions}\n\n"
            f"Present a strong argument *challenging* or *raising concerns about* the following claim/question in the context of customer support:\n"
            f"CLAIM: '{query}'\nAgent:"
        )

        pro_response = self.llm.generate(pro_prompt)
        con_response = self.llm.generate(con_prompt)

        return (
            f"Analyzing your query ('{query}') from multiple perspectives:\n\n"
            f"--- Argument FOR ---\n{pro_response}\n\n"
            f"--- Argument AGAINST / CONCERNS ---\n{con_response}\n\n"
            f"Please consider both viewpoints for a comprehensive understanding."
        )

    def ask(self, query: str, culture: str = "en", use_dense: bool = False, use_debate_style: bool = False) -> str:
        """
        Main method to get a response from the chatbot, applying various design patterns.

        Args:
            query (str): The user's input query.
            culture (str): The desired cultural context (e.g., "en", "es").
            use_dense (bool): Whether to use Demonstration Ensembling for the response.
            use_debate_style (bool): Whether to use Debate-Style Evidence Aggregation for complex queries.

        Returns:
            str: The chatbot's response.
        """
        final_response_content = ""

        if use_debate_style:
            # Apply Debate-Style Evidence Aggregation for complex queries.
            final_response_content = self._generate_debate_style_response(query)
        elif use_dense:
            # Pattern: Demonstration Ensembling
            dense_prompts = self._generate_dense_prompts(query)
            raw_responses = []
            for prompt_variation in dense_prompts:
                # Apply cultural awareness to each prompt variation before sending to LLM
                culturally_aware_prompt = self._apply_cultural_awareness_to_prompt(prompt_variation, culture)
                raw_responses.append(self.llm.generate(culturally_aware_prompt))
            final_response_content = self._aggregate_responses(raw_responses)
        else:
            # Standard few-shot prompting with Balanced Demonstrations and Cultural Awareness
            # A more sophisticated system would classify the 'topic' of the query first.
            topic = random.choice(list(self.balanced_demonstrations.keys())) # For demonstration, pick a random topic
            demonstrations = self._get_demonstrations(topic=topic, num_examples=2)
            
            base_prompt_body = (
                f"{demonstrations}"
                f"Customer Query: {query}\n"
                f"Agent:"
            )
            full_prompt = self._apply_cultural_awareness_to_prompt(base_prompt_body, culture)
            final_response_content = self.llm.generate(full_prompt)

        # Pattern: Bias-Aware Design & Mitigation (post-processing the final response)
        final_response_content = self._check_and_mitigate_bias(final_response_content)

        return final_response_content