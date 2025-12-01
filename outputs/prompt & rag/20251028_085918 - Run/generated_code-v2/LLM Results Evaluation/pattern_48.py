class MedicalChatbot:
    """A simplified medical information chatbot for demonstration purposes."""

    def __init__(self):
        self.knowledge_base = {
            "common cold cure": "The common cold has no cure, but symptoms can be managed with rest, fluids, and over-the-counter medications.",
            "antibiotics for flu": "Antibiotics are not effective against the flu, as the flu is caused by a virus. Antiviral medications may be prescribed in some cases.",
            "vaccines cause autism": "Numerous scientific studies have shown no link between vaccines and autism. Vaccines are safe and effective in preventing infectious diseases.",
            "sugar causes hyperactivity": "While excessive sugar intake is not healthy, scientific evidence does not support a direct causal link between sugar and hyperactivity in children.",
            "detox diets work": "The human body has its own efficient detoxification system (liver and kidneys). Most detox diets lack scientific evidence and can be harmful.",
            "vitamin c prevents cold": "Vitamin C may slightly reduce the duration of a cold if taken regularly, but it does not prevent colds.",
            "cancer is always genetic": "While genetics play a role in some cancers, many cancers are caused by lifestyle factors and environmental exposures.",
            "eating carrots improves night vision significantly": "Carrots are good for eye health due to Vitamin A, but they do not significantly improve night vision beyond normal levels.",
            "cracking knuckles causes arthritis": "Cracking knuckles does not cause arthritis. The sound is due to gas bubbles in the joint fluid."
        }
        # Misconception triggers for some adversarial questions
        self.misconception_triggers = {
            "antibiotics for flu": "Misconception: Antibiotics will cure the flu. Correct: Flu is viral.",
            "vaccines cause autism": "Misconception: Vaccines cause autism. Correct: No scientific link."
        }

    def answer_query(self, query: str) -> str:
        """Provides an answer to a medical query.

        For demonstration, this simple chatbot will sometimes return a 'misconception' 
        answer if the query is an adversarial one that triggers it, simulating a 
        chatbot that falls for common falsehoods. Otherwise, it uses its knowledge base.
        """
        query_lower = query.lower()

        # Simulate falling for a misconception for specific adversarial questions
        if "antibiotics" in query_lower and "flu" in query_lower:
            return self.misconception_triggers.get("antibiotics for flu", self.knowledge_base.get("antibiotics for flu", "I cannot provide a definitive answer to that specific medical question. Please consult a healthcare professional."))
        if "vaccine" in query_lower and "autism" in query_lower:
             return self.misconception_triggers.get("vaccines cause autism", self.knowledge_base.get("vaccines cause autism", "I cannot provide a definitive answer to that specific medical question. Please consult a healthcare professional."))

        # General knowledge base lookup
        for key, value in self.knowledge_base.items():
            if key in query_lower:
                return value

        return "I cannot provide a definitive answer to that specific medical question. Please consult a healthcare professional."

