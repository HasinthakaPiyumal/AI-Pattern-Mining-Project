import random

class LikertEvaluator:
    def __init__(self):
        self.likert_scale = {
            1: "Very Poor",
            2: "Poor",
            3: "Neutral",
            4: "Good",
            5: "Excellent"
        }

    def _simulate_llm_evaluation(self, customer_query: str, chatbot_response: str) -> tuple[int, str]:
        score = 3  # Start with Neutral
        explanation_parts = []

        # Heuristic for relevance (keyword matching)
        query_keywords = set(word.lower() for word in customer_query.split() if len(word) > 2)
        response_keywords = set(word.lower() for word in chatbot_response.split() if len(word) > 2)
        common_keywords = query_keywords.intersection(response_keywords)

        if len(common_keywords) >= 2:
            score += 1 # More relevant
            explanation_parts.append("Response contained relevant keywords.")
        elif not common_keywords:
            score -= 1 # Less relevant
            explanation_parts.append("Response lacked clear relevance to the query.")
        else:
            explanation_parts.append("Response showed some relevance.")

        # Heuristic for helpfulness (response length and presence of solutions)
        if len(chatbot_response) > 80 and ("solution" in chatbot_response.lower() or "steps" in chatbot_response.lower()):
            score += 1 # More helpful
            explanation_parts.append("Response was comprehensive and offered potential solutions.")
        elif len(chatbot_response) < 30:
            score -= 1 # Less helpful
            explanation_parts.append("Response was too brief to be helpful.")
        else:
            explanation_parts.append("Response was adequately helpful.")

        # Heuristic for tone (simple sentiment approximation)
        positive_words = ["great", "excellent", "happy", "resolve", "assist"]
        negative_words = ["sorry", "unable", "difficult", "issue"]

        response_lower = chatbot_response.lower()
        positive_count = sum(response_lower.count(word) for word in positive_words)
        negative_count = sum(response_lower.count(word) for word in negative_words)

        if positive_count > negative_count:
            score += 1 # Positive tone
            explanation_parts.append("Tone was generally positive.")
        elif negative_count > positive_count:
            score -= 1 # Negative tone
            explanation_parts.append("Tone leaned slightly negative.")
        else:
            explanation_parts.append("Tone was neutral.")

        # Clamp score between 1 and 5
        score = max(1, min(5, score))

        final_explanation = " ".join(explanation_parts) if explanation_parts else "No specific evaluation points identified."
        return score, final_explanation

    def evaluate_response(self, customer_query: str, chatbot_response: str) -> dict:
        numerical_score, explanation = self._simulate_llm_evaluation(customer_query, chatbot_response)
        likert_score = self.likert_scale.get(numerical_score, "Unknown")
        return {
            "likert_score": likert_score,
            "numerical_score": numerical_score,
            "explanation": explanation
        }

if __name__ == "__main__":
    evaluator = LikertEvaluator()

    sample_interactions = [
        {
            "query": "My internet is not working. I need help.",
            "response": "I apologize for the inconvenience. Please try restarting your router and modem. If the issue persists, contact our technical support at 1-800-555-1234. We are happy to assist you further."
        },
        {
            "query": "How do I change my billing address?",
            "response": "You can change your billing address in your account settings."
        },
        {
            "query": "I want to know about your premium plan features.",
            "response": "Our premium plan offers unlimited storage, priority support, and advanced analytics. It's a great value for our users and provides excellent features to enhance your experience."
        },
        {
            "query": "My order is late, what should I do?",
            "response": "Order status can be checked online. We are unable to provide exact delivery dates."
        },
        {
            "query": "Can I get a refund for my subscription?",
            "response": "Refunds are processed according to our terms and conditions. Please refer to our policy for more details."
        }
    ]

    print("--- Chatbot Response Evaluation ---")
    for i, interaction in enumerate(sample_interactions):
        query = interaction["query"]
        response = interaction["response"]
        evaluation_result = evaluator.evaluate_response(query, response)

        print(f"\nInteraction {i+1}:")
        print(f"  Customer Query: {query}")
        print(f"  Chatbot Response: {response}")
        print(f"  Evaluation: {evaluation_result['likert_score']} ({evaluation_result['numerical_score']}/5)")
        print(f"  Explanation: {evaluation_result['explanation']}")
    print("\n--- End of Evaluation ---")