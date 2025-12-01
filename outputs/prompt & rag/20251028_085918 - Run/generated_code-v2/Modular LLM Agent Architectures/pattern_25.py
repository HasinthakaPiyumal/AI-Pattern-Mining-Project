class MedKnowRetriever:
    """
    A plug-and-play module for retrieving and summarizing medical information
    from a curated knowledge base.
    """
    def __init__(self):
        # A dummy in-memory medical knowledge base for demonstration
        self._medical_knowledge_base = [
            "Aspirin is commonly used for pain relief, fever reduction, and anti-inflammatory purposes. It's also an antiplatelet agent.",
            "Metformin is a first-line medication for type 2 diabetes, primarily working by decreasing glucose production by the liver.",
            "Hypertension (high blood pressure) is a common condition where the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease.",
            "Symptoms of a common cold include runny nose, sore throat, cough, and congestion. It is caused by viruses.",
            "For acute pain, paracetamol (acetaminophen) is often recommended as a first-line treatment due to its good safety profile.",
            "Insulin is a hormone produced by the pancreas that helps regulate blood sugar. In type 1 diabetes, the body does not produce insulin.",
            "The recommended daily water intake varies but is generally around 8 glasses (2 liters) for adults.",
            "Exercise regularly for at least 30 minutes most days of the week to maintain cardiovascular health."
        ]

    def _get_keywords(self, text):
        """Extract simple keywords from text (for basic matching)."""
        return set(word.lower() for word in text.split() if len(word) > 2)

    def retrieve_info(self, query: str) -> list[str]:
        """
        Retrieves relevant medical information from the knowledge base based on the query.
        """
        query_keywords = self._get_keywords(query)
        relevant_snippets = []
        for snippet in self._medical_knowledge_base:
            snippet_keywords = self._get_keywords(snippet)
            if any(keyword in snippet_keywords for keyword in query_keywords):
                relevant_snippets.append(snippet)
        return relevant_snippets

    def summarize_info(self, retrieved_info: list[str]) -> str:
        """
        Generates a concise summary from the retrieved information.
        (Simplified for demonstration purposes)
        """
        if not retrieved_info:
            return "No specific medical information found relevant to this query."
        
        # In a real scenario, an NLP model would perform advanced summarization.
        # Here, we simply combine and add a introductory phrase.
        summary = "Based on medical knowledge: " + " ".join(retrieved_info)
        return summary

    def process_query(self, query: str) -> str:
        """
        Orchestrates the retrieval and summarization process.
        Returns the augmented context for an LLM.
        """
        retrieved_data = self.retrieve_info(query)
        summarized_context = self.summarize_info(retrieved_data)
        return summarized_context

class HealthcareLLM:
    """
    A mock Large Language Model representing a healthcare-specific LLM
    that can be augmented with external context.
    """
    def __init__(self, name="GenericHealthcareLLM"):
        self.name = name

    def generate_response(self, query: str, augmented_context: str) -> str:
        """
        Generates a response using the original query and augmented medical context.
        """
        print(f"\n--- {self.name} Processing ---")
        print(f"Original Query: '{query}'")
        print(f"Augmented Context Provided: '{augmented_context}'")

        # Simulate LLM's reasoning with the context
        if "No specific medical information found" in augmented_context:
            base_response = f"I'm unable to provide specific medical advice based on the information found, but generally, for '{query}', consult a healthcare professional."
            return base_response
        elif "diabetes" in query.lower() and "Metformin" in augmented_context:
             return f"For '{query}', {self.name} suggests: {augmented_context} Additionally, it's crucial to manage diet and exercise. Always consult a doctor for personalized advice."
        elif "pain relief" in query.lower() and "Aspirin" in augmented_context:
             return f"Regarding '{query}', {self.name} suggests: {augmented_context} Consider over-the-counter options, but always follow dosage instructions."
        elif "blood pressure" in query.lower() and "Hypertension" in augmented_context:
             return f"In response to '{query}', {self.name} states: {augmented_context} Regular monitoring and lifestyle changes are key for managing blood pressure."
        else:
            return f"For '{query}', {self.name} considers: {augmented_context}. My generated advice is to seek professional medical consultation for an accurate diagnosis and treatment plan."

# --- Demonstration of Plug-and-Play Integration ---
def main():
    # Initialize the MedKnowRetriever module
    med_retriever = MedKnowRetriever()

    # Initialize a mock Healthcare LLM
    healthcare_llm = HealthcareLLM(name="MediChatBot")

    print("--- Scenario 1: Query about diabetes medication ---")
    user_query_1 = "What is the primary medication for type 2 diabetes?"
    # The MedKnowRetriever processes the query
    context_1 = med_retriever.process_query(user_query_1)
    # The LLM generates a response using the retrieved context
    response_1 = healthcare_llm.generate_response(user_query_1, context_1)
    print(f"LLM Response: {response_1}")

    print("\n--- Scenario 2: Query about pain relief ---")
    user_query_2 = "What are common over-the-counter options for pain relief?"
    context_2 = med_retriever.process_query(user_query_2)
    response_2 = healthcare_llm.generate_response(user_query_2, context_2)
    print(f"LLM Response: {response_2}")

    print("\n--- Scenario 3: Query about a less specific topic (less direct match) ---")
    user_query_3 = "How can I stay healthy?"
    context_3 = med_retriever.process_query(user_query_3)
    response_3 = healthcare_llm.generate_response(user_query_3, context_3)
    print(f"LLM Response: {response_3}")

    print("\n--- Scenario 4: Query with no direct match in dummy KB ---")
    user_query_4 = "What is the recommended dosage for a specific experimental cancer drug?"
    context_4 = med_retriever.process_query(user_query_4)
    response_4 = healthcare_llm.generate_response(user_query_4, context_4)
    print(f"LLM Response: {response_4}")


if __name__ == "__main__":
    main()