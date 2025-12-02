class InCLTPromptBuilder:
    def build_prompt(
        self,
        user_query: str,
        retrieved_english_faq: str,
        retrieved_spanish_faq: str,
    ) -> str:
        """
        Constructs an InCLT (In-Context Learning Transfer) prompt.
        This prompt leverages both source (English) and target (Spanish) language examples
        to guide the multilingual LLM for cross-lingual tasks.
        """
        prompt_template = f"""You are a helpful multilingual customer support assistant.
I will provide you with a customer's question in Spanish and relevant information about our products/services in both English and Spanish.
Your task is to answer the customer's question in Spanish, using the provided information.

--- In-Context Example ---
English FAQ: What is the estimated delivery time? Delivery usually takes 3-5 business days.
Spanish FAQ: ¿Cuál es el tiempo estimado de entrega? La entrega suele tardar de 3 a 5 días hábiles.
Customer Question: ¿Cuánto tarda en llegar mi pedido?
Assistant Answer: El tiempo de entrega estimado es de 3 a 5 días hábiles. (Estimated delivery time is 3 to 5 business days.)

--- New Customer Query ---
English FAQ: {retrieved_english_faq}
Spanish FAQ: {retrieved_spanish_faq}
Customer Question: {user_query}
Assistant Answer:"""
        return prompt_template.format(
            user_query=user_query,
            retrieved_english_faq=retrieved_english_faq,
            retrieved_spanish_faq=retrieved_spanish_faq,
        )
