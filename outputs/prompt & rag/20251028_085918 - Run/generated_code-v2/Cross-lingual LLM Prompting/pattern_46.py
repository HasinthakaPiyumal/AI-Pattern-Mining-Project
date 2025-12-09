class MultilingualLLMSimulator:
    def __init__(self):
        pass

    def generate_response(self, prompt: str) -> str:
        # Simulate LLM behavior. In a real scenario, this would call a multilingual LLM API.
        # This simulation tries to give a plausible answer based on keywords in the prompt
        # and the presence of cross-lingual examples.
        if "return policy" in prompt.lower() or "política de devoluciones" in prompt.lower():
            if "english" in prompt.lower():
                return "According to cross-lingual understanding, our return policy allows returns within 30 days of purchase."
            elif "spanish" in prompt.lower():
                return "Según la comprensión interlingüe, nuestra política de devoluciones permite devoluciones dentro de los 30 días de la compra."
        elif "track my order" in prompt.lower() or "rastrear mi pedido" in prompt.lower():
            if "english" in prompt.lower():
                return "Leveraging cross-lingual examples, you can track your order using the tracking number in your shipping confirmation."
            elif "spanish" in prompt.lower():
                return "Aprovechando ejemplos interlingües, puede rastrear su pedido utilizando el número de seguimiento en su confirmación de envío."
        elif "delivery time" in prompt.lower() or "tiempo de entrega" in prompt.lower():
            if "english" in prompt.lower():
                return "Based on cross-lingual context, standard delivery takes 5-7 business days."
            elif "spanish" in prompt.lower():
                return "Basado en el contexto interlingüe, la entrega estándar tarda de 5 a 7 días hábiles."
        return "I understand your query with the help of cross-lingual examples and am processing your request. Please provide more details if needed."


class PromptEngineeringModule:
    def __init__(self, in_context_examples: list):
        self.in_context_examples = in_context_examples

    def construct_prompt(self, customer_query: str, query_language: str) -> str:
        prompt_parts = ["Here are some examples of customer queries and their answers in both English and Spanish to help me understand and respond effectively:"]

        for example in self.in_context_examples:
            prompt_parts.append(f"\nEnglish Query: {example['en_query']}")
            prompt_parts.append(f"English Answer: {example['en_answer']}")
            prompt_parts.append(f"Spanish Query: {example['es_query']}")
            prompt_parts.append(f"Spanish Answer: {example['es_answer']}")

        prompt_parts.append(f"\nCustomer Query ({query_language.capitalize()}): {customer_query}")
        prompt_parts.append(f"Response in {query_language.capitalize()}:")

        return "\n".join(prompt_parts)


class ChatbotInterface:
    def __init__(self, llm_simulator: MultilingualLLMSimulator, prompt_engineer: PromptEngineeringModule):
        self.llm_simulator = llm_simulator
        self.prompt_engineer = prompt_engineer

    def get_language(self, text: str) -> str:
        # Simple language detection for demonstration (can be replaced with a proper library)
        # For simplicity, if it contains common Spanish words, assume Spanish, else English.
        spanish_keywords = ["qué", "cómo", "dónde", "cuándo", "por qué", "el", "la", "los", "las", "un", "una", "mi", "su", "usted", "gracias", "hola"]
        if any(keyword in text.lower() for keyword in spanish_keywords):
            return "spanish"
        return "english"

    def chat(self, customer_query: str) -> str:
        query_language = self.get_language(customer_query)
        prompt = self.prompt_engineer.construct_prompt(customer_query, query_language)
        print(f"\n--- PROMPT SENT TO LLM ---\n{prompt}\n---\n") # For demonstration
        response = self.llm_simulator.generate_response(prompt)
        return response


# --- In-Context Learning Examples Database/Store ---
in_context_examples = [
    {
        "en_query": "What is your return policy?",
        "en_answer": "Our return policy allows returns within 30 days of purchase.",
        "es_query": "¿Cuál es su política de devoluciones?",
        "es_answer": "Nuestra política de devoluciones permite devoluciones dentro de los 30 días de la compra."
    },
    {
        "en_query": "How do I track my order?",
        "en_answer": "You can track your order using the tracking number provided in your shipping confirmation email.",
        "es_query": "¿Cómo rastreo mi pedido?",
        "es_answer": "Puede rastrear su pedido utilizando el número de seguimiento proporcionado en su correo electrónico de confirmación de envío."
    },
    {
        "en_query": "What is the estimated delivery time?",
        "en_answer": "Standard delivery typically takes 5-7 business days.",
        "es_query": "¿Cuál es el tiempo estimado de entrega?",
        "es_answer": "La entrega estándar suele tardar de 5 a 7 días hábiles."
    }
]

# --- Main Application Flow ---
if __name__ == "__main__":
    llm_simulator = MultilingualLLMSimulator()
    prompt_engineer = PromptEngineeringModule(in_context_examples)
    chatbot = ChatbotInterface(llm_simulator, prompt_engineer)

    print("Welcome to the Multilingual Customer Support Chatbot! (Type 'exit' to quit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break

        response = chatbot.chat(user_input)
        print(f"Chatbot: {response}")