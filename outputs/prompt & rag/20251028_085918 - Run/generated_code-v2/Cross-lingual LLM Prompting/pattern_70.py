class MockLLM:
    def __init__(self):
        pass

    def generate_response(self, prompt: str) -> str:
        if "returns policy" in prompt.lower() or "politica de devoluciones" in prompt.lower():
            return "According to our returns policy, you can return items within 30 days with a valid receipt. (Spanish: Según nuestra política de devoluciones, puedes devolver artículos dentro de los 30 días con un recibo válido.)"
        elif "shipping information" in prompt.lower() or "información de envío" in prompt.lower():
            return "Shipping usually takes 3-5 business days. You will receive a tracking number via email. (Spanish: El envío suele tardar de 3 a 5 días hábiles. Recibirás un número de seguimiento por correo electrónico.)"
        elif "password reset" in prompt.lower() or "restablecer contraseña" in prompt.lower():
            return "To reset your password, please visit our website and click on 'Forgot Password'. (Spanish: Para restablecer tu contraseña, visita nuestro sitio web y haz clic en 'Olvidé mi contraseña'.)"
        else:
            return "I'm sorry, I don't have enough information to answer that question. Can you please rephrase or ask something else? (Spanish: Lo siento, no tengo suficiente información para responder a esa pregunta. ¿Puedes reformular o preguntar algo más?)"

class PromptConstructor:
    def __init__(self, knowledge_base: dict):
        self.knowledge_base = knowledge_base

    def build_prompt(self, user_query: str, target_language_code: str = "es") -> str:
        prompt_parts = []

        # Simulate intent detection and retrieve relevant examples
        relevant_examples = []
        if "returns" in user_query.lower() or "devoluciones" in user_query.lower():
            relevant_examples.append(self.knowledge_base["returns_policy"])
        elif "shipping" in user_query.lower() or "envío" in user_query.lower():
            relevant_examples.append(self.knowledge_base["shipping_info"])
        elif "password" in user_query.lower() or "contraseña" in user_query.lower():
            relevant_examples.append(self.knowledge_base["password_reset"])
        
        if not relevant_examples:
            # Add some general examples if no specific intent is matched
            relevant_examples.append(self.knowledge_base["returns_policy"])

        for example in relevant_examples:
            prompt_parts.append(f"Example - English Query: {example['en_query']}")
            prompt_parts.append(f"Example - English Answer: {example['en_answer']}")
            prompt_parts.append(f"Example - {target_language_code.upper()} Query: {example[f'{target_language_code}_query']}")
            prompt_parts.append(f"Example - {target_language_code.upper()} Answer: {example[f'{target_language_code}_answer']}")
            prompt_parts.append("\n---\n")

        prompt_parts.append(f"User Query ({target_language_code.upper()}): {user_query}\n")
        prompt_parts.append(f"Chatbot Answer ({target_language_code.upper()}):")

        return "\n".join(prompt_parts)

# Knowledge Base of Cross-lingual Examples
KNOWLEDGE_BASE = {
    "returns_policy": {
        "en_query": "What is your returns policy?",
        "en_answer": "You can return items within 30 days of purchase with a valid receipt for a full refund.",
        "es_query": "¿Cuál es su política de devoluciones?",
        "es_answer": "Puede devolver artículos dentro de los 30 días posteriores a la compra con un recibo válido para un reembolso completo."
    },
    "shipping_info": {
        "en_query": "How long does shipping take?",
        "en_answer": "Standard shipping typically takes 3-5 business days. Expedited options are also available.",
        "es_query": "¿Cuánto tarda el envío?",
        "es_answer": "El envío estándar generalmente toma de 3 a 5 días hábiles. También hay opciones de envío acelerado disponibles."
    },
    "password_reset": {
        "en_query": "How can I reset my password?",
        "en_answer": "To reset your password, go to the login page and click 'Forgot Password'. Follow the instructions sent to your email.",
        "es_query": "¿Cómo puedo restablecer mi contraseña?",
        "es_answer": "Para restablecer su contraseña, vaya a la página de inicio de sesión y haga clic en 'Olvidé mi contraseña'. Siga las instrucciones enviadas a su correo electrónico."
    }
}

def main():
    llm = MockLLM()
    prompt_constructor = PromptConstructor(KNOWLEDGE_BASE)

    print("Welcome to the Multilingual Customer Support Chatbot!")
    print("Type 'exit' to end the conversation.")
    print("Please specify the target language (e.g., 'es' for Spanish). For this demo, we assume Spanish (es).")
    target_lang = "es" # For simplicity, hardcoding target language as per architecture notes.

    while True:
        user_input = input(f"\nYou ({target_lang.upper()}): ")
        if user_input.lower() == 'exit':
            print("Thank you for chatting! Goodbye.")
            break

        # Build the prompt using the InCLT Crosslingual Transfer Prompting pattern
        full_prompt = prompt_constructor.build_prompt(user_input, target_lang)
        
        # Get response from the (mock) LLM
        chatbot_response = llm.generate_response(full_prompt)
        
        print(f"Chatbot ({target_lang.upper()}): {chatbot_response}")

if __name__ == "__main__":
    main()