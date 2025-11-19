
from langchain.prompts import PromptTemplate

# 1. In-Context Examples Store
cross_lingual_examples = {
    "spanish": [
        ("What is your return policy?", "¿Cuál es su política de devoluciones?", "Nuestra política de devoluciones permite cambios o reembolsos dentro de los 30 días posteriores a la compra con el recibo original."),
        ("How can I track my order?", "¿Cómo puedo rastrear mi pedido?", "Puede rastrear su pedido iniciando sesión en su cuenta y visitando la sección 'Mis Pedidos'."),
    ],
    "french": [
        ("What is your return policy?", "Quelle est votre politique de retour ?", "Notre politique de retour permet les échanges ou les remboursements dans les 30 jours suivant l'achat avec le reçu original."),
        ("Do you offer international shipping?", "Proposez-vous la livraison internationale ?", "Oui, nous offrons la livraison internationale vers la plupart des pays. Les frais peuvent varier."),
    ]
}

# 2. LLM Integration (Simulated)
class SimulatedLLM:
    def predict(self, prompt: str) -> str:
        # In a real scenario, this would call a multilingual LLM API or local model.
        # For simulation, we'll try to give a somewhat relevant response based on the prompt content.
        if "¿Cuál es su política de devoluciones?" in prompt and "spanish" in prompt.lower():
            return "Nuestra política de devoluciones permite cambios o reembolsos dentro de los 30 días posteriores a la compra con el recibo original."
        elif "¿Cómo puedo rastrear mi pedido?" in prompt and "spanish" in prompt.lower():
            return "Puede rastrear su pedido iniciando sesión en su cuenta y visitando la sección 'Mis Pedidos'."
        elif "Quelle est votre politique de retour ?" in prompt and "french" in prompt.lower():
            return "Notre politique de retour permet les échanges ou les remboursements dans les 30 jours suivant l'achat avec le reçu original."
        elif "Proposez-vous la livraison internationale ?" in prompt and "french" in prompt.lower():
            return "Oui, nous offrons la livraison internationale vers la plupart des pays. Les frais peuvent varier."
        elif "hello" in prompt.lower() or "hola" in prompt.lower() or "bonjour" in prompt.lower():
            return "Hello! How can I help you today? ¡Hola! ¿Cómo puedo ayudarte hoy? Bonjour! Comment puis-je vous aider aujourd's'hui?"
        return "I'm sorry, I don't have a specific answer for that in the simulated environment. But I understand you're asking about your query in " + prompt.split("\n")[-1].split(": ")[-1]

def load_multilingual_llm():
    return SimulatedLLM()

# 3. Prompt Generator
def generate_inclt_prompt(user_query: str, target_language: str) -> str:
    examples = cross_lingual_examples.get(target_language.lower(), [])

    template_parts = [
        "You are a helpful multilingual customer support assistant. Provide accurate and concise answers in the target language.",
        "Below are examples of customer queries in English and their corresponding answers in the target language. Use these examples for in-context learning."
    ]

    for eng_q, target_q, target_a in examples:
        template_parts.append(f"English Query: {eng_q}")
        template_parts.append(f"Target Language Query ({target_language.capitalize()}): {target_q}")
        template_parts.append(f"Target Language Answer ({target_language.capitalize()}): {target_a}\n")

    template_parts.append(f"New User Query ({target_language.capitalize()}): {{user_query}}")
    template_parts.append(f"Target Language Answer ({target_language.capitalize()}):")

    full_template = "\n".join(template_parts)
    prompt = PromptTemplate(template=full_template, input_variables=["user_query"])
    return prompt.format(user_query=user_query)

# 4. Chatbot Core Logic
def chatbot_response(user_query: str, target_language: str, llm: SimulatedLLM) -> str:
    inclt_prompt = generate_inclt_prompt(user_query, target_language)
    print(f"\n--- Generated InCLT Prompt for '{user_query}' in {target_language.capitalize()} ---")
    print(inclt_prompt)
    print("------------------------------------------------------------------------")
    simulated_response = llm.predict(inclt_prompt)
    return simulated_response

# 5. Main Execution
if __name__ == "__main__":
    print("Loading simulated multilingual LLM...")
    llm_model = load_multilingual_llm()
    print("LLM loaded.\n")

    test_queries = [
        ("¿Cuál es mi estado de pedido?", "spanish"),
        ("¿Qué pasa con mi entrega?", "spanish"),
        ("Je voudrais savoir ma politique de retour.", "french"),
        ("Bonjour, avez-vous des offres spéciales ?", "french"),
        ("Hello, where is my order?", "english") # Example for a language without specific InCLT examples
    ]

    for query, lang in test_queries:
        print(f"\nUser: {query} (Language: {lang.capitalize()})")
        response = chatbot_response(query, lang, llm_model)
        print(f"Chatbot: {response}")
        print("========================================================================")

