# This script demonstrates a basic multilingual customer support chatbot utilizing InCLT Crosslingual Transfer Prompting.
# It constructs a prompt with both English and Spanish in-context examples to help a simulated LLM generate better cross-lingual responses.

# --- In-Context Examples for Cross-lingual Transfer (InCLT) ---
# These examples contain both source (English) and target (Spanish) language queries
# along with the desired Spanish response.
in_context_examples = [
    {
        "english_query": "How do I reset my password?",
        "spanish_query": "¿Cómo restablezco mi contraseña?",
        "spanish_response": "Para restablecer su contraseña, vaya a la sección de 'Configuración de cuenta' y haga clic en 'Restablecer contraseña'."
    },
    {
        "english_query": "Where can I find my order history?",
        "spanish_query": "¿Dónde puedo encontrar mi historial de pedidos?",
        "spanish_response": "Puede encontrar su historial de pedidos en la sección 'Mis pedidos' de su perfil."
    },
    {
        "english_query": "My payment failed, but the funds were deducted. What should I do?",
        "spanish_query": "Mi pago falló, pero se dedujo el dinero. ¿Qué debo hacer?",
        "spanish_response": "Lamentamos el inconveniente. Por favor, espere 24 horas para que el sistema se actualice. Si el problema persiste, contacte a nuestro soporte con el ID de su transacción."
    },
    {
        "english_query": "I want to return an item, what's the policy?",
        "spanish_query": "Quiero devolver un artículo, ¿cuál es la política?",
        "spanish_response": "Nuestra política de devoluciones permite cambios o reembolsos dentro de los 30 días posteriores a la compra, siempre que el artículo esté en su estado original y con el recibo."
    }
]

def create_inclt_prompt(customer_query_spanish: str) -> str:
    """
    Constructs a prompt incorporating InCLT examples for a multilingual LLM.
    The prompt includes both English source and Spanish target examples.
    """
    prompt = "You are a helpful multilingual customer support assistant. Your goal is to provide accurate and culturally appropriate responses in Spanish.\n"
    prompt += "Below are examples of customer queries in English and Spanish, and the appropriate Spanish responses. Use these examples to guide your response to the new customer query."
    prompt += "\n\n-- In-Context Learning Examples --\n"

    for example in in_context_examples:
        prompt += f"Example English Query: {example['english_query']}\n"
        prompt += f"Example Spanish Query: {example['spanish_query']}\n"
        prompt += f"Example Spanish Response: {example['spanish_response']}\n\n"

    prompt += "-- New Customer Query --\n"
    prompt += f"Customer Query (Spanish): {customer_query_spanish}\n"
    prompt += "Response (Spanish):"
    return prompt

def simulate_llm_response(prompt: str) -> str:
    """
    Simulates an LLM's response based on the prompt.
    In a real application, this would involve an API call to a multilingual LLM.
    For this demonstration, it provides a hardcoded response based on keywords
    to illustrate the expected behavior of cross-lingual transfer.
    """
    # Simple keyword-based simulation for demonstration purposes
    if "¿Mi pago falló, pero se dedujo el dinero" in prompt:
        return "Entiendo su preocupación por el pago fallido. Por favor, espere un momento. Si el cargo aparece en su extracto bancario pero el pedido no se procesó, le recomendamos contactar a su banco primero y luego a nuestro equipo de soporte con el comprobante de la transacción para una investigación más profunda."
    elif "¿Cómo restablezco mi contraseña?" in prompt:
        return "Para restablecer su contraseña, diríjase a la sección de 'Configuración de cuenta' y busque la opción 'Cambiar/Restablecer contraseña'. Siga los pasos indicados allí."
    elif "¿Dónde puedo encontrar mi historial de pedidos?" in prompt:
        return "Su historial de pedidos completo está disponible en la sección 'Mis Pedidos' dentro de su perfil de usuario en nuestro sitio web/aplicación."
    elif "¿Mi pedido está retrasado" in prompt:
        return "Lamento escuchar que su pedido está retrasado. Para poder ayudarle, ¿podría proporcionarme el número de su pedido? Así podré verificar el estado y la fecha de entrega estimada."
    elif "Quiero devolver un artículo" in prompt:
        return "Nuestra política de devoluciones permite la devolución de artículos dentro de los 30 días posteriores a la compra, siempre y cuando el artículo esté sin usar y en su embalaje original. Por favor, tenga a mano su recibo o comprobante de compra."
    else:
        return "Gracias por contactarnos. Estoy aquí para ayudarle con su consulta. Por favor, especifique cómo puedo asistirle hoy."

def multilingual_chatbot(customer_query: str):
    """
    Runs the multilingual chatbot with a given customer query.
    """
    print(f"\n--- Customer Query: {customer_query} ---")
    
    # 1. Create the InCLT prompt
    inclt_prompt = create_inclt_prompt(customer_query)
    
    print("\n--- Generated InCLT Prompt (Partial View) ---")
    # Print only a part of the prompt to keep output concise, but show the core elements
    print(inclt_prompt[:500] + "\n... [Truncated for brevity] ...\n" + inclt_prompt[-200:])
    print("------------------------------------------\n")

    # 2. Simulate LLM response using the generated prompt
    llm_response = simulate_llm_response(inclt_prompt)
    
    print("--- Chatbot Response (Spanish) ---")
    print(llm_response)
    print("----------------------------------\n")

if __name__ == "__main__":
    print("\n### Multilingual Customer Support Chatbot (InCLT Crosslingual Transfer Prompting Demo) ###")
    print("This demo illustrates how prompts are constructed using both source (English) and target (Spanish) \nlanguage examples to enhance a simulated LLM's cross-lingual understanding.")
    
    # Test with various customer queries in Spanish
    multilingual_chatbot("¿Mi pedido está retrasado, qué puedo hacer?")
    
    print("\n" + "="*80 + "\n")

    multilingual_chatbot("Mi pago falló, pero se dedujo el dinero. ¿Qué debo hacer?")

    print("\n" + "="*80 + "\n")

    multilingual_chatbot("Quiero devolver un artículo, ¿cuál es la política?")

    print("\n" + "="*80 + "\n")

    multilingual_chatbot("¿Cómo restablezco mi contraseña?")

    print("\n" + "="*80 + "\n")

    multilingual_chatbot("¿Dónde puedo encontrar mi historial de pedidos?")

    print("\n" + "="*80 + "\n")

    multilingual_chatbot("Tengo un problema con la entrega de mi paquete.")