import gradio as gr
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0  # for reproducibility

# 1. In-Context Examples (Data Structure)
in_context_examples = [
    {
        "id": 1,
        "en_query": "My order hasn't arrived yet.",
        "en_response": "Please provide your order number, and I will check the status for you.",
        "es_query": "Mi pedido aún no ha llegado.",
        "es_response": "Por favor, proporcione su número de pedido y verificaré el estado por usted."
    },
    {
        "id": 2,
        "en_query": "How can I return an item?",
        "en_response": "You can initiate a return through your account's order history section.",
        "es_query": "¿Cómo puedo devolver un artículo?",
        "es_response": "Puede iniciar una devolución a través de la sección de historial de pedidos de su cuenta."
    },
    {
        "id": 3,
        "en_query": "I need help with my payment.",
        "en_response": "Could you please specify the issue you are facing with the payment?",
        "es_query": "Necesito ayuda con mi pago.",
        "es_response": "¿Podría especificar el problema que está experimentando con el pago?"
    }
]

# 2. Language Detection Module
def detect_language(text: str) -> str:
    try:
        return detect(text)
    except:
        return "en" # Default to English if detection fails

# 3. InCLT Prompt Constructor Function
def construct_icl_prompt(query: str, detected_language: str, examples: list) -> str:
    prompt_parts = []
    for example in examples:
        # Always include both source and target language versions in the prompt
        prompt_parts.append(f"English Query: {example['en_query']}\nEnglish Response: {example['en_response']}")
        prompt_parts.append(f"Spanish Query: {example['es_query']}\nSpanish Response: {example['es_response']}")
    
    prompt_parts.append(f"\nUser Query in {detected_language.upper()}: {query}\nAssistant Response in {detected_language.upper()}:")
    return "\n".join(prompt_parts)

# 4. Multilingual LLM (Simulated for demonstration)
def generate_multilingual_response_mock(prompt: str, target_language: str) -> str:
    # This is a very basic mock. In a real scenario, a powerful LLM would process the prompt.
    # The mock tries to give a relevant response based on keywords and target language.
    
    if target_language == "es":
        if "pedido" in prompt.lower() or "llegado" in prompt.lower():
            return "Por favor, espere un momento mientras verifico su pedido. ¿Puede proporcionar su número de pedido?"
        elif "devolver" in prompt.lower() or "artículo" in prompt.lower():
            return "Para devoluciones, visite la sección de devoluciones en su cuenta. ¡Es fácil!"
        elif "pago" in prompt.lower() or "ayuda" in prompt.lower():
            return "Para asistencia con el pago, por favor, detalle el problema. Nuestro equipo está listo para ayudar."
        else:
            return "Entiendo. ¿En qué más puedo ayudarle hoy?"
    else: # Default to English
        if "order" in prompt.lower() or "arrived" in prompt.lower():
            return "Please hold on while I check your order status. Can you provide your order number?"
        elif "return" in prompt.lower() or "item" in prompt.lower():
            return "For returns, please visit the returns section in your account. It's straightforward!"
        elif "payment" in prompt.lower() or "help" in prompt.lower():
            return "For payment assistance, please elaborate on the issue. Our team is ready to help."
        else:
            return "I understand. How else may I assist you today?"

# Main chatbot logic
def chatbot_interface(user_query: str) -> str:
    detected_language = detect_language(user_query)
    prompt = construct_icl_prompt(user_query, detected_language, in_context_examples)
    print(f"\n--- Constructed Prompt ---\n{prompt}\n--------------------------") # For debugging and observing the prompt
    response = generate_multilingual_response_mock(prompt, detected_language)
    return response

# 5. Gradio User Interface
if __name__ == "__main__":
    demo = gr.Interface(
        fn=chatbot_interface,
        inputs=gr.Textbox(lines=2, placeholder="Enter your query here in any language..."),
        outputs="text",
        title="Multilingual Customer Support Chatbot (InCLT Prompting)",
        description="This chatbot uses In-Context Learning (InCLT) with both source and target language examples to enhance cross-lingual understanding. Try asking questions in English or Spanish!"
    )
    demo.launch()