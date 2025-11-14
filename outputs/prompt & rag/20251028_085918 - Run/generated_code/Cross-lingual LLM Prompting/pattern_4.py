from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn
from typing import Dict, Any

# --- 1. In-Context Examples Data Store ---
# Curated examples including source (English) and target language translations
in_context_examples: Dict[str, Dict[str, str]] = {
    "order_status_example": {
        "en_query": "My order #12345 has not arrived yet. What is the status?",
        "en_response": "I see that your order #12345 is currently in transit and expected to arrive by [Date].",
        "es_query": "Mi pedido #12345 aún no ha llegado. ¿Cuál es el estado?",
        "es_response": "Veo que su pedido #12345 está actualmente en tránsito y se espera que llegue antes del [Fecha].",
        "fr_query": "Ma commande #12345 n'est pas encore arrivée. Quel est le statut ?",
        "fr_response": "Je vois que votre commande #12345 est actuellement en transit et devrait arriver avant le [Date].",
        "de_query": "Meine Bestellung #12345 ist noch nicht angekommen. Was ist der Status?",
        "de_response": "Ich sehe, dass Ihre Bestellung #12345 derzeit unterwegs ist und voraussichtlich bis zum [Datum] ankommt.",
    },
    "change_address_example": {
        "en_query": "I want to change my shipping address for my latest order.",
        "en_response": "Please provide your order number and the new shipping address. I will update it for you.",
        "es_query": "Quiero cambiar mi dirección de envío para mi último pedido.",
        "es_response": "Por favor, proporcione su número de pedido y la nueva dirección de envío. Lo actualizaré por usted.",
        "fr_query": "Je souhaite modifier mon adresse de livraison pour ma dernière commande.",
        "fr_response": "Veuillez fournir votre numéro de commande et la nouvelle adresse de livraison. Je la mettrai à jour pour vous.",
        "de_query": "Ich möchte meine Lieferadresse für meine letzte Bestellung ändern.",
        "de_response": "Bitte geben Sie Ihre Bestellnummer und die neue Lieferadresse an. Ich werde sie für Sie aktualisieren.",
    },
}

# --- 2. InCLT Prompt Generator Module ---
def generate_inclt_prompt(
    customer_query: str, target_language: str
) -> str:
    """
    Generates a prompt using the InCLT Crosslingual Transfer pattern.
    It includes in-context examples in both English (source) and the target language.
    """
    prompt_parts = []

    # System instructions
    prompt_parts.append(
        "You are a helpful multilingual customer support assistant. Your goal is to provide accurate and helpful responses in the customer's requested language, leveraging examples provided in both English and the target language to understand the context.\n\n"
    )

    prompt_parts.append("Here are some examples of customer queries and their resolutions:\n\n")

    # Add in-context examples
    for example_key, example_data in in_context_examples.items():
        lang_code = target_language.lower()

        # Fallback to English if target language examples are not available for this specific example
        query_target_lang = example_data.get(f"{lang_code}_query", example_data["en_query"])
        response_target_lang = example_data.get(f"{lang_code}_response", example_data["en_response"])

        prompt_parts.append(f"--- Example: {example_key} ---")
        prompt_parts.append(f"English Query: '{example_data["en_query"]}'")
        prompt_parts.append(f"English Response: '{example_data["en_response"]}'")
        prompt_parts.append(f"{target_language.capitalize()} Query: '{query_target_lang}'")
        prompt_parts.append(f"{target_language.capitalize()} Response: '{response_target_lang}'\n")

    prompt_parts.append("--- End of Examples ---\n\n")

    # Add the current customer's query
    prompt_parts.append(
        f"Now, please answer the following customer query in {target_language.capitalize()}:\n"
    )
    prompt_parts.append(f"Customer Query: '{customer_query}'\n")
    prompt_parts.append(f"Response in {target_language.capitalize()}:")

    return "\n".join(prompt_parts)


# --- 3. Multilingual LLM Service (Conceptual/Placeholder) ---
def get_llm_response(prompt: str, target_language: str) -> str:
    """
    Conceptual function to simulate an LLM's response.
    In a real application, this would call an actual LLM API.
    """
    print(f"\n[MOCK LLM] Received prompt for {target_language}:\n{prompt}\n")
    
    # Simple mock logic based on keywords and target language
    if "order status" in prompt.lower() or "estado" in prompt.lower() or "statut" in prompt.lower() or "status" in prompt.lower():
        if target_language.lower() == "es":
            return "Su pedido #12345 está en camino y debería llegar pronto."
        elif target_language.lower() == "fr":
            return "Votre commande #12345 est en route et devrait arriver bientôt."
        elif target_language.lower() == "de":
            return "Ihre Bestellung #12345 ist unterwegs und sollte bald ankommen."
        else:
            return "Your order #12345 is on its way and should arrive soon."
    elif "shipping address" in prompt.lower() or "dirección de envío" in prompt.lower() or "adresse de livraison" in prompt.lower() or "lieferadresse" in prompt.lower():
        if target_language.lower() == "es":
            return "Para cambiar su dirección de envío, por favor, proporcione el nuevo domicilio y el número de pedido."
        elif target_language.lower() == "fr":
            return "Pour changer votre adresse de livraison, veuillez fournir la nouvelle adresse et le numéro de commande."
        elif target_language.lower() == "de":
            return "Um Ihre Lieferadresse zu ändern, geben Sie bitte die neue Adresse und die Bestellnummer an."
        else:
            return "To change your shipping address, please provide the new address and order number."
    else:
        if target_language.lower() == "es":
            return "Gracias por contactarnos. ¿En qué más podemos ayudarle?"
        elif target_language.lower() == "fr":
            return "Merci de nous avoir contactés. Comment pouvons-nous vous aider davantage ?"
        elif target_language.lower() == "de":
            return "Vielen Dank für Ihre Kontaktaufnahme. Wie können wir Ihnen weiterhelfen?"
        else:
            return "Thank you for contacting us. How else can we assist you?"


# --- 4. Web API (FastAPI) ---
app = FastAPI(
    title="Multilingual Customer Support Chatbot",
    description="Chatbot using InCLT Crosslingual Transfer Prompting for enhanced multilingual support.",
    version="1.0.0",
)


class ChatRequest(BaseModel):
    customer_query: str
    target_language: str  # e.g., "es", "fr", "de", "en"


class ChatResponse(BaseModel):
    response: str
    prompt_used: str


@app.post("/chat", response_model=ChatResponse, summary="Get a chat response in the target language")
async def chat_endpoint(request: ChatRequest):
    """
    Receives a customer query and target language, generates an InCLT prompt,
    and returns a simulated LLM response.
    """
    # Generate the prompt using InCLT pattern
    generated_prompt = generate_inclt_prompt(
        request.customer_query, request.target_language
    )

    # Get a simulated response from the LLM service
    llm_response = get_llm_response(generated_prompt, request.target_language)

    return ChatResponse(response=llm_response, prompt_used=generated_prompt)


if __name__ == "__main__":
    # To run this application:
    # 1. Save the code as 'inclt_chatbot_app.py'
    # 2. Open your terminal in the same directory
    # 3. Run: uvicorn inclt_chatbot_app:app --reload --port 8000
    # Then, open your browser to http://127.0.0.1:8000/docs for the interactive API documentation (Swagger UI).
    uvicorn.run(app, host="0.0.0.0", port=8000)
