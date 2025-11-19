from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0 # Ensure consistent language detection

app = FastAPI()

INCLT_EXAMPLES = {
    "order_status": [
        {"es": "Mi pedido no ha llegado.", "en": "My order hasn't arrived.", "response": "Please provide your order number and I will check the status for you."},
        {"es": "¿Dónde está mi paquete?", "en": "Where is my package?", "response": "To track your package, please share your tracking number."}
    ],
    "returns": [
        {"es": "Quiero devolver un artículo.", "en": "I want to return an item.", "response": "To initiate a return, please visit your order history."},
        {"es": "¿Cómo hago una devolución?", "en": "How do I make a return?", "response": "You can find our return policy and instructions on our website's FAQ section."}
    ]
}

class ChatRequest(BaseModel):
    query: str

def detect_language(text: str) -> str:
    try:
        return detect(text)
    except Exception:
        return "en" # Default to English if detection fails

def get_intent_from_query(query: str, lang: str) -> str:
    query_lower = query.lower()
    if lang == "es":
        if any(keyword in query_lower for keyword in ["pedido", "llegado", "paquete"]):
            return "order_status"
        elif any(keyword in query_lower for keyword in ["devolver", "devolución"]):
            return "returns"
    elif lang == "en":
        if any(keyword in query_lower for keyword in ["order", "arrived", "package"]):
            return "order_status"
        elif any(keyword in query_lower for keyword in ["return", "item"]):
            return "returns"
    return "general" # Fallback for unknown intent

def construct_inclt_prompt(customer_query: str, detected_lang: str, intent: str) -> str:
    system_message = """You are a helpful customer support assistant for an e-commerce platform. Your goal is to assist customers with their queries in a helpful and polite manner. You understand multiple languages.

Here are some examples of how to respond to customer queries across different languages:
"""

    inclt_examples_str = []
    if intent in INCLT_EXAMPLES:
        for example in INCLT_EXAMPLES[intent]:
            source_q = example.get(detected_lang, example["en"]) # Prioritize detected lang, fallback to English
            target_q_en = example["en"]
            response = example["response"]
            
            # Present both source and target language in the in-context example
            if detected_lang != "en" and detected_lang in example: # Avoid redundant English if query is already English
                inclt_examples_str.append(f"Human: {source_q}\nHuman: {target_q_en}\nAssistant: {response}")
            else:
                inclt_examples_str.append(f"Human: {target_q_en}\nAssistant: {response}")

    examples_block = "\n\n".join(inclt_examples_str)

    # The final prompt for the LLM includes the system message, InCLT examples, and the current query
    full_prompt = f"{system_message}\n{examples_block}\n\nHuman: {customer_query}\nAssistant:"
    return full_prompt

def mock_llm_response(prompt: str) -> str:
    # Simulate LLM behavior based on the prompt and implicit understanding from InCLT examples
    # In a real scenario, this would involve calling a true multilingual LLM API
    if "order number" in prompt or "track your package" in prompt or "pedido no ha llegado" in prompt or "dónde está mi paquete" in prompt:
        return "I understand you are asking about your order status. Could you please provide your order number or tracking ID so I can assist you further?"
    elif "return an item" in prompt or "devolver un artículo" in prompt or "return policy" in prompt or "cómo hago una devolución" in prompt:
        return "To initiate a return or understand our return policy, please visit the returns section on our website or provide your order details."
    else:
        return "Thank you for contacting customer support. How else can I help you today?"

@app.post("/chat")
async def chat_with_bot(request: ChatRequest):
    customer_query = request.query
    detected_lang = detect_language(customer_query)
    intent = get_intent_from_query(customer_query, detected_lang)

    prompt = construct_inclt_prompt(customer_query, detected_lang, intent)
    llm_response = mock_llm_response(prompt)
    
    return {"response": llm_response, "language_detected": detected_lang, "detected_intent": intent}
