from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from langdetect import detect
import random

# Pydantic models
class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    response: str
    detected_language: str
    prompt_used: str

# FastAPI app
app = FastAPI(title="Multilingual Customer Support Assistant")

# --- LLM and Prompt Generation Service ---
class PromptGenerationService:
    def __init__(self):
        # In-context examples demonstrating cross-lingual transfer
        # Each example has a query and an answer in two languages (e.g., English and Spanish)
        self.in_context_examples = [
            {
                "en_query": "My order #12345 is delayed. When will it arrive?",
                "en_answer": "We apologize for the delay. Your order #12345 is expected to arrive within 2-3 business days. You can track it here: [tracking_link]",
                "es_query": "Mi pedido #12345 está retrasado. ¿Cuándo llegará?",
                "es_answer": "Lamentamos el retraso. Se espera que su pedido #12345 llegue en 2-3 días hábiles. Puede rastrearlo aquí: [enlace_rastreo]"
            },
            {
                "en_query": "How do I return an item?",
                "en_answer": "To return an item, please visit our returns portal at [returns_portal_link] and follow the instructions.",
                "es_query": "¿Cómo devuelvo un artículo?",
                "es_answer": "Para devolver un artículo, visite nuestro portal de devoluciones en [enlace_portal_devoluciones] y siga las instrucciones."
            },
            {
                "en_query": "I need help with a payment issue.",
                "en_answer": "Please describe your payment issue in more detail so we can assist you. Is it about a failed transaction or a refund?",
                "es_query": "Necesito ayuda con un problema de pago.",
                "es_answer": "Por favor, describa su problema de pago con más detalle para que podamos ayudarle. ¿Se trata de una transacción fallida o un reembolso?"
            },
            {
                "en_query": "Where is my invoice?",
                "en_answer": "You can find your invoice in your account's 'Order History' section, or check your email for the confirmation.",
                "es_query": "¿Dónde está mi factura?",
                "es_answer": "Puede encontrar su factura en la sección 'Historial de Pedidos' de su cuenta, o revisar su correo electrónico para la confirmación."
            }
        ]

    def create_in_clt_prompt(self, customer_query: str, detected_lang: str) -> str:
        """
        Creates a prompt using the InCLT Crosslingual Transfer Prompting pattern.
        It includes examples in both source and target languages.
        """
        prompt_parts = []
        prompt_parts.append("You are a helpful multilingual customer support assistant for an e-commerce platform.")
        prompt_parts.append("Below are examples of customer queries and their appropriate responses, demonstrating cross-lingual understanding.\n")

        for example in self.in_context_examples:
            prompt_parts.append(f"Customer (EN): {example['en_query']}")
            prompt_parts.append(f"Assistant (EN): {example['en_answer']}\n")
            if detected_lang in example and f'{detected_lang}_query' in example and f'{detected_lang}_answer' in example:
                prompt_parts.append(f"Customer ({detected_lang.upper()}): {example[f'{detected_lang}_query']}")
                prompt_parts.append(f"Assistant ({detected_lang.upper()}): {example[f'{detected_lang}_answer']}\n")
            else:
                # Fallback if specific language example is not available, use EN as a proxy
                prompt_parts.append(f"Customer ({detected_lang.upper()}): {example['en_query']}")
                prompt_parts.append(f"Assistant ({detected_lang.upper()}): {example['en_answer']}\n")

        prompt_parts.append(f"Now, based on the examples above, please answer the following customer query in {detected_lang.upper()}:\n")
        prompt_parts.append(f"Customer ({detected_lang.upper()}): {customer_query}")
        prompt_parts.append(f"Assistant ({detected_lang.upper()}):")

        return "\n".join(prompt_parts)

# --- Mock LLM Pipeline for Demonstration ---
class MockMultilingualLLMPipeline:
    def __call__(self, prompt: str, max_new_tokens: int = 100, num_return_sequences: int = 1, **kwargs):
        # Simulate LLM behavior: look for the last question and try to respond.
        # This is a very simplified mock focused on demonstrating prompt structure.
        if "Customer (ES): ¿Dónde está mi factura?" in prompt and "Assistant (ES):" in prompt:
            return [{"generated_text": prompt + " Puede encontrar su factura en su cuenta.\n"}]
        elif "Customer (EN): Where is my invoice?" in prompt and "Assistant (EN):" in prompt:
            return [{"generated_text": prompt + " You can find your invoice in your account.\n"}]
        elif "Customer (ES): Mi pedido #12345 está retrasado. ¿Cuándo llegará?" in prompt:
            return [{"generated_text": prompt + " Se espera que su pedido llegue pronto.\n"}]
        elif "Customer (EN): My order #12345 is delayed. When will it arrive?" in prompt:
            return [{"generated_text": prompt + " Your order is expected soon.\n"}]
        elif "Customer (ES): ¿Cómo devuelvo un artículo?" in prompt:
            return [{"generated_text": prompt + " Visite el portal de devoluciones.\n"}]
        elif "Customer (EN): How do I return an item?" in prompt:
            return [{"generated_text": prompt + " Visit the returns portal.\n"}]
        elif "Customer (ES): Necesito ayuda con un problema de pago." in prompt:
            return [{"generated_text": prompt + " Describa su problema de pago.\n"}]
        elif "Customer (EN): I need help with a payment issue." in prompt:
            return [{"generated_text": prompt + " Describe your payment issue.\n"}]

        # Fallback for unexpected queries
        return [{"generated_text": prompt + " I am a mock assistant and cannot fully process this. Please rephrase.\n"}]

llm_pipeline = MockMultilingualLLMPipeline()
prompt_service = PromptGenerationService()


@app.post("/chat", response_model=ChatResponse)
async def chat_with_assistant(request: ChatRequest):
    """
    Handles customer queries and provides multilingual support using InCLT prompting.
    """
    query = request.query
    try:
        detected_lang = detect(query)
        # Ensure the detected language is one we have examples for or handle fallback
        if detected_lang not in ["en", "es"]:
            detected_lang = "en" # Default to English if unsupported language detected
    except Exception:
        detected_lang = "en" # Default to English if detection fails

    prompt = prompt_service.create_in_clt_prompt(customer_query=query, detected_lang=detected_lang)

    try:
        llm_response_raw = llm_pipeline(prompt, max_new_tokens=200, num_return_sequences=1)
        generated_text = llm_response_raw[0]['generated_text']

        # Extract only the assistant's part from the generated text
        assistant_tag = f"Assistant ({detected_lang.upper()}):"
        response_start_index = generated_text.rfind(assistant_tag)
        if response_start_index != -1:
            # Get everything after the last assistant tag, and then clean up any potential leftover prompt parts
            assistant_response = generated_text[response_start_index + len(assistant_tag):].strip()
            # Remove any subsequent customer queries or assistant tags if they appear due to mock behavior
            if "Customer (" in assistant_response:
                assistant_response = assistant_response.split("Customer (")[0].strip()
            if "Assistant (" in assistant_response:
                assistant_response = assistant_response.split("Assistant (")[0].strip()
        else:
            # Fallback if tag not found, try to extract reasonable response
            # This might need more sophisticated parsing for a real LLM
            assistant_response = generated_text.split("Assistant (")[-1].strip() if "Assistant (" in generated_text else generated_text.strip()

    except Exception as e:
        assistant_response = f"Error generating response: {e}. Please try again later."
        print(f"LLM generation failed: {e}")

    return ChatResponse(
        response=assistant_response,
        detected_language=detected_lang,
        prompt_used=prompt
    )