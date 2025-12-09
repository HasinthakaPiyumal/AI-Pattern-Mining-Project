from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict

app = FastAPI()

class InContextExample(BaseModel):
    source_lang_text: str
    target_lang_text: str
    source_lang_response: str
    target_lang_response: str

class PromptEngineeringModule:
    def __init__(self):
        self.cross_lingual_examples: Dict[str, List[InContextExample]] = {
            "es": [
                InContextExample(
                    source_lang_text="What is the return policy for electronics?",
                    target_lang_text="¿Cuál es la política de devolución para productos electrónicos?",
                    source_lang_response="You can return electronics within 30 days of purchase with the original receipt.",
                    target_lang_response="Puede devolver productos electrónicos dentro de los 30 días posteriores a la compra con el recibo original."
                ),
                InContextExample(
                    source_lang_text="How can I track my order?",
                    target_lang_text="¿Cómo puedo rastrear mi pedido?",
                    source_lang_response="You can track your order using the tracking number provided in your shipping confirmation email.",
                    target_lang_response="Puede rastrear su pedido utilizando el número de seguimiento proporcionado en su correo electrónico de confirmación de envío."
                )
            ],
            "fr": [
                InContextExample(
                    source_lang_text="What is the return policy for electronics?",
                    target_lang_text="Quelle est la politique de retour pour l'électronique ?",
                    source_lang_response="You can return electronics within 30 days of purchase with the original receipt.",
                    target_lang_response="Vous pouvez retourner les produits électroniques dans les 30 jours suivant l'achat avec le reçu original."
                ),
                InContextExample(
                    source_lang_text="How can I track my order?",
                    target_lang_text="Comment puis-je suivre ma commande ?",
                    source_lang_response="You can track your order using the tracking number provided in your shipping confirmation email.",
                    target_lang_response="Vous pouvez suivre votre commande en utilisant le numéro de suivi fourni dans votre e-mail de confirmation d'expédition."
                )
            ]
        }

    def create_in_clt_prompt(self, customer_query: str, target_language: str) -> str:
        examples_for_lang = self.cross_lingual_examples.get(target_language.lower(), [])
        if not examples_for_lang:
            # Fallback to a simpler prompt if no specific examples are found
            print(f"Warning: No InCLT examples found for {target_language}. Using direct query.")
            return f"User query in {target_language}: {customer_query}\nAssistant: "

        prompt_parts = []
        for example in examples_for_lang:
            prompt_parts.append(f"User (English): {example.source_lang_text}")
            prompt_parts.append(f"Assistant (English): {example.source_lang_response}")
            prompt_parts.append(f"User ({target_language.capitalize()}): {example.target_lang_text}")
            prompt_parts.append(f"Assistant ({target_language.capitalize()}): {example.target_lang_response}")

        prompt_parts.append(f"User ({target_language.capitalize()}): {customer_query}")
        prompt_parts.append(f"Assistant ({target_language.capitalize()}): ")

        return "\n".join(prompt_parts)

class LLMIntegration:
    def get_llm_response(self, prompt: str) -> str:
        # This is a simulated LLM. In a real application, you would integrate
        # with a service like Google Gemini, OpenAI GPT, etc.
        print(f"Simulating LLM response for prompt:\n---\n{prompt}\n---")

        # Simple rule-based simulation based on query content (for demonstration)
        if "política de devolución" in prompt.lower() or "return policy" in prompt.lower():
            return "Nuestra política de devolución permite devoluciones dentro de los 30 días con el recibo original." if "es" in prompt.lower() else \
                   "Our return policy allows returns within 30 days with the original receipt."
        elif "rastrear mi pedido" in prompt.lower() or "track my order" in prompt.lower():
            return "Puede rastrear su pedido a través del enlace en su correo electrónico de envío." if "es" in prompt.lower() else \
                   "You can track your order using the link in your shipping email."
        else:
            return "Lo siento, no tengo suficiente información para responder a eso. ¿Puede dar más detalles?" if "es" in prompt.lower() else \
                   "I'm sorry, I don't have enough information to answer that. Can you provide more details?"

class CustomerQuery(BaseModel):
    query: str
    target_language: str

class ChatResponse(BaseModel):
    response: str
    generated_prompt: str

@app.post("/chat", response_model=ChatResponse)
async def chat_with_chatbot(customer_query: CustomerQuery):
    prompt_engineer = PromptEngineeringModule()
    llm_integrator = LLMIntegration()

    generated_prompt = prompt_engineer.create_in_clt_prompt(
        customer_query.query,
        customer_query.target_language
    )
    
    llm_response = llm_integrator.get_llm_response(generated_prompt)
    
    return ChatResponse(response=llm_response, generated_prompt=generated_prompt)

# To run this application, save it as main.py and execute:
# uvicorn main:app --reload