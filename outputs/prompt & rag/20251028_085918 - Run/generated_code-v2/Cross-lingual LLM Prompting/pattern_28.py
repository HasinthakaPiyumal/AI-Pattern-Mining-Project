from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

example_database = [
    {
        "source_lang": "en",
        "target_lang": "es",
        "en_query": "Where is my order?",
        "es_query": "¿Dónde está mi pedido?",
        "en_response": "Please provide your order number for tracking.",
        "es_response": "Por favor, proporcione su número de pedido para el seguimiento."
    },
    {
        "source_lang": "en",
        "target_lang": "fr",
        "en_query": "How do I return an item?",
        "fr_query": "Comment retourner un article ?",
        "en_response": "You can initiate a return through your order history.",
        "fr_response": "Vous pouvez initier un retour via l'historique de vos commandes."
    },
    {
        "source_lang": "en",
        "target_lang": "es",
        "en_query": "What is your refund policy?",
        "es_query": "¿Cuál es su política de reembolso?",
        "en_response": "Refunds are processed within 5-7 business days.",
        "es_response": "Los reembolsos se procesan en un plazo de 5 a 7 días hábiles."
    },
    {
        "source_lang": "en",
        "target_lang": "en",
        "en_query": "What are your business hours?",
        "es_query": "N/A",
        "en_response": "Our business hours are Monday to Friday, 9 AM to 5 PM.",
        "es_response": "N/A"
    }
]

def detect_language(text: str) -> str:
    text_lower = text.lower()
    if "pedido" in text_lower or "¿dónde" in text_lower or "reembolso" in text_lower or "devolver" in text_lower:
        return "es"
    if "retourner" in text_lower or "article" in text_lower or "remboursement" in text_lower:
        return "fr"
    return "en"

class InCLTPromptEngineer:
    def __init__(self, examples: list):
        self.examples = examples

    def retrieve_cross_lingual_examples(self, customer_query: str, target_lang: str, num_examples: int = 2) -> list:
        relevant_examples = []
        for example in self.examples:
            if example["target_lang"] == target_lang:
                relevant_examples.append(example)
                if len(relevant_examples) >= num_examples:
                    break
        if not relevant_examples and target_lang != "en":
            for example in self.examples:
                if example["source_lang"] == "en" and example["target_lang"] != target_lang:
                     relevant_examples.append(example)
                     if len(relevant_examples) >= num_examples:
                        break
        return relevant_examples

    def construct_prompt(self, customer_query: str, target_lang: str) -> str:
        prompt_parts = []
        prompt_parts.append("You are a helpful multilingual customer support assistant. Provide concise answers in the language of the customer query.")
        prompt_parts.append("Here are some examples of customer interactions for context:")
        prompt_parts.append("")

        retrieved_examples = self.retrieve_cross_lingual_examples(customer_query, target_lang, num_examples=2)

        for ex in retrieved_examples:
            prompt_parts.append(f"Example Source Query ({ex['source_lang']}): {ex[f'{ex['source_lang']}_query']}")
            if ex['target_lang'] != 'N/A':
                prompt_parts.append(f"Example Target Query ({ex['target_lang']}): {ex[f'{ex['target_lang']}_query']}")
            prompt_parts.append(f"Example Source Response ({ex['source_lang']}): {ex[f'{ex['source_lang']}_response']}")
            if ex['target_lang'] != 'N/A':
                prompt_parts.append(f"Example Target Response ({ex['target_lang']}): {ex[f'{ex['target_lang']}_response']}")
            prompt_parts.append("---")

        prompt_parts.append(f"Customer Query ({target_lang}): {customer_query}")
        prompt_parts.append(f"Response ({target_lang}):")

        return "\n".join(prompt_parts)

class MultilingualLLM:
    def generate_response(self, prompt: str, target_lang: str) -> str:
        lines = prompt.strip().split("\n")
        query_text = ""
        for line in reversed(lines):
            if line.startswith(f"Customer Query ({target_lang}):"):
                query_text = line.replace(f"Customer Query ({target_lang}):", "").strip()
                break

        query_lower = query_text.lower()

        if target_lang == "es":
            if "pedido" in query_lower:
                return "Claro, para rastrear su pedido, por favor, ingrese su número de pedido."
            elif "reembolso" in query_lower:
                return "Nuestra política de reembolso permite devoluciones hasta 30 días después de la compra. ¿Desea iniciar una devolución?"
            elif "devolver" in query_lower or "cambiar" in query_lower:
                return "Para devolver o cambiar un artículo, visite la sección 'Mis Pedidos' en su cuenta."
            elif "gracias" in query_lower:
                return "De nada. ¿Hay algo más en lo que pueda ayudarle?"
        elif target_lang == "fr":
            if "retourner" in query_lower or "article" in query_lower:
                return "Pour retourner un article, veuillez vous connecter à votre compte et accéder à 'Mes Commandes'."
            elif "remboursement" in query_lower:
                return "Notre politique de remboursement est de 30 jours. Souhaitez-vous en savoir plus ?"
            elif "merci" in query_lower:
                return "De rien. Y a-t-il autre chose que je puisse faire pour vous ?"
        else:
            if "order" in query_lower:
                return "Certainly, please provide your order number to track your shipment."
            elif "refund" in query_lower:
                return "Our refund policy allows returns for up to 30 days after purchase. Would you like to start a return?"
            elif "return" in query_lower or "exchange" in query_lower:
                return "To return or exchange an item, please visit the 'My Orders' section in your account."
            elif "thank you" in query_lower:
                return "You're welcome! Is there anything else I can assist you with?"

        if target_lang == "es":
            return "Comprendo. Para poder asistirte mejor, ¿podrías darme más detalles sobre tu consulta?"
        elif target_lang == "fr":
            return "Je comprends. Pour mieux vous aider, pourriez-vous me donner plus de détails sur votre demande ?"
        return "I understand. To assist you better, could you please provide more details about your inquiry."


prompt_engineer = InCLTPromptEngineer(example_database)
llm_model = MultilingualLLM()

class ChatRequest(BaseModel):
    query: str

@app.post("/chat")
async def chat(request: ChatRequest):
    customer_query = request.query
    target_lang = detect_language(customer_query)
    constructed_prompt = prompt_engineer.construct_prompt(customer_query, target_lang)
    chatbot_response = llm_model.generate_response(constructed_prompt, target_lang)
    return {"response": chatbot_response}