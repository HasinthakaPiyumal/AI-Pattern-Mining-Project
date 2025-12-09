from fastapi import FastAPI
from pydantic import BaseModel
from chatbot_core import MultilingualChatbot

app = FastAPI()

# Initialize the chatbot (in a real scenario, this would load an actual LLM)
chatbot = MultilingualChatbot(llm_model_name="mock-multilingual-llm") # Placeholder for a real LLM

# In a real application, these examples might be dynamically loaded from a database or a knowledge base.
# For this demonstration, they are hardcoded.
icl_examples = [
    {
        "source_query": "My order is delayed.",
        "target_response": "Su pedido se ha retrasado. Permítame verificar el estado.",
        "source_lang_example": "en",
        "target_lang_example": "es",
    },
    {
        "source_query": "J'ai besoin d'aide avec un remboursement.",
        "target_response": "You need help with a refund. Please provide your order number.",
        "source_lang_example": "fr",
        "target_lang_example": "en",
    },
    {
        "source_query": "¿Cuál es el estado de mi envío?",
        "target_response": "The status of your shipment. Could you give me the tracking ID?",
        "source_lang_example": "es",
        "target_lang_example": "en",
    },
    {
        "source_query": "Où est ma commande numéro 98765?",
        "target_response": "Pour votre commande numéro 98765, veuillez patienter. Je vérifie l'état.",
        "source_lang_example": "fr",
        "target_lang_example": "fr",
    },
]

class ChatRequest(BaseModel):
    user_query: str
    source_lang: str
    target_lang: str

@app.post("/chat")
async def chat_with_bot(request: ChatRequest):
    """
    Endpoint for interacting with the multilingual customer support chatbot.
    Receives a user query and returns a response in the specified target language,
    leveraging InCLT Crosslingual Transfer Prompting.
    """
    response = chatbot.get_response(
        user_query=request.user_query,
        source_lang=request.source_lang,
        target_lang=request.target_lang,
        icl_examples=icl_examples,
    )
    return {"response": response}