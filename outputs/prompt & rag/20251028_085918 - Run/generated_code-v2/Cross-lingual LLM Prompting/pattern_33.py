
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


class InCLT_Prompting_Module:
    def __init__(self, in_context_examples):
        self.in_context_examples = in_context_examples

    def construct_prompt(self, customer_query: str, customer_lang: str) -> str:
        prompt_parts = [
            "You are a helpful multilingual customer support assistant.",
            "Translate the customer query to English, and then provide an appropriate response in English and in the customer's original language."
        ]

        if self.in_context_examples:
            prompt_parts.append("Here are some examples of customer queries, their English translations, and appropriate responses in both English and the customer's language:")
            for i, example in enumerate(self.in_context_examples):
                prompt_parts.append(f"\nExample {i+1}:")
                prompt_parts.append(f"Customer Query ({example['source_lang']}): {example['source_query']}")
                prompt_parts.append(f"English Translation: {example['target_query']}")
                prompt_parts.append(f"Agent Response (English): {example['source_response']}")
                prompt_parts.append(f"Agent Response ({example['target_lang']}): {example['target_response']}")

        prompt_parts.append(f"\nNow, process the following new customer query:\n")
        prompt_parts.append(f"Customer Query ({customer_lang}): {customer_query}")
        prompt_parts.append(f"English Translation:")
        prompt_parts.append(f"Agent Response (English):")
        prompt_parts.append(f"Agent Response ({customer_lang}):")

        return "\n".join(prompt_parts)


class Multilingual_LLM_Core:
    def __init__(self, model_name="facebook/mbart-large-50-many-to-many-mmt"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def generate_response(self, prompt: str, src_lang: str, tgt_lang: str) -> str:
        # MBART expects language codes for source and target
        # For example, 'en_XX' for English, 'es_XX' for Spanish, 'fr_XX' for French
        # This assumes the LLM's tokenizer understands these codes implicitly from the prompt structure

        # A more robust solution would set src_lang and tgt_lang tokens if the model explicitly supports them
        # For this example, we rely on the prompt instructing the model.

        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=200,
            num_beams=5,
            early_stopping=True
        )
        response = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return response


# In-Context Example Database/Store
in_context_examples_data = [
    {
        "source_lang": "es_XX",
        "target_lang": "es_XX",
        "source_query": "¿Cómo puedo rastrear mi pedido?",
        "target_query": "How can I track my order?",
        "source_response": "You can track your order using the tracking number provided in your shipping confirmation email.",
        "target_response": "Puede rastrear su pedido utilizando el número de seguimiento proporcionado en su correo electrónico de confirmación de envío."
    },
    {
        "source_lang": "fr_XX",
        "target_lang": "fr_XX",
        "source_query": "Je souhaite annuler ma commande.",
        "target_query": "I want to cancel my order.",
        "source_response": "To cancel your order, please provide your order number.",
        "target_response": "Pour annuler votre commande, veuillez fournir votre numéro de commande."
    },
    {
        "source_lang": "de_DE",
        "target_lang": "de_DE",
        "source_query": "Meine Bestellung ist beschädigt angekommen.",
        "target_query": "My order arrived damaged.",
        "source_response": "We apologize for the inconvenience. Please send us photos of the damaged item and packaging.",
        "target_response": "Wir entschuldigen uns für die Unannehmlichkeiten. Bitte senden Sie uns Fotos des beschädigten Artikels und der Verpackung."
    }
]

# Initialize Modules
llm_core = Multilingual_LLM_Core()
prompt_module = InCLT_Prompting_Module(in_context_examples_data)

app = FastAPI()


class ChatRequest(BaseModel):
    customer_query: str
    customer_lang: str # e.g., "es_XX", "fr_XX", "de_DE", "en_XX"


@app.post("/chat")
async def chat_with_customer(request: ChatRequest):
    try:
        # Construct the InCLT enhanced prompt
        prompt = prompt_module.construct_prompt(request.customer_query, request.customer_lang)

        # Generate response using the LLM
        # For mbart-large-50-many-to-many-mmt, the target language for generation is often set during tokenizer/model setup
        # However, for the purpose of this architecture, we are asking the LLM *within the prompt* to generate in the customer's language.
        # The src_lang and tgt_lang parameters here are mostly indicative, as the prompt itself guides the LLM.
        response_text = llm_core.generate_response(prompt, src_lang=request.customer_lang, tgt_lang=request.customer_lang)

        # In a real application, you might parse response_text to extract structured English translation and native language response.
        # For this demonstration, we return the raw LLM output, assuming it follows the requested format in the prompt.

        return {"status": "success", "response": response_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# To run this application:
# 1. Save the code as `chatbot_app.py`.
# 2. Install necessary libraries: `pip install transformers fastapi uvicorn pydantic`
# 3. Run from your terminal: `uvicorn chatbot_app:app --reload`
# 4. Access the API at http://127.0.0.1:8000/docs for Swagger UI.