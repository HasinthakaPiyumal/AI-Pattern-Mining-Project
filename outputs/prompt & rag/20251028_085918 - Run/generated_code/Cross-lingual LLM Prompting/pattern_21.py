from fastapi import FastAPI
from pydantic import BaseModel
from langdetect import detect, LangDetectException
from transformers import pipeline

example_data = [
    {
        "source_lang": "es",
        "target_lang": "en",
        "source_query": "Necesito ayuda con mi factura.",
        "target_answer": "I need help with my bill."
    },
    {
        "source_lang": "es",
        "target_lang": "en",
        "source_query": "¿Cuál es el estado de mi pedido?",
        "target_answer": "What is the status of my order?"
    },
    {
        "source_lang": "fr",
        "target_lang": "en",
        "source_query": "Je voudrais annuler ma commande.",
        "target_answer": "I would like to cancel my order."
    },
    {
        "source_lang": "fr",
        "target_lang": "en",
        "source_query": "Quel est le problème avec mon compte ?",
        "target_answer": "What is the problem with my account?"
    },
    {
        "source_lang": "de",
        "target_lang": "en",
        "source_query": "Ich möchte mein Abonnement kündigen.",
        "target_answer": "I want to cancel my subscription."
    },
    {
        "source_lang": "de",
        "target_lang": "en",
        "source_query": "Wo ist meine Lieferung?",
        "target_answer": "Where is my delivery?"
    }
]

class PromptGenerator:
    def __init__(self, examples: list):
        self.examples = examples

    def generate_prompt(self, customer_query: str, source_lang: str, target_lang: str) -> str:
        prompt_parts = [
            "You are a helpful multilingual customer support assistant. Your task is to understand the customer's query in the source language, and provide a helpful response in the target language. Use the following examples to guide your response.\n\n"
            "Examples:\n"
        ]

        relevant_examples = [
            ex for ex in self.examples
            if ex["source_lang"] == source_lang and ex["target_lang"] == target_lang
        ]

        for ex in relevant_examples:
            prompt_parts.append(f"Source ({ex['source_lang']}): {ex['source_query']}\n")
            prompt_parts.append(f"Target ({ex['target_lang']}): {ex['target_answer']}\n\n")

        prompt_parts.append(f"Source ({source_lang}): {customer_query}\n")
        prompt_parts.append(f"Target ({target_lang}):")

        return "".join(prompt_parts)

class LLMIntegration:
    def __init__(self, model_name: str = "t5-small"):
        self.generator = pipeline("text2text-generation", model=model_name)

    def get_response(self, prompt: str) -> str:
        response = self.generator(prompt, max_new_tokens=100, num_return_sequences=1, do_sample=True, top_k=50, top_p=0.95)
        return response[0]["generated_text"].strip()

class LanguageDetector:
    def detect_language(self, text: str) -> str:
        try:
            return detect(text)
        except LangDetectException:
            return "unknown"

app = FastAPI()

class ChatRequest(BaseModel):
    query: str
    target_lang: str = "en"

language_detector = LanguageDetector()
prompt_generator = PromptGenerator(example_data)
llm_integration = LLMIntegration(model_name="t5-small")

@app.post("/chat")
async def chat(request: ChatRequest):
    customer_query = request.query
    target_lang = request.target_lang

    source_lang = language_detector.detect_language(customer_query)
    if source_lang == "unknown":
        return {"error": "Could not detect source language. Please provide a clear query."}

    in_context_prompt = prompt_generator.generate_prompt(customer_query, source_lang, target_lang)

    try:
        llm_response = llm_integration.get_response(in_context_prompt)
    except Exception as e:
        return {"error": f"Error during LLM response generation: {str(e)}"}

    return {"response": llm_response, "source_language": source_lang}