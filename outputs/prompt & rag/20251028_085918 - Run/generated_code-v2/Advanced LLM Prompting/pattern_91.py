from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from transformers import pipeline

class DictionaryLookupModule:
    def get_definitions(self, word: str, source_lang: str, target_lang: str) -> dict:
        mock_definitions = {
            "bank": {
                "en": [
                    "A financial institution that accepts deposits and makes loans.",
                    "The land alongside or sloping down to a river or lake."
                ],
                "fr": [
                    "Établissement financier qui accepte les dépôts et accorde des prêts.",
                    "Bord d'une rivière ou d'un lac."
                ]
            },
            "apple": {
                "en": [
                    "A common, edible fruit, produced by the apple tree."
                ],
                "fr": [
                    "Fruit comestible, généralement rond, à peau lisse, de l'arbre pommier."
                ]
            },
            "default": {
                "en": [
                    f"No specific definition found for '{word}'."
                ],
                "fr": [
                    f"Aucune définition spécifique trouvée pour '{word}'."
                ]
            }
        }
        return mock_definitions.get(word.lower(), mock_definitions["default"])

class PromptBuilder:
    def build_translation_prompt(self, original_query: str, source_defs: list, target_defs: list) -> str:
        prompt_parts = []
        prompt_parts.append("Translate the following text, considering the provided definitions to resolve any ambiguity.\n")

        if source_defs:
            prompt_parts.append("--- Source Language Definitions ---")
            for i, definition in enumerate(source_defs):
                prompt_parts.append(f"Definition {i+1} (Source): {definition}")

        if target_defs:
            if source_defs:
                prompt_parts.append("")
            prompt_parts.append("--- Target Language Definitions ---")
            for i, definition in enumerate(target_defs):
                prompt_parts.append(f"Definition {i+1} (Target): {definition}")

        if source_defs or target_defs:
            prompt_parts.append("-----------------------------------")

        prompt_parts.append(f"Original Text: {original_query}")
        prompt_parts.append(f"Translation:")

        final_prompt = "\n".join(prompt_parts)
        return final_prompt

class TranslationModule:
    def __init__(self, model_name: str = "Helsinki-NLP/opus-mt-en-fr"):
        self.translator = pipeline("translation", model=model_name)

    def translate_text(self, text_to_translate: str, target_lang_code: str = "fr") -> str:
        try:
            result = self.translator(text_to_translate)
            translated_text = result[0]["translation_text"] if result else ""
            return translated_text
        except Exception as e:
            return f"[Translation Error] Could not translate: {e}"

app = FastAPI(
    title="DiPMT Customer Support Chatbot",
    description="Multi-language customer support chatbot using Dictionary-based Prompting for Machine Translation (DiPMT)."
)

dictionary_lookup = DictionaryLookupModule()
prompt_builder = PromptBuilder()
translation_module = TranslationModule()

class ChatQuery(BaseModel):
    text: str
    source_language: str = "en"
    target_language: str = "fr"
    keyword_for_lookup: str = ""

@app.post("/chat/query")
async def handle_customer_query(query: ChatQuery) -> Dict[str, Any]:
    word_to_lookup = query.keyword_for_lookup if query.keyword_for_lookup else query.text.split()[0]

    definitions = dictionary_lookup.get_definitions(
        word=word_to_lookup,
        source_lang=query.source_language,
        target_lang=query.target_language
    )
    source_defs = definitions.get(query.source_language, [])
    target_defs = definitions.get(query.target_language, [])

    enhanced_prompt = prompt_builder.build_translation_prompt(
        original_query=query.text,
        source_defs=source_defs,
        target_defs=target_defs
    )

    translated_response = translation_module.translate_text(
        text_to_translate=enhanced_prompt,
        target_lang_code=query.target_language
    )

    return {
        "original_query": query.text,
        "source_language": query.source_language,
        "target_language": query.target_language,
        "keyword_looked_up": word_to_lookup,
        "source_definitions": source_defs,
        "target_definitions": target_defs,
        "enhanced_prompt_sent_to_mt": enhanced_prompt,
        "translated_response": translated_response
    }

@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "DiPMT Chatbot Service is running."}

# To run this service:
# 1. Install dependencies: pip install fastapi uvicorn transformers pydantic
# 2. Run: uvicorn dipmt_chatbot_service:app --reload --port 8000
# 3. Access: http://127.0.0.1:8000/docs for Swagger UI

# Example usage with curl:
# curl -X POST "http://localhost:8000/chat/query" -H "Content-Type: application/json" -d '{
#   "text": "The company needs to secure its bank assets.",
#   "source_language": "en",
#   "target_language": "fr",
#   "keyword_for_lookup": "bank"
# }'
