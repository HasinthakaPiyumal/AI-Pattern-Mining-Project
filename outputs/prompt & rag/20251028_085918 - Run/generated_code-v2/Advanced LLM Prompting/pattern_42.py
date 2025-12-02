from fastapi import FastAPI
from pydantic import BaseModel
import re

app = FastAPI()

class TranslationRequest(BaseModel):
    source_phrase: str
    source_language: str
    target_language: str

class TranslationResponse(BaseModel):
    translated_text: str

SIMULATED_DICTIONARY = {
    "apple": {
        "en": ["A common, edible fruit, produced by the apple tree.", "The tree which bears apples."],
        "fr": ["Pomme: Fruit charnu du pommier.", "Pommier: Arbre qui produit des pommes."],
        "es": ["Manzana: Fruto comestible del manzano.", "Manzano: Árbol que produce manzanas."]
    },
    "bank": {
        "en": ["A financial institution that accepts deposits and makes loans.", "The land alongside or sloping down to a river or lake."],
        "fr": ["Banque: Établissement financier.", "Rive: Bord d'un cours d'eau."],
        "es": ["Banco: Entidad financiera.", "Ribera: Orilla de un río o mar."]
    },
    "book": {
        "en": ["A set of printed or written pages bound together along one edge and encased between covers.", "To reserve (a seat, hotel room, etc.) in advance."],
        "fr": ["Livre: Ensemble de feuilles imprimées reliées.", "Réserver: Prendre d'avance une place."],
        "es": ["Libro: Conjunto de hojas escritas o impresas.", "Reservar: Guardar con antelación."]
    },
    "river": {
        "en": ["A large natural stream of water flowing in a channel to the sea, a lake, or another river."],
        "fr": ["Fleuve: Grand cours d'eau se jetant dans la mer.", "Rivière: Cours d'eau moins important qu'un fleuve."],
        "es": ["Río: Corriente natural de agua que fluye.", "Ribera: Orilla de un río."]
    }
}

def get_definitions(word: str, language: str) -> list[str]:
    word_lower = word.lower()
    if word_lower in SIMULATED_DICTIONARY:
        return SIMULATED_DICTIONARY[word_lower].get(language, [])
    return []

def extract_keywords(phrase: str, language: str) -> list[str]:
    words = re.findall(r'\b\w+\b', phrase.lower())
    stop_words_en = {"a", "an", "the", "is", "are", "was", "were", "and", "or", "but", "for", "nor", "on", "in", "at", "to", "from", "of", "with", "by", "as", "it", "he", "she", "we", "you", "they", "i", "me", "him", "her", "us", "them", "my", "your", "his", "her", "our", "their", "this", "that", "these", "those"}
    
    stop_words = stop_words_en

    keywords = [word for word in words if word not in stop_words and len(word) > 2]
    return list(set(keywords))

def create_prompt(
    source_phrase: str,
    source_language: str,
    target_language: str,
    keyword_definitions: dict[str, dict[str, list[str]]]
) -> str:
    prompt_parts = [
        "Given the following dictionary definitions, translate the phrase:"
    ]

    for word, definitions_by_lang in keyword_definitions.items():
        if definitions_by_lang.get(source_language) or definitions_by_lang.get(target_language):
            prompt_parts.append(f"\nSource Word: {word}")
            if source_language in definitions_by_lang:
                for def_text in definitions_by_lang[source_language]:
                    prompt_parts.append(f"Source Definition ({source_language}): {def_text}")

            if target_language in definitions_by_lang:
                for def_text in definitions_by_lang[target_language]:
                    prompt_parts.append(f"Target Definition ({target_language}): {def_text}")

    prompt_parts.append(f"\nPhrase to translate from {source_language} to {target_language}: \"{source_phrase}\"")
    return "\n".join(prompt_parts)

def call_genai(prompt: str) -> str:
    print(f"--- Simulating GenAI Call with Prompt ---\n{prompt}\n--- End Simulating GenAI Call ---")
    if "apple" in prompt.lower() and "fruit" in prompt.lower() and "fr" in prompt.lower():
        return "Je voudrais acheter une pomme (le fruit)."
    elif "bank" in prompt.lower() and "river" in prompt.lower() and "es" in prompt.lower():
        return "Me senté en la ribera (orilla del río)."
    elif "bank" in prompt.lower() and "financial" in prompt.lower() and "es" in prompt.lower():
        return "Fui al banco (institución financiera)."
    elif "book" in prompt.lower() and "reserve" in prompt.lower() and "fr" in prompt.lower():
        return "Je vais réserver (prendre) une table."
    
    return f"Simulated translation of '{prompt.split('Phrase to translate from')[-1].split('"')[1]}'"


@app.post("/translate", response_model=TranslationResponse)
async def translate_phrase(request: TranslationRequest):
    keywords = extract_keywords(request.source_phrase, request.source_language)
    
    keyword_definitions = {}
    for word in keywords:
        source_defs = get_definitions(word, request.source_language)
        target_defs = get_definitions(word, request.target_language)
        if source_defs or target_defs:
            keyword_definitions[word] = {
                request.source_language: source_defs,
                request.target_language: target_defs
            }
            
    prompt = create_prompt(
        request.source_phrase,
        request.source_language,
        request.target_language,
        keyword_definitions
    )
    
    translated_text = call_genai(prompt)
    
    return TranslationResponse(translated_text=translated_text)