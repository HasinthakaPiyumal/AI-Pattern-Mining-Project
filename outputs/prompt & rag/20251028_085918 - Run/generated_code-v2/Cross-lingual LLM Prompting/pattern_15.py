from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

in_context_examples = [
    {
        "source_lang": "es",
        "source_query": "¿Cuál es el tiempo de entrega de este producto?",
        "target_lang": "en",
        "target_context": "Product A: Delivery within 3-5 business days. Product B: Delivery within 7-10 business days.",
        "target_answer": "For Product A, delivery is 3-5 business days. For Product B, it's 7-10 business days.",
        "translated_source_answer": "Para el Producto A, la entrega es de 3 a 5 días hábiles. Para el Producto B, es de 7 a 10 días hábiles."
    },
    {
        "source_lang": "fr",
        "source_query": "Comment puis-je retourner un article ?",
        "target_lang": "en",
        "target_context": "Returns Policy: Items can be returned within 30 days of purchase if unused and in original packaging. Contact customer support to initiate a return.",
        "target_answer": "You can return an item within 30 days if it's unused and in original packaging. Please contact customer support to start the return process.",
        "translated_source_answer": "Vous pouvez retourner un article dans les 30 jours s'il n'est pas utilisé et dans son emballage d'origine. Veuillez contacter le service client pour initier le processus de retour."
    },
    {
        "source_lang": "de",
        "source_query": "Gibt es diesen Artikel in einer anderen Farbe?",
        "target_lang": "en",
        "target_context": "Product C colors available: Red, Blue, Green. Product D colors available: Black, White.",
        "target_answer": "Product C is available in Red, Blue, and Green. Product D is available in Black and White.",
        "translated_source_answer": "Produkt C ist in Rot, Blau und Grün erhältlich. Produkt D ist in Schwarz und Weiß erhältlich."
    }
]

def _simulate_language_detection(text: str) -> str:
    if "¿" in text or "ñ" in text.lower():
        return "es"
    elif "comment" in text.lower() or "s'il vous plaît" in text.lower():
        return "fr"
    elif "gibt es" in text.lower() or "farbe" in text.lower():
        return "de"
    return "en"

def _construct_prompt(customer_query: str, detected_lang: str) -> str:
    prompt_parts = []
    for example in in_context_examples:
        # Only include examples relevant to the detected language or general ones
        if example["source_lang"] == detected_lang or example["source_lang"] == "en": # Simplified relevance
            prompt_parts.append(f"Customer Query ({example['source_lang']}): {example['source_query']}")
            prompt_parts.append(f"Context (en): {example['target_context']}")
            prompt_parts.append(f"Assistant Response (en): {example['target_answer']}")
            prompt_parts.append(f"Assistant Response ({example['source_lang']}): {example['translated_source_answer']}\n---")

    prompt_parts.append(f"Customer Query ({detected_lang}): {customer_query}")
    prompt_parts.append(f"Context (en): <Relevant product info goes here - simulated>")
    prompt_parts.append(f"Assistant Response (en):")
    
    return "\n".join(prompt_parts)

def _call_llm(prompt: str) -> str:
    # This function simulates an LLM call. In a real application, you would integrate with a multilingual LLM API.
    # For this demo, we'll return a mock response based on the prompt.
    if "¿Cuál es el tiempo de entrega" in prompt:
        return "For Product A, delivery is 3-5 business days. For Product B, it's 7-10 business days."
    elif "Comment puis-je retourner" in prompt:
        return "You can return an item within 30 days if it's unused and in original packaging. Please contact customer support to start the return process."
    elif "Gibt es diesen Artikel in einer anderen Farbe" in prompt:
        return "Product C is available in Red, Blue, and Green. Product D is in Black and White."
    else:
        return "I'm sorry, I couldn't find a direct answer based on the provided context. Please provide more details or contact a human agent."

@app.post("/assist")
async def assist_customer(request: QueryRequest):
    customer_query = request.query
    detected_lang = _simulate_language_detection(customer_query)
    
    # Construct prompt using InCLT strategy
    prompt_for_llm = _construct_prompt(customer_query, detected_lang)
    
    # Simulate LLM call
    llm_response_en = _call_llm(prompt_for_llm)
    
    # In a real scenario, you'd translate the LLM's English response back to the detected_lang
    # For simplicity, we'll just return the English response and the constructed prompt.
    
    return {
        "customer_query": customer_query,
        "detected_language": detected_lang,
        "constructed_prompt": prompt_for_llm,
        "simulated_llm_response_en": llm_response_en,
        "message": "This is a demonstration of InCLT prompting. The LLM response is simulated."
    }