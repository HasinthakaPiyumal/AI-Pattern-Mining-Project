from fastapi import FastAPI
from pydantic import BaseModel
from langdetect import detect
import pandas as pd
from langchain.prompts import PromptTemplate, FewShotPromptTemplate

app = FastAPI()

examples = pd.DataFrame([
    {"source_lang": "en", "target_lang": "en", "query": "What is the return policy?", "response": "Our return policy allows returns within 30 days of purchase with a valid receipt."},
    {"source_lang": "es", "target_lang": "es", "query": "¿Cuál es la política de devoluciones?", "response": "Nuestra política de devoluciones permite devoluciones dentro de los 30 días de la compra con un recibo válido."},
    {"source_lang": "en", "target_lang": "es", "query": "How do I track my order?", "response": "Para rastrear su pedido, por favor use el número de seguimiento proporcionado en su correo electrónico de confirmación."},
    {"source_lang": "es", "target_lang": "en", "query": "¿Cómo rastreo mi pedido?", "response": "To track your order, please use the tracking number provided in your confirmation email."},
    {"source_lang": "fr", "target_lang": "fr", "query": "Comment suivre ma commande ?", "response": "Pour suivre votre commande, veuillez utiliser le numéro de suivi fourni dans votre email de confirmation."},
    {"source_lang": "en", "target_lang": "fr", "query": "Do you offer international shipping?", "response": "Oui, nous offrons la livraison internationale vers de nombreux pays. Veuillez consulter notre section d'expédition pour plus de détails."},
    {"source_lang": "fr", "target_lang": "en", "query": "Offrez-vous la livraison internationale ?", "response": "Yes, we offer international shipping to many countries. Please check our shipping section for more details."},
])

def generate_inclt_prompt(user_query: str, source_lang: str, target_lang: str) -> str:
    relevant_examples = []
    relevant_examples.extend(examples[((examples["source_lang"] == source_lang) & (examples["target_lang"] == target_lang)) |
                                    ((examples["source_lang"] == target_lang) & (examples["target_lang"] == source_lang)) | 
                                    ((examples["source_lang"] == source_lang) & (examples["target_lang"] == source_lang))
                                    ].to_dict(orient="records"))
    
    unique_examples = []
    seen = set()
    for ex in relevant_examples:
        example_tuple = (ex["query"], ex["response"])
        if example_tuple not in seen:
            unique_examples.append(ex)
            seen.add(example_tuple)
    
    selected_examples = unique_examples[:3]

    example_formatter_template = "User query in {source_lang}: {query}\nResponse in {target_lang}: {response}"
    example_prompt = PromptTemplate(
        input_variables=["source_lang", "target_lang", "query", "response"],
        template=example_formatter_template
    )

    few_shot_prompt = FewShotPromptTemplate(
        examples=selected_examples,
        example_prompt=example_prompt,
        prefix="You are a helpful customer support assistant. Respond to the user's query.\n"
               "Here are some examples of how to respond to queries in different languages, "
               "leveraging cross-lingual understanding:",
        suffix=f"User query in {source_lang}: {user_query}\nResponse in {target_lang}:",
        input_variables=["user_query", "source_lang", "target_lang"],
        example_separator="\n\n"
    )

    return few_shot_prompt.format(user_query=user_query, source_lang=source_lang, target_lang=target_lang)

def generate_response_from_llm(prompt: str) -> str:
    if "return policy" in prompt.lower() and "response in en" in prompt.lower():
        return "Our return policy allows returns within 30 days of purchase."
    elif "politica de devoluciones" in prompt.lower() and "response in es" in prompt.lower():
        return "Nuestra política de devoluciones permite devoluciones dentro de los 30 días."
    elif "track my order" in prompt.lower() and "response in en" in prompt.lower():
        return "Please use the tracking number provided in your confirmation email to track your order."
    elif "suivre ma commande" in prompt.lower() and "response in fr" in prompt.lower():
        return "Veuillez utiliser le numéro de suivi de votre e-mail de confirmation pour suivre votre commande."
    elif "shipping" in prompt.lower() and "response in en" in prompt.lower():
        return "Yes, we offer international shipping to many countries."
    elif "livraison internationale" in prompt.lower() and "response in fr" in prompt.lower():
        return "Oui, nous offrons la livraison internationale vers de nombreux pays."
    else:
        return f"Thank you for your query. (Simulated response for: '{prompt.splitlines()[-1]}')"

class ChatbotRequest(BaseModel):
    query: str
    target_language: str = "en"

@app.post("/chat")
async def chat_with_bot(request: ChatbotRequest):
    user_query = request.query
    target_lang = request.target_language

    try:
        source_lang = detect(user_query)
    except Exception:
        source_lang = "en"

    full_prompt = generate_inclt_prompt(user_query, source_lang, target_lang)

    llm_response = generate_response_from_llm(full_prompt)

    return {"response": llm_response, "source_language_detected": source_lang, "target_language": target_lang, "full_prompt_sent_to_llm": full_prompt} 