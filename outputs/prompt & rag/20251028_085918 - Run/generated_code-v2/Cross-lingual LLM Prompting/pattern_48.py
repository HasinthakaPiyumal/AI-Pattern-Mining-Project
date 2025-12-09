import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

KNOWLEDGE_BASE = {
    "shipping": {
        "en": "Our standard shipping takes 3-5 business days. Express shipping takes 1-2 business days. Free shipping on orders over $50.",
        "es": "Nuestro envío estándar tarda de 3 a 5 días hábiles. El envío exprés tarda de 1 a 2 días hábiles. Envío gratuito en pedidos superiores a $50.",
        "fr": "Notre expédition standard prend 3-5 jours ouvrables. L'expédition express prend 1-2 jours ouvrables. Livraison gratuite pour les commandes de plus de 50 $.",
    },
    "return": {
        "en": "You can return items within 30 days of purchase with a valid receipt. Items must be in their original condition.",
        "es": "Puede devolver artículos dentro de los 30 días posteriores a la compra con un recibo válido. Los artículos deben estar en su estado original.",
        "fr": "Vous pouvez retourner les articles dans les 30 jours suivant l'achat avec un reçu valide. Les articles doivent être dans leur état d'origine.",
    },
    "contact": {
        "en": "You can contact our support team via email at support@example.com or call us at 1-800-123-4567.",
        "es": "Puede ponerse en contacto con nuestro equipo de soporte por correo electrónico en support@example.com o llamarnos al 1-800-123-4567.",
        "fr": "Vous pouvez contacter notre équipe d'assistance par e-mail à support@example.com ou nous appeler au 1-800-123-4567.",
    },
    "product": {
        "en": "We offer a wide range of electronics, apparel, and home goods.",
        "es": "Ofrecemos una amplia gama de productos electrónicos, ropa y artículos para el hogar.",
        "fr": "Nous proposons une large gamme d'électronique, de vêtements et d'articles pour la maison.",
    }
}

model_name = "google/mt5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def detect_language_placeholder(text):
    text_lower = text.lower()
    if any(word in text_lower for word in ["envío", "pedido", "devolver", "español", "dónde", "cuál"]):
        return "es"
    elif any(word in text_lower for word in ["expédition", "retourner", "produit", "français", "comment", "quelle"]):
        return "fr"
    return "en"

def retrieve_knowledge(query, lang):
    query_lower = query.lower()
    for key, translations in KNOWLEDGE_BASE.items():
        if key in query_lower or any(q_word in query_lower for q_word in [key, "policy", "contact", "product", "cost", "status", "return"]):
            if lang in translations:
                return translations[lang], lang
            return translations.get("en", "No specific information found."), "en"
    return "No specific information found.", None

def create_inclt_prompt(user_query, detected_lang, kb_info, kb_info_lang):
    prompt_template = f"""
As a multilingual e-commerce customer support assistant, answer the user's question accurately and concisely. Use the provided knowledge base information if it is relevant. Generate the response strictly in the user's language.

---
In-Context Learning Examples (demonstrating cross-lingual transfer):

User Query (en): What is the shipping cost?
Knowledge Base (es): El envío estándar cuesta $5. Envío exprés $10.
Assistant Response (en): Standard shipping costs $5. Express shipping costs $10.
---
User Query (es): ¿Dónde está mi pedido?
Knowledge Base (en): Your order is currently in transit and is expected to arrive within 2 days.
Assistant Response (es): Su pedido está actualmente en tránsito y se espera que llegue en 2 días.
---
User Query (fr): Comment puis-je retourner un produit défectueux?
Knowledge Base (fr): Pour retourner un produit défectueux, veuillez contacter notre service client avec votre numéro de commande.
Assistant Response (fr): Pour retourner un produit défectueux, veuillez contacter notre service client avec votre numéro de commande.
---
User Query (en): How can I return an item?
Knowledge Base (es): Puede devolver artículos dentro de los 30 días posteriores a la compra con un recibo válido.
Assistant Response (en): You can return items within 30 days of purchase with a valid receipt.
---

---
Current Customer Interaction:

User Query ({detected_lang}): {user_query}
"""
    if kb_info and kb_info_lang:
        prompt_template += f"Knowledge Base Information ({kb_info_lang}): {kb_info}\n"
    else:
        prompt_template += "Knowledge Base Information: No specific information found.\n"

    prompt_template += f"Assistant Response ({detected_lang}):"
    return prompt_template.strip()

def multilingual_chatbot(user_query, history):
    detected_lang = detect_language_placeholder(user_query)
    kb_info, kb_info_lang = retrieve_knowledge(user_query, detected_lang)
    prompt = create_inclt_prompt(user_query, detected_lang, kb_info, kb_info_lang)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_new_tokens=100, num_beams=5, early_stopping=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    history = history or []
    history.append((user_query, response))
    return "", history

gr.ChatInterface(
    multilingual_chatbot,
    title="Multilingual E-commerce Support Chatbot (InCLT Prompting)",
    description="Ask questions in English, Spanish, or French about shipping, returns, or products. The chatbot uses cross-lingual in-context learning.",
).launch()