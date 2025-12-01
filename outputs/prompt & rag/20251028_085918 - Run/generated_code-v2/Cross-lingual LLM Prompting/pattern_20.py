import random
import math

# 1. FAQ Knowledge Base
faq_knowledge_base = [
    {"id": 1, "question_en": "What are your operating hours?", "answer_en": "Our operating hours are Monday to Friday, 9 AM to 5 PM."},
    {"id": 2, "question_en": "How can I contact customer support?", "answer_en": "You can contact us via email at support@example.com or call us at 1-800-123-4567."},
    {"id": 3, "question_en": "Do you offer international shipping?", "answer_en": "Yes, we offer international shipping to most countries. Shipping costs and delivery times vary by destination."},
    {"id": 4, "question_en": "What is your return policy?", "answer_en": "Items can be returned within 30 days of purchase with a valid receipt for a full refund."}, 
    {"id": 5, "question_en": "What payment methods do you accept?", "answer_en": "We accept major credit cards (Visa, Mastercard, Amex), PayPal, and bank transfers."},
    {"id": 6, "question_en": "How do I track my order?", "answer_en": "You can track your order using the tracking number provided in your shipping confirmation email."}
]

# 2. Mock Embedding Model
def get_embedding_mock(text):
    # In a real scenario, this would use a sentence-transformers model (e.g., distiluse-base-multilingual-cased-v1)
    # For demonstration, return a random vector of a fixed size
    random.seed(hash(text) % (2**32 - 1)) # Seed for consistent mock embeddings for the same text
    return [random.uniform(-1, 1) for _ in range(768)] # A common embedding dimension

# Pre-calculate embeddings for FAQ questions (mock)
faq_embeddings = []
for faq in faq_knowledge_base:
    faq_embeddings.append({
        "id": faq["id"],
        "embedding": get_embedding_mock(faq["question_en"])
    })

# 3. Mock Vector Database & Similarity Search
def cosine_similarity(vec1, vec2):
    dot_product = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(v**2 for v in vec1))
    magnitude2 = math.sqrt(sum(v**2 for v in vec2))
    if magnitude1 == 0 or magnitude2 == 0:
        return 0
    return dot_product / (magnitude1 * magnitude2)

def retrieve_relevant_faq_mock(query_embedding):
    highest_similarity = -1
    most_relevant_faq_id = None

    for faq_emb in faq_embeddings:
        similarity = cosine_similarity(query_embedding, faq_emb["embedding"])
        if similarity > highest_similarity:
            highest_similarity = similarity
            most_relevant_faq_id = faq_emb["id"]
    
    if most_relevant_faq_id is not None:
        return next(faq for faq in faq_knowledge_base if faq["id"] == most_relevant_faq_id)
    return None

# 4. Language Detection Module (Simplified)
def detect_language_mock(text):
    text_lower = text.lower()
    if "hola" in text_lower or "¿qué tal" in text_lower or "¿cuál es" in text_lower or "horario" in text_lower or "contactar" in text_lower:
        return "es"
    if "bonjour" in text_lower or "comment" in text_lower or "quelle est" in text_lower or "heures" in text_lower or "contacter" in text_lower:
        return "fr"
    if "hallo" in text_lower or "wie geht" in text_lower or "was ist" in text_lower or "öffnungszeiten" in text_lower or "kontaktieren" in text_lower:
        return "de"
    return "en" # Default to English

# 5. Prompt Engineering Module (Core of InCLT)
def construct_inclt_prompt(user_query, detected_language, retrieved_faq_question_en, retrieved_faq_answer_en):
    # Hand-crafted in-context examples for cross-lingual transfer
    in_context_examples = [
        {
            "user_query_lang": "es",
            "user_query_text": "¿Cuál es el horario de la tienda?",
            "retrieved_faq_q_en": "What are your operating hours?",
            "retrieved_faq_a_en": "Our operating hours are Monday to Friday, 9 AM to 5 PM.",
            "chatbot_response_lang": "es",
            "chatbot_response_text": "Nuestros horarios de atención son de lunes a viernes, de 9 a. m. a 5 p. m."
        },
        {
            "user_query_lang": "fr",
            "user_query_text": "Comment puis-je contacter le support client?",
            "retrieved_faq_q_en": "How can I contact customer support?",
            "retrieved_faq_a_en": "You can contact us via email at support@example.com or call us at 1-800-123-4567.",
            "chatbot_response_lang": "fr",
            "chatbot_response_text": "Vous pouvez nous contacter par e-mail à support@example.com ou nous appeler au 1-800-123-4567."
        },
        {
            "user_query_lang": "de",
            "user_query_text": "Bieten Sie internationalen Versand an?",
            "retrieved_faq_q_en": "Do you offer international shipping?",
            "retrieved_faq_a_en": "Yes, we offer international shipping to most countries. Shipping costs and delivery times vary by destination.",
            "chatbot_response_lang": "de",
            "chatbot_response_text": "Ja, wir bieten internationalen Versand in die meisten Länder an. Die Versandkosten und Lieferzeiten variieren je nach Zielort."
        }
    ]

    prompt_parts = [
        "You are a helpful multilingual customer support assistant.",
        "Your task is to answer user questions in their original language, by finding relevant information from the provided English FAQ and translating it appropriately.",
        "Here are some examples of how to respond:"
    ]

    for ex in in_context_examples:
        prompt_parts.append(f"User's Query ({ex['user_query_lang']}): \"{ex['user_query_text']}\"")
        prompt_parts.append(f"Retrieved FAQ (en):\nQuestion: \"{ex['retrieved_faq_q_en']}\"\nAnswer: \"{ex['retrieved_faq_a_en']}\"")
        prompt_parts.append(f"Chatbot Response ({ex['chatbot_response_lang']}): \"{ex['chatbot_response_text']}\"")
        prompt_parts.append("") # Add a newline for separation

    prompt_parts.append(f"User's Query ({detected_language}): \"{user_query}\"")
    prompt_parts.append(f"Retrieved FAQ (en):\nQuestion: \"{retrieved_faq_question_en}\"\nAnswer: \"{retrieved_faq_answer_en}\"")
    prompt_parts.append(f"Chatbot Response ({detected_language}):")

    return "\n".join(prompt_parts)

# 6. Multilingual Large Language Model (Mock)
def mock_llm_response(prompt):
    # Extract detected language from the prompt
    target_lang = "en"
    lang_match_str = "Chatbot Response ("
    lang_start_idx = prompt.rfind(lang_match_str)
    if lang_start_idx != -1:
        lang_start_idx += len(lang_match_str)
        lang_end_idx = prompt.find("):", lang_start_idx)
        if lang_end_idx != -1:
            target_lang = prompt[lang_start_idx:lang_end_idx].strip()

    # Extract the retrieved FAQ answer from the prompt
    retrieved_answer_en = "Information not available."
    faq_ans_match_str = "Answer: \""
    faq_ans_start_idx = prompt.rfind(faq_ans_match_str)
    if faq_ans_start_idx != -1:
        faq_ans_start_idx += len(faq_ans_match_str)
        faq_ans_end_idx = prompt.find("\"", faq_ans_start_idx)
        if faq_ans_end_idx != -1:
            retrieved_answer_en = prompt[faq_ans_start_idx:faq_ans_end_idx]

    # Simple heuristic-based translation for demonstration
    if "operating hours" in retrieved_answer_en.lower():
        if target_lang == "es":
            return "Nuestros horarios de atención son de lunes a viernes, de 9 a. m. a 5 p. m."
        elif target_lang == "fr":
            return "Nos heures d'ouverture sont du lundi au vendredi, de 9h à 17h."
        elif target_lang == "de":
            return "Unsere Öffnungszeiten sind Montag bis Freitag, 9 Uhr bis 17 Uhr."
        else: # Default or English
            return retrieved_answer_en
    elif "contact us" in retrieved_answer_en.lower() or "contact customer support" in retrieved_answer_en.lower():
        if target_lang == "es":
            return "Puede contactarnos por correo electrónico a support@example.com o llamarnos al 1-800-123-4567."
        elif target_lang == "fr":
            return "Vous pouvez nous contacter par e-mail à support@example.com ou nous appeler au 1-800-123-4567."
        elif target_lang == "de":
            return "Sie können uns per E-Mail unter support@example.com kontaktieren oder uns unter 1-800-123-4567 anrufen."
        else: # Default or English
            return retrieved_answer_en
    elif "international shipping" in retrieved_answer_en.lower():
        if target_lang == "es":
            return "Sí, ofrecemos envío internacional a la mayoría de los países. Los costos de envío y los tiempos de entrega varían según el destino."
        elif target_lang == "fr":
            return "Oui, nous proposons l'expédition internationale vers la plupart des pays. Les frais d'expédition et les délais de livraison varient selon la destination."
        elif target_lang == "de":
            return "Ja, wir bieten internationalen Versand in die meisten Länder an. Die Versandkosten und Lieferzeiten variieren je nach Zielort."
        else: # Default or English
            return retrieved_answer_en
    elif "return policy" in retrieved_answer_en.lower():
        if target_lang == "es":
            return "Los artículos se pueden devolver dentro de los 30 días posteriores a la compra con un recibo válido para un reembolso completo."
        elif target_lang == "fr":
            return "Les articles peuvent être retournés dans les 30 jours suivant l'achat avec un reçu valide pour un remboursement complet."
        elif target_lang == "de":
            return "Artikel können innerhalb von 30 Tagen nach dem Kauf mit einem gültigen Beleg für eine vollständige Rückerstattung zurückgegeben werden."
        else: # Default or English
            return retrieved_answer_en
    elif "payment methods" in retrieved_answer_en.lower():
        if target_lang == "es":
            return "Aceptamos las principales tarjetas de crédito (Visa, Mastercard, Amex), PayPal y transferencias bancarias."
        elif target_lang == "fr":
            return "Nous acceptons les principales cartes de crédit (Visa, Mastercard, Amex), PayPal et les virements bancaires."
        elif target_lang == "de":
            return "Wir akzeptieren gängige Kreditkarten (Visa, Mastercard, Amex), PayPal und Banküberweisungen."
        else: # Default or English
            return retrieved_answer_en
    elif "track my order" in retrieved_answer_en.lower():
        if target_lang == "es":
            return "Puede rastrear su pedido utilizando el número de seguimiento proporcionado en su correo electrónico de confirmación de envío."
        elif target_lang == "fr":
            return "Vous pouvez suivre votre commande en utilisant le numéro de suivi fourni dans votre e-mail de confirmation d'expédition."
        elif target_lang == "de":
            return "Sie können Ihre Bestellung mit der Sendungsverfolgungsnummer verfolgen, die in Ihrer Versandbestätigungs-E-Mail angegeben ist."
        else: # Default or English
            return retrieved_answer_en

    # Fallback if no specific translation rule applies
    if target_lang == "es":
        return f"Según la información, la respuesta es: {retrieved_answer_en}. (Traducción simulada)"
    elif target_lang == "fr":
        return f"Selon les informations, la réponse est : {retrieved_answer_en}. (Traduction simulée)"
    elif target_lang == "de":
        return f"Laut den Informationen lautet die Antwort: {retrieved_answer_en}. (Simulierte Übersetzung)"
    else:
        return retrieved_answer_en

def run_chatbot(user_query):
    # 1. Language Detection
    detected_language = detect_language_mock(user_query)
    print(f"Detected language: {detected_language}")

    # 2. Embedding User Query
    query_embedding = get_embedding_mock(user_query)

    # 3. FAQ Retrieval
    relevant_faq = retrieve_relevant_faq_mock(query_embedding)

    if not relevant_faq:
        return "Sorry, I could not find relevant information for your query."

    retrieved_faq_question_en = relevant_faq["question_en"]
    retrieved_faq_answer_en = relevant_faq["answer_en"]
    print(f"Retrieved FAQ (EN): Q: \"{retrieved_faq_question_en}\" A: \"{retrieved_faq_answer_en}\"")

    # 4. Prompt Construction (InCLT)
    prompt = construct_inclt_prompt(
        user_query,
        detected_language,
        retrieved_faq_question_en,
        retrieved_faq_answer_en
    )
    # print("\n--- Constructed LLM Prompt ---\n", prompt, "\n----------------------------\n") # Uncomment to see the full prompt

    # 5. LLM Inference
    chatbot_response = mock_llm_response(prompt)

    return chatbot_response

if __name__ == "__main__":
    print("Multilingual Customer Support Chatbot (InCLT Demo)")
    print("Type 'exit' to end the conversation.")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            break
        
        response = run_chatbot(user_input)
        print(f"Chatbot: {response}")
