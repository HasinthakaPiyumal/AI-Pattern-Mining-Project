from transformers import pipeline

# In-Context Example Database (in-memory for demonstration)
in_context_examples = [
    {
        "source_language_query": "My internet is not working.",
        "source_language_response": "Please restart your router and modem. If the issue persists, contact technical support.",
        "target_language_code": "fr",
        "target_language_query": "Mon internet ne fonctionne pas.",
        "target_language_response": "Veuillez redémarrer votre routeur et votre modem. Si le problème persiste, contactez le support technique."
    },
    {
        "source_language_query": "How do I check my bill?",
        "source_language_response": "You can view your latest bill in your online account portal under 'Billing'.",
        "target_language_code": "fr",
        "target_language_query": "Comment consulter ma facture ?",
        "target_language_response": "Vous pouvez consulter votre dernière facture dans votre portail de compte en ligne sous 'Facturation'."
    },
    {
        "source_language_query": "I want to upgrade my plan.",
        "source_language_response": "You can explore upgrade options on our website or speak to a sales representative.",
        "target_language_code": "es",
        "target_language_query": "Quiero mejorar mi plan.",
        "target_language_response": "Puede explorar las opciones de actualización en nuestro sitio web o hablar con un representante de ventas."
    },
    {
        "source_language_query": "What are your operating hours?",
        "source_language_response": "Our customer support is available from 9 AM to 6 PM, Monday to Friday.",
        "target_language_code": "es",
        "target_language_query": "¿Cuáles son sus horas de operación?",
        "target_language_response": "Nuestro servicio de atención al cliente está disponible de 9 a.m. a 6 p.m., de lunes a viernes."
    }
]

# Initialize translation pipelines
en_fr_translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")
fr_en_translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
en_es_translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es")
es_en_translator = pipeline("translation", model="Helsinki-NLP/opus-mt-es-en")

def translate_text(text, src_lang, tgt_lang):
    if src_lang == tgt_lang:
        return text
    if src_lang == "en" and tgt_lang == "fr":
        return en_fr_translator(text)[0]["translation_text"]
    elif src_lang == "fr" and tgt_lang == "en":
        return fr_en_translator(text)[0]["translation_text"]
    elif src_lang == "en" and tgt_lang == "es":
        return en_es_translator(text)[0]["translation_text"]
    elif src_lang == "es" and tgt_lang == "en":
        return es_en_translator(text)[0]["translation_text"]
    else:
        # Fallback for unsupported language pairs or more robust model needed
        print(f"Warning: Translation for {src_lang} to {tgt_lang} not explicitly handled. Returning original text.")
        return text

def generate_inplt_prompt(query_english, query_target_lang, target_lang_code, examples):
    prompt = f"Here are some examples of customer queries and their resolutions in both English and {target_lang_code.upper()}:\n\n"

    for ex in examples:
        if ex["target_language_code"] == target_lang_code:
            prompt += f"English Query: {ex["source_language_query"]}\n"
            prompt += f"{target_lang_code.upper()} Query: {ex["target_language_query"]}\n"
            prompt += f"English Response: {ex["source_language_response"]}\n"
            prompt += f"{target_lang_code.upper()} Response: {ex["target_language_response"]}\n\n"
    
    prompt += f"English Query: {query_english}\n"
    prompt += f"{target_lang_code.upper()} Query: {query_target_lang}\n"
    prompt += f"English Response: [Expected English Response from LLM]\n"
    prompt += f"{target_lang_code.upper()} Response:"
    return prompt

def simulate_llm_response(prompt, target_lang_code):
    # In a real scenario, this would call an actual LLM (e.g., OpenAI API, Gemini API)
    # and process its completion. For this simulation, we'll extract the core query
    # and provide a generic translated response.
    
    # Simple heuristic to get the user's main query from the end of the prompt
    lines = prompt.strip().split('\n')
    # Look for the last 'X Query: ' where X is the target language
    last_target_query = ""
    for i in range(len(lines) - 1, -1, -1):
        if f"{target_lang_code.upper()} Query: " in lines[i]:
            last_target_query = lines[i].replace(f"{target_lang_code.upper()} Query: ", "").strip()
            break

    if "internet not working" in last_target_query.lower() or "internet ne fonctionne pas" in last_target_query.lower():
        en_response = "Please restart your router and modem. If the issue persists, contact technical support."
    elif "check my bill" in last_target_query.lower() or "consulter ma facture" in last_target_query.lower():
        en_response = "You can view your latest bill in your online account portal under 'Billing'."
    elif "upgrade my plan" in last_target_query.lower() or "mejorar mi plan" in last_target_query.lower():
        en_response = "You can explore upgrade options on our website or speak to a sales representative."
    elif "operating hours" in last_target_query.lower() or "horas de operación" in last_target_query.lower():
        en_response = "Our customer support is available from 9 AM to 6 PM, Monday to Friday."
    else:
        en_response = "I understand you have a query. Our team will be happy to assist you further."

    # Translate the simulated English response to the target language
    simulated_translated_response = translate_text(en_response, "en", target_lang_code)
    return simulated_translated_response

def get_chatbot_response(user_query, target_lang_code="en"):
    # 1. Translate the user query to English (if not already English)
    query_english = translate_text(user_query, target_lang_code, "en")
    
    # 2. Translate the English query to the target language (if not already target lang)
    query_target_lang = translate_text(query_english, "en", target_lang_code)

    # 3. Filter relevant examples (for this demo, we use all examples for the target_lang_code)
    relevant_examples = [ex for ex in in_context_examples if ex["target_language_code"] == target_lang_code]

    # 4. Generate the InCLT prompt
    prompt = generate_inplt_prompt(query_english, query_target_lang, target_lang_code, relevant_examples)
    
    # print(f"\n--- Generated Prompt for {target_lang_code.upper()} ---\n{prompt}\n---\n") # For debugging

    # 5. Simulate LLM response based on the prompt
    chatbot_response = simulate_llm_response(prompt, target_lang_code)
    
    return chatbot_response

if __name__ == "__main__":
    print("\n--- Multilingual Customer Support Chatbot (InCLT) ---\n")

    # Example 1: French Query
    user_query_fr = "J'ai un problème avec ma connexion internet."
    target_lang_fr = "fr"
    response_fr = get_chatbot_response(user_query_fr, target_lang_fr)
    print(f"Customer ({target_lang_fr.upper()}): {user_query_fr}")
    print(f"Chatbot ({target_lang_fr.upper()}): {response_fr}\n")

    # Example 2: Spanish Query
    user_query_es = "¿Quisiera saber sobre los horarios de atención?"
    target_lang_es = "es"
    response_es = get_chatbot_response(user_query_es, target_lang_es)
    print(f"Customer ({target_lang_es.upper()}): {user_query_es}")
    print(f"Chatbot ({target_lang_es.upper()}): {response_es}\n")

    # Example 3: English Query (should still use InCLT with English examples)
    user_query_en = "My monthly bill seems too high."
    target_lang_en = "en"
    response_en = get_chatbot_response(user_query_en, target_lang_en)
    print(f"Customer ({target_lang_en.upper()}): {user_query_en}")
    print(f"Chatbot ({target_lang_en.upper()}): {response_en}\n")

    # Example 4: A new query not directly in examples, in French
    user_query_fr_new = "Je ne trouve pas mon identifiant de client."
    target_lang_fr_new = "fr"
    response_fr_new = get_chatbot_response(user_query_fr_new, target_lang_fr_new)
    print(f"Customer ({target_lang_fr_new.upper()}): {user_query_fr_new}")
    print(f"Chatbot ({target_lang_fr_new.upper()}): {response_fr_new}\n")
