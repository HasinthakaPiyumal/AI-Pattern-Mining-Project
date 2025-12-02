medical_dictionary = {
    "fr": {
        "malin": {
            "definition": "Caractérise une tumeur ou une maladie qui a tendance à s'aggraver, à envahir les tissus voisins ou à métastaser, menaçant la vie du patient.",
            "en_equivalent": "malignant"
        },
        "bénin": {
            "definition": "Caractérise une tumeur ou une maladie qui n'est pas cancéreuse, ne se propage pas et n'est généralement pas mortelle.",
            "en_equivalent": "benign"
        }
    },
    "en": {
        "malignant": {
            "definition": "Tending to invade normal tissue and recur or spread to distant parts of the body : cancerous.",
            "fr_equivalent": "malin"
        },
        "benign": {
            "definition": "Of a growth or tumor) not harmful in effect : in particular, not malignant.",
            "fr_equivalent": "bénin"
        }
    }
}

def get_medical_term_definition(term, lang, target_lang):
    if lang in medical_dictionary and term in medical_dictionary[lang]:
        term_data = medical_dictionary[lang][term]
        definition = term_data["definition"]
        target_term = term_data.get(f"{target_lang}_equivalent", "")
        target_definition = ""
        if target_term and target_lang in medical_dictionary and target_term in medical_dictionary[target_lang]:
            target_definition = medical_dictionary[target_lang][target_term]["definition"]
        return definition, target_term, target_definition
    return None, None, None

def identify_medical_terms(text, source_lang, predefined_terms):
    found_terms = []
    text_lower = text.lower()
    for term in predefined_terms:
        if term.lower() in text_lower:
            found_terms.append(term)
    return found_terms

def construct_translation_prompt(original_text, source_lang, target_lang, identified_terms_data):
    prompt_parts = []
    for term_data in identified_terms_data:
        source_term = term_data["source_term"]
        source_def = term_data["source_definition"]
        target_term = term_data["target_term"]
        target_def = term_data["target_definition"]

        prompt_parts.append(f"Source Language Term: {source_term}")
        if source_def:
            prompt_parts.append(f"Source Language Definition: {source_def}")
        if target_term:
            prompt_parts.append(f"Target Language Term: {target_term}")
        if target_def:
            prompt_parts.append(f"Target Language Definition: {target_def}")
        prompt_parts.append("")

    prompt_parts.append(f"Translate the following medical text from {source_lang.upper()} to {target_lang.upper()}:")
    prompt_parts.append(original_text)
    return "\n".join(prompt_parts)

def simulated_machine_translation(enriched_prompt):
    print("\n--- SIMULATED MT MODEL INPUT (Enriched Prompt) ---")
    print(enriched_prompt)
    print("---------------------------------------------------\n")

    if "Source Language Term: malin" in enriched_prompt and "Target Language Term: malignant" in enriched_prompt:
        return "The patient has a malignant tumor that requires immediate attention."
    elif "Source Language Term: bénin" in enriched_prompt and "Target Language Term: benign" in enriched_prompt:
        return "The diagnosis revealed a benign cyst, which is not a cause for concern."
    else:
        return "Simulated translation (context applied): The document discusses medical conditions."

def medical_document_translator(document_text, source_lang, target_lang):
    predefined_terms = list(medical_dictionary.get(source_lang, {}).keys())
    identified_terms = identify_medical_terms(document_text, source_lang, predefined_terms)

    identified_terms_data = []
    for term in identified_terms:
        source_def, target_term_equivalent, target_def = get_medical_term_definition(term, source_lang, target_lang)
        if source_def:
            identified_terms_data.append({
                "source_term": term,
                "source_definition": source_def,
                "target_term": target_term_equivalent,
                "target_definition": target_def
            })

    enriched_prompt = construct_translation_prompt(document_text, source_lang, target_lang, identified_terms_data)
    translated_text = simulated_machine_translation(enriched_prompt)

    return translated_text

# Example Usage:
if __name__ == "__main__":
    source_text_1 = "Le patient présente une tumeur maligne au poumon. Il faut intervenir rapidement."
    translated_text_1 = medical_document_translator(source_text_1, "fr", "en")
    print(f"Original (FR): {source_text_1}")
    print(f"Translated (EN): {translated_text_1}\n")

    source_text_2 = "Le rapport indique une lésion bénigne, sans signe de malignité."
    translated_text_2 = medical_document_translator(source_text_2, "fr", "en")
    print(f"Original (FR): {source_text_2}")
    print(f"Translated (EN): {translated_text_2}\n")

    source_text_3 = "The patient's malignant tumor required surgery."
    translated_text_3 = medical_document_translator(source_text_3, "en", "fr")
    print(f"Original (EN): {source_text_3}")
    print(f"Translated (FR): {translated_text_3}\n")

    source_text_4 = "This is a general medical statement."
    translated_text_4 = medical_document_translator(source_text_4, "en", "fr")
    print(f"Original (EN): {source_text_4}")
    print(f"Translated (FR): {translated_text_4}\n")
