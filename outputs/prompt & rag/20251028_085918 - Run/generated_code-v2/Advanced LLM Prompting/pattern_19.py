import re

# Simulated Medical Dictionary
simulated_medical_dictionary = {
    "fever": {
        "en": "An abnormally high body temperature, usually accompanied by shivering, headache, and in severe instances, delirium.",
        "es": "Temperatura corporal anormalmente alta, generalmente acompañada de escalofríos, dolor de cabeza y, en casos graves, delirio."
    },
    "hypertension": {
        "en": "Abnormally high blood pressure.",
        "es": "Presión arterial anormalmente alta."
    },
    "diagnosis": {
        "en": "The identification of the nature of an illness or other problem by examination of the symptoms.",
        "es": "La identificación de la naturaleza de una enfermedad u otro problema mediante el examen de los síntomas."
    },
    "patient": {
        "en": "A person receiving or registered to receive medical treatment.",
        "es": "Una persona que recibe o está registrada para recibir tratamiento médico."
    }
}

def extract_medical_terms(document):
    extracted_terms = []
    # A simple approach: check if words in the document exist in our simulated dictionary
    words = re.findall(r'\b\w+\b', document.lower())
    for word in words:
        if word in simulated_medical_dictionary:
            extracted_terms.append(word)
    return list(set(extracted_terms))

def get_definitions(term, source_lang, target_lang):
    if term in simulated_medical_dictionary:
        definitions = simulated_medical_dictionary[term]
        source_def = definitions.get(source_lang, f"No {source_lang} definition found for '{term}'.")
        target_def = definitions.get(target_lang, f"No {target_lang} definition found for '{term}'.")
        return {"term": term, "source_definition": source_def, "target_definition": target_def}
    return None

def construct_dipmt_prompt(document, extracted_terms_with_definitions):
    prompt_parts = [f"Original Document to Translate: {document}", "\n--- Medical Terminology Context ---"]

    for term_info in extracted_terms_with_definitions:
        prompt_parts.append(f"Term: {term_info['term']}")
        prompt_parts.append(f"  Source Definition: {term_info['source_definition']}")
        prompt_parts.append(f"  Target Definition: {term_info['target_definition']}")
    
    prompt_parts.append("\n--- End Context ---")
    prompt_parts.append(f"Please translate the original document, incorporating the provided medical context for accuracy.")

    return "\n".join(prompt_parts)

def simulate_machine_translation(dipmt_prompt):
    # This is a highly simplified simulation of an MT model's output
    # In a real application, this would involve calling a sophisticated MT API/model.
    if "fever" in dipmt_prompt and "en" in dipmt_prompt and "es" in dipmt_prompt:
        return 