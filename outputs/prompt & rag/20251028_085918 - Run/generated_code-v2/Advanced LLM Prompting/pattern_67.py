import streamlit as st
import re

# --- Simulated Medical Dictionary Service ---
MEDICAL_DICTIONARY = {
    "angina": {
        "en": "a condition marked by severe pain in the chest, often also spreading to the shoulders, arms, and neck, caused by an inadequate blood supply to the heart",
        "es": "afección caracterizada por un dolor intenso en el pecho, que a menudo se extiende a los hombros, brazos y cuello, causada por un suministro insuficiente de sangre al corazón"
    },
    "myocardium": {
        "en": "the muscular tissue of the heart",
        "es": "el tejido muscular del corazón"
    },
    "hypertension": {
        "en": "abnormally high blood pressure",
        "es": "presión arterial anormalmente alta"
    },
    "diabetes": {
        "en": "a metabolic disease in which the body's inability to produce any or enough insulin causes elevated levels of glucose in the blood.",
        "es": "una enfermedad metabólica en la que la incapacidad del cuerpo para producir suficiente insulina o ninguna causa niveles elevados de glucosa en la sangre."
    }
}

# --- Simulated Translation Service (Placeholder for LLM/API) ---
def translate_text_simulated(prompt, source_lang, target_lang):
    original_text_match = re.search(r"Original text: '(.*?)'", prompt, re.DOTALL)
    if not original_text_match:
        return "Error: Could not extract original text from prompt."
    
    original_text = original_text_match.group(1)
    translated_parts = []

    # A very simplistic word-by-word translation for demonstration.
    # In a real scenario, an LLM or a sophisticated MT model would handle this.
    # We prioritize dictionary definitions if available.

    # Extract definitions from the prompt for direct use in translation
    term_definitions = {}
    definition_matches = re.findall(r"Definition for (.*?)\(({}): '(.*?)'\)", prompt)
    for match in definition_matches:
        term, lang, definition = match[0], match[1], match[2]
        if lang == target_lang:
            term_definitions[term.lower()] = definition
    
    # Attempt to use definitions for translation where available
    words = original_text.split()
    for word in words:
        cleaned_word = re.sub(r'[^\w]', '', word).lower()
        if cleaned_word in MEDICAL_DICTIONARY and target_lang in MEDICAL_DICTIONARY[cleaned_word]:
             translated_parts.append(MEDICAL_DICTIONARY[cleaned_word][target_lang].split(' ')[0]) # Just take the first word of definition as a simple replacement
        else:
            # Placeholder for actual translation of non-medical words
            # In a real LLM, the prompt would guide the LLM to translate normally
            translated_parts.append(word) # Keep original if no translation provided
    
    basic_translation = ' '.join(translated_parts)
    
    # Append the full definitions as context for the user
    full_definitions_context = []
    for term, def_text in term_definitions.items():
        full_definitions_context.append(f"  - {term.capitalize()} ({target_lang}): {def_text}")

    if full_definitions_context:
        return f"{basic_translation}\n\nRelevant Definitions:\n" + "\n".join(full_definitions_context)
    else:
        return basic_translation

# --- Term Identification & Prompt Engineering Module ---
def identify_medical_terms(text, medical_dictionary):
    identified_terms = []
    text_lower = text.lower()
    for term in medical_dictionary.keys():
        if re.search(r'\b' + re.escape(term) + r'\b', text_lower):
            identified_terms.append(term)
    return identified_terms

def craft_prompt(text, terms_with_definitions, source_lang, target_lang):
    prompt_parts = []
    prompt_parts.append(f"Translate the following medical text from {source_lang} to {target_lang}.\n")
    prompt_parts.append(f"Original text: '{text}'\n")

    if terms_with_definitions:
        prompt_parts.append("\nContextual definitions to aid translation:\n")
        for term, definitions in terms_with_definitions.items():
            if source_lang in definitions:
                prompt_parts.append(f"Definition for {term} ({source_lang}): '{definitions[source_lang]}'\n")
            if target_lang in definitions:
                prompt_parts.append(f"Definition for {term} ({target_lang}): '{definitions[target_lang]}'\n")
    
    prompt_parts.append("Please provide an accurate translation, leveraging the provided definitions where applicable.")
    return "".join(prompt_parts)

# --- Streamlit Application ---
st.set_page_config(layout="wide")
st.title("🩺 Medical Translation Platform (DiPMT)")
st.markdown("Translate complex medical reports and diagnoses with contextual dictionary definitions.")

with st.sidebar:
    st.header("Configuration")
    source_lang = st.selectbox("Source Language", [("English", "en"), ("Spanish", "es")], format_func=lambda x: x[0])
    target_lang = st.selectbox("Target Language", [("Spanish", "es"), ("English", "en")], format_func=lambda x: x[0])

    source_lang_code = source_lang[1]
    target_lang_code = target_lang[1]

if source_lang_code == target_lang_code:
    st.warning("Source and Target languages cannot be the same. Please select different languages.")
else:
    input_text = st.text_area("Enter Medical Text Here:", height=200)

    if st.button("Translate") and input_text:
        st.subheader("Translation Result")
        
        # 1. Term Identification
        identified_terms = identify_medical_terms(input_text, MEDICAL_DICTIONARY)
        
        terms_with_definitions = {}
        if identified_terms:
            for term in identified_terms:
                terms_with_definitions[term] = MEDICAL_DICTIONARY[term]

        # 2. Prompt Engineering
        enriched_prompt = craft_prompt(input_text, terms_with_definitions, source_lang_code, target_lang_code)
        
        st.text_area("Generated Prompt for Translation Model (for debug/demonstration)", enriched_prompt, height=250)

        # 3. Simulated Translation
        translated_output = translate_text_simulated(enriched_prompt, source_lang_code, target_lang_code)
        
        st.success("Translation Complete!")
        st.markdown("### Translated Text:")
        st.write(translated_output)

        if terms_with_definitions:
            st.markdown("### Definitions Used in Context:")
            for term, defs in terms_with_definitions.items():
                st.markdown(f"**{term.capitalize()}**")
                if source_lang_code in defs:
                    st.write(f"- **{source_lang[0]}**: {defs[source_lang_code]}")
                if target_lang_code in defs:
                    st.write(f"- **{target_lang[0]}**: {defs[target_lang_code]}")
    elif st.button("Translate") and not input_text:
        st.warning("Please enter some text to translate.")