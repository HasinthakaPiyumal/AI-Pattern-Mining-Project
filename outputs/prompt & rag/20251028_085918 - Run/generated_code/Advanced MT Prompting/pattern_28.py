"""main.py: Orchestrates the Multi-Strategy Translation Enhancement for Global Health Insights.

This script demonstrates the end-to-end process of translating medical documents,
incorporating pre-processing, prompt augmentation, task decomposition, and iterative refinement.
"""

from typing import List, Dict
import os

# Assuming these modules are in the same directory
from translation_module import preprocess_text, translate_segment
from augmentation_module import augment_prompt
from decomposition_module import segment_text
from refinement_module import assess_quality, request_human_feedback

def translate_document(source_text: str, source_lang: str, target_lang: str) -> Dict:
    """
    Translates a medical document using the multi-strategy enhancement pattern.

    Args:
        source_text (str): The original medical text to be translated.
        source_lang (str): The language code of the source text (e.g., 'es', 'fr', 'sw').
        target_lang (str): The language code for the desired translation (e.g., 'en', 'fr').

    Returns:
        Dict: A dictionary containing the translated text and a summary of the process.
    """
    print(f"\n--- Starting Translation Process for {source_lang} to {target_lang} ---")

    processed_text_for_genai = source_text
    preprocessed_flag = False

    # 1. Input Pre-processing: Translate to a high-resource language if source is low-resource
    # For this example, we'll assume English ('en') is the high-resource language for internal processing
    if source_lang != 'en':
        print(f"[Step 1] Pre-processing: Translating from {source_lang} to English for better GenAI input...")
        processed_text_for_genai = preprocess_text(source_text, source_lang, 'en')
        preprocessed_flag = True
        if not processed_text_for_genai:
            print("[ERROR] Pre-processing failed. Aborting.")
            return {"translated_text": "", "summary": "Pre-processing failed."}
        print("[Step 1] Pre-processing successful.")
    else:
        print("[Step 1] Pre-processing: Source language is already English. Skipping to segmentation.")

    # 2. Task Decomposition and Planning: Segment the (potentially pre-processed) text
    print("\n[Step 2] Task Decomposition: Segmenting text into manageable chunks...")
    segments = segment_text(processed_text_for_genai, 'en' if preprocessed_flag else source_lang)
    print(f"[Step 2] Text segmented into {len(segments)} parts.")

    translated_segments = []
    translation_notes = []

    for i, segment in enumerate(segments):
        print(f"\n--- Processing Segment {i+1}/{len(segments)} ---")

        # 3. Prompt Augmentation: Add external contextual information
        print("[Step 3] Prompt Augmentation: Retrieving medical context and definitions...")
        augmented_context = augment_prompt(segment, source_lang if not preprocessed_flag else 'en', target_lang)
        print("[Step 3] Augmentation complete. Context added.")

        # 4. GenAI Core Translation (incorporating pre-processing and augmentation)
        print("[Step 4] GenAI Translation: Translating segment with augmented prompt...")
        current_translated_segment = translate_segment(segment, target_lang, augmented_context)
        if not current_translated_segment:
            print(f"[ERROR] Translation failed for segment {i+1}. Skipping this segment.")
            translated_segments.append(f"[Translation Error for Segment {i+1}]")
            translation_notes.append(f"Segment {i+1} failed to translate.")
            continue
        print("[Step 4] Translation successful.")

        # 5. Iterative Refinement: Quality assessment and potential human feedback
        print("[Step 5] Iterative Refinement: Assessing translation quality...")
        quality_score = assess_quality(segment, current_translated_segment)
        translation_notes.append(f"Segment {i+1} quality score: {quality_score:.2f}.")
        print(f"[Step 5] Quality score for segment {i+1}: {quality_score:.2f}")

        if quality_score < 0.7:  # Example threshold for human review
            print("[Step 5] Low quality detected. Requesting human feedback...")
            human_feedback = request_human_feedback(
                segment,
                current_translated_segment,
                quality_score
            )
            if human_feedback:
                current_translated_segment = human_feedback # Incorporate human correction
                translation_notes.append(f"Segment {i+1} refined with human feedback.")
                print("[Step 5] Human feedback incorporated.")
            else:
                print("[Step 5] No human feedback provided or deemed necessary.")
        else:
            print("[Step 5] Quality is satisfactory. No human feedback needed.")

        translated_segments.append(current_translated_segment)

    final_translation = " ".join(translated_segments)

    summary = {
        "source_language": source_lang,
        "target_language": target_lang,
        "preprocessed_to_english": preprocessed_flag,
        "number_of_segments": len(segments),
        "translation_notes": translation_notes,
        "final_translated_text_length": len(final_translation)
    }

    print("\n--- Translation Process Complete ---")
    return {"translated_text": final_translation, "summary": summary}

if __name__ == "__main__":
    # Example Usage
    # Note: For actual execution, ensure you have internet access for downloading models
    # and that `nltk` data (e.g., 'punkt') is downloaded (nltk.download('punkt'))

    # Example 1: Low-resource language (Swahili) to English
    swahili_medical_text = (
        "Ugonjwa wa kisukari ni hali sugu inayojitokeza wakati kongosho halitoi insulini ya kutosha "
        "au wakati mwili hauwezi kutumia insulini inayotengenezwa vizuri. Hali hii husababisha "
        "kiwango cha sukari kwenye damu kuwa juu."
    )
    print("\n--- Running Example 1: Swahili to English ---")
    result1 = translate_document(swahili_medical_text, 'sw', 'en')
    print("\n--- Final Translation (Swahili to English) ---")
    print("Translated Text:", result1["translated_text"])
    print("Summary:", result1["summary"])

    # Example 2: Spanish to English (high-resource to high-resource, still benefits from augmentation/refinement)
    spanish_medical_text = (
        "La hipertensión arterial, o presión arterial alta, es una condición común "
        "en la que la fuerza a largo plazo de la sangre contra las paredes de las "
        "arterias es lo suficientemente alta como para eventualmente causar problemas de salud, "
        "como enfermedades cardíacas. Los síntomas a menudo no se presentan, lo que la convierte "
        "en una enfermedad 'silenciosa'."
    )
    print("\n--- Running Example 2: Spanish to English ---")
    result2 = translate_document(spanish_medical_text, 'es', 'en')
    print("\n--- Final Translation (Spanish to English) ---")
    print("Translated Text:", result2["translated_text"])
    print("Summary:", result2["summary"])

    # Example 3: English to French (demonstrating direct translation path)
    english_medical_text = (
        "Dengue fever is a mosquito-borne tropical disease caused by the dengue virus. "
        "Symptoms typically include a high fever, headache, vomiting, muscle and joint pains, "
        "and a characteristic skin rash."
    )
    print("\n--- Running Example 3: English to French ---")
    result3 = translate_document(english_medical_text, 'en', 'fr')
    print("\n--- Final Translation (English to French) ---")
    print("Translated Text:", result3["translated_text"])
    print("Summary:", result3["summary"])
