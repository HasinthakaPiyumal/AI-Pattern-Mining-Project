class TextExtractor:
    def extract_text(self, document_content):
        return document_content

class TextChunker:
    def chunk_text(self, text, max_chunk_size=500):
        sentences = []
        current_chunk = []
        current_length = 0

        # Simple sentence tokenizer by splitting at common delimiters.
        # This is a very basic simulation, real NLTK/SpaCy would be used.
        temp_sentences = text.replace(". ", ".<SPLIT>").replace("! ", "!<SPLIT>").replace("? ", "?<SPLIT>").split("<SPLIT>")
        
        for sent in temp_sentences:
            if not sent.strip():
                continue
            # Re-add the delimiter that was stripped for splitting
            if not (sent.strip().endswith('.') or sent.strip().endswith('!') or sent.strip().endswith('?')):
                sent = sent.strip() + "."
            else:
                sent = sent.strip()
                
            if current_length + len(sent) < max_chunk_size:
                current_chunk.append(sent)
                current_length += len(sent)
            else:
                if current_chunk:
                    sentences.append(" ".join(current_chunk))
                current_chunk = [sent]
                current_length = len(sent)
        if current_chunk:
            sentences.append(" ".join(current_chunk))
        return sentences

class LLMSimulator:
    def __init__(self):
        self.few_shot_medical_examples = {
            "en_to_es": [
                {"source": "The patient exhibits symptoms of severe pneumonia.", "target": "El paciente presenta síntomas de neumonía severa.", "term": "pneumonia"},
                {"source": "Administer 200mg of ibuprofen every 8 hours.", "target": "Administrar 200 mg de ibuprofeno cada 8 horas.", "term": "ibuprofen"}
            ],
            "en_to_de": [
                {"source": "The biopsy results confirm malignancy.", "target": "Die Biopsieergebnisse bestätigen Malignität.", "term": "malignancy"},
                {"source": "Prescribe medication for hypertension.", "target": "Medikamente gegen Bluthochdruck verschreiben.", "term": "hypertension"}
            ]
        }
        self.medical_lexicon = {
            "en_to_es": {"pneumonia": "neumonía", "hypertension": "hipertensión", "malignancy": "malignidad", "ibuprofen": "ibuprofeno", "diagnosis": "diagnóstico"},
            "en_to_de": {"pneumonia": "Pneumonie", "hypertension": "Bluthochdruck", "malignancy": "Malignität", "ibuprofen": "Ibuprofen", "diagnosis": "Diagnose"}
        }

    def _get_few_shot_prompt(self, chunk, target_language):
        examples = self.few_shot_medical_examples.get(f"en_to_{target_language}", [])
        prompt_parts = ["Translate the following medical text. Ensure medical terminology is accurate and consistent."]
        
        for ex in examples:
            prompt_parts.append(f"Source: {ex['source']}\nTarget: {ex['target']}")
        
        prompt_parts.append(f"Source: {chunk}\nTarget:")
        return "\n\n".join(prompt_parts)

    def translate_chunk(self, chunk, target_language):
        if not chunk:
            return ""
        
        # Simulate LLM translation with basic term replacement and appending a suffix
        translated_chunk = chunk
        lang_lexicon = self.medical_lexicon.get(f"en_to_{target_language}", {})

        for en_term, target_term in lang_lexicon.items():
            translated_chunk = translated_chunk.replace(en_term, target_term)
        
        if target_language == "es":
            translated_chunk = translated_chunk.replace("patient", "paciente").replace("symptoms", "síntomas")
            translated_chunk = translated_chunk + " (simulated Spanish translation)"
        elif target_language == "de":
            translated_chunk = translated_chunk.replace("patient", "Patient").replace("symptoms", "Symptome")
            translated_chunk = translated_chunk + " (simulated German translation)"
        else:
            translated_chunk = chunk + f" (simulated translation to {target_language})"

        return translated_chunk

    def refine_translation(self, translated_text, original_text, target_language):
        # Simulate refinement for consistency and coherence
        # In a real scenario, this would involve another LLM call or complex NLP logic
        lang_lexicon = self.medical_lexicon.get(f"en_to_{target_language}", {})
        
        refined_text = translated_text
        for en_term, target_term in lang_lexicon.items():
            if target_term not in refined_text and en_term in original_text:
                # Simple rule: if an original term's translation is missing, add a note
                refined_text += f" [CONSISTENCY_NOTE: Ensure '{target_term}' is present for '{en_term}']"
        
        refined_text += " [CONTEXTUAL_REVIEWED]"
        return refined_text

class ContextualMerger:
    def __init__(self, llm_simulator):
        self.llm_simulator = llm_simulator
        self.medical_ontology = {
            "pneumonia": ["lung infection", "respiratory disease"],
            "hypertension": ["high blood pressure", "cardiovascular condition"],
            "malignancy": ["cancerous", "tumor"]
        }

    def _check_terminology_consistency(self, translated_chunks, target_language):
        # A very basic simulation of consistency check
        issues = []
        target_lexicon = self.llm_simulator.medical_lexicon.get(f"en_to_{target_language}", {})
        
        all_translated_text = " ".join(translated_chunks).lower()
        for en_term, target_term in target_lexicon.items():
            # Check if the translated term appears at all in the merged text
            if target_term.lower() not in all_translated_text:
                issues.append(f"Warning: Term '{en_term}' (translated as '{target_term}') might be missing or inconsistent.")
        return issues

    def merge_and_refine(self, translated_chunks, original_text, target_language):
        initial_merged_translation = " ".join(translated_chunks)
        
        consistency_issues = self._check_terminology_consistency(translated_chunks, target_language)
        
        # Simulate refinement using the LLM Simulator
        final_refined_translation = self.llm_simulator.refine_translation(
            initial_merged_translation, original_text, target_language
        )
        
        if consistency_issues:
            final_refined_translation += "\n\nConsistency Review Notes:\n" + "\n".join(consistency_issues)
            
        return final_refined_translation

class MedicalDocumentTranslator:
    def __init__(self):
        self.text_extractor = TextExtractor()
        self.text_chunker = TextChunker()
        self.llm_simulator = LLMSimulator()
        self.contextual_merger = ContextualMerger(self.llm_simulator)

    def translate_document(self, document_content, target_language):
        print(f"\n--- Starting translation for target language: {target_language.upper()} ---")
        
        # 1. Extract Text
        raw_text = self.text_extractor.extract_text(document_content)
        print(f"Original Document (first 200 chars): {raw_text[:200]}...")

        # 2. Chunk Text
        chunks = self.text_chunker.chunk_text(raw_text)
        print(f"Document chunked into {len(chunks)} parts.")

        translated_chunks = []
        for i, chunk in enumerate(chunks):
            # 3. Independent Translation (Few-shot Prompting simulated)
            print(f"\nTranslating chunk {i+1}/{len(chunks)}: {chunk[:100]}...")
            translated_chunk = self.llm_simulator.translate_chunk(chunk, target_language)
            translated_chunks.append(translated_chunk)
            print(f"Translated chunk {i+1}: {translated_chunk[:100]}...")

        # 4. Contextual Merging & Refinement
        print("\n--- Merging and refining translated chunks ---")
        final_translation = self.contextual_merger.merge_and_refine(
            translated_chunks, raw_text, target_language
        )
        
        print("\n--- Translation Complete ---")
        return final_translation

# --- Example Usage ---
if __name__ == "__main__":
    medical_document = (
        "The patient, John Doe, was admitted on 2023-10-26 with severe pneumonia. "
        "Diagnosis confirmed Streptococcus pneumoniae infection. The recommended treatment "
        "includes a 10-day course of Amoxicillin 500mg, three times daily. "
        "Further evaluation for underlying cardiovascular conditions, specifically hypertension, "
        "is advised. The patient's medical history indicates a previous diagnosis of "
        "mild hypertension managed with lifestyle changes. No signs of malignancy were observed "
        "during the initial assessment. The prognosis is good with proper adherence to medication. "
        "Follow-up appointment scheduled in two weeks. All tests were conclusive. "
        "The nurse recorded stable vital signs throughout the stay. The discharge summary will be issued tomorrow."
    )

    translator = MedicalDocumentTranslator()

    # Translate to Spanish
    spanish_translation = translator.translate_document(medical_document, "es")
    print("\n*** Final Spanish Translation ***")
    print(spanish_translation)

    print("\n" + "="*80 + "\n")

    # Translate to German
    german_translation = translator.translate_document(medical_document, "de")
    print("\n*** Final German Translation ***")
    print(german_translation)

