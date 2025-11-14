import re

class MedicalTranslator:
    def __init__(self, medical_ontology: dict):
        self.medical_ontology = medical_ontology
        self.translation_memory = {}

    def _segment_text(self, text: str) -> list:
        """Segments a given text into sentences."""
        # A simple regex for sentence segmentation. More sophisticated methods exist.
        sentences = re.split(r'(?<=[.!?]) +', text.strip())
        return [s for s in sentences if s]

    def _enrich_context(self, segment: str, source_lang: str = "en") -> dict:
        """Extracts medical terms from a segment and retrieves their contextual information from the ontology."""
        context_data = {}
        # This is a simplified term extraction. In a real system, NLP would be used.
        words = re.findall(r'\b\w+\b', segment.lower())
        for word in words:
            if word in self.medical_ontology:
                context_data[word] = self.medical_ontology[word]
        return context_data

    def _mock_translate_segment(self, segment: str, context: dict, target_lang: str = "es") -> str:
        """Simulates translation of a single segment, using context."""
        translated_segment = segment
        # Apply translation memory first
        if segment in self.translation_memory:
            return self.translation_memory[segment]

        # Prioritize terms from the medical ontology
        for term, translation in context.items():
            # Simple replacement for demonstration. A real system would be more nuanced.
            translated_segment = re.sub(r'\b' + re.escape(term) + r'\b', translation, translated_segment, flags=re.IGNORECASE)
        
        # Mock generic translation for remaining parts if no specific context was found for entire segment
        # For demonstration, we'll just append a language indicator if no specific translation happened
        if translated_segment == segment: # If no ontology terms were replaced
            translated_segment = f"[Translated to {target_lang}: {segment}]"
        
        return translated_segment

    def _automated_feedback(self, original_segment: str, translated_segment: str) -> dict:
        """Simulates automated checks for translation quality (e.g., flagging unknown terms)."""
        feedback = {"issues": []}
        # Example: Check if any part still looks like the original language (very basic)
        if "[Translated to" not in translated_segment and translated_segment == original_segment:
             feedback["issues"].append("Potential untranslated segment.")
        # In a real system, this would involve back-translation, term consistency checks, etc.
        return feedback

    def _human_in_the_loop_review(self, original_segment: str, translated_segment: str) -> str:
        """Simulates human review and correction."""
        print(f"\n--- Human Review Required ---")
        print(f"Original: {original_segment}")
        print(f"Proposed Translation: {translated_segment}")
        # For a real application, this would be a UI prompt.
        # Here, we'll simulate a simple correction or acceptance.
        # input_correction = input("Enter corrected translation (or press Enter to accept): ")
        # if input_correction:
        #     return input_correction
        # else:
        #     return translated_segment
        
        # For this script, we'll just simulate a minor correction if it meets a condition
        if "unknown_term" in original_segment.lower() and "[Translated to" in translated_segment:
            print("Simulating human correction for 'unknown_term'.")
            return translated_segment.replace("[Translated to", "[Human Corrected to")
        return translated_segment

    def translate_document(self, document: str, source_lang: str = "en", target_lang: str = "es", enable_human_review: bool = True) -> list:
        """Translates a document using context-augmentation and iterative feedback."""
        print(f"Starting translation from {source_lang} to {target_lang}...")
        segments = self._segment_text(document)
        translated_document_segments = []

        for i, segment in enumerate(segments):
            print(f"\nProcessing segment {i+1}/{len(segments)}: '{segment}'")
            context = self._enrich_context(segment, source_lang)
            print(f"  Enriched Context: {context}")

            initial_translation = self._mock_translate_segment(segment, context, target_lang)
            print(f"  Initial Translation: '{initial_translation}'")

            # Automated Feedback Loop
            automated_feedback = self._automated_feedback(segment, initial_translation)
            print(f"  Automated Feedback: {automated_feedback}")
            
            current_translation = initial_translation
            if automated_feedback["issues"] and enable_human_review:
                print("  Automated feedback detected issues. Initiating human review.")
                current_translation = self._human_in_the_loop_review(segment, initial_translation)
            
            # Update translation memory for next time
            self.translation_memory[segment] = current_translation
            translated_document_segments.append(current_translation)
            print(f"  Final Segment Translation: '{current_translation}'")
            
        print("\nTranslation complete.")
        return translated_document_segments

# Example Usage:
if __name__ == "__main__":
    # Simplified medical ontology (term: translation/definition)
    medical_ontology = {
        "hypertension": "hipertensión",
        "myocardial infarction": "infarto de miocardio",
        "diabetes mellitus": "diabetes mellitus",
        "diagnosis": "diagnóstico",
        "treatment": "tratamiento",
        "patient": "paciente",
        "symptoms": "síntomas"
    }

    translator = MedicalTranslator(medical_ontology)

    medical_document = (
        "The patient presented with severe hypertension and chest pain. "
        "A provisional diagnosis of myocardial infarction was made. "
        "Treatment will involve medication and lifestyle changes. "
        "Further tests are required for a definitive diagnosis. "
        "There is an unknown_term in this document."
    )

    translated_segments = translator.translate_document(medical_document, source_lang="en", target_lang="es", enable_human_review=True)

    print("\n--- Full Translated Document ---")
    print(" ".join(translated_segments))

    # Example with automated feedback triggering human review
    print("\n--- Demonstrating human review for an untranslated segment ---")
    medical_document_with_issue = "The patient has an unmapped_condition. Immediate treatment is vital."
    translator_issue = MedicalTranslator(medical_ontology) # New instance for clean state
    translated_segments_issue = translator_issue.translate_document(medical_document_with_issue, source_lang="en", target_lang="es", enable_human_review=True)
    print(" ".join(translated_segments_issue))
