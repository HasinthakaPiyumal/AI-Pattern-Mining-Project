import json
import re

class MedicalReportTranslator:
    def __init__(self):
        # --- Simulated Context Augmentation Data ----
        self.medical_terminology_db = {
            "en": {
                "fever": "elevated body temperature",
                "hypertension": "high blood pressure",
                "diabetes": "a condition where the body does not properly process food for energy",
                "diagnosis": "identification of the nature of an illness or other problem by examination of the symptoms",
                "treatment": "medical care given for an illness or injury",
                "infection": "the invasion of an organism's body tissues by disease-causing agents, their multiplication, and the reaction of host tissues to these organisms and the toxins they produce",
                "dosage": "the size or frequency of a dose of a medicine or drug.",
                "symptoms": "a physical or mental feature which is regarded as indicating a condition of disease, particularly such a feature that is apparent to the patient.",
                "prognosis": "the likely course of a medical condition or disease."
            },
            "es": {
                "fiebre": "temperatura corporal elevada",
                "hipertensión": "presión arterial alta",
                "diabetes": "una condición donde el cuerpo no procesa adecuadamente los alimentos para obtener energía",
                "diagnóstico": "identificación de la naturaleza de una enfermedad u otro problema mediante el examen de los síntomas",
                "tratamiento": "atención médica brindada por una enfermedad o lesión",
                "infección": "la invasión de los tejidos del cuerpo de un organismo por agentes causantes de enfermedades, su multiplicación y la reacción de los tejidos del huésped a estos organismos y las toxinas que producen",
                "dosis": "el tamaño o la frecuencia de una dosis de un medicamento o droga.",
                "síntomas": "una característica física o mental que se considera indicativa de una condición de enfermedad, particularmente una característica que es evidente para el paciente.",
                "pronóstico": "el curso probable de una afección médica o enfermedad."
            }
            # ... add more languages and terms
        }

        self.cross_lingual_exemplars = [
            ("The patient has a fever.", "El paciente tiene fiebre."),
            ("Diagnosis: Hypertension", "Diagnóstico: Hipertensión"),
            ("Treatment plan includes medication.", "El plan de tratamiento incluye medicación."),
            ("Patient reported severe headaches.", "El paciente refirió fuertes dolores de cabeza."),
            ("Blood test results are normal.", "Los resultados del análisis de sangre son normales.")
        ]

        # --- Feedback Storage ---
        self.feedback_log = []

    def _detect_language(self, text: str) -> str:
        """Simulated language detection. In a real system, use langdetect or similar."""
        # Placeholder: Always returns Spanish for demonstration purposes or a simple check.
        if "fiebre" in text.lower() or "diagnóstico" in text.lower():
            return "es"
        return "en"

    def _segment_text(self, text: str) -> list[str]:
        """Simulated text segmentation. Breaks text into paragraphs/sections."""
        # Simple splitting by double newline for paragraph-like segmentation
        segments = [s.strip() for s in re.split(r'\n\n|\n\r\n', text) if s.strip()]
        return segments

    def _prioritize_segments(self, segments: list[str]) -> list[tuple[str, int]]:
        """Prioritizes segments based on medical keywords."""
        prioritized_segments = []
        for segment in segments:
            priority = 0
            # Higher priority for key medical sections
            if any(keyword in segment.lower() for keyword in ["diagnosis", "diagnóstico", "treatment", "tratamiento", "prognosis", "pronóstico"]):
                priority = 2
            elif any(keyword in segment.lower() for keyword in ["symptoms", "síntomas", "history", "historial", "results", "resultados"]):
                priority = 1
            prioritized_segments.append((segment, priority))

        # Sort by priority in descending order
        return sorted(prioritized_segments, key=lambda x: x[1], reverse=True)

    def _retrieve_exemplars(self, segment: str, source_lang: str) -> list[str]:
        """Simulated retrieval of cross-lingual exemplars based on simple string matching."""
        relevant_exemplars = []
        for src_ex, tgt_ex in self.cross_lingual_exemplars:
            if source_lang == "en" and src_ex.lower() in segment.lower():
                relevant_exemplars.append(f"Example: '{src_ex}' -> '{tgt_ex}'")
            elif source_lang == "es" and tgt_ex.lower() in segment.lower(): # Assuming tgt_ex might be present in Spanish source for matching
                 relevant_exemplars.append(f"Example: '{tgt_ex}' -> '{src_ex}'")
        return relevant_exemplars

    def _lookup_term(self, term: str, lang: str) -> str | None:
        """Looks up a term's definition in the medical terminology database."""
        return self.medical_terminology_db.get(lang, {}).get(term.lower())

    def _translate_segment(self, segment: str, source_lang: str, target_lang: str, context_info: dict) -> tuple[str, float]:
        """Simulated NMT translation. Returns translated text and a confidence score."""
        print(f"[SIMULATING NMT for {source_lang} -> {target_lang}] Segment: \"{segment}\"")
        print(f"  Context used: {json.dumps(context_info)}")

        # Placeholder: Simple translation logic. In a real system, integrate with an NMT model.
        # For demonstration, it just prepends and adds some context.
        translated_text = f"[Translated from {source_lang}]: {segment}"
        confidence = 0.85 # Default simulated confidence

        # Simulate better translation if exact exemplar found
        for src_ex, tgt_ex in self.cross_lingual_exemplars:
            if source_lang == "en" and target_lang == "es" and src_ex.lower() == segment.lower():
                translated_text = tgt_ex
                confidence = 0.98
                break
            elif source_lang == "es" and target_lang == "en" and tgt_ex.lower() == segment.lower():
                translated_text = src_ex
                confidence = 0.98
                break

        # Simulate term replacement based on dictionary lookup for better context
        for term, definition in self.medical_terminology_db.get(source_lang, {}).items():
            if term.lower() in segment.lower():
                # Simple replacement for demonstration, NMT would handle this more gracefully
                if source_lang == "en" and target_lang == "es":
                    spanish_term = next((k for k, v in self.medical_terminology_db.get('es', {}).items() if v == definition), term)
                    translated_text = translated_text.replace(term, spanish_term)
                elif source_lang == "es" and target_lang == "en":
                    english_term = next((k for k, v in self.medical_terminology_db.get('en', {}).items() if v == definition), term)
                    translated_text = translated_text.replace(term, english_term)


        # Lower confidence if specific medical terms are not found in context_info or are complex
        if "contextual_terms" in context_info and not context_info["contextual_terms"]:
            confidence -= 0.1

        return translated_text, max(0.0, min(1.0, confidence))

    def _validate_translation_rules(self, original_segment: str, translated_segment: str, source_lang: str, target_lang: str) -> list[str]:
        """Simulated rule-based validation for medical context."""
        flags = []
        # Example rule: If original contains 'fever', translated should contain 'fiebre' (if target is es)
        if source_lang == "en" and target_lang == "es":
            if "fever" in original_segment.lower() and "fiebre" not in translated_segment.lower():
                flags.append("Missing 'fiebre' for 'fever' in translation.")
            if "diagnosis" in original_segment.lower() and "diagnóstico" not in translated_segment.lower():
                flags.append("Missing 'diagnóstico' for 'diagnosis' in translation.")
        elif source_lang == "es" and target_lang == "en":
            if "fiebre" in original_segment.lower() and "fever" not in translated_segment.lower():
                flags.append("Missing 'fever' for 'fiebre' in translation.")
            if "diagnóstico" in original_segment.lower() and "diagnosis" not in translated_segment.lower():
                flags.append("Missing 'diagnosis' for 'diagnóstico' in translation.")

        return flags

    def translate_and_analyze_report(self, medical_report: str, target_lang: str = "en") -> dict:
        """Main function to translate and analyze a medical report iteratively."""
        source_lang = self._detect_language(medical_report)
        print(f"Detected source language: {source_lang.upper()}")

        segments = self._segment_text(medical_report)
        prioritized_segments = self._prioritize_segments(segments)

        full_translated_report = []
        analysis_results = []

        for segment, priority in prioritized_segments:
            print(f"\nProcessing segment (Priority: {priority}): \"{segment}\"")

            # --- Context Augmentation ---
            exemplars = self._retrieve_exemplars(segment, source_lang)
            contextual_terms = {}
            for term in re.findall(r'\b[a-zA-Z]+\b', segment):
                definition = self._lookup_term(term, source_lang)
                if definition:
                    contextual_terms[term] = definition
            
            context_info = {
                "exemplars": exemplars,
                "contextual_terms": contextual_terms,
                "source_lang_terms": self.medical_terminology_db.get(source_lang, {})
            }

            # --- Neural Machine Translation ---
            translated_segment, confidence = self._translate_segment(segment, source_lang, target_lang, context_info)

            # --- Automated Feedback --- 
            validation_flags = self._validate_translation_rules(segment, translated_segment, source_lang, target_lang)
            
            segment_analysis = {
                "original_segment": segment,
                "translated_segment": translated_segment,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "priority": priority,
                "confidence_score": confidence,
                "automated_flags": validation_flags,
                "exemplars_used": exemplars,
                "contextual_terms_found": contextual_terms,
                "human_corrected": False,
                "human_feedback_text": None
            }
            analysis_results.append(segment_analysis)
            full_translated_report.append(translated_segment)

            # Log for potential human review
            self.feedback_log.append(segment_analysis)

        return {
            "full_translated_report": "\n\n".join(full_translated_report),
            "detailed_analysis": analysis_results
        }

    def provide_human_feedback(self, segment_id: int, corrected_text: str, notes: str = None):
        """Allows a human to provide feedback and correct a translated segment."""
        if 0 <= segment_id < len(self.feedback_log):
            entry = self.feedback_log[segment_id]
            print(f"\n--- Human Feedback for Segment ID {segment_id} ---")
            print(f"Original: {entry['original_segment']}")
            print(f"Automated Translation: {entry['translated_segment']}")
            print(f"Corrected by Human: {corrected_text}")
            
            entry["translated_segment"] = corrected_text
            entry["human_corrected"] = True
            entry["human_feedback_text"] = notes
            entry["confidence_score"] = 1.0 # Assume human correction is perfect
            entry["automated_flags"] = [] # Clear automated flags if human corrected
            print(f"Feedback recorded for segment: \"{entry['original_segment']}\"")
        else:
            print(f"Error: Segment ID {segment_id} out of bounds.")

    def get_feedback_log(self):
        """Returns the current feedback log."""
        return self.feedback_log


if __name__ == "__main__":
    translator = MedicalReportTranslator()

    # Example Medical Report 1 (English)
    english_report = """
    Patient Name: John Doe
    Date: 2023-10-27

    **Patient History:**
    Mr. Doe, a 45-year-old male, presented with symptoms of high fever, persistent cough, and general fatigue for the past 3 days. He has a history of mild hypertension but no other significant medical conditions.

    **Diagnosis:**
    Acute Bronchitis, likely viral infection. Further tests for bacterial infection are pending.

    **Treatment Plan:**
    Prescribed Amoxicillin 500mg, three times a day for 7 days. Advised rest and increased fluid intake. Follow-up in one week. The dosage is critical.

    **Lab Results:**
    White blood cell count slightly elevated.
    """

    print("\n===== Translating English Report to Spanish =====")
    translated_output_es = translator.translate_and_analyze_report(english_report, target_lang="es")
    print("\n--- Full Translated Report (Spanish) ---")
    print(translated_output_es["full_translated_report"])
    print("\n--- Detailed Analysis (Spanish) ---")
    # for i, segment in enumerate(translated_output_es["detailed_analysis"]):
    #     print(f"Segment {i}: {json.dumps(segment, indent=2)}")
    
    # Simulate human feedback for a specific segment (e.g., segment 1 in the analysis results)
    print("\n===== Simulating Human Feedback =====")
    # Assuming the second segment (index 1) in analysis_results refers to 'Diagnosis'
    # The actual index might vary based on prioritization, so a real UI would map this.
    # Let's assume after sorting, 'Diagnosis' becomes segment 0 or 1 for this example.
    
    # Find the diagnosis segment to provide feedback
    diagnosis_segment_id = -1
    for i, seg_data in enumerate(translated_output_es["detailed_analysis"]):
        if "diagnosis" in seg_data['original_segment'].lower():
            diagnosis_segment_id = i
            break

    if diagnosis_segment_id != -1:
        translator.provide_human_feedback(
            segment_id=diagnosis_segment_id, 
            corrected_text="Diagnóstico: Bronquitis Aguda, probablemente infección viral. Pruebas adicionales para infección bacteriana están pendientes.",
            notes="Corrected the nuance of 'likely viral' and ensured correct medical term usage."
        )
    else:
        print("Diagnosis segment not found for feedback simulation.")

    print("\n--- Updated Feedback Log (after human correction) ---")
    # print(json.dumps(translator.get_feedback_log(), indent=2))
    # You can inspect the full log for changes
    for entry in translator.get_feedback_log():
        if entry['human_corrected']:
            print(f"[Corrected by Human] Original: '{entry['original_segment']}' -> Corrected: '{entry['translated_segment']}'")
        elif entry['automated_flags']:
            print(f"[Flagged] Original: '{entry['original_segment']}' -> Flags: {entry['automated_flags']}")


    # Example Medical Report 2 (Spanish)
    spanish_report = """
    Nombre del Paciente: María García
    Fecha: 2023-10-28

    **Historial del Paciente:**
    La Sra. García, mujer de 60 años, acudió con síntomas de fiebre leve, tos seca y dolor de garganta durante 2 días. Tiene antecedentes de diabetes tipo 2 bien controlada.

    **Diagnóstico:**
    Faringitis aguda.

    **Tratamiento:**
    Se recetó Paracetamol 500mg cada 8 horas. Reposo y abundante líquidos. El pronóstico es bueno.

    **Resultados de Laboratorio:**
    Recuento de glóbulos blancos normal.
    """

    print("\n===== Translating Spanish Report to English =====")
    translated_output_en = translator.translate_and_analyze_report(spanish_report, target_lang="en")
    print("\n--- Full Translated Report (English) ---")
    print(translated_output_en["full_translated_report"])
    print("\n--- Detailed Analysis (English) ---")
    # for i, segment in enumerate(translated_output_en["detailed_analysis"]):
    #     print(f"Segment {i}: {json.dumps(segment, indent=2)}")

    print("\n--- Final Feedback Log --- ")
    # print(json.dumps(translator.get_feedback_log(), indent=2))
    for i, entry in enumerate(translator.get_feedback_log()):
        status = "Human Corrected" if entry['human_corrected'] else f"Confidence: {entry['confidence_score']:.2f}"
        flags = f"Flags: {', '.join(entry['automated_flags'])}" if entry['automated_flags'] else "No Flags"
        print(f"Segment {i}: Original='{entry['original_segment'][:50]}...' -> Translated='{entry['translated_segment'][:50]}...' ({status}, {flags})")
