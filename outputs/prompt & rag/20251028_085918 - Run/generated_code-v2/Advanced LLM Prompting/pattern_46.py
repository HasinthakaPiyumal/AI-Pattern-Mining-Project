import re

class MedicalTranslationSystem:
    def __init__(self):
        # Mock LLM - in a real scenario, this would be an actual LLM client (e.g., OpenAI, HuggingFace transformers)
        self.llm = self._mock_llm_translate

        # Mock Medical Terminology Database
        self.medical_terminology_db = {
            "hypertension": "उच्च रक्तचाप", # High blood pressure
            "diabetes mellitus": "मधुमेह", # Diabetes
            "cardiac arrest": "हृदय गति रुकना", # Heart attack
            "renal failure": "गुर्दे की विफलता", # Kidney failure
            "diagnosis": "निदान", # Diagnosis
            "medication": "दवा", # Medication
            "patient": "रोगी", # Patient
            "symptoms": "लक्षण", # Symptoms
            "therapy": "चिकित्सा", # Therapy
            "physician": "चिकित्सक", # Physician
            "anemia": "रक्ताल्पता", # Anemia
            "fracture": "हड्डी टूटना", # Fracture
            "inflammation": "सूजन" # Inflammation
        }

    def _mock_llm_translate(self, text, context=None):
        """
        A very basic mock LLM translation function.
        In a real application, this would interact with an actual LLM API.
        It attempts a simple English to Hindi (Devanagari script) substitution for demonstration.
        If context is provided, it tries to incorporate it.
        """
        # Simple word-by-word (or phrase-by-phrase) mock translation
        mock_translation_map = {
            "The patient has high blood pressure.": "रोगी को उच्च रक्तचाप है।",
            "Initial diagnosis indicates diabetes.": "प्रारंभिक निदान मधुमेह का संकेत देता है।",
            "Report suggests renal failure.": "रिपोर्ट गुर्दे की विफलता का सुझाव देती है।",
            "Review medication history.": "दवा के इतिहास की समीक्षा करें।",
            "The patient": "रोगी",
            "has": "को है",
            "high blood pressure": "उच्च रक्तचाप",
            "Initial diagnosis": "प्रारंभिक निदान",
            "indicates": "संकेत देता है",
            "diabetes": "मधुमेह",
            "Report suggests": "रिपोर्ट सुझाव देती है",
            "renal failure": "गुर्दे की विफलता",
            "Review": "समीक्षा करें",
            "medication history": "दवा का इतिहास",
            "Please refine": "कृपया परिष्कृत करें",
            "based on this feedback": "इस प्रतिक्रिया के आधार पर",
            "Original": "मूल",
            "Draft": "मसौदा",
            "Refined Translation": "परिष्कृत अनुवाद",
            # Add more basic translations for common words that might appear in medical context
            "a": "एक", "is": "है", "and": "और", "with": "के साथ", "for": "के लिए",
            "history": "इतिहास", "blood": "रक्त", "pressure": "दबाव",
            "The": "द", "an": "एक"
        }

        translated_text = text
        if context == "refinement":
            # Simple attempt to incorporate context for refinement
            if "Refine the following translation" in text:
                # Extract feedback and draft from the prompt
                match = re.search(r"Refine the following translation based on this feedback: '(.*?)'.\nOriginal: (.*?)\nDraft: (.*?)\nRefined Translation:", text, re.DOTALL)
                if match:
                    feedback = match.group(1)
                    original = match.group(2)
                    draft = match.group(3)
                    
                    print(f"LLM received feedback for refinement: '{feedback}'")
                    
                    # Apply very basic "refinement" logic by trying to incorporate feedback
                    refined_draft = draft
                    if "correct 'उच्च रक्तचाप' to 'हाई ब्लड प्रेशर'" in feedback.lower():
                        refined_draft = refined_draft.replace("उच्च रक्तचाप", "हाई ब्लड प्रेशर")
                    elif "add 'severe' to hypertension" in feedback.lower():
                        if "उच्च रक्तचाप" in refined_draft:
                            refined_draft = refined_draft.replace("उच्च रक्तचाप", "गंभीर उच्च रक्तचाप") # severe hypertension
                        elif "हाई ब्लड प्रेशर" in refined_draft:
                            refined_draft = refined_draft.replace("हाई ब्लड प्रेशर", "गंभीर हाई ब्लड प्रेशर")

                    # If no specific refinement, just slightly alter the original draft to show an LLM interaction
                    if refined_draft == draft:
                        refined_draft += " (refined by LLM)" # Placeholder for actual LLM improvement
                    return refined_draft
        
        # General translation for initial draft
        for eng_phrase, hindi_phrase in mock_translation_map.items():
            translated_text = translated_text.replace(eng_phrase, hindi_phrase)

        return translated_text

    def _retrieve_automated_feedback(self, current_translation, source_text):
        """
        Simulates retrieving automated feedback by cross-referencing with a medical terminology database.
        Identifies key medical terms in the source text and checks if their translation in the current_translation
        matches the preferred terminology.
        """
        feedback = []
        source_words = set(re.findall(r'\b\w+\b', source_text.lower()))

        for term_eng, term_hindi_preferred in self.medical_terminology_db.items():
            if term_eng in source_text.lower():
                # Check if the preferred Hindi term is *not* in the current translation
                if term_hindi_preferred not in current_translation:
                    feedback.append(f"Suggestion: For '{term_eng}', consider using the preferred term '{term_hindi_preferred}'.")
        return feedback

    def _apply_automated_feedback(self, current_translation, automated_suggestions):
        """
        Applies automated feedback to the translation. For this mock, it simply returns the current_translation.
        In a real system, these suggestions would typically form part of a re-prompt for the LLM.
        """
        return current_translation

    def translate_document(self, source_text, max_iterations=3):
        """
        Translates a medical document using an iterative prompting approach.
        Integrates automated retrieval signals and human feedback.
        """
        print(f"Translating: '{source_text}'")
        current_translation = self.llm(source_text)
        print(f"Initial Draft: {current_translation}\n")

        for i in range(max_iterations):
            print(f"--- Iteration {i+1} ---")
            
            # --- Automated Feedback Step ---
            automated_suggestions = self._retrieve_automated_feedback(current_translation, source_text)
            if automated_suggestions:
                print("Automated Review Suggestions:")
                for suggestion in automated_suggestions:
                    print(f"- {suggestion}")
                
                # In a real system, the LLM could be re-prompted here with the suggestions.
                # For this mock, we'll just show the suggestions.
                current_translation = self._apply_automated_feedback(current_translation, automated_suggestions)
                print(f"Draft after automated review consideration: {current_translation}\n")
            
            # --- Human Feedback Step ---
            human_input = input("Please review the current draft and provide refinements (e.g., 'correct X to Y'), or type 'APPROVE' to finish: ")
            
            if human_input.upper() == 'APPROVE':
                print("\nTranslation approved by human.")
                break
            elif human_input:
                print(f"Human feedback received: '{human_input}'")
                # Re-prompt the LLM with the original text, current draft, and human feedback
                # The LLM's role here is to intelligently incorporate the feedback.
                feedback_prompt = (
                    f"Refine the following translation based on this feedback: '{human_input}'.\n"
                    f"Original: {source_text}\n"
                    f"Draft: {current_translation}\n"
                    f"Refined Translation:"
                )
                current_translation = self.llm(feedback_prompt, context="refinement")
                print(f"Draft after LLM refinement with human feedback: {current_translation}\n")
            else:
                print("No human feedback provided for this iteration.\n")
        
        print(f"Final Translation: {current_translation}")
        return current_translation

# Example Usage:
if __name__ == "__main__":
    translator = MedicalTranslationSystem()
    
    # Example 1: Basic translation with potential automated and human refinement
    print("\n--- Example 1: Patient Medical History ---")
    source_text_1 = "The patient has high blood pressure. Initial diagnosis indicates diabetes. Review medication history."
    translator.translate_document(source_text_1)
    
    # Example 2: Another medical report snippet
    print("\n--- Example 2: Renal Failure Report ---")
    source_text_2 = "Report suggests renal failure."
    translator.translate_document(source_text_2, max_iterations=2)
