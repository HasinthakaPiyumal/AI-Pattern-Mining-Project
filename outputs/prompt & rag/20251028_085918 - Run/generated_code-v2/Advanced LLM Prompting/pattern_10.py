import random

class MedicalDiagnosticAssistant:
    def _clarify_question(self, symptoms: str, medical_history: str) -> str:
        clarified_q = f"Given the symptoms: '{symptoms}' and medical history: '{medical_history}', the core question is to identify the most probable medical condition(s). Is this understanding correct?"
        return clarified_q

    def _preliminary_judgment(self, clarified_question: str) -> dict:
        # Simulate initial differential diagnoses
        if "fever" in clarified_question.lower() and "cough" in clarified_question.lower():
            diagnoses = {
                "Common Cold": "Symptoms align with viral infection.",
                "Influenza": "Could be a more severe viral infection, warrants further checks."
            }
        elif "headache" in clarified_question.lower() and "nausea" in clarified_question.lower():
            diagnoses = {
                "Migraine": "Classic symptoms, especially if pulsating.",
                "Tension Headache": "Less severe but common, often stress-related."
            }
        else:
            diagnoses = {
                "General Malaise": "Non-specific symptoms, needs more investigation.",
                "Stress-Related": "Many physical symptoms can stem from stress."
            }
        return diagnoses

    def _evaluate_response(self, preliminary_judgment: dict) -> dict:
        evaluated_diagnoses = preliminary_judgment.copy()
        evaluation_notes = []

        for diag, reason in evaluated_diagnoses.items():
            if "viral infection" in reason.lower():
                evaluation_notes.append(f"Considering {diag}: Need to rule out bacterial infections due to potential overlap in early stages. Also consider patient's age and immune status.")
            elif "headache" in diag.lower():
                evaluation_notes.append(f"For {diag}: Are there any visual disturbances or neurological signs that might suggest something more serious?")

        if not evaluation_notes:
            evaluation_notes.append("Preliminary judgments appear reasonable, no immediate red flags but always consider individual patient context.")
            
        evaluated_diagnoses["evaluation_notes"] = evaluation_notes
        return evaluated_diagnoses

    def _confirm_decision(self, evaluation_result: dict) -> str:
        # Simulate confirming the most probable diagnosis
        # For simplicity, we'll pick the first diagnosis from the preliminary judgment
        # unless evaluation notes strongly suggest otherwise (simulated here).
        diagnoses = [d for d in evaluation_result.keys() if d != "evaluation_notes"]
        
        final_diagnosis = ""
        if "rule out bacterial infections" in str(evaluation_result.get("evaluation_notes", [])) and "Common Cold" in diagnoses:
            final_diagnosis = "Common Cold (viral, but consider bacterial if symptoms worsen)"
        elif "Migraine" in diagnoses:
            final_diagnosis = "Migraine"
        else:
            final_diagnosis = diagnoses[0] if diagnoses else "Undetermined Condition"

        return final_diagnosis

    def _assess_confidence(self, final_diagnosis: str, preliminary_judgment_details: dict) -> tuple[float, str]:
        # Simulate confidence based on complexity/specificity
        confidence_score = 0.0
        explanation = ""

        if "Common Cold" in final_diagnosis or "Tension Headache" in final_diagnosis:
            confidence_score = round(random.uniform(0.7, 0.9), 2) # Higher confidence for common issues
            explanation = "High confidence due to classic, well-aligned symptoms and ruling out more severe conditions based on initial information."
        elif "Migraine" in final_diagnosis:
            confidence_score = round(random.uniform(0.65, 0.85), 2)
            explanation = "Moderate to high confidence, typical presentation, but always mindful of individual variations and triggers."
        elif "General Malaise" in final_diagnosis or "Undetermined Condition" in final_diagnosis:
            confidence_score = round(random.uniform(0.3, 0.5), 2) # Lower confidence for vague cases
            explanation = "Lower confidence due to non-specific symptoms and the need for further diagnostic tests or patient history."
        else:
            confidence_score = round(random.uniform(0.5, 0.75), 2)
            explanation = "Moderate confidence, based on available data, but further specialist consultation may be beneficial."
        
        return confidence_score, explanation

    def diagnose(self, symptoms: str, medical_history: str) -> dict:
        clarified_q = self._clarify_question(symptoms, medical_history)
        prelim_judgments = self._preliminary_judgment(clarified_q)
        evaluated_response = self._evaluate_response(prelim_judgments)
        final_diagnosis = self._confirm_decision(evaluated_response)
        confidence_score, confidence_explanation = self._assess_confidence(final_diagnosis, prelim_judgments)

        return {
            "clarified_question": clarified_q,
            "preliminary_judgments": prelim_judgments,
            "evaluation_details": evaluated_response.get("evaluation_notes", []), # Extracting just the notes for clearer output
            "final_diagnosis": final_diagnosis,
            "confidence_score": confidence_score,
            "confidence_explanation": confidence_explanation
        }

