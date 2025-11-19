import functools

# --- 1. Knowledge Bases (Simulated) ---
medical_literature_kb = {
    "flu": {"symptoms": "fever, cough, body aches, fatigue", "causes": "influenza virus", "diagnosis": "clinical evaluation, rapid flu test"},
    "common cold": {"symptoms": "runny nose, sore throat, cough", "causes": "rhinovirus", "diagnosis": "clinical evaluation"},
    "pneumonia": {"symptoms": "fever, cough with phlegm, shortness of breath, chest pain", "causes": "bacteria, viruses, fungi", "diagnosis": "chest X-ray, blood tests"},
    "diabetes": {"symptoms": "frequent urination, increased thirst, unexplained weight loss", "causes": "insulin resistance or deficiency", "diagnosis": "blood sugar tests"},
    "hypertension": {"symptoms": "often none (silent killer), headaches, shortness of breath", "causes": "various, including genetics, lifestyle", "diagnosis": "blood pressure readings"},
    "paracetamol": {"usage": "pain relief, fever reduction", "dosage": "500-1000mg every 4-6 hours", "side_effects": "liver damage (overdose)"},
    "ibuprofen": {"usage": "pain relief, inflammation reduction", "dosage": "200-400mg every 4-6 hours", "side_effects": "stomach upset, kidney issues"},
}

patient_history_kb = {
    "patient_a": {"history": "35-year-old male with recent onset fever, cough, and body aches.", "symptoms": "fever, cough, body aches", "conditions": "flu-like symptoms"},
    "patient_b": {"history": "60-year-old female with chronic high blood pressure and occasional headaches.", "symptoms": "headaches", "conditions": "hypertension"},
    "patient_c": {"history": "12-year-old child with runny nose, sore throat, and mild cough for 2 days.", "symptoms": "runny nose, sore throat, cough", "conditions": "common cold symptoms"}
}

clinical_guidelines_kb = {
    "flu_treatment": {"recommendation": "antivirals (oseltamivir) if within 48 hours, rest, fluids, symptomatic relief", "dosage": "oseltamivir 75mg twice daily for 5 days"},
    "common_cold_treatment": {"recommendation": "rest, fluids, over-the-counter medications for symptoms (decongestants, pain relievers)"},
    "pneumonia_treatment": {"recommendation": "antibiotics (bacterial), antivirals (viral), oxygen therapy, hospitalisation if severe"},
    "diabetes_management": {"recommendation": "diet, exercise, oral medications (metformin), insulin therapy"},
    "hypertension_management": {"recommendation": "lifestyle changes (diet, exercise), ACE inhibitors, ARBs, diuretics"}
}

# --- 2. Simulated LLM Component ---
def simulated_llm(context: str, query: str) -> str:
    if not context:
        return "I need more information to provide a helpful response." + f" Original query: {query}"
    
    response_map = {
        "symptoms of flu": "The symptoms of flu typically include fever, cough, body aches, and fatigue.",
        "treatment for flu": "For flu, treatment often involves antivirals if started early, along with rest and fluids.",
        "diagnosis for flu": "Flu is diagnosed by clinical evaluation and sometimes a rapid flu test.",
        "diagnosis for patient with fever, cough, body aches": "Based on the symptoms of fever, cough, and body aches, a likely diagnosis could be flu.",
        "treatment for patient with flu-like symptoms": "Considering flu-like symptoms, rest, fluids, and possibly antivirals are recommended.",
        "dosage for paracetamol": "The recommended dosage for paracetamol is 500-1000mg every 4-6 hours."
    }
    
    # Simple heuristic to match query to a predefined response based on context
    for key, value in response_map.items():
        if key in query.lower() or any(term in context.lower() for term in key.split()):
            return value

    return f"Based on the retrieved information: '{context}', I can suggest the following: Please consult a healthcare professional for a definitive diagnosis and treatment plan."


# --- 3. Retrieval Mechanism ---

class RetrievalSystem:
    def __init__(self):
        self.medical_literature_kb = medical_literature_kb
        self.patient_history_kb = patient_history_kb
        self.clinical_guidelines_kb = clinical_guidelines_kb

    @functools.lru_cache(maxsize=128)
    def _search_medical_literature(self, query_terms):
        results = []
        for term in query_terms:
            for condition, data in self.medical_literature_kb.items():
                if term.lower() in condition.lower():
                    results.append(data)
                for field, text in data.items():
                    if term.lower() in text.lower():
                        results.append(data)
        return results

    @functools.lru_cache(maxsize=128)
    def _search_patient_history(self, query_symptoms):
        results = []
        for patient_id, data in self.patient_history_kb.items():
            if any(symp.lower() in data["symptoms"].lower() for symp in query_symptoms):
                results.append(data)
        return results

    @functools.lru_cache(maxsize=128)
    def _search_clinical_guidelines(self, query_terms):
        results = []
        for term in query_terms:
            for guideline, data in self.clinical_guidelines_kb.items():
                if term.lower() in guideline.lower():
                    results.append(data)
                for field, text in data.items():
                    if term.lower() in text.lower():
                        results.append(data)
        return results

    def query_analysis(self, query: str) -> dict:
        query_lower = query.lower()
        if "symptoms of" in query_lower or "what are symptoms" in query_lower:
            return {"intent": "symptoms", "entity": query_lower.replace("symptoms of", "").strip().replace("what are", "").strip()}
        elif "diagnosis for" in query_lower or "likely diagnosis" in query_lower:
            return {"intent": "diagnosis", "entity": query_lower.replace("diagnosis for", "").strip().replace("likely", "").strip()}
        elif "treatment for" in query_lower or "how to treat" in query_lower:
            return {"intent": "treatment", "entity": query_lower.replace("treatment for", "").strip().replace("how to treat", "").strip()}
        elif "patient" in query_lower and ("fever" in query_lower or "cough" in query_lower or "aches" in query_lower or "blood pressure" in query_lower):
            return {"intent": "patient_case", "symptoms": [term for term in ["fever", "cough", "body aches", "runny nose", "sore throat", "headaches", "high blood pressure"] if term in query_lower]}
        elif "dosage for" in query_lower:
            return {"intent": "dosage", "entity": query_lower.replace("dosage for", "").strip()}
        return {"intent": "general", "entity": query}

    def dynamic_retrieval(self, parsed_query: dict, iteration=0) -> str:
        context = []
        intent = parsed_query["intent"]
        entity = parsed_query.get("entity")
        symptoms = parsed_query.get("symptoms", [])

        if intent == "symptoms" and entity:
            results = self._search_medical_literature([entity])
            if results:
                context.append(f"Medical Literature: {results[0].get('symptoms', 'No symptoms found.')}")
        elif intent == "diagnosis" and entity:
            results = self._search_medical_literature([entity])
            if results:
                context.append(f"Medical Literature: {results[0].get('diagnosis', 'No diagnosis info found.')}")
        elif intent == "treatment" and entity:
            results = self._search_clinical_guidelines([entity])
            if results:
                context.append(f"Clinical Guidelines: {results[0].get('recommendation', 'No treatment recommendation found.')}")
        elif intent == "dosage" and entity:
            results = self._search_medical_literature([entity])
            if results:
                context.append(f"Drug Info: {results[0].get('dosage', 'No dosage info found.')}")
        elif intent == "patient_case" and symptoms:
            patient_results = self._search_patient_history(symptoms)
            if patient_results:
                for pr in patient_results:
                    context.append(f"Patient History: {pr['history']}")
                # Try to infer conditions from patient symptoms to get treatment/diagnosis
                for symp in symptoms:
                    lit_results = self._search_medical_literature([symp])
                    if lit_results:
                        for lr in lit_results:
                            if "causes" in lr: context.append(f"Medical Literature (related to {symp}): {lr['causes']}")
                            if "diagnosis" in lr: context.append(f"Medical Literature (related to {symp}): {lr['diagnosis']}")
                # Now try to get guidelines based on potential conditions
                for pr in patient_results:
                    for condition_keyword in pr.get("conditions", "").split():
                        guideline_results = self._search_clinical_guidelines([condition_keyword])
                        if guideline_results:
                            for gr in guideline_results:
                                context.append(f"Clinical Guidelines (related to {condition_keyword}): {gr['recommendation']}")
            
        elif intent == "general" and entity:
            lit_results = self._search_medical_literature([entity])
            if lit_results:
                context.append(f"Medical Literature: {lit_results[0]}")

        # Iterative Refinement
        if not context and iteration < 1:
            # If initial retrieval fails, try a broader search or different KB
            print("\n--- Iterative Refinement: Broadening search ---")
            if entity:
                broad_results_lit = self._search_medical_literature(entity.split())
                if broad_results_lit: context.append(f"Broad Medical Literature: {broad_results_lit[0]}")
                broad_results_guidelines = self._search_clinical_guidelines(entity.split())
                if broad_results_guidelines: context.append(f"Broad Clinical Guidelines: {broad_results_guidelines[0]}")

        return " ".join(context) if context else "No relevant information found."


# --- 4. Self-Reflection and Confidence Module ---
def self_reflect_and_decide(retrieved_context: str, generated_response: str, query: str) -> tuple[str, float, str]:
    confidence_score = 0.0
    decision = "generate_answer"
    reason = ""

    # Heuristic 1: Check for presence of key medical terms in the response based on query
    medical_terms = ["diagnosis", "treatment", "symptoms", "dosage", "flu", "pneumonia", "diabetes", "hypertension"]
    query_terms = [term for term in medical_terms if term in query.lower()]
    response_contains_key_terms = any(term in generated_response.lower() for term in query_terms)
    if response_contains_key_terms: 
        confidence_score += 0.4
        reason += "Key medical terms found in response. "

    # Heuristic 2: Check if context was actually utilized
    if "Based on the retrieved information:" in generated_response and retrieved_context != "No relevant information found.":
        confidence_score += 0.3
        reason += "Retrieved context explicitly used. "

    # Heuristic 3: Check for explicit 