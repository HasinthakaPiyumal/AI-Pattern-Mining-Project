class MedicalDataTools:
    def get_patient_record(self, patient_id):
        if patient_id == "P001":
            return {"id": "P001", "name": "Alice Smith", "age": 45, "diagnosis": "Hypertension", "medications": [{"name": "Lisinopril", "dosage": "10mg"}]}
        return None

    def get_drug_info(self, drug_name):
        if drug_name == "Lisinopril":
            return {"name": "Lisinopril", "class": "ACE Inhibitor", "uses": "High blood pressure", "side_effects": "Dizziness, cough"}
        return None

    def get_research_paper_abstract(self, topic):
        if topic == "Hypertension treatment":
            return "A recent study showed that a combination of lifestyle changes and ACE inhibitors significantly reduced blood pressure in patients with moderate hypertension. Further research is needed to evaluate long-term cardiovascular outcomes. This abstract is quite long and contains detailed medical terminology suitable for clinicians."
        return None

    def get_lab_results(self, patient_id):
        if patient_id == "P001":
            return {"patient_id": "P001", "blood_pressure": "140/90 mmHg", "cholesterol_ldl": "130 mg/dL"}
        return None

class LLMCore:
    def simulate_llm_response(self, prompt, complexity="normal"):
        if "summarize" in prompt.lower() and complexity == "patient":
            return f"Simplified summary: {prompt[:50]}... (for patient)"
        elif "summarize" in prompt.lower() and complexity == "clinician":
            return f"Detailed summary for clinicians: {prompt[:100]}... (for clinician)"
        elif "explain" in prompt.lower() and complexity == "patient":
            return f"In simple terms, {prompt.replace('explain', '')}."
        elif "explain" in prompt.lower() and complexity == "clinician":
            return f"From a medical perspective, {prompt.replace('explain', '')}."
        elif "simplify" in prompt.lower():
            return f"Here's a simpler version of: {prompt.replace('simplify', '')}"
        return f"LLM processed: {prompt}"

class SynthesisAndFormattingLayer:
    def __init__(self, llm_core):
        self.llm_core = llm_core

    def direct_insert(self, data):
        return str(data)

    def compress_text(self, text, user_role):
        prompt = f"Summarize the following medical text for a {user_role}: {text}"
        return self.llm_core.simulate_llm_response(prompt, complexity=user_role)

    def simplify_schema(self, data, schema_keys):
        simplified_data = {key: data.get(key) for key in schema_keys if key in data}
        return simplified_data

class UserRoleAndOutputTailoringModule:
    def __init__(self, llm_core):
        self.llm_core = llm_core

    def tailor_response(self, synthesized_content, user_role):
        if user_role == "patient":
            prompt = f"Explain this medical information in easy-to-understand language for a patient: {synthesized_content}"
            return self.llm_core.simulate_llm_response(prompt, complexity="patient")
        elif user_role == "clinician":
            prompt = f"Provide a detailed, medically accurate explanation for a clinician based on: {synthesized_content}"
            return self.llm_core.simulate_llm_response(prompt, complexity="clinician")
        return synthesized_content

class FeedbackAndRefinementMechanism:
    def record_feedback(self, query_id, feedback_text, rating):
        return f"Feedback recorded for query {query_id}: '{feedback_text}' with rating {rating}."

class MedicalInformationSynthesizer:
    def __init__(self):
        self.medical_tools = MedicalDataTools()
        self.llm_core = LLMCore()
        self.synthesis_layer = SynthesisAndFormattingLayer(self.llm_core)
        self.tailoring_module = UserRoleAndOutputTailoringModule(self.llm_core)
        self.feedback_mechanism = FeedbackAndRefinementMechanism()
        self.query_counter = 0

    def process_query(self, user_query, user_role):
        self.query_counter += 1
        current_query_id = f"Q{self.query_counter}"
        raw_outputs = {}
        synthesized_parts = []

        if "patient record" in user_query.lower() or "diagnosis" in user_query.lower() or "medications" in user_query.lower():
            patient_data = self.medical_tools.get_patient_record("P001")
            if patient_data:
                raw_outputs["patient_record"] = patient_data
                if user_role == "patient":
                    simplified_patient_data = self.synthesis_layer.simplify_schema(patient_data, ["name", "diagnosis", "medications"])
                    synthesized_parts.append(f"Your diagnosis is {simplified_patient_data.get('diagnosis')}. Your medications include {', '.join([m['name'] for m in simplified_patient_data.get('medications', [])])}.")
                else:
                    synthesized_parts.append(self.synthesis_layer.direct_insert(patient_data))

        if "drug info" in user_query.lower() or "lisinopril" in user_query.lower():
            drug_data = self.medical_tools.get_drug_info("Lisinopril")
            if drug_data:
                raw_outputs["drug_info"] = drug_data
                if user_role == "patient":
                    simplified_drug_data = self.synthesis_layer.simplify_schema(drug_data, ["name", "uses", "side_effects"])
                    synthesized_parts.append(f"About {simplified_drug_data.get('name')}: It's used for {simplified_drug_data.get('uses')}. Possible side effects include {simplified_drug_data.get('side_effects')}.")
                else:
                    synthesized_parts.append(self.synthesis_layer.direct_insert(drug_data))

        if "research" in user_query.lower() or "hypertension treatment" in user_query.lower():
            research_abstract = self.medical_tools.get_research_paper_abstract("Hypertension treatment")
            if research_abstract:
                raw_outputs["research_abstract"] = research_abstract
                synthesized_parts.append(self.synthesis_layer.compress_text(research_abstract, user_role))

        if "lab results" in user_query.lower():
            lab_results = self.medical_tools.get_lab_results("P001")
            if lab_results:
                raw_outputs["lab_results"] = lab_results
                if user_role == "patient":
                    synthesized_parts.append(f"Your blood pressure is {lab_results.get('blood_pressure')} and LDL cholesterol is {lab_results.get('cholesterol_ldl')}.")
                else:
                    synthesized_parts.append(self.synthesis_layer.direct_insert(lab_results))

        if not synthesized_parts:
            synthesized_content = self.llm_core.simulate_llm_response(f"No specific medical tools found for '{user_query}'. Please provide general information based on your knowledge related to: {user_query}", complexity=user_role)
        else:
            combined_synthesized_content = "\n".join(synthesized_parts)
            synthesized_content = self.llm_core.simulate_llm_response(f"Synthesize the following information for a {user_role}: {combined_synthesized_content}", complexity=user_role)

        final_response = self.tailoring_module.tailor_response(synthesized_content, user_role)

        return {"query_id": current_query_id, "response": final_response}

if __name__ == "__main__":
    synthesizer = MedicalInformationSynthesizer()

    # Patient query example
    patient_query_1 = "Can you explain my diagnosis and medications in simple terms?"
    patient_response_1 = synthesizer.process_query(patient_query_1, "patient")
    print(f"Patient Query: '{patient_query_1}'")
    print(f"Patient Response ({patient_response_1['query_id']}): {patient_response_1['response']}\n")

    patient_query_2 = "What is Lisinopril for and what are its side effects?"
    patient_response_2 = synthesizer.process_query(patient_query_2, "patient")
    print(f"Patient Query: '{patient_query_2}'")
    print(f"Patient Response ({patient_response_2['query_id']}): {patient_response_2['response']}\n")

    patient_query_3 = "What are my recent lab results?"
    patient_response_3 = synthesizer.process_query(patient_query_3, "patient")
    print(f"Patient Query: '{patient_query_3}'")
    print(f"Patient Response ({patient_response_3['query_id']}): {patient_response_3['response']}\n")

    # Clinician query example
    clinician_query_1 = "Provide a detailed summary of patient P001's record and relevant research on hypertension treatment."
    clinician_response_1 = synthesizer.process_query(clinician_query_1, "clinician")
    print(f"Clinician Query: '{clinician_query_1}'")
    print(f"Clinician Response ({clinician_response_1['query_id']}): {clinician_response_1['response']}\n")

    clinician_query_2 = "Give me the full drug information for Lisinopril."
    clinician_response_2 = synthesizer.process_query(clinician_query_2, "clinician")
    print(f"Clinician Query: '{clinician_query_2}'")
    print(f"Clinician Response ({clinician_response_2['query_id']}): {clinician_response_2['response']}\n")

    clinician_query_3 = "What are the lab results for P001?"
    clinician_response_3 = synthesizer.process_query(clinician_query_3, "clinician")
    print(f"Clinician Query: '{clinician_query_3}'")
    print(f"Clinician Response ({clinician_response_3['query_id']}): {clinician_response_3['response']}\n")

    # Test feedback mechanism
    feedback_result = synthesizer.feedback_mechanism.record_feedback(patient_response_1['query_id'], "Response was clear and helpful.", 5)
    print(feedback_result)