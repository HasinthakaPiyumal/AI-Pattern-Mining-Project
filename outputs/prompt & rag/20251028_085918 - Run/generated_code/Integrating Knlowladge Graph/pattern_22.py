from medical_kg import MedicalKnowledgeGraph
from llm_interface import LLMInterface

class MedicalDiagnosticSystem:
    def __init__(self, kg: MedicalKnowledgeGraph, llm: LLMInterface):
        self.kg = kg
        self.llm = llm
        print("MedicalDiagnosticSystem initialized with MedicalKnowledgeGraph and LLMInterface.")

    def _get_relevant_kg_facts(self, entities):
        """
        Retrieves relevant facts from the Knowledge Graph based on a list of entities.
        This simulates RAG for KGs by grounding LLM with relevant KG facts.
        """
        all_facts = []
        for entity in entities:
            # Get facts where entity is subject
            subject_facts = self.kg.query_triples(subject=entity)
            all_facts.extend(subject_facts)
            # Get facts where entity is object
            object_facts = self.kg.query_triples(object=entity)
            all_facts.extend(object_facts)

            # Get entity attributes
            attributes = self.kg.get_entity_attributes(entity)
            if attributes:
                for attr, value in attributes.items():
                    if attr != "type": # Avoid adding type as a 'fact' if it's already a node property
                        all_facts.append((entity, f"has_{attr}", value))

        # Remove duplicates by converting to set then back to list of tuples
        return list(set(all_facts))

    def diagnose_and_recommend(self, patient_query: str):
        """
        Main function to diagnose and recommend treatment based on patient query.
        Leverages LLM-KG integration patterns.
        """
        print(f"\n--- Processing Patient Query: {patient_query} ---")

        # 1. LLM-based Topic Entity Extraction
        extracted_entities = self.llm.extract_entities(patient_query)
        print(f"Extracted Entities: {extracted_entities}")

        if not extracted_entities:
            return "I couldn't extract any relevant medical entities from your query. Can you please rephrase?"

        # 2. Retrieval-Augmented Generation (RAG) for KGs & Initial RoG
        # Get initial relevant facts from KG based on extracted entities
        kg_context_facts = self._get_relevant_kg_facts(extracted_entities)
        print(f"Initial KG Context Facts: {kg_context_facts}")

        # 3. Knowledge-Driven Chain-of-Thought (KDCoT) / Reasoning on Graphs (RoG)
        # Use LLM to reason over symptoms and KG context to suggest potential diagnoses
        diagnosis_prompt = f"Given the symptoms '{', '.join(extracted_entities)}', and the following medical knowledge graph facts, what are the most likely diagnoses and why?"
        
        # Simulate ThinkonGraph and iterative reasoning: initial diagnosis attempt
        llm_diagnosis_response = self.llm.generate_response(diagnosis_prompt, context=kg_context_facts)
        print(f"LLM Initial Diagnosis Response: {llm_diagnosis_response}")

        # Extract potential diagnosis from LLM's response (mock)
        potential_diagnosis = "Unknown Disease"
        if "Common Cold" in llm_diagnosis_response:
            potential_diagnosis = "Common Cold"
        elif "Diabetes Mellitus Type 2" in llm_diagnosis_response:
            potential_diagnosis = "Diabetes Mellitus Type 2"
        elif "Flu" in llm_diagnosis_response:
            potential_diagnosis = "Flu"

        if potential_diagnosis != "Unknown Disease":
            print(f"Identified Potential Diagnosis: {potential_diagnosis}")
            # Retrieve further KG context related to the potential diagnosis
            diagnosis_related_facts = self._get_relevant_kg_facts([potential_diagnosis])
            combined_kg_context = list(set(kg_context_facts + diagnosis_related_facts))

            # 4. LLM-KG Tight-Coupling Paradigm (ThinkonGraph) & Iterative Prompting
            # Refine diagnosis and ask for treatment based on enhanced context
            refined_prompt = (
                f"Considering the patient's symptoms '{', '.join(extracted_entities)}' "
                f"and a potential diagnosis of '{potential_diagnosis}', "
                f"and the comprehensive medical knowledge graph facts below, "
                f"provide a confirmed diagnosis, explain the reasoning, and suggest initial treatment recommendations, including drug interaction warnings if applicable."
            )
            final_llm_response = self.llm.generate_response(refined_prompt, context=combined_kg_context)
        else:
            final_llm_response = llm_diagnosis_response # If no specific diagnosis, just return initial LLM response

        return final_llm_response

    def check_drug_interactions(self, drug1: str, drug2: str):
        """
        Checks for drug interactions using KG and LLM for explanation.
        """
        print(f"\n--- Checking Drug Interactions: {drug1} and {drug2} ---")

        interaction_facts = self.kg.query_triples(subject=drug1, predicate="interacts_with", object=drug2) or \
                            self.kg.query_triples(subject=drug2, predicate="interacts_with", object=drug1)

        if interaction_facts:
            prompt = f"Explain the interaction between {drug1} and {drug2} based on these facts:"
            explanation = self.llm.generate_response(prompt, context=interaction_facts)
            return f"Potential interaction found: {drug1} and {drug2}. {explanation}"
        else:
            return f"No direct interaction found between {drug1} and {drug2} in the knowledge graph."

    def semantic_query(self, natural_language_query: str):
        """
        Performs semantic parsing to convert natural language into KG queries.
        """
        print(f"\n--- Performing Semantic Query: {natural_language_query} ---")
        parsed_query = self.llm.semantic_parse(natural_language_query)
        print(f"Parsed Query: {parsed_query}")

        query_type = parsed_query.get("query_type")

        if query_type == "symptoms":
            entity = parsed_query.get("entity")
            if entity: 
                symptom_facts = self.kg.query_triples(subject=entity, predicate="has_symptom")
                if symptom_facts:
                    return f"Symptoms for {entity}: {', '.join([o for s, p, o in symptom_facts])}.\nLLM Explanation: {self.llm.generate_response(f"Explain symptoms of {entity}", context=symptom_facts)}"
                else:
                    return f"No symptoms found for {entity} in KG."
            else: return "Could not identify entity for symptom query."
        elif query_type == "treatments":
            entity = parsed_query.get("entity")
            if entity:
                treatment_facts = self.kg.query_triples(object=entity, predicate="treated_by")
                if treatment_facts:
                    return f"Treatments for {entity}: {', '.join([s for s, p, o in treatment_facts])}.\nLLM Explanation: {self.llm.generate_response(f"Explain treatments for {entity}", context=treatment_facts)}"
                else:
                    return f"No treatments found for {entity} in KG."
            else: return "Could not identify entity for treatment query."
        elif query_type == "drug_interaction":
            drugs = parsed_query.get("drugs")
            if drugs and len(drugs) == 2:
                return self.check_drug_interactions(drugs[0], drugs[1])
            else: return "Could not identify drugs for interaction query."
        else:
            return f"Semantic parser could not process the query: {natural_language_query}. LLM response: {self.llm.generate_response(natural_language_query, context=[])}"


# --- Demo Usage ---
if __name__ == "__main__":
    # 1. Initialize Medical Knowledge Graph
    medical_kg = MedicalKnowledgeGraph()

    # Populate KG with demo data
    medical_kg.add_entity("Diabetes Mellitus Type 2", "Disease", {"description": "A chronic condition that affects the way the body processes blood sugar (glucose)."})
    medical_kg.add_entity("Common Cold", "Disease", {"description": "A viral infectious disease of the upper respiratory tract that primarily affects the nose, throat, sinuses, and larynx."})
    medical_kg.add_entity("Flu", "Disease", {"description": "A common viral infection that can be deadly, especially in high-risk groups."})
    medical_kg.add_entity("Fever", "Symptom")
    medical_kg.add_entity("Cough", "Symptom")
    medical_kg.add_entity("Fatigue", "Symptom")
    medical_kg.add_entity("High Blood Sugar", "Symptom")
    medical_kg.add_entity("Frequent Urination", "Symptom")
    medical_kg.add_entity("Headache", "Symptom")

    medical_kg.add_entity("Metformin", "Drug")
    medical_kg.add_entity("Acetaminophen", "Drug")
    medical_kg.add_entity("Ibuprofen", "Drug")
    medical_kg.add_entity("Warfarin", "Drug")

    medical_kg.add_relationship("Diabetes Mellitus Type 2", "High Blood Sugar", "has_symptom")
    medical_kg.add_relationship("Diabetes Mellitus Type 2", "Frequent Urination", "has_symptom")
    medical_kg.add_relationship("Diabetes Mellitus Type 2", "Metformin", "treated_by")

    medical_kg.add_relationship("Common Cold", "Fever", "has_symptom")
    medical_kg.add_relationship("Common Cold", "Cough", "has_symptom")
    medical_kg.add_relationship("Common Cold", "Fatigue", "has_symptom")
    medical_kg.add_relationship("Common Cold", "Acetaminophen", "treated_by")

    medical_kg.add_relationship("Flu", "Fever", "has_symptom")
    medical_kg.add_relationship("Flu", "Cough", "has_symptom")
    medical_kg.add_relationship("Flu", "Fatigue", "has_symptom")
    medical_kg.add_relationship("Flu", "Headache", "has_symptom")
    medical_kg.add_relationship("Flu", "Acetaminophen", "treated_by")
    medical_kg.add_relationship("Flu", "Ibuprofen", "treated_by")

    medical_kg.add_relationship("Ibuprofen", "Warfarin", "interacts_with", {"severity": "High", "mechanism": "Increased bleeding risk"})

    # 2. Initialize LLM Interface
    llm_interface = LLMInterface()

    # 3. Initialize Medical Diagnostic System
    system = MedicalDiagnosticSystem(kg=medical_kg, llm=llm_interface)

    # --- Demo Queries ---

    # Scenario 1: Diagnosis and Treatment Recommendation
    print(system.diagnose_and_recommend("I have a fever, cough, and I'm feeling very tired."))
    print(system.diagnose_and_recommend("My blood sugar is high and I'm urinating frequently."))
    print(system.diagnose_and_recommend("I have a headache and fatigue.")) # Might be ambiguous
    print(system.diagnose_and_recommend("I have a sore throat.")) # No entities extracted initially, will be handled by LLM mock

    # Scenario 2: Semantic Parsing for KGQA
    print(system.semantic_query("What are the symptoms of diabetes?"))
    print(system.semantic_query("What are the treatments for common cold?"))
    print(system.semantic_query("Do ibuprofen and warfarin interact?"))

    # Scenario 3: Direct Drug Interaction Check
    print(system.check_drug_interactions("Ibuprofen", "Warfarin"))
    print(system.check_drug_interactions("Acetaminophen", "Metformin"))

