class MedicalKnowledgeGraph:
    def __init__(self):
        self.triples = [] # (subject, predicate, object)

    def add_triple(self, subject, predicate, obj):
        self.triples.append((subject.lower(), predicate.lower(), obj.lower()))

    def query(self, entity=None, predicate=None, obj=None):
        results = []
        for s, p, o in self.triples:
            match = True
            if entity and entity.lower() not in [s, o]:
                match = False
            if predicate and predicate.lower() != p:
                match = False
            if obj and obj.lower() != o and obj.lower() != s: # check if obj matches either s or o
                match = False

            if match:
                results.append((s, p, o))
        return results

class LLMAgent:
    def __init__(self):
        # In a real scenario, this would be an actual LLM client (e.g., OpenAI, Hugging Face)
        pass

    def generate_initial_queries(self, symptoms):
        queries = []
        for symptom in symptoms:
            queries.append({"entity": symptom, "predicate": "is_symptom_of"})
        return queries

    def process_kg_response(self, kg_results, current_context):
        # Simulate LLM interpreting KG results and updating context
        diseases_found = set()
        for s, p, o in kg_results:
            if p == "is_symptom_of":
                diseases_found.add(o) # o is the disease
            elif p == "has_symptom":
                diseases_found.add(s) # s is the disease
        
        updated_context = current_context
        updated_context["potential_diseases"].update(diseases_found)
        return updated_context

    def refine_queries_and_prune(self, current_context, all_kg_triples):
        # Simulate iterative refinement and pruning
        # For simplicity, we'll "prune" by only considering triples relevant to potential diseases.
        relevant_triples = []
        potential_diseases = current_context["potential_diseases"]
        
        for s, p, o in all_kg_triples:
            if s in potential_diseases or o in potential_diseases:
                relevant_triples.append((s,p,o))
        
        if not potential_diseases: # If no diseases found yet, don't prune
            return all_kg_triples, [] # Return all triples if no diseases, and no refined queries yet

        # Further refinement: Generate queries for treatments for potential diseases
        new_queries = []
        for disease in potential_diseases:
            new_queries.append({"entity": disease, "predicate": "has_treatment"})
        
        return relevant_triples, new_queries


    def reason_and_diagnose(self, symptoms, kg_data, context):
        # Simulate LLM reasoning
        # This is a very simplified heuristic for demonstration
        
        disease_scores = {}
        for disease in context["potential_diseases"]:
            disease_scores[disease] = 0
            # Count how many symptoms match for each potential disease
            for s, p, o in kg_data:
                if p == "has_symptom" and s == disease and o in symptoms:
                    disease_scores[disease] += 1
                elif p == "is_symptom_of" and s in symptoms and o == disease:
                    disease_scores[disease] += 1
        
        if not disease_scores:
            return "No clear diagnosis based on provided symptoms and knowledge graph.", []
        
        # Select the disease with the highest score
        most_likely_disease = max(disease_scores, key=disease_scores.get)
        
        # Gather treatments for the most likely disease
        treatments = set()
        for s, p, o in kg_data:
            if p == "has_treatment" and s == most_likely_disease:
                treatments.add(o)

        return most_likely_disease, list(treatments)

    def generate_explanation(self, diagnosis, treatment, kg_facts):
        explanation = f"Based on your symptoms, the most likely diagnosis is {diagnosis}.\n"
        
        if treatment:
            explanation += f"Suggested treatments include: {', '.join(treatment)}.\n"
        else:
            explanation += "No specific treatments found in the knowledge base for this condition.\n"

        explanation += "\nReasoning details (from Knowledge Graph):\n"
        relevant_facts = [f"{s} {p} {o}" for s, p, o in kg_facts if diagnosis in [s, o] or (treatment and any(t in [s, o] for t in treatment))]
        
        if relevant_facts:
            explanation += "\n".join(relevant_facts)
        else:
            explanation += "No specific relevant facts found for explanation."

        return explanation

class MedicalDiagnosticAssistant:
    def __init__(self):
        self.kg = MedicalKnowledgeGraph()
        self.llm_agent = LLMAgent()
        self._initialize_knowledge_graph()

    def _initialize_knowledge_graph(self):
        # A small, illustrative medical knowledge graph
        self.kg.add_triple("Fever", "is_symptom_of", "Flu")
        self.kg.add_triple("Cough", "is_symptom_of", "Flu")
        self.kg.add_triple("Sore Throat", "is_symptom_of", "Flu")
        self.kg.add_triple("Headache", "is_symptom_of", "Flu")
        self.kg.add_triple("Flu", "has_symptom", "Fever")
        self.kg.add_triple("Flu", "has_symptom", "Cough")
        self.kg.add_triple("Flu", "has_symptom", "Sore Throat")
        self.kg.add_triple("Flu", "has_symptom", "Headache")
        self.kg.add_triple("Flu", "has_treatment", "Rest")
        self.kg.add_triple("Flu", "has_treatment", "Fluids")
        self.kg.add_triple("Flu", "has_treatment", "Antivirals")

        self.kg.add_triple("Headache", "is_symptom_of", "Migraine")
        self.kg.add_triple("Nausea", "is_symptom_of", "Migraine")
        self.kg.add_triple("Sensitivity to Light", "is_symptom_of", "Migraine")
        self.kg.add_triple("Migraine", "has_symptom", "Headache")
        self.kg.add_triple("Migraine", "has_symptom", "Nausea")
        self.kg.add_triple("Migraine", "has_symptom", "Sensitivity to Light")
        self.kg.add_triple("Migraine", "has_treatment", "Painkillers")
        self.kg.add_triple("Migraine", "has_treatment", "Triptans")

        self.kg.add_triple("Skin Rash", "is_symptom_of", "Allergy")
        self.kg.add_triple("Itching", "is_symptom_of", "Allergy")
        self.kg.add_triple("Sneezing", "is_symptom_of", "Allergy")
        self.kg.add_triple("Allergy", "has_symptom", "Skin Rash")
        self.kg.add_triple("Allergy", "has_symptom", "Itching")
        self.kg.add_triple("Allergy", "has_symptom", "Sneezing")
        self.kg.add_triple("Allergy", "has_treatment", "Antihistamines")

    def diagnose(self, symptoms):
        print(f"User Symptoms: {', '.join(symptoms)}")
        
        current_context = {"potential_diseases": set(), "retrieved_facts": []}
        
        # Step 1: LLM Agent generates initial queries based on symptoms
        initial_queries = self.llm_agent.generate_initial_queries(symptoms)
        print(f"\nLLM Agent generating initial queries: {initial_queries}")

        # Step 2: Query KG and LLM Agent processes response
        for query in initial_queries:
            kg_results = self.kg.query(entity=query["entity"], predicate=query["predicate"])
            current_context = self.llm_agent.process_kg_response(kg_results, current_context)
            current_context["retrieved_facts"].extend(kg_results)
        
        print(f"\nInitial KG exploration results (potential diseases): {current_context['potential_diseases']}")

        # Step 3: LLM Agent refines queries and prunes graph (iterative exploration)
        # For simplicity, we'll re-query the KG with the refined criteria
        all_kg_triples = self.kg.triples # Get all triples to simulate pruning over the entire graph
        
        # Simulate pruning by only keeping relevant triples for potential diseases
        pruned_triples, refined_queries = self.llm_agent.refine_queries_and_prune(current_context, all_kg_triples)
        current_context["retrieved_facts"] = list(set(current_context["retrieved_facts"]) | set(pruned_triples)) # Update retrieved facts with pruned ones and de-duplicate
        
        print(f"\nLLM Agent refining queries and pruning. Pruned facts count: {len(pruned_triples)}")
        print(f"Refined queries: {refined_queries}")

        # Execute refined queries
        for query in refined_queries:
            kg_results = self.kg.query(entity=query["entity"], predicate=query["predicate"])
            current_context = self.llm_agent.process_kg_response(kg_results, current_context)
            current_context["retrieved_facts"].extend(kg_results) # Add new facts


        # Step 4: LLM Agent reasons and diagnoses
        diagnosis, treatments = self.llm_agent.reason_and_diagnose(
            symptoms, current_context["retrieved_facts"], current_context
        )
        print(f"\nLLM Agent reasoning: Diagnosis = {diagnosis}, Treatments = {treatments}")

        # Step 5: LLM Agent generates explanation
        explanation = self.llm_agent.generate_explanation(diagnosis, treatments, current_context["retrieved_facts"])
        print("\n--- Diagnosis Report ---")
        print(explanation)
        return diagnosis, treatments, explanation

# Example Usage:
if __name__ == "__main__":
    assistant = MedicalDiagnosticAssistant()
    
    # Test case 1
    print("\n##### TEST CASE 1: Flu Symptoms #####")
    symptoms_1 = ["fever", "cough", "headache"]
    assistant.diagnose(symptoms_1)

    print("\n##### TEST CASE 2: Migraine Symptoms #####")
    symptoms_2 = ["headache", "nausea", "sensitivity to light"]
    assistant.diagnose(symptoms_2)

    print("\n##### TEST CASE 3: Allergy Symptoms #####")
    symptoms_3 = ["skin rash", "itching"]
    assistant.diagnose(symptoms_3)

    print("\n##### TEST CASE 4: Mixed/Insufficient Symptoms #####")
    symptoms_4 = ["headache", "itching"]
    assistant.diagnose(symptoms_4)

    print("\n##### TEST CASE 5: Unknown Symptom #####")
    symptoms_5 = ["stomach ache"]
    assistant.diagnose(symptoms_5)