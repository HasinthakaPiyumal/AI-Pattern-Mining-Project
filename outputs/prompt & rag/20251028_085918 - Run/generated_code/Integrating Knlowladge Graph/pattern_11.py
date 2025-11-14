import networkx as nx

class MedicalKnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.populate_sample_data()

    def populate_sample_data(self):
        # Diseases
        self.add_triple("Influenza", "has_symptom", "Fever")
        self.add_triple("Influenza", "has_symptom", "Cough")
        self.add_triple("Influenza", "has_symptom", "Body_Aches")
        self.add_triple("Influenza", "treated_by", "Antivirals")
        self.add_triple("Influenza", "treated_by", "Rest")

        self.add_triple("Common_Cold", "has_symptom", "Cough")
        self.add_triple("Common_Cold", "has_symptom", "Runny_Nose")
        self.add_triple("Common_Cold", "treated_by", "Symptomatic_Relief")

        self.add_triple("Pneumonia", "has_symptom", "Fever")
        self.add_triple("Pneumonia", "has_symptom", "Cough")
        self.add_triple("Pneumonia", "has_symptom", "Chest_Pain")
        self.add_triple("Pneumonia", "treated_by", "Antibiotics")

        # Symptoms
        self.add_triple("Fever", "indicates", "Infection")
        self.add_triple("Cough", "part_of", "Respiratory_Issues")

        # Treatments
        self.add_triple("Antivirals", "treats_type", "Viral_Infection")
        self.add_triple("Antibiotics", "treats_type", "Bacterial_Infection")

        # Labs (simplified)
        self.add_triple("CBC", "diagnoses", "Infection")
        self.add_triple("Chest_XRay", "diagnoses", "Pneumonia")

    def add_triple(self, s, p, o):
        self.graph.add_edge(s, o, relation=p)

    def query(self, entity, relation=None, max_hops=1):
        results = set()
        if entity in self.graph:
            # Direct outgoing edges
            for neighbor in self.graph.neighbors(entity):
                edge_data = self.graph.get_edge_data(entity, neighbor)
                if edge_data and (relation is None or edge_data["relation"] == relation):
                    results.add((entity, edge_data["relation"], neighbor))

            # Direct incoming edges (if looking for what causes/treats entity)
            for predecessor in self.graph.predecessors(entity):
                edge_data = self.graph.get_edge_data(predecessor, entity)
                if edge_data and (relation is None or edge_data["relation"] == relation):
                    results.add((predecessor, edge_data["relation"], entity))
            
            # Explore further if max_hops > 1
            if max_hops > 1:
                for s, p, o in list(results):
                    # For each result, query its object as a new entity
                    results.update(self.query(o, max_hops=max_hops - 1))
                    # And its subject as a new entity (if it was an incoming edge)
                    results.update(self.query(s, max_hops=max_hops - 1))
        return list(results)


class LLMAgent:
    def __init__(self):
        # In a real system, this would initialize an actual LLM client
        pass

    def generate_query(self, patient_context):
        symptoms = patient_context.get("symptoms", [])
        if not symptoms:
            return None
        
        # Simple heuristic: prioritize symptoms for initial queries
        # In a real LLM, this would involve natural language understanding
        # and converting to a structured query for the KG.
        first_symptom = symptoms[0].replace(" ", "_") # Make compatible with KG entity naming
        print(f"  [LLM Agent] Generating initial query for: {first_symptom}")
        return {"entity": first_symptom, "relation": "has_symptom", "max_hops": 2}

    def reason(self, kg_triples, patient_context):
        diagnosis_candidates = {}
        recommendation_candidates = {}
        
        patient_symptoms = [s.replace(" ", "_") for s in patient_context.get("symptoms", [])]

        for s, p, o in kg_triples:
            if p == "has_symptom" and o in patient_symptoms:
                # If a disease has a symptom the patient has, it's a candidate
                diagnosis_candidates[s] = diagnosis_candidates.get(s, 0) + 1
            elif p == "treated_by" and s in diagnosis_candidates:
                # If a treatment treats a candidate disease
                recommendation_candidates[o] = recommendation_candidates.get(o, 0) + 1
            elif p == "diagnoses" and o in diagnosis_candidates:
                # If a lab test diagnoses a candidate disease
                pass # Could suggest labs here

        # Simple reasoning: the disease with most matching symptoms is the likely one
        if diagnosis_candidates:
            likely_diagnosis = max(diagnosis_candidates, key=diagnosis_candidates.get)
            
            # Find treatments for the likely diagnosis
            treatments_for_diagnosis = [o for s, p, o in kg_triples if s == likely_diagnosis and p == "treated_by"]
            
            return {"diagnosis": likely_diagnosis, "treatments": treatments_for_diagnosis}
        return {}

    def prune_results(self, kg_triples, patient_context):
        # A simple pruning strategy: keep only triples directly relevant to patient's symptoms
        # or potential diagnoses/treatments identified so far.
        patient_symptoms = [s.replace(" ", "_") for s in patient_context.get("symptoms", [])]
        pruned_triples = []

        for s, p, o in kg_triples:
            if any(sym in [s, o] for sym in patient_symptoms) or \
               any(item in [s, o] for item in patient_context.get("current_diagnosis_candidates", [])) or \
               any(item in [s, o] for item in patient_context.get("current_treatment_candidates", [])):
                pruned_triples.append((s, p, o))
        return pruned_triples

    def generate_explanation(self, diagnosis_recommendation, relevant_triples):
        explanation = "Based on the patient's symptoms and reasoning over the medical knowledge graph:\n"
        
        if diagnosis_recommendation.get("diagnosis"):
            diagnosis = diagnosis_recommendation["diagnosis"]
            explanation += f"  *   **Diagnosis:** {diagnosis.replace("_", " ")}\n"
            
            symptoms_found = set()
            for s, p, o in relevant_triples:
                if s == diagnosis and p == "has_symptom":
                    symptoms_found.add(o.replace("_", " "))
            if symptoms_found:
                explanation += f"      (Supported by symptoms: {', '.join(symptoms_found)})\n"

            treatments = diagnosis_recommendation.get("treatments", [])
            if treatments:
                explanation += f"  *   **Recommended Treatments:** {', '.join([t.replace(' ', '_') for t in treatments])}\n"

        explanation += "\nRelevant knowledge graph facts considered:\n"
        for s, p, o in relevant_triples:
            explanation += f"  - {s.replace("_", " ")} {p.replace("_", " ")} {o.replace("_", " ")}\n"
        
        return explanation


class OrchestrationController:
    def __init__(self, kg_layer, llm_agent):
        self.kg_layer = kg_layer
        self.llm_agent = llm_agent
        self._context = {}
        self._prompts = {
            "initial_query": "Given the patient's symptoms {symptoms}, what initial medical entity should be explored?",
            "refine_query": "Based on the retrieved facts {facts} and current patient context {context}, what further information should be queried?",
            "reasoning": "Analyze these facts {facts} in the context of patient's {symptoms} and {history} to propose a diagnosis and treatment."
        }

    def process_patient_case(self, symptoms, history, max_iterations=3):
        self._context = {
            "symptoms": symptoms,
            "history": history,
            "explored_triples": set(),
            "current_diagnosis_candidates": [],
            "current_treatment_candidates": []
        }
        
        print("\n--- Starting Clinical Decision Support Process ---")
        print(f"Patient Symptoms: {', '.join(symptoms)}")
        print(f"Patient History: {history if history else 'None'}")

        all_relevant_triples = set()
        diagnosis_recommendation = {}

        for i in range(max_iterations):
            print(f"\nIteration {i+1}/{max_iterations}")
            
            # 1. LLM Agent generates initial/refined query
            query_params = self.llm_agent.generate_query(self._context)
            if not query_params:
                print("  [Orchestrator] No query generated. Exiting loop.")
                break

            # 2. KG querying and retrieval
            print(f"  [Orchestrator] Querying KG for entity: {query_params['entity']} (max_hops={query_params.get('max_hops', 1)}) ...")
            retrieved_triples = self.kg_layer.query(query_params["entity"], max_hops=query_params.get("max_hops", 1))
            print(f"  [Orchestrator] Retrieved {len(retrieved_triples)} triples.")
            
            # Add to all_relevant_triples for final explanation
            for triple in retrieved_triples:
                all_relevant_triples.add(triple)

            # 3. LLM Agent reasoning and pruning
            pruned_triples = self.llm_agent.prune_results(retrieved_triples, self._context)
            print(f"  [LLM Agent] Pruned to {len(pruned_triples)} relevant triples.")

            reasoning_output = self.llm_agent.reason(pruned_triples, self._context)
            if reasoning_output:
                self._context["current_diagnosis_candidates"] = [reasoning_output.get("diagnosis")] if reasoning_output.get("diagnosis") else []
                self._context["current_treatment_candidates"] = reasoning_output.get("treatments", [])
                diagnosis_recommendation = reasoning_output
                print(f"  [LLM Agent] Reasoning suggests: Diagnosis={reasoning_output.get('diagnosis')}, Treatments={reasoning_output.get('treatments')}")

            # Decision to continue or stop based on confidence or new info (simplified)
            if diagnosis_recommendation.get("diagnosis") and len(diagnosis_recommendation.get("treatments", [])) > 0:
                print("  [Orchestrator] Sufficient information for recommendation. Breaking loop.")
                break

        # 7. & 8. Generate final recommendation and explanation
        final_explanation = self.llm_agent.generate_explanation(diagnosis_recommendation, list(all_relevant_triples))
        print("\n--- Final Recommendation and Explanation ---")
        print(final_explanation)
        print("\n--- End of Clinical Decision Support Process ---")


if __name__ == "__main__":
    kg_layer = MedicalKnowledgeGraph()
    llm_agent = LLMAgent()
    controller = OrchestrationController(kg_layer, llm_agent)

    while True:
        print("\n--- Enter Patient Information ---")
        symptoms_input = input("Enter patient symptoms (comma-separated, e.g., Fever, Cough): ")
        history_input = input("Enter patient medical history (optional): ")

        symptoms = [s.strip() for s in symptoms_input.split(",") if s.strip()]

        if not symptoms:
            print("Please enter at least one symptom.")
            continue

        controller.process_patient_case(symptoms, history_input)

        another_case = input("Process another patient case? (yes/no): ").lower()
        if another_case != "yes":
            break
