
import networkx as nx

class MedicalKnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self._populate_kg()

    def _populate_kg(self):
        # Diseases
        self.graph.add_node("Influenza", type="disease")
        self.graph.add_node("Common Cold", type="disease")
        self.graph.add_node("Pneumonia", type="disease")
        self.graph.add_node("Diabetes Type 2", type="disease")
        self.graph.add_node("Hypertension", type="disease")
        self.graph.add_node("Migraine", type="disease")
        self.graph.add_node("Asthma", type="disease")

        # Symptoms
        self.graph.add_node("Fever", type="symptom")
        self.graph.add_node("Cough", type="symptom")
        self.graph.add_node("Sore Throat", type="symptom")
        self.graph.add_node("Runny Nose", type="symptom")
        self.graph.add_node("Headache", type="symptom")
        self.graph.add_node("Fatigue", type="symptom")
        self.graph.add_node("Shortness of Breath", type="symptom")
        self.graph.add_node("High Blood Sugar", type="symptom")
        self.graph.add_node("High Blood Pressure", type="symptom")
        self.graph.add_node("Nausea", type="symptom")
        self.graph.add_node("Chest Pain", type="symptom")
        self.graph.add_node("Wheezing", type="symptom")

        # Treatments
        self.graph.add_node("Antivirals", type="treatment")
        self.graph.add_node("Rest", type="treatment")
        self.graph.add_node("Fluids", type="treatment")
        self.graph.add_node("Pain Relievers", type="treatment")
        self.graph.add_node("Antibiotics", type="treatment")
        self.graph.add_node("Insulin", type="treatment")
        self.graph.add_node("Statins", type="treatment")
        self.graph.add_node("Beta-blockers", type="treatment")
        self.graph.add_node("Inhalers", type="treatment")

        # Relations
        self.graph.add_edge("Influenza", "causes", "Fever")
        self.graph.add_edge("Influenza", "causes", "Cough")
        self.graph.add_edge("Influenza", "causes", "Sore Throat")
        self.graph.add_edge("Influenza", "causes", "Fatigue")
        self.graph.add_edge("Influenza", "treats", "Antivirals")
        self.graph.add_edge("Influenza", "treats", "Rest")
        self.graph.add_edge("Influenza", "treats", "Fluids")

        self.graph.add_edge("Common Cold", "causes", "Runny Nose")
        self.graph.add_edge("Common Cold", "causes", "Sore Throat")
        self.graph.add_edge("Common Cold", "causes", "Cough")
        self.graph.add_edge("Common Cold", "treats", "Rest")
        self.graph.add_edge("Common Cold", "treats", "Fluids")
        self.graph.add_edge("Common Cold", "treats", "Pain Relievers")

        self.graph.add_edge("Pneumonia", "causes", "Fever")
        self.graph.add_edge("Pneumonia", "causes", "Cough")
        self.graph.add_edge("Pneumonia", "causes", "Shortness of Breath")
        self.graph.add_edge("Pneumonia", "causes", "Chest Pain")
        self.graph.add_edge("Pneumonia", "treats", "Antibiotics")
        self.graph.add_edge("Pneumonia", "treats", "Rest")

        self.graph.add_edge("Diabetes Type 2", "causes", "High Blood Sugar")
        self.graph.add_edge("Diabetes Type 2", "treats", "Insulin")

        self.graph.add_edge("Hypertension", "causes", "High Blood Pressure")
        self.graph.add_edge("Hypertension", "treats", "Beta-blockers")
        self.graph.add_edge("Hypertension", "treats", "Statins") # often co-prescribed

        self.graph.add_edge("Migraine", "causes", "Headache")
        self.graph.add_edge("Migraine", "causes", "Nausea")
        self.graph.add_edge("Migraine", "treats", "Pain Relievers")

        self.graph.add_edge("Asthma", "causes", "Shortness of Breath")
        self.graph.add_edge("Asthma", "causes", "Wheezing")
        self.graph.add_edge("Asthma", "treats", "Inhalers")

        self.graph.add_edge("Fever", "indicates", "Influenza")
        self.graph.add_edge("Cough", "indicates", "Influenza")
        self.graph.add_edge("Cough", "indicates", "Common Cold")
        self.graph.add_edge("Cough", "indicates", "Pneumonia")
        self.graph.add_edge("Sore Throat", "indicates", "Influenza")
        self.graph.add_edge("Sore Throat", "indicates", "Common Cold")
        self.graph.add_edge("Runny Nose", "indicates", "Common Cold")
        self.graph.add_edge("Headache", "indicates", "Influenza")
        self.graph.add_edge("Headache", "indicates", "Migraine")
        self.graph.add_edge("Fatigue", "indicates", "Influenza")
        self.graph.add_edge("Shortness of Breath", "indicates", "Pneumonia")
        self.graph.add_edge("Shortness of Breath", "indicates", "Asthma")
        self.graph.add_edge("High Blood Sugar", "indicates", "Diabetes Type 2")
        self.graph.add_edge("High Blood Pressure", "indicates", "Hypertension")
        self.graph.add_edge("Nausea", "indicates", "Migraine")
        self.graph.add_edge("Chest Pain", "indicates", "Pneumonia")
        self.graph.add_edge("Wheezing", "indicates", "Asthma")


    def query(self, source_entity, relation_type=None, target_type=None):
        results = []
        if source_entity not in self.graph:
            return []

        for neighbor in self.graph.neighbors(source_entity):
            # Check if there's an edge from source_entity to neighbor
            if self.graph.has_edge(source_entity, neighbor):
                edge_data = self.graph.get_edge_data(source_entity, neighbor)
                # The relation type is the attribute of the edge
                # For this simple graph, we assume the edge itself is the relation
                # In networkx, edge attributes are dictionaries.
                # We need to adapt this if we encoded relation type as an edge attribute.
                # For now, let's assume direct edges mean a relation.
                # To make it more explicit for our conceptual model:
                # We use the edge weight as the relation type (simulated)
                # self.graph.add_edge("Influenza", "Fever", relation="causes")

                # Re-designing edge storage for explicit relations:
                # In networkx, we can store edge attributes.
                # Let's assume edges are stored like (u, v, {'relation': 'type'})
                # However, the current population uses add_edge(u, relation, v)
                # This is actually creating a path: u -> relation -> v
                # This is a common way to represent reified relationships in KGs using triple stores.
                # So, a query for 'source -> relation -> target' would be looking for paths of length 2.

                # Let's adjust query to handle the current _populate_kg structure (source -> relation_node -> target_node)
                for relation_node in self.graph.successors(source_entity):
                    if self.graph.nodes[relation_node].get("type") == "relation":
                        if relation_type and relation_node != relation_type:
                            continue
                        for target_node in self.graph.successors(relation_node):
                            if target_type and self.graph.nodes[target_node].get("type") != target_type:
                                continue
                            results.append((source_entity, relation_node, target_node))
            else:
                # If the graph was built with direct edges like self.graph.add_edge(source, target, relation='type')
                # But our current implementation uses reified relations.
                pass # The outer loop for neighbors is not directly used in the reified approach.

        # Re-implementing query based on the specific triple-like structure in _populate_kg
        # Our graph structure is: Source -> Relation_Node -> Target
        final_results = []
        if source_entity in self.graph:
            for intermediate_node in self.graph.successors(source_entity):
                # Check if the intermediate_node is actually a relation type we're interested in
                # In our current setup, the 'relation_node' IS the relation_type itself.
                if relation_type and intermediate_node != relation_type:
                    continue
                
                # Check if this intermediate node is part of a relation triple
                # For simplicity, we assume any node directly connected from source_entity that has outgoing edges
                # to other entities is a 'relation node' in this context.

                # It's better to explicitly model relation nodes. Let's adjust populate_kg for this idea for clarity:
                # self.graph.add_node("causes", type="relation")
                # self.graph.add_node("treats", type="relation")
                # self.graph.add_edge("Influenza", "causes")
                # self.graph.add_edge("causes", "Fever")
                # This is more explicit but requires changes to how _populate_kg adds edges.

                # Sticking to the current _populate_kg:  self.graph.add_edge("Influenza", "causes", "Fever") is NOT how networkx works
                # It means self.graph.add_edge(u, v) - there is an edge between u and v.
                # So current _populate_kg creates direct edges like:
                # Influenza -> Fever (with a label of 'causes' if that were possible via attributes)
                # Influenza -> Antivirals (with a label of 'treats')

                # Let's re-populate KG with explicit edge attributes for relations
                # and modify the query method accordingly.

        # Re-initialise and re-populate the graph with explicit edge attributes
        self.graph = nx.DiGraph()
        self._populate_kg_with_attributes()

        for u, v, data in self.graph.edges(data=True):
            if u == source_entity:
                if relation_type is None or data.get('relation') == relation_type:
                    if target_type is None or self.graph.nodes[v].get('type') == target_type:
                        final_results.append((u, data.get('relation'), v))
        return final_results

    def _populate_kg_with_attributes(self):
        # Diseases
        self.graph.add_node("Influenza", type="disease")
        self.graph.add_node("Common Cold", type="disease")
        self.graph.add_node("Pneumonia", type="disease")
        self.graph.add_node("Diabetes Type 2", type="disease")
        self.graph.add_node("Hypertension", type="disease")
        self.graph.add_node("Migraine", type="disease")
        self.graph.add_node("Asthma", type="disease")

        # Symptoms
        self.graph.add_node("Fever", type="symptom")
        self.graph.add_node("Cough", type="symptom")
        self.graph.add_node("Sore Throat", type="symptom")
        self.graph.add_node("Runny Nose", type="symptom")
        self.graph.add_node("Headache", type="symptom")
        self.graph.add_node("Fatigue", type="symptom")
        self.graph.add_node("Shortness of Breath", type="symptom")
        self.graph.add_node("High Blood Sugar", type="symptom")
        self.graph.add_node("High Blood Pressure", type="symptom")
        self.graph.add_node("Nausea", type="symptom")
        self.graph.add_node("Chest Pain", type="symptom")
        self.graph.add_node("Wheezing", type="symptom")

        # Treatments
        self.graph.add_node("Antivirals", type="treatment")
        self.graph.add_node("Rest", type="treatment")
        self.graph.add_node("Fluids", type="treatment")
        self.graph.add_node("Pain Relievers", type="treatment")
        self.graph.add_node("Antibiotics", type="treatment")
        self.graph.add_node("Insulin", type="treatment")
        self.graph.add_node("Statins", type="treatment")
        self.graph.add_node("Beta-blockers", type="treatment")
        self.graph.add_node("Inhalers", type="treatment")

        # Relations
        self.graph.add_edge("Influenza", "Fever", relation="causes")
        self.graph.add_edge("Influenza", "Cough", relation="causes")
        self.graph.add_edge("Influenza", "Sore Throat", relation="causes")
        self.graph.add_edge("Influenza", "Fatigue", relation="causes")
        self.graph.add_edge("Influenza", "Antivirals", relation="treats")
        self.graph.add_edge("Influenza", "Rest", relation="treats")
        self.graph.add_edge("Influenza", "Fluids", relation="treats")

        self.graph.add_edge("Common Cold", "Runny Nose", relation="causes")
        self.graph.add_edge("Common Cold", "Sore Throat", relation="causes")
        self.graph.add_edge("Common Cold", "Cough", relation="causes")
        self.graph.add_edge("Common Cold", "Rest", relation="treats")
        self.graph.add_edge("Common Cold", "Fluids", relation="treats")
        self.graph.add_edge("Common Cold", "Pain Relievers", relation="treats")

        self.graph.add_edge("Pneumonia", "Fever", relation="causes")
        self.graph.add_edge("Pneumonia", "Cough", relation="causes")
        self.graph.add_edge("Pneumonia", "Shortness of Breath", relation="causes")
        self.graph.add_edge("Pneumonia", "Chest Pain", relation="causes")
        self.graph.add_edge("Pneumonia", "Antibiotics", relation="treats")
        self.graph.add_edge("Pneumonia", "Rest", relation="treats")

        self.graph.add_edge("Diabetes Type 2", "High Blood Sugar", relation="causes")
        self.graph.add_edge("Diabetes Type 2", "Insulin", relation="treats")

        self.graph.add_edge("Hypertension", "High Blood Pressure", relation="causes")
        self.graph.add_edge("Hypertension", "Beta-blockers", relation="treats")
        self.graph.add_edge("Hypertension", "Statins", relation="treats")

        self.graph.add_edge("Migraine", "Headache", relation="causes")
        self.graph.add_edge("Migraine", "Nausea", relation="causes")
        self.graph.add_edge("Migraine", "Pain Relievers", relation="treats")

        self.graph.add_edge("Asthma", "Shortness of Breath", relation="causes")
        self.graph.add_edge("Asthma", "Wheezing", relation="causes")
        self.graph.add_edge("Asthma", "Inhalers", relation="treats")

        # Inverse relations (for symptom -> disease lookup)
        self.graph.add_edge("Fever", "Influenza", relation="indicates")
        self.graph.add_edge("Cough", "Influenza", relation="indicates")
        self.graph.add_edge("Cough", "Common Cold", relation="indicates")
        self.graph.add_edge("Cough", "Pneumonia", relation="indicates")
        self.graph.add_edge("Sore Throat", "Influenza", relation="indicates")
        self.graph.add_edge("Sore Throat", "Common Cold", relation="indicates")
        self.graph.add_edge("Runny Nose", "Common Cold", relation="indicates")
        self.graph.add_edge("Headache", "Influenza", relation="indicates")
        self.graph.add_edge("Headache", "Migraine", relation="indicates")
        self.graph.add_edge("Fatigue", "Influenza", relation="indicates")
        self.graph.add_edge("Shortness of Breath", "Pneumonia", relation="indicates")
        self.graph.add_edge("Shortness of Breath", "Asthma", relation="indicates")
        self.graph.add_edge("High Blood Sugar", "Diabetes Type 2", relation="indicates")
        self.graph.add_edge("High Blood Pressure", "Hypertension", relation="indicates")
        self.graph.add_edge("Nausea", "Migraine", relation="indicates")
        self.graph.add_edge("Chest Pain", "Pneumonia", relation="indicates")
        self.graph.add_edge("Wheezing", "Asthma", relation="indicates")


class LLMAgent:
    def __init__(self, kg):
        self.kg = kg
        self.context = {}

    def _simulate_llm_response(self, prompt_type, **kwargs):
        # A highly simplified LLM simulation using string matching and basic logic
        if prompt_type == "generate_query":
            symptoms = kwargs.get("symptoms", [])
            if "Fever" in symptoms and "Cough" in symptoms:
                return {"query_type": "find_diseases_by_symptoms", "symptoms": symptoms}
            elif "Headache" in symptoms:
                return {"query_type": "find_diseases_by_symptoms", "symptoms": symptoms}
            return {"query_type": "find_diseases_by_symptoms", "symptoms": symptoms}

        elif prompt_type == "interpret_results":
            kg_results = kwargs.get("kg_results", [])
            query = kwargs.get("query", {})
            interpreted_info = {"diseases": [], "treatments": [], "evidence": []}

            for u, rel, v in kg_results:
                interpreted_info["evidence"].append(f"{u} {rel} {v}")
                if self.kg.graph.nodes[v].get("type") == "disease":
                    interpreted_info["diseases"].append(v)
                elif self.kg.graph.nodes[v].get("type") == "treatment":
                    interpreted_info["treatments"].append(v)

            # Simple hypothesis generation based on disease counts
            disease_counts = {d: interpreted_info["diseases"].count(d) for d in set(interpreted_info["diseases"])}
            sorted_diseases = sorted(disease_counts.items(), key=lambda item: item[1], reverse=True)
            if sorted_diseases:
                interpreted_info["primary_diagnosis_hypothesis"] = sorted_diseases[0][0]
            
            return interpreted_info

        elif prompt_type == "reason_and_explain":
            diagnosis_hypotheses = kwargs.get("diagnosis_hypotheses", "")
            kg_evidence = kwargs.get("kg_evidence", [])
            treatments = kwargs.get("treatments", [])

            explanation = f"Based on the gathered knowledge from the medical graph:\n"
            explanation += f"  - Potential diagnosis: {diagnosis_hypotheses}\n"
            if treatments:
                explanation += f"  - Recommended treatments: {', '.join(treatments)}\n"
            explanation += f"  - Supporting evidence: {'; '.join(kg_evidence)}"
            return {"explanation": explanation, "diagnosis": diagnosis_hypotheses, "treatments": treatments}

        elif prompt_type == "prune_paths":
            kg_results = kwargs.get("kg_results", [])
            focus = kwargs.get("current_focus", "")
            pruned_results = []
            for triple in kg_results:
                # Simple pruning: keep only results related to the current focus (e.g., a specific disease hypothesis)
                if focus.lower() in str(triple).lower():
                    pruned_results.append(triple)
            return pruned_results

        return {}

    def generate_query(self, symptoms, history=""):
        # Simulates LLM generating a query based on symptoms and history
        prompt_context = {"symptoms": symptoms, "history": history}
        llm_output = self._simulate_llm_response("generate_query", **prompt_context)
        return llm_output

    def interpret_results(self, kg_results, query):
        # Simulates LLM interpreting KG query results
        prompt_context = {"kg_results": kg_results, "query": query}
        llm_output = self._simulate_llm_response("interpret_results", **prompt_context)
        return llm_output

    def reason_and_explain(self, diagnosis_hypotheses, kg_evidence, treatments):
        # Simulates LLM performing reasoning and generating an explanation
        prompt_context = {"diagnosis_hypotheses": diagnosis_hypotheses, "kg_evidence": kg_evidence, "treatments": treatments}
        llm_output = self._simulate_llm_response("reason_and_explain", **prompt_context)
        return llm_output
    
    def prune_knowledge(self, kg_results, current_focus):
        # Simulates LLM-driven pruning of irrelevant knowledge
        prompt_context = {"kg_results": kg_results, "current_focus": current_focus}
        llm_output = self._simulate_llm_response("prune_paths", **prompt_context)
        return llm_output


class KGARMedicalSystem:
    def __init__(self):
        self.kg = MedicalKnowledgeGraph()
        self.llm_agent = LLMAgent(self.kg)

    def diagnose_patient(self, symptoms, patient_history=""):
        print(f"\n--- Starting KGAR Diagnosis for Symptoms: {', '.join(symptoms)} ---")
        
        # Iteration 1: Initial Query Generation and Exploration
        print("\n[LLM Agent] Generating initial query...")
        initial_query = self.llm_agent.generate_query(symptoms, patient_history)
        print(f"[LLM Agent] Generated query: {initial_query}")

        kg_exploration_results = []
        if initial_query.get("query_type") == "find_diseases_by_symptoms":
            for symptom in initial_query["symptoms"]:
                # Query KG for diseases indicated by each symptom
                symptom_indications = self.kg.query(symptom, relation_type="indicates", target_type="disease")
                kg_exploration_results.extend(symptom_indications)
        
        print("\n[KG Module] Initial KG exploration results (Symptom -> Disease):")
        for res in kg_exploration_results:
            print(f"  - {res[0]} {res[1]} {res[2]}")

        # Iteration 2: Interpret Results and Form Hypotheses
        print("\n[LLM Agent] Interpreting initial KG results and forming hypotheses...")
        interpretation = self.llm_agent.interpret_results(kg_exploration_results, initial_query)
        potential_diagnosis = interpretation.get("primary_diagnosis_hypothesis")
        current_kg_evidence = interpretation.get("evidence", [])
        print(f"[LLM Agent] Primary diagnosis hypothesis: {potential_diagnosis}")
        print(f"[LLM Agent] Initial evidence gathered: {len(current_kg_evidence)} triples")

        # Iteration 3: Refine and Prune based on Hypothesis (Simulated)
        if potential_diagnosis:
            print(f"\n[LLM Agent] Refining knowledge around '{potential_diagnosis}' and pruning...")
            # Query KG for symptoms caused by the potential diagnosis
            disease_causes_symptoms = self.kg.query(potential_diagnosis, relation_type="causes", target_type="symptom")
            kg_exploration_results.extend(disease_causes_symptoms)
            
            # Query KG for treatments for the potential diagnosis
            disease_treatments = self.kg.query(potential_diagnosis, relation_type="treats", target_type="treatment")
            kg_exploration_results.extend(disease_treatments)

            # Simulate pruning: keep only triples directly related to the potential diagnosis
            pruned_results = self.llm_agent.prune_knowledge(kg_exploration_results, potential_diagnosis)
            current_kg_evidence = [f"{u} {rel} {v}" for u, rel, v in pruned_results]
            
            # Re-interpret with pruned and expanded knowledge
            re_interpretation = self.llm_agent.interpret_results(pruned_results, initial_query)
            final_diagnosis_hypothesis = re_interpretation.get("primary_diagnosis_hypothesis", potential_diagnosis)
            recommended_treatments = list(set(re_interpretation.get("treatments", [])))

            print("\n[KG Module] Refined KG exploration results (Disease -> Symptoms/Treatments):")
            for res in pruned_results:
                print(f"  - {res[0]} {res[1]} {res[2]}")

            print("\n[LLM Agent] Generating final reasoning and explanation...")
            final_output = self.llm_agent.reason_and_explain(final_diagnosis_hypothesis, current_kg_evidence, recommended_treatments)
            
            print("\n--- Diagnosis and Treatment Recommendation ---")
            print(f"Diagnosis: {final_output['diagnosis']}")
            print(f"Recommended Treatments: {', '.join(final_output['treatments']) if final_output['treatments'] else 'None'}")
            print("Explanation:")
            print(final_output['explanation'])
        else:
            print("\n[System] Could not determine a primary diagnosis hypothesis based on initial symptoms.")

        print("\n--- KGAR Diagnosis Complete ---")


# User Interface (Basic Command Line)
def main():
    system = KGARMedicalSystem()

    while True:
        print("\n-------------------------------------------------")
        print("Medical Diagnostic and Treatment Recommendation System")
        print("-------------------------------------------------")
        symptoms_input = input("Enter patient symptoms (comma-separated, e.g., Fever,Cough,Fatigue): ")
        if not symptoms_input:
            print("No symptoms entered. Exiting.")
            break
        
        symptoms = [s.strip() for s in symptoms_input.split(',') if s.strip()]
        patient_history = input("Enter patient history (optional): ")

        system.diagnose_patient(symptoms, patient_history)

        another_case = input("\nDiagnose another patient? (yes/no): ").lower()
        if another_case != 'yes':
            break

if __name__ == "__main__":
    main()
