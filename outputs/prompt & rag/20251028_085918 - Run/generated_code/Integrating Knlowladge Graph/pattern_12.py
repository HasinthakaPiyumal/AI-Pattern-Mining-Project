import networkx as nx

class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_triple(self, subject, predicate, obj):
        self.graph.add_edge(subject, obj, relation=predicate)

    def query(self, entity=None, max_depth=2):
        """Simulates querying the KG to find related information up to a certain depth."""
        if entity not in self.graph:
            return []

        found_triples = set()
        visited = {entity}
        queue = [(entity, 0)]  # (node, current_depth)

        while queue:
            current_node, current_depth = queue.pop(0)

            if current_depth < max_depth:
                for neighbor in self.graph.neighbors(current_node):
                    relation = self.graph[current_node][neighbor]["relation"]
                    found_triples.add((current_node, relation, neighbor))
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, current_depth + 1))
                for predecessor in self.graph.predecessors(current_node):
                    relation = self.graph[predecessor][current_node]["relation"]
                    found_triples.add((predecessor, relation, current_node))
                    if predecessor not in visited:
                        visited.add(predecessor)
                        queue.append((predecessor, current_depth + 1))

        return list(found_triples)

class QueryGenerationAgent:
    def generate_initial_query(self, natural_language_query):
        """Simulates generating an initial KG query (e.g., identifying key entities)."""
        # Very simple keyword extraction for simulation
        if "fever" in natural_language_query.lower():
            return "Symptom:Fever"
        if "cough" in natural_language_query.lower():
            return "Symptom:Cough"
        if "patient x" in natural_language_query.lower():
            return "Patient:X"
        return None

    def refine_query(self, current_query, retrieved_data, context, iteration):
        """Simulates refining the query based on retrieved data and context."""
        # For simulation, just add depth to exploration or stop after a few iterations
        if iteration < 2 and retrieved_data:
            # Try to find new entities from the retrieved data
            new_entities = set()
            for s, p, o in retrieved_data:
                if s.startswith("Symptom:") or s.startswith("Disease:") or s.startswith("Drug:") or s.startswith("Patient:"):
                    new_entities.add(s)
                if o.startswith("Symptom:") or o.startswith("Disease:") or o.startswith("Drug:") or o.startswith("Patient:"):
                    new_entities.add(o)
            if new_entities:
                # This is a simplification; a real LLM would choose the most relevant ones
                return list(new_entities)[0] # Just pick one new entity to expand from
        return None # Signal to stop further exploration

class RelevancePruningAgent:
    def prune(self, retrieved_triples, context):
        """Simulates pruning irrelevant triples based on context."""
        pruned_triples = []
        patient_symptoms = context.get("symptoms", [])
        for s, p, o in retrieved_triples:
            # Simple pruning: keep triples relevant to patient's symptoms or common diseases
            is_relevant = False
            for symptom in patient_symptoms:
                if symptom.lower() in s.lower() or symptom.lower() in o.lower():
                    is_relevant = True
                    break
            if "disease" in s.lower() or "disease" in o.lower() or "treatment" in p.lower() or is_relevant:
                pruned_triples.append((s,p,o))
        return pruned_triples

class ReasoningAndExplanationAgent:
    def reason(self, pruned_data, patient_context):
        """Simulates reasoning to derive diagnosis/treatment."""
        diagnosis = "Unknown Disease"
        treatment_plan = "Consult a doctor for further examination."
        drug_interactions = "None identified."

        found_diseases = set()
        found_treatments = set()
        found_drug_interactions = set()

        for s, p, o in pruned_data:
            if p == "indicates" and s.startswith("Symptom:") and o.startswith("Disease:"):
                found_diseases.add(o)
            elif p == "treats" and s.startswith("Drug:") and o.startswith("Disease:"):
                found_treatments.add(f"{s} for {o}")
            elif p == "interacts_with" and s.startswith("Drug:") and o.startswith("Drug:"):
                found_drug_interactions.add(f"{s} interacts with {o}")

        if found_diseases:
            diagnosis = f"Possible diseases: {', '.join(list(found_diseases))}"
        if found_treatments:
            treatment_plan = f"Suggested treatments: {', '.join(list(found_treatments))}"
        if found_drug_interactions:
            drug_interactions = f"Potential drug interactions: {', '.join(list(found_drug_interactions))}"

        return diagnosis, treatment_plan, drug_interactions

    def explain(self, diagnosis, treatment_plan, drug_interactions, used_facts):
        """Simulates generating an interpretable explanation."""
        explanation_parts = [
            "Based on the information gathered from the medical knowledge graph:"
        ]
        explanation_parts.append(f"Diagnosis: {diagnosis}")
        explanation_parts.append(f"Treatment Plan: {treatment_plan}")
        explanation_parts.append(f"Drug Interactions: {drug_interactions}")

        if used_facts:
            explanation_parts.append("\nSupporting facts from Knowledge Graph:")
            for fact in used_facts:
                explanation_parts.append(f"- {fact[0]} {fact[1]} {fact[2]}")
        else:
            explanation_parts.append("\nNo specific supporting facts could be traced directly for this outcome (due to simulation limitations).")

        return "\n".join(explanation_parts)

class MediReasonerSystem:
    def __init__(self):
        self.kg = KnowledgeGraph()
        self.query_agent = QueryGenerationAgent()
        self.pruning_agent = RelevancePruningAgent()
        self.reasoning_agent = ReasoningAndExplanationAgent()
        self._initialize_knowledge_graph()

    def _initialize_knowledge_graph(self):
        # Sample medical data
        self.kg.add_triple("Symptom:Fever", "indicates", "Disease:Influenza")
        self.kg.add_triple("Symptom:Fever", "indicates", "Disease:CommonCold")
        self.kg.add_triple("Symptom:Cough", "indicates", "Disease:Influenza")
        self.kg.add_triple("Symptom:Fatigue", "indicates", "Disease:Influenza")
        self.kg.add_triple("Disease:Influenza", "treated_by", "Drug:Oseltamivir")
        self.kg.add_triple("Disease:CommonCold", "treated_by", "Drug:Paracetamol")
        self.kg.add_triple("Drug:Paracetamol", "alleviates", "Symptom:Fever")
        self.kg.add_triple("Drug:Oseltamivir", "is_antiviral_for", "Disease:Influenza")
        self.kg.add_triple("Drug:Warfarin", "interacts_with", "Drug:Aspirin")
        self.kg.add_triple("Patient:X", "has_symptom", "Symptom:Fever")
        self.kg.add_triple("Patient:X", "has_symptom", "Symptom:Cough")
        self.kg.add_triple("Patient:X", "has_symptom", "Symptom:Fatigue")
        self.kg.add_triple("Patient:Y", "has_symptom", "Symptom:Fever")
        self.kg.add_triple("Patient:Y", "on_medication", "Drug:Warfarin")
        self.kg.add_triple("Patient:Y", "on_medication", "Drug:Aspirin")

    def process_patient_query(self, natural_language_query, patient_context=None):
        print(f"\nProcessing query: '{natural_language_query}'")
        if patient_context is None:
            patient_context = {"symptoms": [], "medications": []}

        # 1. Initial Query Generation
        initial_entity = self.query_agent.generate_initial_query(natural_language_query)
        if not initial_entity:
            return "Could not identify a starting point for the query.", "No explanation available."

        current_entity_for_exploration = initial_entity
        all_retrieved_triples = set()
        max_exploration_iterations = 3
        
        print(f"Initial entity for exploration: {initial_entity}")

        # 2. Iterative KG Exploration & Pruning
        for i in range(max_exploration_iterations):
            print(f"  Exploration Iteration {i+1}: Querying KG with entity '{current_entity_for_exploration}'")
            retrieved_data = self.kg.query(entity=current_entity_for_exploration, max_depth=1) # Explore one step at a time
            all_retrieved_triples.update(retrieved_data)

            # Simulate LLM agent refining the query based on current findings
            next_entity_to_explore = self.query_agent.refine_query(current_entity_for_exploration, retrieved_data, patient_context, i)

            if not next_entity_to_explore or next_entity_to_explore == current_entity_for_exploration:
                print("  Query agent decided to stop or no new entity to explore.")
                break
            current_entity_for_exploration = next_entity_to_explore

        # 3. Relevance Pruning on aggregated data
        print("\nApplying relevance pruning...")
        pruned_triples = self.pruning_agent.prune(list(all_retrieved_triples), patient_context)
        print(f"  Pruned {len(all_retrieved_triples) - len(pruned_triples)} triples. Remaining: {len(pruned_triples)} triples.")

        # 4. Reasoning & Explanation
        print("\nPerforming reasoning and explanation...")
        diagnosis, treatment_plan, drug_interactions = self.reasoning_agent.reason(pruned_triples, patient_context)
        explanation = self.reasoning_agent.explain(diagnosis, treatment_plan, drug_interactions, pruned_triples)

        return diagnosis, treatment_plan, drug_interactions, explanation

# Main execution block
if __name__ == "__main__":
    system = MediReasonerSystem()

    print("Welcome to Medi-Reasoner: AI-Powered Clinical Decision Support System")
    print("Enter a patient query (e.g., 'diagnose patient x with fever and cough' or 'check medications for patient y'):")
    print("Type 'exit' to quit.")

    while True:
        user_query = input("\nYour query: ")
        if user_query.lower() == 'exit':
            break

        # Simulate extracting patient context from the query for the pruning agent
        patient_context = {"symptoms": [], "medications": []}
        if "patient x" in user_query.lower():
            patient_context["symptoms"].extend(["fever", "cough", "fatigue"])
        if "patient y" in user_query.lower():
            patient_context["symptoms"].extend(["fever"])
            patient_context["medications"].extend(["warfarin", "aspirin"])

        diagnosis, treatment, drug_interactions, explanation = system.process_patient_query(user_query, patient_context)

        print("\n--- Medi-Reasoner Output ---")
        print(f"Diagnosis: {diagnosis}")
        print(f"Treatment: {treatment}")
        print(f"Drug Interactions: {drug_interactions}")
        print(explanation)
        print("--------------------------")