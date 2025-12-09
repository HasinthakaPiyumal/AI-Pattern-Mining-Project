import networkx as nx
import random

class MedicalKnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self._populate_graph()

    def _populate_graph(self):
        # Nodes: Medical entities
        entities = [
            "Patient A", "Fever", "Cough", "Headache", "Fatigue", "Sore Throat",
            "COVID-19", "Influenza", "Common Cold", "Pneumonia",
            "Antibiotics", "Antivirals", "Rest", "Fluids", "Paracetamol",
            "High Blood Pressure", "Diabetes Type 2", "Obesity",
            "Insulin", "Metformin", "Diet Modification", "Exercise"
        ]
        self.graph.add_nodes_from(entities)

        # Edges: Relations between entities
        self.graph.add_edges_from([
            ("Patient A", "has_symptom", "Fever"),
            ("Patient A", "has_symptom", "Cough"),
            ("Patient A", "has_symptom", "Fatigue"),
            ("Fever", "is_symptom_of", "COVID-19"),
            ("Fever", "is_symptom_of", "Influenza"),
            ("Cough", "is_symptom_of", "COVID-19"),
            ("Cough", "is_symptom_of", "Influenza"),
            ("Headache", "is_symptom_of", "COVID-19"),
            ("Fatigue", "is_symptom_of", "COVID-19"),
            ("Sore Throat", "is_symptom_of", "Common Cold"),
            ("COVID-19", "treatable_by", "Antivirals"),
            ("Influenza", "treatable_by", "Antivirals"),
            ("Common Cold", "treatable_by", "Rest"),
            ("Common Cold", "treatable_by", "Fluids"),
            ("Common Cold", "treatable_by", "Paracetamol"),
            ("COVID-19", "leads_to_complication", "Pneumonia"),
            ("Diabetes Type 2", "treated_by", "Insulin"),
            ("Diabetes Type 2", "treated_by", "Metformin"),
            ("Diabetes Type 2", "managed_by", "Diet Modification"),
            ("Diabetes Type 2", "managed_by", "Exercise"),
            ("High Blood Pressure", "has_risk_factor", "Obesity"),
            ("Obesity", "is_risk_factor_for", "Diabetes Type 2"),
            ("Fever", "associated_with", "Headache"),
            ("Cough", "associated_with", "Sore Throat"),
        ])

    def get_outgoing_relations(self, entity):
        return list(self.graph.out_edges(entity, data='relation'))

    def get_entities(self):
        return list(self.graph.nodes())


class LLMSimulator:
    def extract_entities(self, query: str) -> list[str]:
        found_entities = []
        kg_entities = MedicalKnowledgeGraph().get_entities()
        for entity in kg_entities:
            if entity.lower() in query.lower():
                found_entities.append(entity)
        if not found_entities and "patient a" in query.lower():
            found_entities.append("Patient A")
        if not found_entities:
            if "fever" in query.lower() or "cough" in query.lower():
                found_entities.append("Fever")
        return list(set(found_entities))

    def score_relations(self, entity: str, relations: list[str], question: str) -> dict[str, float]:
        scores = {rel_type: 0.5 + random.random() * 0.5 for _, _, rel_type in relations}
        return scores

    def score_entities(self, relation: str, entities: list[str], question: str) -> dict[str, float]:
        scores = {entity: 0.5 + random.random() * 0.5 for entity in entities}
        return scores

    def evaluate_paths(self, question: str, paths: list[list[tuple[str, str, str]]]) -> tuple[bool, str]:
        relevant_diagnoses = ["COVID-19", "Influenza", "Common Cold", "Diabetes Type 2"]
        relevant_treatments = ["Antivirals", "Rest", "Insulin", "Metformin"]

        if not paths:
            return False, "No clear pathway found. Please provide more information."

        for path in paths:
            if not path:
                continue
            last_entity = path[-1][2]

            if "diagnose" in question.lower() and last_entity in relevant_diagnoses:
                explanation = f"Based on the symptoms and medical knowledge, a possible diagnosis is {last_entity}."
                return True, explanation
            if "treat" in question.lower() and last_entity in relevant_treatments:
                explanation = f"For {path[-2][2] if len(path) > 1 else 'the condition'}, {last_entity} is a relevant treatment option."
                return True, explanation
            if "pathway" in question.lower() and (last_entity in relevant_diagnoses or last_entity in relevant_treatments):
                 explanation = f"A potential clinical pathway leads to {last_entity}."
                 return True, explanation

        return False, "The current reasoning paths are not yet sufficient to provide a definitive answer. Further exploration may be needed."


class ThinkonGraphFramework:
    def __init__(self, kg: MedicalKnowledgeGraph, llm_simulator: LLMSimulator, beam_width: int = 3, max_depth: int = 3):
        self.kg = kg
        self.llm_simulator = llm_simulator
        self.beam_width = beam_width
        self.max_depth = max_depth

    def run(self, question: str) -> tuple[bool, str, list[list[tuple[str, str, str]]]]:
        initial_entities = self.llm_simulator.extract_entities(question)

        if not initial_entities:
            return False, "Could not identify initial topic entities from the question.", []

        # Initialize beam with starting entities (path, score)
        # A path is a list of (source, relation, target) tuples
        beam = [([(entity, "initial", entity)], 1.0) for entity in initial_entities]
        all_explored_paths = []

        for depth in range(self.max_depth):
            new_beam_candidates = []
            for current_path, current_score in beam:
                last_entity = current_path[-1][2]
                outgoing_edges = self.kg.get_outgoing_relations(last_entity)

                if not outgoing_edges:
                    new_beam_candidates.append((current_path, current_score))
                    continue

                scored_relations = self.llm_simulator.score_relations(last_entity, outgoing_edges, question)

                for source, target, relation_type in outgoing_edges:
                    relation_score = scored_relations.get(relation_type, 0.5) # Default score if not in simulator
                    
                    # Simulate scoring entities connected by this relation
                    # In a real LLM, this would be more nuanced, here we just use the target
                    entity_score = self.llm_simulator.score_entities(relation_type, [target], question).get(target, 0.5)

                    new_path = current_path + [(source, relation_type, target)]
                    new_score = current_score + relation_score + entity_score
                    new_beam_candidates.append((new_path, new_score))

            # Prune step: Select top-N paths for the next iteration
            beam = sorted(new_beam_candidates, key=lambda x: x[1], reverse=True)[:self.beam_width]
            
            for path, _ in beam:
                all_explored_paths.append(path)
            
            # Check if current paths are sufficient
            is_sufficient, answer = self.llm_simulator.evaluate_paths(question, [p for p, _ in beam])
            if is_sufficient:
                return True, answer, [p for p, _ in beam]

        # Final reasoning after max depth reached
        final_is_sufficient, final_answer = self.llm_simulator.evaluate_paths(question, [p for p, _ in beam])
        return final_is_sufficient, final_answer, [p for p, _ in beam]


if __name__ == "__main__":
    medical_kg = MedicalKnowledgeGraph()
    llm_sim = LLMSimulator()
    tog_framework = ThinkonGraphFramework(medical_kg, llm_sim, beam_width=2, max_depth=3)

    print("Clinical Pathway Navigator (ToG Algorithmic Framework Prototype)")
    print("Enter your medical query (e.g., 'diagnose Patient A with fever and cough', 'how to treat Common Cold', 'pathway for Diabetes Type 2'):")
    print("Type 'exit' to quit.")

    while True:
        user_query = input("\nQuery: ")
        if user_query.lower() == 'exit':
            break

        print(f"Processing query: '{user_query}'...")
        sufficient, response_text, reasoning_paths = tog_framework.run(user_query)

        print(f"\nSufficiency: {sufficient}")
        print(f"Response: {response_text}")

        if reasoning_paths:
            print("\nReasoning Paths Found:")
            for i, path in enumerate(reasoning_paths):
                print(f"  Path {i+1}:")
                for step in path:
                    print(f"    {step[0]} --({step[1]})--> {step[2]}")
        else:
            print("\nNo explicit reasoning paths found.")
