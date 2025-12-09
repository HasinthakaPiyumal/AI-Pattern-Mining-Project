import networkx as nx
import random
from typing import List, Tuple, Dict, Any
from tqdm import tqdm

class MedicalKnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_entity(self, entity_id: str, entity_type: str, attributes: Dict[str, Any] = None):
        if attributes is None:
            attributes = {}
        attributes["entity_type"] = entity_type
        self.graph.add_node(entity_id, **attributes)

    def add_relation(self, source_entity_id: str, target_entity_id: str, relation_type: str, attributes: Dict[str, Any] = None):
        if attributes is None:
            attributes = {}
        attributes["relation_type"] = relation_type
        self.graph.add_edge(source_entity_id, target_entity_id, **attributes)

    def get_neighbors(self, entity_id: str) -> List[Tuple[str, str, str]]:
        neighbors = []
        for target_node in self.graph.neighbors(entity_id):
            edge_data = self.graph.get_edge_data(entity_id, target_node)
            relation_type = edge_data.get("relation_type", "unknown")
            target_entity_type = self.graph.nodes[target_node].get("entity_type", "unknown")
            neighbors.append((relation_type, target_node, target_entity_type))
        return neighbors

class Path:
    def __init__(self, steps: List[Tuple[str, str]] = None, current_score: float = 0.0):
        self.steps = steps if steps is not None else []
        self.current_score = current_score

    def get_last_entity(self) -> str:
        if not self.steps:
            return None
        return self.steps[-1][1]

    def add_step(self, relation_type: str, entity_id: str) -> "Path":
        new_steps = list(self.steps)
        new_steps.append((relation_type, entity_id))
        return Path(new_steps, self.current_score)

    def get_path_description(self) -> str:
        description_parts = []
        for i, (relation, entity) in enumerate(self.steps):
            if i == 0:
                description_parts.append(f"Start: {entity}")
            else:
                description_parts.append(f" -[{relation}]-> {entity}")
        return "".join(description_parts)

    def __repr__(self):
        return f"Path(score={self.current_score:.2f}, steps={' -> '.join([s[1] for s in self.steps])})"

    def __lt__(self, other):
        return self.current_score < other.current_score

class LLMPruner:
    def __init__(self):
        pass

    def prune_candidates(self, patient_case: str, current_path_context: str, candidate_paths: List[Path], top_n: int) -> List[Path]:
        scored_paths = []
        for path in candidate_paths:
            path_description = path.get_path_description()
            
            # Mock LLM scoring logic:
            # Assign a higher score if entities in the path are related to the patient case keywords
            score = random.uniform(0.1, 0.5) # Base random score
            
            # Simple keyword matching for demonstration
            for keyword in patient_case.lower().split():
                if keyword in path_description.lower():
                    score += 0.4 # Boost score for keyword match
            
            # Further boost if the path context is very relevant (mocked)
            if "rare disease" in current_path_context.lower() and "gene" in path_description.lower():
                score += 0.3
                
            path.current_score = min(score, 1.0) # Cap score at 1.0
            scored_paths.append(path)

        # Sort by score in descending order and select top_n
        scored_paths.sort(key=lambda p: p.current_score, reverse=True)
        return scored_paths[:top_n]

class LLMGuidanceBeamSearch:
    def __init__(self, knowledge_graph: MedicalKnowledgeGraph, llm_pruner: LLMPruner, beam_width: int, max_depth: int):
        self.knowledge_graph = knowledge_graph
        self.llm_pruner = llm_pruner
        self.beam_width = beam_width
        self.max_depth = max_depth

    def search(self, start_entity_id: str, patient_case: str) -> List[Path]:
        if start_entity_id not in self.knowledge_graph.graph:
            print(f"Error: Start entity '{start_entity_id}' not found in the knowledge graph.")
            return []
        
        initial_path = Path(steps=[("starts_at", start_entity_id)])
        beam: List[Path] = [initial_path]

        print(f"Starting beam search for '{patient_case}' from '{start_entity_id}'...")

        for depth in tqdm(range(self.max_depth), desc="Beam Search Depth"):
            new_candidates: List[Path] = []
            current_beam_descriptions = [p.get_path_description() for p in beam]
            context_for_llm = "\n".join(current_beam_descriptions) if current_beam_descriptions else ""
            
            for path in beam:
                last_entity_id = path.get_last_entity()
                if last_entity_id is None: # Should not happen after initial path, but good for safety
                    continue

                neighbors = self.knowledge_graph.get_neighbors(last_entity_id)
                for relation_type, neighbor_entity_id, _ in neighbors:
                    new_path = path.add_step(relation_type, neighbor_entity_id)
                    new_candidates.append(new_path)
            
            if not new_candidates:
                print(f"No new candidates found at depth {depth+1}. Stopping search.")
                break

            # Prune candidates using the LLM Pruner
            beam = self.llm_pruner.prune_candidates(patient_case, context_for_llm, new_candidates, self.beam_width)
            
            if not beam:
                print(f"Beam became empty after pruning at depth {depth+1}. Stopping search.")
                break

        print("Beam search completed.")
        return beam


if __name__ == "__main__":
    # 1. Initialize Medical Knowledge Graph
    mkg = MedicalKnowledgeGraph()

    # Add entities (symptoms, genes, diseases, drugs)
    mkg.add_entity("Patient_A", "Patient")
    mkg.add_entity("Fatigue", "Symptom")
    mkg.add_entity("Muscle_Weakness", "Symptom")
    mkg.add_entity("Gene_X", "Gene")
    mkg.add_entity("Gene_Y", "Gene")
    mkg.add_entity("Disease_R", "Disease", {"rarity": "rare"})
    mkg.add_entity("Disease_C", "Disease", {"rarity": "common"})
    mkg.add_entity("Drug_Z", "Drug")
    mkg.add_entity("Drug_A", "Drug")
    mkg.add_entity("Protein_P", "Protein")
    mkg.add_entity("Mutation_M", "GeneticMutation")

    # Add relations
    mkg.add_relation("Patient_A", "Fatigue", "has_symptom")
    mkg.add_relation("Patient_A", "Muscle_Weakness", "has_symptom")
    
    mkg.add_relation("Fatigue", "Disease_R", "associated_with")
    mkg.add_relation("Muscle_Weakness", "Disease_R", "associated_with")
    mkg.add_relation("Gene_X", "Disease_R", "causes")
    mkg.add_relation("Mutation_M", "Gene_X", "occurs_in")
    mkg.add_relation("Disease_R", "Protein_P", "affects")
    mkg.add_relation("Drug_Z", "Disease_R", "treats")

    mkg.add_relation("Gene_Y", "Disease_C", "causes")
    mkg.add_relation("Fatigue", "Disease_C", "associated_with") # Common disease also has fatigue
    mkg.add_relation("Drug_A", "Disease_C", "treats")

    # 2. Initialize LLM Pruner (Mocked)
    llm_pruner = LLMPruner()

    # 3. Initialize LLM-Guided Beam Search
    beam_width = 3
    max_depth = 4
    beam_search = LLMGuidanceBeamSearch(mkg, llm_pruner, beam_width, max_depth)

    # 4. Define Patient Case and Start Entity
    patient_case = "Patient presents with chronic fatigue and muscle weakness. Suspecting a rare genetic disorder related to Gene X."
    start_entity = "Patient_A"

    # 5. Run the Beam Search
    results = beam_search.search(start_entity, patient_case)

    # 6. Print Results
    print("\n--- Top Reasoning Paths Found ---")
    if results:
        for i, path in enumerate(results):
            print(f"Path {i+1} (Score: {path.current_score:.2f}): {path.get_path_description()}")
    else:
        print("No relevant paths found.")

    # Example with a different patient case
    print("\n--- Another Search: Common Disease related to Gene Y ---")
    patient_case_2 = "Patient has fatigue and fever, possibly a common viral infection or a disorder related to Gene Y."
    start_entity_2 = "Patient_A"
    results_2 = beam_search.search(start_entity_2, patient_case_2)

    print("\n--- Top Reasoning Paths Found for second case ---")
    if results_2:
        for i, path in enumerate(results_2):
            print(f"Path {i+1} (Score: {path.current_score:.2f}): {path.get_path_description()}")
    else:
        print("No relevant paths found.")
