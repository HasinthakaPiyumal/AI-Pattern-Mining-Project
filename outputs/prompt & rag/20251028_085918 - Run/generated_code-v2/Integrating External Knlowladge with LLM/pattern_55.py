import networkx as nx
import random

class Config:
    RELATION_PRUNING_THRESHOLD = 0.6
    RANDOM_ENTITY_SAMPLE_SIZE = 2
    MAX_CHAIN_LENGTH = 3

class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()

    def load_kg_from_data(self, data):
        for s, r, t, attrs in data:
            self.graph.add_edge(s, t, relation=r, **attrs)

    def get_outgoing_edges(self, node):
        return list(self.graph.out_edges(node, data=True))

class RelationChainDiscovery:
    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg

    def find_relation_chains(self, start_entities, max_length=Config.MAX_CHAIN_LENGTH, entity_sample_size=Config.RANDOM_ENTITY_SAMPLE_SIZE):
        all_chains = []
        queue = []

        for entity in start_entities:
            queue.append((entity, [], 0))

        while queue:
            current_entity, current_path, current_depth = queue.pop(0)

            if current_depth >= max_length:
                continue

            outgoing_edges = self.kg.get_outgoing_edges(current_entity)

            edges_to_process = []
            if len(outgoing_edges) > entity_sample_size:
                relations_to_targets = {}
                for u, v, data in outgoing_edges:
                    rel = data.get("relation")
                    if rel not in relations_to_targets:
                        relations_to_targets[rel] = []
                    relations_to_targets[rel].append((v, data))

                sampled_edges = []
                for rel, targets_with_data in relations_to_targets.items():
                    if len(targets_with_data) > 1:
                        sampled_targets = random.sample(targets_with_data, min(len(targets_with_data), entity_sample_size // len(relations_to_targets) + 1))
                    else:
                        sampled_targets = targets_with_data
                    for target_entity, data in sampled_targets:
                        sampled_edges.append((current_entity, target_entity, data))
                
                edges_to_process = sampled_edges
            else:
                edges_to_process = outgoing_edges

            for _, neighbor, edge_data in edges_to_process:
                relation = edge_data.get("relation")
                if relation:
                    new_path_segment = (current_entity, relation, neighbor)
                    new_chain = current_path + [new_path_segment]
                    all_chains.append(new_chain)
                    
                    if current_depth + 1 < max_length:
                        queue.append((neighbor, new_chain, current_depth + 1))
        
        unique_chains = []
        seen_chains = set()
        for chain in all_chains:
            chain_tuple = tuple(tuple(segment) for segment in chain)
            if chain_tuple not in seen_chains:
                unique_chains.append(chain)
                seen_chains.add(chain_tuple)

        return unique_chains

class LLMBasedRelationPruner:
    def __init__(self, threshold=Config.RELATION_PRUNING_THRESHOLD):
        self.threshold = threshold

    def prune_relations(self, relation_chains, patient_context):
        pruned_chains = []
        for chain in relation_chains:
            relevance_score = random.uniform(0.3, 0.9)

            if len(chain) == 1 and chain[0][1] == "associated_with":
                relevance_score *= 0.8
            
            if any(seg[1] in ["indicative_of", "causes"] for seg in chain):
                relevance_score *= 1.1

            if relevance_score > self.threshold:
                pruned_chains.append(chain)
        return pruned_chains

class RandomEntityPruner:
    def __init__(self, sample_size=Config.RANDOM_ENTITY_SAMPLE_SIZE):
        self.sample_size = sample_size

    def prune_entities(self, relation_chains):
        if not relation_chains:
            return []

        grouped_chains = {}
        for chain in relation_chains:
            if chain:
                start_entity = chain[0][0]
                end_entity = chain[-1][2]
                key = (start_entity, end_entity)
                if key not in grouped_chains:
                    grouped_chains[key] = []
                grouped_chains[key].append(chain)

        pruned_chains = []
        for key, chains_list in grouped_chains.items():
            if len(chains_list) > self.sample_size:
                pruned_chains.extend(random.sample(chains_list, self.sample_size))
            else:
                pruned_chains.extend(chains_list)
        return pruned_chains

class PatientInputHandler:
    def get_patient_data(self):
        return {
            "symptoms": ["fatigue", "muscle weakness"],
            "genetic_markers": ["mutation_gene_x"]
        }

class ResultPresenter:
    def present_diagnosis(self, ranked_diseases_with_chains):
        print("\n--- Potential Rare Disease Diagnoses ---")
        if not ranked_diseases_with_chains:
            print("No significant diagnoses found based on available data.")
            return

        for i, (disease, supporting_chains) in enumerate(ranked_diseases_with_chains):
            print(f"\n{i+1}. Disease: {disease}")
            print("   Supporting Relation Chains:")
            for chain in supporting_chains:
                path_str = " -> ".join([f"{e1} -[{r}]-> {e2}" for e1, r, e2 in chain])
                print(f"     - {path_str}")

class DiagnosticOrchestrator:
    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg
        self.relation_discovery = RelationChainDiscovery(kg)
        self.llm_pruner = LLMBasedRelationPruner()
        self.entity_pruner = RandomEntityPruner()
        self.result_presenter = ResultPresenter()

    def diagnose(self, patient_data):
        initial_entities = patient_data["symptoms"] + patient_data["genetic_markers"]
        
        print(f"Starting diagnosis for entities: {initial_entities}")

        discovered_chains = self.relation_discovery.find_relation_chains(
            initial_entities, 
            max_length=Config.MAX_CHAIN_LENGTH,
            entity_sample_size=Config.RANDOM_ENTITY_SAMPLE_SIZE
        )
        print(f"Discovered {len(discovered_chains)} initial relation chains (with random entity sampling during discovery).")

        pruned_by_llm = self.llm_pruner.prune_relations(discovered_chains, patient_data)
        print(f"After LLM-based relation pruning: {len(pruned_by_llm)} chains remaining.")

        final_chains = self.entity_pruner.prune_entities(pruned_by_llm)
        print(f"After secondary random entity (chain) pruning: {len(final_chains)} chains remaining.")

        disease_scores = {}
        disease_supporting_chains = {}

        for chain in final_chains:
            if chain:
                final_entity = chain[-1][2]
                
                if "disease" in final_entity.lower(): 
                    disease = final_entity
                    disease_scores[disease] = disease_scores.get(disease, 0) + 1
                    if disease not in disease_supporting_chains:
                        disease_supporting_chains[disease] = []
                    disease_supporting_chains[disease].append(chain)

        ranked_diseases = sorted(
            disease_scores.items(), key=lambda item: item[1], reverse=True
        )

        ranked_output = []
        for disease, score in ranked_diseases:
            ranked_output.append((disease, disease_supporting_chains[disease]))

        self.result_presenter.present_diagnosis(ranked_output)

if __name__ == "__main__":
    kg = KnowledgeGraph()
    sample_kg_data = [
        ("fatigue", "associated_with", "anemia", {}),
        ("fatigue", "associated_with", "thyroid_disorder", {}),
        ("fatigue", "manifests_as", "chronic_fatigue_syndrome", {}),
        ("muscle weakness", "associated_with", "myopathy", {}),
        ("muscle weakness", "indicative_of", "neuromuscular_disease", {}),
        ("mutation_gene_x", "linked_to", "protein_alpha", {}),
        ("mutation_gene_x", "is_a_risk_factor_for", "disease_y", {}),
        ("protein_alpha", "causes", "cell_dysfunction_beta", {}),
        ("cell_dysfunction_beta", "indicative_of", "disease_y", {}),
        ("thyroid_disorder", "manifests_as", "disease_z", {}),
        ("myopathy", "indicative_of", "disease_w", {}),
        ("neuromuscular_disease", "can_lead_to", "disease_v", {}),
        ("anemia", "complication_of", "celiac_disease", {}),
        ("celiac_disease", "is_a_rare_disease", "True", {}),
        ("disease_y", "is_a_rare_disease", "True", {}),
        ("disease_z", "is_a_rare_disease", "False", {}),
        ("disease_w", "is_a_rare_disease", "True", {}),
        ("disease_v", "is_a_rare_disease", "False", {}),
    ]
    kg.load_kg_from_data(sample_kg_data)
    print("Knowledge Graph loaded.")

    patient_handler = PatientInputHandler()
    patient_data = patient_handler.get_patient_data()

    orchestrator = DiagnosticOrchestrator(kg)
    orchestrator.diagnose(patient_data)