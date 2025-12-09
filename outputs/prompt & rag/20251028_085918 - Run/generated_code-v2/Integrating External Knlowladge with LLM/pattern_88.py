import networkx as nx

class MedicalKnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_symptom_disease_relation(self, symptom, disease):
        self.graph.add_edge(symptom, disease, relation="causes")
        self.graph.add_edge(disease, symptom, relation="has_symptom")

    def add_disease_treatment_relation(self, disease, treatment):
        self.graph.add_edge(disease, treatment, relation="treated_by")
        self.graph.add_edge(treatment, disease, relation="treats")

    def add_fact(self, entity1, relation, entity2):
        self.graph.add_edge(entity1, entity2, relation=relation)

    def get_neighbors(self, entity, relation=None):
        if entity not in self.graph:
            return []
        neighbors = []
        for neighbor in self.graph.neighbors(entity):
            edge_data = self.graph.get_edge_data(entity, neighbor)
            if relation is None or edge_data.get("relation") == relation:
                neighbors.append((neighbor, edge_data.get("relation")))
        return neighbors

    def search_facts(self, query_entities, max_hops=2):
        relevant_facts = set()
        for entity in query_entities:
            if entity in self.graph:
                # Explore direct neighbors
                for neighbor, relation in self.get_neighbors(entity):
                    relevant_facts.add(f"{entity} {relation} {neighbor}")
                    # Explore one more hop from neighbors
                    if max_hops > 1:
                        for second_neighbor, second_relation in self.get_neighbors(neighbor):
                            if second_neighbor != entity: # Avoid trivial cycles
                                relevant_facts.add(f"{neighbor} {second_relation} {second_neighbor}")
        return list(relevant_facts)

# Example Usage (for demonstration)
if __name__ == "__main__":
    mkg = MedicalKnowledgeGraph()
    mkg.add_symptom_disease_relation("fever", "influenza")
    mkg.add_symptom_disease_relation("cough", "influenza")
    mkg.add_symptom_disease_relation("headache", "influenza")
    mkg.add_symptom_disease_relation("sore throat", "strep throat")
    mkg.add_disease_treatment_relation("influenza", "rest")
    mkg.add_disease_treatment_relation("influenza", "flu antiviral")
    mkg.add_disease_treatment_relation("strep throat", "antibiotics")

    print("Facts about influenza:")
    print(mkg.search_facts(["influenza"]))

    print("Facts related to fever and cough:")
    print(mkg.search_facts(["fever", "cough"]))
