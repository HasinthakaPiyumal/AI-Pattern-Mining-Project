import networkx as nx

class MedicalKnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_triple(self, head, relation, tail):
        self.graph.add_edge(head, tail, relation=relation)

    def get_neighbors(self, entity, relation=None):
        neighbors = []
        for neighbor in self.graph.neighbors(entity):
            if relation is None or self.graph[entity][neighbor]["relation"] == relation:
                neighbors.append((self.graph[entity][neighbor]["relation"], neighbor))
        return neighbors

    def get_facts_about_entity(self, entity):
        facts = []
        # Outgoing edges
        for u, v, data in self.graph.out_edges(entity, data=True):
            facts.append(f"({u} --{data["relation"]}--> {v})")
        # Incoming edges (for relationships where entity is the tail)
        for u, v, data in self.graph.in_edges(entity, data=True):
            facts.append(f"({u} --{data["relation"]}--> {v})")
        return facts

    def find_paths(self, start_entity, end_entity=None, max_depth=3):
        paths = []
        # Simple DFS for demonstration, can be extended for beam search
        if start_entity not in self.graph:
            return []

        queue = [(start_entity, [start_entity])]
        
        while queue:
            current_node, current_path = queue.pop(0)

            if len(current_path) -1 >= max_depth:
                continue

            if end_entity and current_node == end_entity:
                paths.append(current_path)
                continue # Continue searching for other paths
            
            for neighbor in self.graph.neighbors(current_node):
                if neighbor not in current_path: # Avoid cycles
                    relation = self.graph[current_node][neighbor]["relation"]
                    new_path_segment = (current_node, relation, neighbor)
                    new_path = current_path + [new_path_segment, neighbor]
                    queue.append((neighbor, new_path))
        
        # Convert raw networkx paths to a more readable triple-based format
        formatted_paths = []
        for path in paths:
            formatted_segment = []
            for i in range(len(path) - 1):
                if isinstance(path[i+1], tuple):
                    formatted_segment.append(f"({path[i]} --{path[i+1][1]}--> {path[i+1][2]})")
                else:
                    # This case should ideally not be hit with the new path construction logic
                    pass # Handle if path structure changes
            if formatted_segment: # Only add if there are actual triples
                formatted_paths.append(" -> ".join(formatted_segment))
        
        return formatted_paths

    def _get_path_triples(self, path_nodes):
        triples = []
        for i in range(len(path_nodes) - 1):
            u, v = path_nodes[i], path_nodes[i+1]
            if self.graph.has_edge(u, v):
                relation = self.graph[u][v]["relation"]
                triples.append(f"({u} --{relation}--> {v})")
        return triples


# Example Usage (for testing)
if __name__ == "__main__":
    kg = MedicalKnowledgeGraph()
    kg.add_triple("Patient_A", "has_symptom", "Fever")
    kg.add_triple("Fever", "indicates_condition", "Influenza")
    kg.add_triple("Influenza", "treated_by", "Antivirals")
    kg.add_triple("Patient_A", "has_medical_history", "Asthma")
    kg.add_triple("Asthma", "contraindicates_drug", "Beta-Blockers")
    kg.add_triple("Influenza", "causes", "Fatigue")
    kg.add_triple("Fatigue", "is_symptom_of", "Anemia")

    print("Facts about Influenza:", kg.get_facts_about_entity("Influenza"))
    print("Neighbors of Patient_A:", kg.get_neighbors("Patient_A"))
    print("Paths from Patient_A to Antivirals:", kg.find_paths("Patient_A", "Antivirals"))
    print("Paths from Fever (exploratory, no end_entity):")
    print(kg.find_paths("Fever", max_depth=2))
