import networkx as nx

# 1. Knowledge Graph (KG) Module
class MedicalKnowledgeGraph:
    def __init__(self):
        self.graph = nx.Graph()

    def load_triples(self, triples_data):
        """Loads a list of (subject, predicate, object) triples into the graph."""
        for s, p, o in triples_data:
            self.graph.add_edge(s, o, relation=p)
            # Ensure nodes exist even if they are only subjects or objects initially
            if s not in self.graph: self.graph.add_node(s)
            if o not in self.graph: self.graph.add_node(o)

    def get_neighbors(self, entity):
        """Returns a list of neighbors for a given entity."""
        if entity in self.graph:
            return list(self.graph.neighbors(entity))
        return []

    def find_paths(self, start_entity, end_entity=None, max_depth=2, relation_filter=None):
        """Finds simple paths from a start_entity up to a max_depth.
           If end_entity is provided, paths between them are found.
           relation_filter can be a list of relations to consider.
        """
        if start_entity not in self.graph:
            return []

        all_found_paths = []

        # BFS-like traversal to find paths up to max_depth
        queue = [(start_entity, [start_entity])]
        visited_paths = set()

        while queue:
            current_node, current_path = queue.pop(0)

            if len(current_path) - 1 > max_depth: # path length is number of nodes - 1
                continue

            path_tuple = tuple(current_path)
            if path_tuple in visited_paths:
                continue
            visited_paths.add(path_tuple)

            if current_node == end_entity and end_entity is not None and len(current_path) > 1:
                all_found_paths.append(current_path)
                continue # Found path to end_entity, continue exploring for other paths
            elif end_entity is None and len(current_path) > 1: # If no end_entity, collect all valid paths
                 all_found_paths.append(current_path)

            for neighbor in self.graph.neighbors(current_node):
                edge_data = self.graph.get_edge_data(current_node, neighbor)
                relation = edge_data.get("relation") if edge_data else None

                if relation_filter and relation not in relation_filter:
                    continue

                if neighbor not in current_path: # Avoid cycles in simple paths
                    new_path = current_path + [neighbor]
                    queue.append((neighbor, new_path))

        # If end_entity was specified, we only return paths that reach it
        if end_entity is not None:
            return [p for p in all_found_paths if p[-1] == end_entity]
        # Otherwise, we return all paths found within max_depth
        # Filter out single-node paths if they are not meaningful for 