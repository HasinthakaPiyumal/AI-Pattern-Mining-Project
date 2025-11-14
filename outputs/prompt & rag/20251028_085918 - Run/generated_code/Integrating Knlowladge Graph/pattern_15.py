class MedicalKnowledgeGraph:
    def __init__(self):
        self.entities = {}
        self.relationships = {}

    def add_entity(self, name, entity_type):
        if name not in self.entities:
            self.entities[name] = {"type": entity_type}
            self.relationships[name] = {}
        else:
            print(f"Warning: Entity '{name}' already exists.")

    def add_relationship(self, entity1_name, relationship_type, entity2_name):
        if entity1_name not in self.entities:
            self.add_entity(entity1_name, "unknown") # Auto-add if not present
        if entity2_name not in self.entities:
            self.add_entity(entity2_name, "unknown") # Auto-add if not present

        if relationship_type not in self.relationships[entity1_name]:
            self.relationships[entity1_name][relationship_type] = []
        if entity2_name not in self.relationships[entity1_name][relationship_type]:
            self.relationships[entity1_name][relationship_type].append(entity2_name)
        
        # For bidirectional relationships (e.g., 'causes' implies 'is_caused_by'),
        # you might add the inverse here. For simplicity, we'll keep it unidirectional for now.
        # Example: if relationship_type == "causes":
        #     self.add_relationship(entity2_name, "is_caused_by", entity1_name)

    def get_neighbors(self, entity_name, relationship_type=None):
        if entity_name not in self.entities:
            return []
        
        results = []
        if relationship_type:
            if relationship_type in self.relationships[entity_name]:
                results.extend([(entity_name, relationship_type, target) for target in self.relationships[entity_name][relationship_type]])
        else:
            for rel_type, targets in self.relationships[entity_name].items():
                results.extend([(entity_name, rel_type, target) for target in targets])
        return results

    def get_triples_around_entity(self, entity_name, depth=1):
        if entity_name not in self.entities:
            return []
        
        visited_triples = set()
        queue = [(entity_name, 0)] # (entity, current_depth)
        all_triples = []

        while queue:
            current_entity, current_depth = queue.pop(0)

            if current_depth >= depth and current_entity != entity_name: # Only explore further if within depth, but always process start entity
                continue

            for rel_type, targets in self.relationships.get(current_entity, {}).items():
                for target in targets:
                    triple = (current_entity, rel_type, target)
                    if triple not in visited_triples:
                        all_triples.append(triple)
                        visited_triples.add(triple)
                        if target in self.entities and current_depth + 1 < depth:
                            queue.append((target, current_depth + 1))
            
            # Also consider relationships where current_entity is the target
            for source_entity, rel_map in self.relationships.items():
                for rel_type, targets in rel_map.items():
                    if current_entity in targets:
                        triple = (source_entity, rel_type, current_entity)
                        if triple not in visited_triples:
                            all_triples.append(triple)
                            visited_triples.add(triple)
                            if source_entity in self.entities and current_depth + 1 < depth:
                                queue.append((source_entity, current_depth + 1))
        
        # Filter to only include unique triples involving the original entity or its direct connections
        return sorted(list(set(all_triples)))

    def find_paths(self, start_entity, end_entity, max_depth=3, current_path=None, all_paths=None):
        if current_path is None:
            current_path = [(start_entity, None, None)] # (entity, relation_to_prev, prev_entity)
        if all_paths is None:
            all_paths = []

        if start_entity == end_entity:
            all_paths.append(current_path)
            return

        if len(current_path) -1 >= max_depth: # current_path length includes start, so -1 for edges
            return
        
        visited_entities = {node[0] for node in current_path}

        for rel_type, targets in self.relationships.get(start_entity, {}).items():
            for next_entity in targets:
                if next_entity not in visited_entities:
                    new_path = current_path + [(next_entity, rel_type, start_entity)]
                    self.find_paths(next_entity, end_entity, max_depth, new_path, all_paths)
        
        return all_paths

    def get_all_entities(self):
        return list(self.entities.keys())

    def get_entity_type(self, entity_name):
        return self.entities.get(entity_name, {}).get("type")
