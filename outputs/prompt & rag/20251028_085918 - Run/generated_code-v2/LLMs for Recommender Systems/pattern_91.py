import networkx as nx
import json

class TravelKnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_entity(self, entity_id, entity_type, attributes=None):
        if not self.graph.has_node(entity_id):
            self.graph.add_node(entity_id, type=entity_type, **(attributes if attributes else {}))
            return True
        return False

    def add_relation(self, source_id, target_id, relation_type, attributes=None):
        if self.graph.has_node(source_id) and self.graph.has_node(target_id):
            self.graph.add_edge(source_id, target_id, type=relation_type, **(attributes if attributes else {}))
            return True
        return False

    def get_related_entities(self, entity_id, relation_type=None):
        if not self.graph.has_node(entity_id):
            return []
        
        related = []
        for neighbor in self.graph.neighbors(entity_id):
            edge_data = self.graph.get_edge_data(entity_id, neighbor)
            if relation_type is None or edge_data.get('type') == relation_type:
                related.append(neighbor)
        return related

    def get_neighbors_with_relations(self, entity_id):
        if not self.graph.has_node(entity_id):
            return []
        
        neighbors = []
        for neighbor in self.graph.neighbors(entity_id):
            edge_data = self.graph.get_edge_data(entity_id, neighbor)
            neighbors.append((neighbor, edge_data.get('type')))
        return neighbors

    def visualize(self):
        # In a real application, this would use a visualization library like matplotlib or pyvis
        print("--- Knowledge Graph Structure ---")
        print(f"Nodes ({len(self.graph.nodes)}):")
        for node, data in self.graph.nodes(data=True):
            print(f"  - {node} (Type: {data.get('type')}, Attributes: {json.dumps({k: v for k, v in data.items() if k != 'type'})})")
        print(f"Edges ({len(self.graph.edges)}):")
        for u, v, data in self.graph.edges(data=True):
            print(f"  - {u} -> {v} (Type: {data.get('type')})")
        print("---------------------------------")


class LLMKnowledgeExtractor:
    def __init__(self, mock_llm_responses=None):
        # In a real application, this would initialize a LangChain or Transformers model
        self.mock_llm_responses = mock_llm_responses if mock_llm_responses is not None else {}

    def _simulate_llm_call(self, prompt):
        # Simulate LLM output based on prompt content or predefined mock responses
        for key, response in self.mock_llm_responses.items():
            if key in prompt:
                return response
        
        if "extract entities and relations" in prompt.lower():
            return "ENTITIES: [('Eiffel Tower', 'Landmark'), ('Paris', 'City'), ('French cuisine', 'Cuisine')]. RELATIONS: [('Eiffel Tower', 'located_in', 'Paris'), ('Paris', 'known_for', 'French cuisine')]."
        elif "infer missing facts" in prompt.lower():
            return "INFERRED_FACTS: [('French cuisine', 'popular_in', 'France')]."
        return ""

    def extract_knowledge(self, text):
        prompt = f"Given the following text, extract entities and relations: {text}"
        response = self._simulate_llm_call(prompt)
        
        entities = []
        relations = []
        
        if "ENTITIES:" in response:
            try:
                entity_str = response.split("ENTITIES:")[1].split("RELATIONS:")[0].strip()
                # A very simple parsing, actual LLM output would need more robust parsing
                # This expects a format like [('entity', 'type'), ...]
                entities_list = eval(entity_str) if entity_str else []
                for ent_name, ent_type in entities_list:
                    entities.append({'id': ent_name, 'type': ent_type})
            except Exception as e:
                print(f"Error parsing entities: {e}")

        if "RELATIONS:" in response:
            try:
                relation_str = response.split("RELATIONS:")[1].strip()
                # This expects a format like [('source', 'relation', 'target'), ...]
                relations_list = eval(relation_str) if relation_str else []
                for src, rel, tgt in relations_list:
                    relations.append({'source': src, 'target': tgt, 'type': rel})
            except Exception as e:
                print(f"Error parsing relations: {e}")

        return entities, relations

    def infer_facts(self, existing_facts_context):
        prompt = f"Given these existing facts: {existing_facts_context}, infer any missing facts."
        response = self._simulate_llm_call(prompt)
        inferred_facts = []
        if "INFERRED_FACTS:" in response:
            try:
                fact_str = response.split("INFERRED_FACTS:")[1].strip()
                inferred_facts_list = eval(fact_str) if fact_str else []
                for src, rel, tgt in inferred_facts_list:
                    inferred_facts.append({'source': src, 'target': tgt, 'type': rel})
            except Exception as e:
                print(f"Error parsing inferred facts: {e}")
        return inferred_facts


class TravelRecommendationSystem:
    def __init__(self, knowledge_graph, llm_extractor):
        self.kg = knowledge_graph
        self.llm_extractor = llm_extractor
        self.user_profiles = {}

    def ingest_text_data(self, text_id, text_content):
        print(f"\n--- Ingesting text data for '{text_id}' ---")
        entities, relations = self.llm_extractor.extract_knowledge(text_content)
        
        for entity in entities:
            self.kg.add_entity(entity['id'], entity['type'])
            print(f"  Added entity: {entity['id']} ({entity['type']})")
        for relation in relations:
            self.kg.add_relation(relation['source'], relation['target'], relation['type'])
            print(f"  Added relation: {relation['source']} --{relation['type']}--> {relation['target']}")

    def enrich_knowledge_graph(self):
        print("\n--- Enriching Knowledge Graph with LLM Inference ---")
        current_facts = [(u, v, self.kg.graph.get_edge_data(u, v)['type']) for u, v in self.kg.graph.edges]
        inferred_facts = self.llm_extractor.infer_facts(current_facts)
        for fact in inferred_facts:
            self.kg.add_entity(fact['source'], 'Concept') # Assume 'Concept' type if not specified
            self.kg.add_entity(fact['target'], 'Concept') # Assume 'Concept' type if not specified
            if self.kg.add_relation(fact['source'], fact['target'], fact['type']):
                print(f"  Inferred and added fact: {fact['source']} --{fact['type']}--> {fact['target']}")

    def add_user_preference(self, user_id, preferences):
        self.user_profiles[user_id] = preferences
        print(f"\n--- User '{user_id}' preferences added: {preferences} ---")
        # Add user preferences as entities and relations in the KG if they are not already there
        for pref in preferences:
            if not self.kg.graph.has_node(pref):
                self.kg.add_entity(pref, 'Preference')
            self.kg.add_relation(user_id, pref, 'prefers')

    def recommend(self, user_id, num_recommendations=3):
        print(f"\n--- Generating recommendations for user '{user_id}' ---")
        if user_id not in self.user_profiles:
            print("  User profile not found.")
            return []

        user_prefs = self.user_profiles[user_id]
        potential_recommendations = set()

        for pref in user_prefs:
            # Get entities directly related to preferences
            direct_relations = self.kg.get_related_entities(pref)
            potential_recommendations.update(direct_relations)
            
            # Explore further via relations (e.g., if user likes 'history', recommend cities 'known_for' history)
            for neighbor, rel_type in self.kg.get_neighbors_with_relations(pref):
                if rel_type == 'known_for': # Example: 'history buff' is 'known_for' 'ancient ruins'
                     potential_recommendations.add(neighbor)

            # Get entities related to user directly
            user_related = self.kg.get_related_entities(user_id, 'prefers')
            for ur in user_related:
                potential_recommendations.update(self.kg.get_related_entities(ur))

        # Filter out preferences themselves and ensure they are actual places/activities
        filtered_recommendations = [rec for rec in list(potential_recommendations) if 
                                    self.kg.graph.has_node(rec) and 
                                    self.kg.graph.nodes[rec].get('type') not in ['Preference', 'Concept', 'User'] and
                                    rec not in user_prefs]
        
        # Simple ranking (e.g., take the first N unique ones)
        final_recommendations = list(set(filtered_recommendations))[:num_recommendations]
        
        print(f"  Recommendations: {final_recommendations}")
        return final_recommendations

# --- Example Usage ---
if __name__ == "__main__":
    kg = TravelKnowledgeGraph()
    
    # Mock LLM responses for demonstration
    mock_llm_responses = {
        "travel blog about Paris": "ENTITIES: [('Eiffel Tower', 'Landmark'), ('Paris', 'City'), ('French cuisine', 'Cuisine'), ('Louvre Museum', 'Museum')]. RELATIONS: [('Eiffel Tower', 'located_in', 'Paris'), ('Louvre Museum', 'located_in', 'Paris'), ('Paris', 'known_for', 'French cuisine')].",
        "review about Rome": "ENTITIES: [('Colosseum', 'Historic Site'), ('Rome', 'City'), ('Italian pasta', 'Cuisine')]. RELATIONS: [('Colosseum', 'located_in', 'Rome'), ('Rome', 'known_for', 'Italian pasta')].",
        "infer any missing facts": "INFERRED_FACTS: [('French cuisine', 'popular_in', 'France'), ('Italian pasta', 'popular_in', 'Italy'), ('Eiffel Tower', 'has_attribute', 'iconic_landmark')]." # Example inference
    }
    llm_extractor = LLMKnowledgeExtractor(mock_llm_responses=mock_llm_responses)
    recommender = TravelRecommendationSystem(kg, llm_extractor)

    # Step 1: Ingest various text sources
    recommender.ingest_text_data("paris_blog", "A wonderful travel blog about Paris, highlighting the Eiffel Tower and the amazing French cuisine, as well as the Louvre Museum.")
    recommender.ingest_text_data("rome_review", "An excellent review of Rome, focusing on the historical Colosseum and delicious Italian pasta.")

    # Step 2: Enrich the Knowledge Graph using LLM inference
    recommender.enrich_knowledge_graph()

    # Step 3: Add user profiles and preferences
    recommender.add_user_preference("user1", ["Paris", "history buff", "French cuisine"])
    recommender.add_user_preference("user2", ["Rome", "foodie", "historic sites"])

    # Add user nodes to KG (explicitly, as they are sources of preferences)
    kg.add_entity("user1", 'User')
    kg.add_entity("user2", 'User')

    # Visualize the (simplified) Knowledge Graph
    kg.visualize()

    # Step 4: Generate recommendations
    recommender.recommend("user1")
    recommender.recommend("user2")

    # Example of a cross-domain recommendation if the KG allowed for it (e.g., 'history buff' -> 'ancient ruins' -> 'restaurant near ancient ruins')
    # For this simplified example, we'll just demonstrate current capabilities.
    recommender.add_user_preference("user3", ["Eiffel Tower", "iconic_landmark"])
    kg.add_entity("user3", 'User')
    recommender.recommend("user3")
