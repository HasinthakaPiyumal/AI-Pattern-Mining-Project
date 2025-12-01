import networkx as nx

class KnowledgeGraphBuilder:
    def __init__(self):
        self.kg = nx.DiGraph()

    def _simulate_llm_extraction(self, text):
        """
        Simulates LLM's entity and relation extraction from text.
        In a real scenario, this would involve calling a fine-tuned LLM
        with a carefully crafted prompt.
        """
        entities = set()
        relations = []

        # Simple keyword-based simulation for demonstration
        if "laptop" in text.lower():
            entities.add("Laptop")
            entities.add("Electronics")
            if "gaming" in text.lower():
                entities.add("Gaming Laptop")
                relations.append(("Gaming Laptop", "is_a", "Laptop"))
                relations.append(("Laptop", "has_feature", "High Performance"))
            if "lightweight" in text.lower():
                entities.add("Ultraportable Laptop")
                relations.append(("Ultraportable Laptop", "is_a", "Laptop"))
                relations.append(("Laptop", "has_feature", "Lightweight"))
            if "intel i7" in text.lower():
                entities.add("Intel i7 Processor")
                relations.append(("Laptop", "has_processor", "Intel i7 Processor"))
            if "apple" in text.lower() or "macbook" in text.lower():
                entities.add("Apple")
                entities.add("MacBook")
                relations.append(("MacBook", "is_brand", "Apple"))

        if "smartphone" in text.lower():
            entities.add("Smartphone")
            entities.add("Electronics")
            if "android" in text.lower():
                entities.add("Android Phone")
                relations.append(("Android Phone", "runs_os", "Android"))
            if "ios" in text.lower() or "iphone" in text.lower():
                entities.add("iOS Phone")
                entities.add("iPhone")
                relations.append(("iPhone", "runs_os", "iOS"))
            if "camera" in text.lower():
                relations.append(("Smartphone", "has_feature", "High Resolution Camera"))

        if "camera" in text.lower():
            entities.add("Camera")
            if "dslr" in text.lower():
                entities.add("DSLR Camera")
                relations.append(("DSLR Camera", "is_a", "Camera"))
                relations.append(("Camera", "has_feature", "Interchangeable Lenses"))

        # Example for coreference resolution (simplified)
        text_lower = text.lower()
        if "this product" in text_lower or "the item" in text_lower:
            # In a real LLM, it would resolve "this product" to the actual product entity
            pass # Placeholder for actual coref resolution logic

        return list(entities), relations

    def add_product_data(self, product_id, description, reviews):
        self.kg.add_node(product_id, type="product", description=description)

        # Process description with LLM simulation
        desc_entities, desc_relations = self._simulate_llm_extraction(description)
        for entity in desc_entities:
            self.kg.add_node(entity, type="attribute")
            self.kg.add_edge(product_id, entity, relation="has_attribute")
        for subj, pred, obj in desc_relations:
            self.kg.add_node(subj, type="attribute")
            self.kg.add_node(obj, type="attribute")
            self.kg.add_edge(subj, obj, relation=pred)

        # Process reviews with LLM simulation
        for review in reviews:
            review_entities, review_relations = self._simulate_llm_extraction(review)
            for entity in review_entities:
                self.kg.add_node(entity, type="attribute")
                self.kg.add_edge(product_id, entity, relation="mentioned_in_review")
            for subj, pred, obj in review_relations:
                self.kg.add_node(subj, type="attribute")
                self.kg.add_node(obj, type="attribute")
                self.kg.add_edge(subj, obj, relation=pred)

    def get_knowledge_graph(self):
        return self.kg

    def visualize_kg(self):
        print("\n--- Knowledge Graph Nodes ---")
        for node, data in self.kg.nodes(data=True):
            print(f"Node: {node}, Type: {data.get('type', 'unknown')}")

        print("\n--- Knowledge Graph Edges ---")
        for u, v, data in self.kg.edges(data=True):
            print(f"({u}) --[{data.get('relation', 'unknown')}]--> ({v})")