import json
import networkx as nx

class LLMBasedKnowledgeExtractor:
    def __init__(self, llm_model=None, tokenizer=None):
        self.llm_model = llm_model
        self.tokenizer = tokenizer

    def _mock_llm_response(self, prompt):
        if "extract entities" in prompt.lower():
            return "{{\"entities\": [\"Laptop X\", \"16GB RAM\", \"512GB SSD\", \"Intel i7\", \"Windows 11\"]}}"
        elif "resolve coreferences" in prompt.lower():
            return "{{\"coreferences\": [{\"original\": \"it\", \"resolved\": \"Laptop X\"}]}}"
        elif "extract relations" in prompt.lower():
            return "{{\"relations\": [[\"Laptop X\", \"has_feature\", \"16GB RAM\"], [\"Laptop X\", \"has_feature\", \"512GB SSD\"], [\"Laptop X\", \"runs_on\", \"Windows 11\"]]}}"
        return "{}"

    def extract_entities(self, text: str) -> list:
        prompt = f"Given the following product description, extract entities: {text}"
        response = self._mock_llm_response(prompt)
        try:
            data = json.loads(response)
            return data.get("entities", [])
        except json.JSONDecodeError:
            return []

    def resolve_coreferences(self, text: str) -> dict:
        prompt = f"Given the following product description, resolve coreferences: {text}"
        response = self._mock_llm_response(prompt)
        try:
            data = json.loads(response)
            return {item["original"]: item["resolved"] for item in data.get("coreferences", [])}
        except json.JSONDecodeError:
            return {}

    def extract_relations(self, text: str) -> list:
        prompt = f"Given the following product description, extract relations in (subject, predicate, object) format: {text}"
        response = self._mock_llm_response(prompt)
        try:
            data = json.loads(response)
            return data.get("relations", [])
        except json.JSONDecodeError:
            return []

class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_entity(self, entity_id: str, entity_type: str, attributes: dict = None):
        if not self.graph.has_node(entity_id):
            node_attributes = {"type": entity_type}
            if attributes:
                node_attributes.update(attributes)
            self.graph.add_node(entity_id, **node_attributes)
            return True
        return False

    def add_relation(self, source_id: str, relation_type: str, target_id: str, attributes: dict = None):
        if self.graph.has_node(source_id) and self.graph.has_node(target_id):
            edge_attributes = {"type": relation_type}
            if attributes:
                edge_attributes.update(attributes)
            self.graph.add_edge(source_id, target_id, **edge_attributes)
            return True
        return False

    def get_neighbors(self, entity_id: str, relation_type: str = None) -> list:
        neighbors = []
        if self.graph.has_node(entity_id):
            for neighbor in self.graph.neighbors(entity_id):
                edge_data = self.graph.get_edge_data(entity_id, neighbor)
                if relation_type is None or (edge_data and edge_data.get("type") == relation_type):
                    neighbors.append(neighbor)
        return neighbors

    def get_entity_attributes(self, entity_id: str) -> dict:
        if self.graph.has_node(entity_id):
            return self.graph.nodes[entity_id]
        return {}

    def save_graph(self, filename: str):
        graph_data = {
            "nodes": [{
                "id": node_id,
                **self.graph.nodes[node_id]
            } for node_id in self.graph.nodes],
            "edges": [{
                "source": u,
                "target": v,
                **self.graph.get_edge_data(u, v)
            } for u, v in self.graph.edges]
        }
        with open(filename, "w") as f:
            json.dump(graph_data, f, indent=4)
        print(f"Knowledge graph saved to {filename}")

    def load_graph(self, filename: str):
        with open(filename, "r") as f:
            graph_data = json.load(f)

        self.graph = nx.DiGraph()
        for node_data in graph_data.get("nodes", []):
            node_id = node_data.pop("id")
            self.graph.add_node(node_id, **node_data)

        for edge_data in graph_data.get("edges", []):
            source = edge_data.pop("source")
            target = edge_data.pop("target")
            self.graph.add_edge(source, target, **edge_data)
        print(f"Knowledge graph loaded from {filename}")

    def visualize(self):
        try:
            import matplotlib.pyplot as plt
            pos = nx.spring_layout(self.graph)
            nx.draw(self.graph, pos, with_labels=True, node_color="lightblue", node_size=1500, font_size=8)
            edge_labels = nx.get_edge_attributes(self.graph, "type")
            nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)
            plt.show()
        except ImportError:
            print("Matplotlib not installed. Cannot visualize graph. Please install with 'pip install matplotlib'.")

class ProductRecommender:
    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.kg = knowledge_graph

    def get_recommendations(self, product_id: str, num_recommendations: int = 5) -> list:
        if not self.kg.graph.has_node(product_id):
            print(f"Product {product_id} not found in the knowledge graph.")
            return []

        recommended_products = set()

        bought_with = self.kg.get_neighbors(product_id, relation_type="often_bought_with")
        for p in bought_with:
            if p != product_id:
                recommended_products.add(p)
                if len(recommended_products) >= num_recommendations:
                    return list(recommended_products)[:num_recommendations]

        compatible_with = self.kg.get_neighbors(product_id, relation_type="is_compatible_with")
        for p in compatible_with:
            if p != product_id:
                recommended_products.add(p)
                if len(recommended_products) >= num_recommendations:
                    return list(recommended_products)[:num_recommendations]

        product_features = self.kg.get_neighbors(product_id, relation_type="has_feature")
        for feature in product_features:
            for other_product in self.kg.graph.nodes:
                if other_product != product_id and self.kg.graph.has_edge(other_product, feature, key=None) and self.kg.graph.get_edge_data(other_product, feature).get("type") == "has_feature":
                    recommended_products.add(other_product)
                    if len(recommended_products) >= num_recommendations:
                        return list(recommended_products)[:num_recommendations]

        return list(recommended_products)[:num_recommendations]

def main():
    print("Initializing E-commerce Recommender System...")

    extractor = LLMBasedKnowledgeExtractor()
    kg = KnowledgeGraph()

    products_data = [
        {
            "id": "P001",
            "name": "Gaming Laptop X",
            "description": "Powerful Gaming Laptop X with 16GB RAM, 512GB SSD, Intel i7 processor, and NVIDIA RTX graphics. Runs on Windows 11. Ideal for serious gamers. It comes with a 15.6-inch display."
        },
        {
            "id": "P002",
            "name": "Wireless Gaming Mouse",
            "description": "Ergonomic wireless gaming mouse with customizable DPI and RGB lighting. Compatible with Laptop X."
        },
        {
            "id": "P003",
            "name": "External SSD 1TB",
            "description": "High-speed 1TB external SSD for expanded storage. Often bought with laptops for extra space."
        },
        {
            "id": "P004",
            "name": "Mechanical Keyboard",
            "description": "Tactile mechanical keyboard for gaming and typing. Features durable keycaps and RGB backlighting."
        },
        {
            "id": "P005",
            "name": "Monitor 27-inch 144Hz",
            "description": "27-inch gaming monitor with a 144Hz refresh rate and QHD resolution. Enhances gaming experience for Laptop X users."
        }
    ]

    print("\n--- Building Knowledge Graph ---")
    for product in products_data:
        product_id = product["id"]
        product_name = product["name"]
        description = product["description"]

        kg.add_entity(product_id, "product", {"name": product_name, "description": description})
        print(f"Added product entity: {product_name} ({product_id})")

        print(f"  Extracting knowledge for {product_name}...")
        entities = extractor.extract_entities(description)
        coreferences = extractor.resolve_coreferences(description)
        relations = extractor.extract_relations(description)

        for entity in entities:
            resolved_entity = coreferences.get(entity, entity)
            if resolved_entity != product_id:
                kg.add_entity(resolved_entity, "feature")

        for sub, pred, obj in relations:
            resolved_sub = coreferences.get(sub, sub)
            resolved_obj = coreferences.get(obj, obj)

            if not kg.graph.has_node(resolved_sub):
                kg.add_entity(resolved_sub, "unknown_entity")
            if not kg.graph.has_node(resolved_obj):
                kg.add_entity(resolved_obj, "unknown_entity")

            final_sub = product_id if product_name.lower() in resolved_sub.lower() else resolved_sub
            final_obj = product_id if product_name.lower() in resolved_obj.lower() else resolved_obj

            if final_sub == product_id or final_obj == product_id:
                if final_sub == product_id and final_obj != product_id:
                     kg.add_relation(final_sub, pred.replace(" ", "_"), final_obj)
                elif final_obj == product_id and final_sub != product_id:
                     kg.add_relation(final_sub, pred.replace(" ", "_"), final_obj)
                elif final_sub != product_id and final_obj != product_id:
                     kg.add_relation(final_sub, pred.replace(" ", "_"), final_obj)

        if product_id == "P001":
            kg.add_relation("P001", "is_compatible_with", "P002")
            kg.add_relation("P001", "often_bought_with", "P003")
            kg.add_relation("P001", "has_feature", "16GB RAM")
            kg.add_relation("P001", "has_feature", "512GB SSD")
            kg.add_relation("P001", "has_feature", "Intel i7")
            kg.add_relation("P001", "runs_on", "Windows 11")
            kg.add_relation("P001", "has_display_size", "15.6-inch display")
        elif product_id == "P002":
            kg.add_relation("P002", "often_bought_with", "P004")
        elif product_id == "P003":
            kg.add_relation("P003", "is_compatible_with", "P001")
        elif product_id == "P005":
            kg.add_relation("P005", "is_compatible_with", "P001")
            kg.add_relation("P005", "has_feature", "27-inch display")
            kg.add_relation("P005", "has_feature", "144Hz refresh rate")
            kg.add_relation("P005", "has_feature", "QHD resolution")

    kg.save_graph("product_knowledge_graph.json")

    recommender = ProductRecommender(kg)

    print("\n--- Generating Recommendations ---")
    test_product_id = "P001"
    recommendations = recommender.get_recommendations(test_product_id)
    print(f"Recommendations for {kg.get_entity_attributes(test_product_id).get("name", test_product_id)}:")
    if recommendations:
        for rec_id in recommendations:
            print(f"- {kg.get_entity_attributes(rec_id).get("name", rec_id)}")
    else:
        print("No recommendations found.")

    test_product_id_2 = "P002"
    recommendations_2 = recommender.get_recommendations(test_product_id_2)
    print(f"\nRecommendations for {kg.get_entity_attributes(test_product_id_2).get("name", test_product_id_2)}:")
    if recommendations_2:
        for rec_id in recommendations_2:
            print(f"- {kg.get_entity_attributes(rec_id).get("name", rec_id)}")
    else:
        print("No recommendations found.")

if __name__ == "__main__":
    main()
