import networkx as nx
import pandas as pd
import streamlit as st
import random

# --- 1. Mock Data Layer ---
products_data = {
    "p1": {"name": "Laptop X1", "category": "Electronics", "price": 1200, "brand": "TechCorp"},
    "p2": {"name": "Wireless Mouse", "category": "Electronics", "price": 35, "brand": "TechCorp"},
    "p3": {"name": "Mechanical Keyboard", "category": "Electronics", "price": 110, "brand": "KeyMaster"},
    "p4": {"name": "Desk Lamp", "category": "Home Office", "price": 45, "brand": "BrightLite"},
    "p5": {"name": "Ergonomic Chair", "category": "Home Office", "price": 300, "brand": "ComfySit"},
    "p6": {"name": "Webcam HD", "category": "Electronics", "price": 70, "brand": "ZoomPro"},
    "p7": {"name": "USB-C Hub", "category": "Electronics", "price": 50, "brand": "ConnectAll"},
    "p8": {"name": "Coffee Mug", "category": "Kitchen", "price": 15, "brand": "MugLife"},
    "p9": {"name": "Noise Cancelling Headphones", "category": "Electronics", "price": 250, "brand": "SoundOff"},
}

reviews_data = {
    "p1": ["Great laptop, very fast!", "Good for coding and gaming.", "Screen quality is amazing."],
    "p2": ["Smooth and responsive mouse.", "Battery life is excellent."],
    "p3": ["Love the clicky sound!", "Solid build quality."],
    "p4": ["Perfect for late-night work.", "Adjustable brightness is a plus."],
    "p5": ["Very comfortable for long hours.", "Easy to assemble."],
    "p6": ["Clear video calls.", "Autofocus works well."],
    "p7": ["All ports work fine.", "Compact design."],
    "p8": ["Keeps coffee hot.", "Nice design."],
    "p9": ["Amazing sound quality, blocks out everything.", "Comfortable earcups."],
}

# --- 2. Knowledge Graph Layer ---
class ProductKnowledgeGraph:
    def __init__(self):
        self.graph = nx.Graph()

    def add_product(self, product_id, attributes):
        self.graph.add_node(product_id, **attributes)

    def add_relationship(self, product_id1, product_id2, relation_type, weight=1):
        self.graph.add_edge(product_id1, product_id2, type=relation_type, weight=weight)

    def get_product_attributes(self, product_id):
        return self.graph.nodes.get(product_id, {})

    def get_related_products(self, product_id, relation_type=None):
        related = []
        for neighbor in self.graph.neighbors(product_id):
            edge_data = self.graph.get_edge_data(product_id, neighbor)
            if relation_type is None or edge_data.get("type") == relation_type:
                related.append(neighbor)
        return related

# --- 3. LLM Service Layer (Mock) ---
class MockLLMService:
    def infer_missing_attributes(self, product_name, existing_attributes, context):
        # Simulate LLM inferring attributes
        if "Laptop" in product_name and "RAM" not in existing_attributes:
            return {"RAM": "16GB", "Storage": "512GB SSD"}
        if "Mouse" in product_name and "connectivity" not in existing_attributes:
            return {"connectivity": "Bluetooth"}
        if "Keyboard" in product_name and "switch_type" not in existing_attributes:
            return {"switch_type": "Tactile"}
        return {}

    def identify_implicit_relationships(self, product1_name, product2_name, context):
        # Simulate LLM identifying relationships (e.g., 'frequently bought together')
        if ("Laptop" in product1_name and "Mouse" in product2_name) or \
           ("Mouse" in product1_name and "Laptop" in product2_name):
            return {"type": "frequently_bought_together", "reason": "Common office/computing setup"}
        if ("Laptop" in product1_name and "Webcam" in product2_name) or \
           ("Webcam" in product1_name and "Laptop" in product2_name):
            return {"type": "frequently_bought_together", "reason": "Remote work accessories"}
        return {}

    def extract_new_features(self, product_name, review_text):
        # Simulate LLM extracting features from reviews
        extracted_features = []
        if "fast" in review_text.lower() or "responsive" in review_text.lower():
            extracted_features.append("performance")
        if "quality" in review_text.lower() or "clear" in review_text.lower():
            extracted_features.append("quality")
        if "comfortable" in review_text.lower() or "ergonomic" in review_text.lower():
            extracted_features.append("comfort")
        return {"extracted_features": list(set(extracted_features))}


# --- 4. KG Enrichment & Completion Module ---
class KGEnrichmentModule:
    def __init__(self, kg, llm_service):
        self.kg = kg
        self.llm_service = llm_service

    def enrich_from_llm(self):
        st.sidebar.subheader("KG Enrichment Log")
        st.sidebar.text("Starting KG Enrichment...")

        # Infer missing attributes
        for product_id, attrs in list(self.kg.graph.nodes(data=True)):
            inferred_attrs = self.llm_service.infer_missing_attributes(attrs.get("name", ""), attrs, None)
            if inferred_attrs:
                self.kg.graph.nodes[product_id].update(inferred_attrs)
                st.sidebar.text(f"Inferred attributes for {product_id}: {inferred_attrs}")

        # Identify implicit relationships
        product_ids = list(self.kg.graph.nodes())
        for i in range(len(product_ids)):
            for j in range(i + 1, len(product_ids)):
                p_id1, p_id2 = product_ids[i], product_ids[j]
                product1_name = self.kg.graph.nodes[p_id1].get("name", "")
                product2_name = self.kg.graph.nodes[p_id2].get("name", "")

                if not self.kg.graph.has_edge(p_id1, p_id2, key="frequently_bought_together"):
                    relationship = self.llm_service.identify_implicit_relationships(product1_name, product2_name, None)
                    if relationship and relationship["type"] == "frequently_bought_together":
                        self.kg.add_relationship(p_id1, p_id2, "frequently_bought_together", weight=0.8)
                        st.sidebar.text(f"Added 'frequently_bought_together' between {p_id1} and {p_id2}")

        # Extract new features from reviews
        for product_id, reviews in reviews_data.items():
            current_attrs = self.kg.get_product_attributes(product_id)
            existing_features = current_attrs.get("features", [])
            for review in reviews:
                extracted = self.llm_service.extract_new_features(current_attrs.get("name", ""), review)
                if extracted.get("extracted_features"):
                    for feature in extracted["extracted_features"]:
                        if feature not in existing_features:
                            existing_features.append(feature)
                            st.sidebar.text(f"Extracted feature '{feature}' for {product_id} from review.")
            self.kg.graph.nodes[product_id]["features"] = existing_features
        st.sidebar.text("KG Enrichment complete.")

# --- 5. Recommendation Engine Module ---
class RecommendationEngine:
    def __init__(self, kg):
        self.kg = kg

    def get_recommendations(self, user_product_id, num_recommendations=3):
        recommendations = set()

        # 1. Recommend products frequently bought together
        frequently_bought = self.kg.get_related_products(user_product_id, relation_type="frequently_bought_together")
        for prod_id in frequently_bought:
            recommendations.add(prod_id)

        # 2. Recommend similar category products
        user_category = self.kg.get_product_attributes(user_product_id).get("category")
        if user_category:
            for product_id in self.kg.graph.nodes():
                if product_id != user_product_id and self.kg.get_product_attributes(product_id).get("category") == user_category:
                    recommendations.add(product_id)

        # 3. Recommend products with shared inferred features (if any)
        user_features = self.kg.get_product_attributes(user_product_id).get("features", [])
        if user_features:
            for product_id in self.kg.graph.nodes():
                if product_id != user_product_id:
                    other_features = self.kg.get_product_attributes(product_id).get("features", [])
                    if any(f in user_features for f in other_features):
                        recommendations.add(product_id)

        # Filter out the user's current product and existing recommendations up to num_recommendations
        final_recommendations = [prod_id for prod_id in list(recommendations) if prod_id != user_product_id]
        random.shuffle(final_recommendations)
        return final_recommendations[:num_recommendations]

# --- Main Application Logic (Streamlit UI) ---
def main():
    st.set_page_config(layout="wide")
    st.title("🛒 LLM-Enhanced E-commerce Recommender")

    # Initialize KG and LLM service
    kg = ProductKnowledgeGraph()
    for p_id, attrs in products_data.items():
        kg.add_product(p_id, attrs)

    llm_service = MockLLMService()
    kg_enrichment = KGEnrichmentModule(kg, llm_service)
    recommender = RecommendationEngine(kg)

    st.sidebar.header("Configuration & Actions")
    if st.sidebar.button("Enrich Knowledge Graph with LLM"):
        kg_enrichment.enrich_from_llm()
        st.sidebar.success("Knowledge Graph Enrichment Complete!")

    st.header("Product Catalog")
    df_products = pd.DataFrame.from_dict(products_data, orient='index')
    st.dataframe(df_products)

    st.header("Explore Knowledge Graph")
    st.write("Select a product to view its enriched attributes and relationships.")
    selected_product_id = st.selectbox("Choose a Product:", list(kg.graph.nodes()))

    if selected_product_id:
        st.subheader(f"Details for {kg.get_product_attributes(selected_product_id).get('name')}")
        st.json(kg.get_product_attributes(selected_product_id))

        st.subheader("Directly Related Products")
        related_products = kg.get_related_products(selected_product_id)
        if related_products:
            related_names = [kg.get_product_attributes(pid).get('name') for pid in related_products]
            st.write(", ".join(related_names))
        else:
            st.write("No direct relationships found (yet).")

        st.subheader("Recommended Products")
        recommendations = recommender.get_recommendations(selected_product_id)
        if recommendations:
            rec_df_data = []
            for rec_id in recommendations:
                attrs = kg.get_product_attributes(rec_id)
                rec_df_data.append({"ID": rec_id, "Name": attrs.get('name'), "Category": attrs.get('category'), "Price": attrs.get('price')})
            st.dataframe(pd.DataFrame(rec_df_data))
        else:
            st.write("No recommendations found. Try enriching the KG!")

    st.sidebar.markdown("--- ")
    st.sidebar.markdown("Developed using LLMs as Knowledge Base pattern.")

if __name__ == "__main__":
    main()