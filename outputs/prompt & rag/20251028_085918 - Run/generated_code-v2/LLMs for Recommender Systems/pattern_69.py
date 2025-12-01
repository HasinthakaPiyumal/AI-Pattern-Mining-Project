import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

def load_and_preprocess_data():
    product_data = {
        "product_id": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        "name": [
            "Wireless Bluetooth Headphones", "Ergonomic Office Chair",
            "Stainless Steel Water Bottle", "Smart LED Desk Lamp",
            "Noise Cancelling Earbuds", "Portable Blender",
            "High-Speed SSD Drive", "Yoga Mat Eco-Friendly",
            "Smartwatch Fitness Tracker", "Electric Kettle"
        ],
        "category": [
            "Electronics", "Office", "Kitchen", "Home", "Electronics",
            "Kitchen", "Electronics", "Fitness", "Electronics", "Kitchen"
        ],
        "description": [
            "Premium audio with long battery life and comfortable design.",
            "Adjustable lumbar support and breathable mesh for all-day comfort.",
            "Double-walled insulation keeps drinks cold for 24 hours.",
            "Customizable brightness and color temperature via app.",
            "Compact earbuds with active noise cancellation and clear sound.",
            "Blend smoothies on the go with this powerful battery-operated blender.",
            "Boost your computer's performance with lightning-fast data transfer.",
            "Non-slip surface, made from sustainable materials for your practice.",
            "Track heart rate, steps, and notifications with a sleek design.",
            "Boil water rapidly with automatic shut-off and sleek design."
        ]
    }
    products_df = pd.DataFrame(product_data)
    products_df = products_df.set_index("product_id")

    user_item_data = {
        "user_id": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 1, 2, 3, 4, 1, 2, 3, 4],
        "product_id": [101, 103, 105, 101, 102, 106, 104, 107, 109, 102, 108, 110, 104, 107, 101, 105, 106, 103, 108, 109],
        "rating": [5, 4, 5, 4, 5, 3, 5, 4, 5, 3, 5, 4, 3, 5, 4, 5, 4, 3, 4, 5]
    }
    ratings_df = pd.DataFrame(user_item_data)

    return products_df, ratings_df

class ItemBasedRecommender:
    def __init__(self, ratings_df, products_df):
        self.ratings_df = ratings_df
        self.products_df = products_df
        self.item_similarity_matrix = None
        self._prepare_data()

    def _prepare_data(self):
        self.user_item_matrix = self.ratings_df.pivot_table(
            index="user_id", columns="product_id", values="rating"
        ).fillna(0)

        item_features = self.user_item_matrix.T
        self.item_similarity_matrix = cosine_similarity(item_features)
        self.item_similarity_df = pd.DataFrame(
            self.item_similarity_matrix,
            index=item_features.index,
            columns=item_features.index
        )

    def recommend_products(self, user_id, num_recommendations=5):
        if user_id not in self.user_item_matrix.index:
            popular_items = self.ratings_df["product_id"].value_counts().head(num_recommendations).index.tolist()
            return popular_items, [{"type": "popular"}] * num_recommendations

        user_ratings = self.user_item_matrix.loc[user_id]
        rated_products = user_ratings[user_ratings > 0].index.tolist()

        if not rated_products:
            popular_items = self.ratings_df["product_id"].value_counts().head(num_recommendations).index.tolist()
            return popular_items, [{"type": "popular"}] * num_recommendations

        scores = defaultdict(float)
        for item_id in rated_products:
            if item_id in self.item_similarity_df.index:
                similar_items = self.item_similarity_df[item_id].sort_values(ascending=False)
                for sim_item, sim_score in similar_items.items():
                    if sim_item not in rated_products and user_ratings.get(sim_item, 0) == 0:
                        scores[sim_item] += sim_score * user_ratings[item_id]

        recommended_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:num_recommendations]
        recommended_product_ids = [item[0] for item in recommended_items]
        
        reasons_for_llm = []
        for rec_id in recommended_product_ids:
            user_liked_items_names = [self.products_df.loc[pid]["name"] for pid in rated_products if pid in self.products_df.index]
            reasons_for_llm.append({
                "type": "item_similarity",
                "similar_to_user_likes": user_liked_items_names
            })

        return recommended_product_ids, reasons_for_llm


class MockLLMExplanationGenerator:
    def __init__(self, products_df):
        self.products_df = products_df

    def generate_explanation(self, recommended_product_id, reasons, user_feedback=None):
        product_info = self.products_df.loc[recommended_product_id]
        product_name = product_info["name"]
        product_category = product_info["category"]
        product_description = product_info["description"]

        explanation_parts = [f"We recommend the {product_name}."]

        if reasons and isinstance(reasons, dict):
            if reasons.get("type") == "item_similarity" and reasons.get("similar_to_user_likes"):
                liked_items_str = ", ".join(reasons["similar_to_user_likes"][:2])
                if liked_items_str:
                    explanation_parts.append(f"It's similar to other items you've enjoyed, like {liked_items_str}.")
                explanation_parts.append(f"Specifically, we think you'll appreciate its features, such as '{product_description}'.")
            elif reasons.get("type") == "popular":
                explanation_parts.append(f"This is a popular choice among our users, especially in the {product_category} category.")
                explanation_parts.append(f"Many find its '{product_description}' to be a great value.")
            else:
                explanation_parts.append(f"This {product_category} product has great reviews and features like '{product_description}'.")
        else:
            explanation_parts.append(f"This {product_category} product has great reviews and features like '{product_description}'.")

        if user_feedback:
            explanation_parts.append(f"Your previous feedback ('{user_feedback}') helped us refine this recommendation for you.")

        return " ".join(explanation_parts)

    def handle_feedback(self, feedback_text):
        if "more detail" in feedback_text.lower():
            return "You're looking for more in-depth reasons? We're constantly improving our explanations to provide richer insights."
        elif "not interested" in feedback_text.lower():
            return "Understood. We'll try to learn from your preferences to avoid similar recommendations in the future."
        else:
            return "Thank you for your feedback! We'll use this to improve your future recommendations and explanations."

def main():
    st.set_page_config(page_title="Explainable Product Recommender", layout="wide")
    st.title("🛍️ Explainable Product Recommender")

    products_df, ratings_df = load_and_preprocess_data()
    recommender = ItemBasedRecommender(ratings_df, products_df)
    llm_explainer = MockLLMExplanationGenerator(products_df)

    st.sidebar.header("User Selection")
    all_users = sorted(ratings_df["user_id"].unique().tolist())
    selected_user = st.sidebar.selectbox("Select a User ID", all_users)

    st.sidebar.header("Recommendation Settings")
    num_recs = st.sidebar.slider("Number of Recommendations", 1, 10, 5)

    if st.sidebar.button("Get Recommendations"):           
        recommended_ids, reasons_data = recommender.recommend_products(selected_user, num_recommendations=num_recs)
        st.session_state["recommended_product_ids"] = recommended_ids
        st.session_state["reasons_for_llm"] = reasons_data
        st.session_state["user_feedback_history"] = {}

    st.header(f"Recommendations for User {selected_user}")

    if "recommended_product_ids" in st.session_state:
        for i, product_id in enumerate(st.session_state["recommended_product_ids"]):
            product_info = products_df.loc[product_id]
            reason_data = st.session_state["reasons_for_llm"][i] if i < len(st.session_state["reasons_for_llm"]) else {}

            st.subheader(f"{i+1}. {product_info['name']} (ID: {product_id})")
            
            explanation = llm_explainer.generate_explanation(
                product_id,
                reason_data,
                st.session_state["user_feedback_history"].get(product_id)
            )
            st.info(f"**Explanation:** {explanation}")

            st.write(f"**Category:** {product_info['category']}")
            st.write(f"**Description:** {product_info['description']}")

            feedback_input_key = f"feedback_{product_id}"
            user_feedback_text = st.text_input(
                f"Your feedback for {product_info['name']} (e.g., 'more detail', 'not interested'):",
                key=feedback_input_key
            )
            if user_feedback_text:
                llm_response_to_feedback = llm_explainer.handle_feedback(user_feedback_text)
                st.success(f"System's response to your feedback: {llm_response_to_feedback}")
                st.session_state["user_feedback_history"][product_id] = user_feedback_text
            st.markdown("---")
    else:
        st.write("Select a user and click 'Get Recommendations' to see personalized product suggestions.")

if __name__ == "__main__":
    main()