import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline, set_seed


# Data Generation
@st.cache_data
def load_data():
    products_data = {
        'product_id': range(1, 11),
        'name': [
            'Laptop Pro X', 'Wireless Mouse', 'Mechanical Keyboard', '4K Monitor', 'Webcam HD',
            'Gaming Headset', 'Ergonomic Chair', 'External SSD', 'USB-C Hub', 'Smart Speaker'
        ],
        'description': [
            'Powerful laptop for professionals with high-end specs.',
            'Ergonomic wireless mouse with customizable buttons.',
            'Durable mechanical keyboard with RGB lighting and tactile switches.',
            'Stunning 4K display for immersive viewing and productivity.',
            'High-definition webcam for clear video calls and streaming.',
            'Immersive gaming headset with surround sound and noise-cancelling mic.',
            'Comfortable ergonomic chair for long hours of work or gaming.',
            'Fast and portable external SSD for quick data transfers.',
            'Versatile USB-C hub with multiple ports for connectivity.',
            'Voice-controlled smart speaker with premium audio quality.'
        ],
        'category': [
            'Electronics', 'Electronics', 'Electronics', 'Electronics', 'Electronics',
            'Electronics', 'Office', 'Electronics', 'Electronics', 'Smart Home'
        ]
    }
    products_df = pd.DataFrame(products_data)

    # Simulate user interaction data
    user_interactions_data = {
        'user_id': [
            1, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5, 5
        ],
        'product_id': [
            1, 2, 3, 4, 5, 1, 6, 7, 8, 2, 9, 3, 10, 1
        ],
        'rating': [
            5, 4, 5, 3, 4, 5, 4, 5, 4, 5, 3, 4, 5, 5
        ]
    }
    user_interactions_df = pd.DataFrame(user_interactions_data)

    return products_df, user_interactions_df

products_df, user_interactions_df = load_data()

# Recommendation Engine
def get_recommendations(user_id, num_recommendations=5):
    user_liked_products = user_interactions_df[user_interactions_df['user_id'] == user_id]['product_id'].tolist()
    
    if not user_liked_products:
        # If no interactions, recommend popular items or random
        return products_df.sample(num_recommendations)['product_id'].tolist()

    # Get descriptions of liked products
    liked_product_descriptions = products_df[products_df['product_id'].isin(user_liked_products)]['description'].tolist()

    # Vectorize product descriptions
    vectorizer = TfidfVectorizer(stop_words='english')
    product_vectors = vectorizer.fit_transform(products_df['description'])
    liked_product_vectors = vectorizer.transform(liked_product_descriptions)

    # Calculate average vector for liked products
    avg_liked_vector = liked_product_vectors.mean(axis=0)

    # Calculate similarity with all products
    similarities = cosine_similarity(avg_liked_vector, product_vectors).flatten()

    # Get top recommendations (excluding already liked products)
    recommended_product_indices = similarities.argsort()[::-1]
    
    recommended_products = []
    for idx in recommended_product_indices:
        product_id = products_df.iloc[idx]['product_id']
        if product_id not in user_liked_products:
            recommended_products.append(product_id)
        if len(recommended_products) >= num_recommendations:
            break

    return recommended_products

# Explanation Generation Layer (using a lightweight LLM)
@st.cache_resource
def load_llm_pipeline():
    set_seed(42)
    return pipeline("text-generation", model="distilgpt2")

generator = load_llm_pipeline()

def generate_explanation(product_name, product_description, user_context="", reason=""):    
    prompt = f"Explain why a user, who {user_context}, might be interested in \"{product_name}\", a product described as '{product_description}'. The recommendation is because {reason}. Provide a concise and natural explanation.\nExplanation:"
    
    # Generate text with LLM
    output = generator(prompt, max_new_tokens=100, num_return_sequences=1, truncation=True)
    explanation = output[0]['generated_text'].replace(prompt, '').strip()
    
    # Post-process to make it sound more natural and remove potential prompt repetition
    explanation = explanation.split('\n')[0] # Take first line
    if explanation.endswith('.'): # Ensure it ends with a period
        return explanation
    else:
        return explanation + '.'

# Streamlit App
st.set_page_config(layout="wide", page_title="Explainable E-commerce Recommender")
st.title("🛒 Explainable E-commerce Product Recommender")

st.sidebar.header("User Selection")
user_id_selection = st.sidebar.selectbox(
    "Select a User ID",
    user_interactions_df['user_id'].unique()
)

st.header(f"Recommendations for User {user_id_selection}")

if user_id_selection:
    st.subheader("Your Recent Interactions:")
    user_recent_items = user_interactions_df[user_interactions_df['user_id'] == user_id_selection]
    if not user_recent_items.empty:
        for idx, row in user_recent_items.iterrows():
            product_info = products_df[products_df['product_id'] == row['product_id']].iloc[0]
            st.write(f"- **{product_info['name']}** (Category: {product_info['category']}) - Rating: {row['rating']}")
    else:
        st.write("No recent interactions found for this user.")

    st.subheader("Personalized Recommendations:")
    recommended_product_ids = get_recommendations(user_id_selection)

    if recommended_product_ids:
        for prod_id in recommended_product_ids:
            product = products_df[products_df['product_id'] == prod_id].iloc[0]
            
            # Simulate a reason for recommendation (could be more sophisticated)
            user_interactions_str = f"liked items in the {products_df[products_df['product_id'].isin(user_interactions_df[user_interactions_df['user_id'] == user_id_selection]['product_id'].tolist() )]['category'].unique()[0]} category"
            recommendation_reason = f"similar to other {product['category']} products you've shown interest in"

            explanation = generate_explanation(
                product_name=product['name'],
                product_description=product['description'],
                user_context=user_interactions_str,
                reason=recommendation_reason
            )
            
            st.markdown(f"### {product['name']}")
            st.write(f"**Category**: {product['category']}")
            st.write(f"**Description**: {product['description']}")
            st.info(f"**Why we recommend this**: {explanation}")
            st.markdown("--- ")
    else:
        st.write("Could not generate recommendations for this user.")

else:
    st.write("Please select a user from the sidebar to get recommendations.")
