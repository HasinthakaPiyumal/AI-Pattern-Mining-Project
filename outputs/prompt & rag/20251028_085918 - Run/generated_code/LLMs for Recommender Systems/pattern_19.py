import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline
import streamlit as st
import os

# --- Simulated product_data.py content ---

@st.cache_resource
def generate_product_data_and_embeddings():
    """Generates synthetic product data and their embeddings."""
    st.write("Generating synthetic product data and embeddings...")
    products = [
        {"id": 1, "name": "Smartwatch X", "description": "A sleek smartwatch with health tracking and notifications.", "category": "Electronics", "price": 299.99},
        {"id": 2, "name": "Wireless Earbuds Pro", "description": "Premium noise-cancelling earbuds for immersive audio experience.", "category": "Electronics", "price": 199.99},
        {"id": 3, "name": "Ergonomic Office Chair", "description": "Adjustable chair designed for maximum comfort during long work hours.", "category": "Home & Office", "price": 450.00},
        {"id": 4, "name": "4K Smart TV 55 inch", "description": "Vibrant 4K display with smart features and voice control.", "category": "Electronics", "price": 799.00},
        {"id": 5, "name": "Robot Vacuum Cleaner", "description": "Automated vacuum for effortless cleaning of various floor types.", "category": "Home & Office", "price": 350.00},
        {"id": 6, "name": "Coffee Maker Deluxe", "description": "Programmable coffee maker with built-in grinder for fresh brews.", "category": "Home & Office", "price": 120.00},
        {"id": 7, "name": "Gaming Laptop RGB", "description": "High-performance gaming laptop with vibrant RGB keyboard and powerful graphics.", "category": "Electronics", "price": 1200.00},
        {"id": 8, "name": "Yoga Mat Eco-Friendly", "description": "Non-slip, durable, and environmentally friendly yoga mat for all levels.", "category": "Sports & Outdoors", "price": 35.00},
        {"id": 9, "name": "Bluetooth Speaker Portable", "description": "Compact and waterproof Bluetooth speaker with rich sound and long battery life.", "category": "Electronics", "price": 75.00},
        {"id": 10, "name": "Air Fryer XL", "description": "Large capacity air fryer for healthy and crispy meals with less oil.", "category": "Home & Office", "price": 150.00},
    ]
    products_df = pd.DataFrame(products)

    model_name = "all-MiniLM-L6-v2"
    try:
        embedding_model = SentenceTransformer(model_name)
    except Exception as e:
        st.error(f"Could not load SentenceTransformer model {model_name}. Please check your internet connection or model availability: {e}")
        st.stop()

    product_descriptions = products_df["description"].tolist()
    product_embeddings = embedding_model.encode(product_descriptions, show_progress_bar=False)

    st.success("Product data and embeddings generated!")
    return products_df, product_embeddings, embedding_model

# --- Simulated recommendation_engine.py content ---

@st.cache_resource
def initialize_recommendation_engine(product_embeddings, embedding_model):
    """Initializes Faiss index and the LLM for explanations."""
    st.write("Initializing recommendation engine (Faiss and LLM)...")

    # Initialize Faiss index
    dimension = product_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity
    index.add(product_embeddings) # Add product embeddings to the index

    # Initialize LLM for explanations
    llm_model_name = "distilgpt2"
    try:
        explainer_pipeline = pipeline("text-generation", model=llm_model_name, tokenizer=llm_model_name, device=0 if st.session_state.get("cuda_available", False) else -1)
    except Exception as e:
        st.error(f"Could not load LLM model {llm_model_name}. Please check your internet connection or model availability: {e}")
        st.stop()

    st.success("Recommendation engine initialized!")
    return index, explainer_pipeline

def get_recommendations_and_explanations(query: str, products_df: pd.DataFrame, embedding_model, faiss_index, llm_pipeline, k: int = 5):
    """Generates personalized recommendations and LLM-powered explanations."""
    st.write(f"Searching for recommendations for: '{query}'...")

    # 1. Embed the user query
    query_embedding = embedding_model.encode([query])

    # 2. Search for similar products using Faiss
    distances, indices = faiss_index.search(query_embedding, k) # Find k nearest neighbors

    recommended_products = products_df.iloc[indices[0]].copy()

    # 3. Generate LLM-powered explanations for each recommendation
    recommended_products["explanation"] = ""
    for i, row in recommended_products.iterrows():
        prompt = f"Explain why a user interested in '{query}' might like the product '{row['name']}' which is a '{row['description']}'. Be concise." # Added 'Be concise.'
        try:
            explanation = llm_pipeline(prompt, max_new_tokens=50, num_return_sequences=1, do_sample=True, temperature=0.7)[0]['generated_text']
            # Clean up the explanation: sometimes LLMs repeat the prompt or generate extra text
            explanation = explanation.replace(prompt, "").strip()
            # Further simple cleanup for common LLM artifacts
            if explanation.startswith(row['name'] + ". "):
                explanation = explanation[len(row['name'] + ". "):]
            if explanation.startswith("The product ") and explanation.find(" is a ") != -1:
                 explanation = explanation[explanation.find(" is a ") + len(" is a "):]
            if explanation.startswith("This product is a ") and explanation.find(" is a ") != -1:
                 explanation = explanation[explanation.find(" is a ") + len(" is a "):]
            explanation = explanation.split("\n\n")[0] # Take only the first paragraph
            recommended_products.loc[i, "explanation"] = explanation if explanation else "No specific explanation generated."
        except Exception as e:
            recommended_products.loc[i, "explanation"] = f"Error generating explanation: {e}"

    st.success(f"Found {len(recommended_products)} recommendations with explanations!")
    return recommended_products

# --- app.py content ---

def main():
    st.set_page_config(layout="wide", page_title="LLM-Enhanced E-commerce Recommender")
    st.title("🛒 LLM-Enhanced E-commerce Recommender")
    st.markdown("This system provides personalized product recommendations and human-centric explanations using Large Language Models.")

    # Check for CUDA availability (optional, for device placement)
    if "cuda_available" not in st.session_state:
        st.session_state.cuda_available = False
        try:
            import torch
            if torch.cuda.is_available():
                st.session_state.cuda_available = True
                st.info("CUDA is available! LLM will run on GPU.")
            else:
                st.info("CUDA not available. LLM will run on CPU.")
        except ImportError:
            st.info("PyTorch not installed. LLM will run on CPU.")

    # 1. Data Ingestion and Processing
    products_df, product_embeddings, embedding_model = generate_product_data_and_embeddings()

    # 2. Recommendation Core (LLM-Enhanced)
    faiss_index, llm_pipeline = initialize_recommendation_engine(product_embeddings, embedding_model)

    st.header("What are you looking for today?")
    user_query = st.text_input("Enter your shopping interest (e.g., 'comfortable office setup', 'gadgets for fitness', 'healthy cooking'):", "")

    if user_query:
        if st.button("Get Recommendations"): # Moved button inside user_query check
            with st.spinner("Generating personalized recommendations..."):
                recommendations = get_recommendations_and_explanations(
                    user_query, products_df, embedding_model, faiss_index, llm_pipeline, k=3
                )

                st.subheader("Your Personalized Recommendations:")
                for i, row in recommendations.iterrows():
                    st.markdown(f"### {row['name']} (Category: {row['category']})")
                    st.write(f"**Description:** {row['description']}")
                    st.write(f"**Price:** ${row['price']:.2f}")
                    st.info(f"**Why you might like this:** {row['explanation']}")
                    st.markdown("--- ")
    else:
        st.info("Please enter your shopping interest to get recommendations.")


if __name__ == "__main__":
    main()
