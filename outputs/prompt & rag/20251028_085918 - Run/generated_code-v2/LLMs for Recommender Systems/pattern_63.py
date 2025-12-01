import gradio as gr
import random
import json

# --- 1. Data Layer Simulation ---
# In a real application, these would be connected to databases (e.g., SQL, NoSQL)

product_catalog = {
    "prod_001": {"name": "Smartwatch X", "description": "Feature-rich smartwatch with health tracking and notifications.", "category": "Electronics", "price": 299.99, "features": ["health tracking", "notifications", "waterproof", "long battery life"]},
    "prod_002": {"name": "Organic Coffee Beans", "description": "Premium Arabica beans, medium roast, ethically sourced.", "category": "Groceries", "price": 15.50, "features": ["organic", "arabica", "medium roast", "ethically sourced"]},
    "prod_003": {"name": "Noise-Cancelling Headphones", "description": "Immersive audio experience with active noise cancellation.", "category": "Electronics", "price": 199.00, "features": ["noise cancellation", "comfortable", "high-fidelity audio"]},
    "prod_004": {"name": "Yoga Mat Pro", "description": "High-density, non-slip yoga mat for all types of practice.", "category": "Sports & Outdoors", "price": 45.00, "features": ["non-slip", "high-density", "durable", "eco-friendly"]},
    "prod_005": {"name": "Fantasy Novel Set", "description": "Epic fantasy series, 5-book collection, acclaimed author.", "category": "Books", "price": 75.00, "features": ["fantasy", "series", "bestseller", "adventure"]},
    "prod_006": {"name": "Wireless Mouse", "description": "Ergonomic wireless mouse with adjustable DPI.", "category": "Electronics", "price": 25.00, "features": ["wireless", "ergonomic", "adjustable DPI"]},
    "prod_007": {"name": "Protein Powder", "description": "Whey protein isolate, vanilla flavor, 25g protein per serving.", "category": "Health & Nutrition", "price": 40.00, "features": ["whey protein", "vanilla flavor", "high protein", "muscle recovery"]}
}

user_profiles = {
    "user_A": {"purchase_history": ["prod_001", "prod_003"], "browsing_history": ["prod_006", "prod_001"], "preferences": {"category": ["Electronics", "Books"], "price_range": "medium-high"}},
    "user_B": {"purchase_history": ["prod_002", "prod_007"], "browsing_history": ["prod_004", "prod_002"], "preferences": {"category": ["Groceries", "Health & Nutrition", "Sports & Outdoors"], "price_range": "low-medium"}}
}

def get_product_info(product_id):
    return product_catalog.get(product_id, {})

def get_user_profile(user_id):
    return user_profiles.get(user_id, {})

# --- 2. Recommendation Engine (Simplified/Simulated) ---
# This is a basic rule-based recommender for demonstration purposes.
# In a real system, this would be a more sophisticated ML model (e.g., collaborative filtering, deep learning).

def get_recommendations(user_id, num_recommendations=3):
    user_profile = get_user_profile(user_id)
    if not user_profile:
        return random.sample(list(product_catalog.keys()), num_recommendations) # Fallback to random

    recommended_product_ids = set()
    user_categories = user_profile.get("preferences", {}).get("category", [])
    user_history_ids = user_profile.get("purchase_history", []) + user_profile.get("browsing_history", [])

    # Recommend based on preferred categories
    for prod_id, prod_info in product_catalog.items():
        if prod_id not in user_history_ids and prod_info.get("category") in user_categories:
            recommended_product_ids.add(prod_id)
            if len(recommended_product_ids) >= num_recommendations:
                break

    # If not enough, add from general popular items (or just more random for this simulation)
    while len(recommended_product_ids) < num_recommendations:
        random_prod = random.choice(list(product_catalog.keys()))
        if random_prod not in user_history_ids:
            recommended_product_ids.add(random_prod)

    return list(recommended_product_ids)[:num_recommendations]

# --- 3. LLM Explainer (Simulated) ---
# This function simulates an LLM call to generate explanations.
# In a real application, you'd integrate with an actual LLM API (e.g., OpenAI, Gemini, Hugging Face).

def generate_llm_explanation(user_id, recommendations, user_feedback=None):
    user_profile = get_user_profile(user_id)
    explanation_parts = []

    explanation_parts.append(f"Hello {user_id}! Here are some personalized product recommendations for you:")

    for i, prod_id in enumerate(recommendations):
        prod_info = get_product_info(prod_id)
        if not prod_info: continue

        prod_name = prod_info.get("name", "a product")
        prod_category = prod_info.get("category", "")
        prod_features = ", ".join(prod_info.get("features", []))

        base_explanation = f"- We recommend '{prod_name}' ({prod_category}) because it offers features like {prod_features}."

        if user_profile:
            if prod_category in user_profile.get("preferences", {}).get("category", []):
                base_explanation += f" This aligns with your past interest in {prod_category} items."
            if prod_id in user_profile.get("purchase_history", []):
                base_explanation += " This is similar to something you've purchased before."
            elif prod_id in user_profile.get("browsing_history", []):
                base_explanation += " This is related to items you've recently browsed."

        explanation_parts.append(base_explanation)

    if user_feedback:
        explanation_parts.append(f"\nThank you for your feedback: '{user_feedback}'. We'll use this to refine future recommendations and explanations.")
        # In a real system, user_feedback would be processed and potentially used to update the LLM prompt or user profile.

    explanation_parts.append("\nWe hope these recommendations enhance your shopping experience!")

    # Simulate a more complex LLM behavior by slightly varying the output based on a 'thought process'
    llm_thought_process = (
        f"\n[LLM Internal Thought: User {user_id} seems interested in {', '.join(user_profile.get('preferences',{}).get('category', []))} "
        f"and has a history of interacting with {', '.join(user_profile.get('purchase_history',[]))} and {', '.join(user_profile.get('browsing_history',[]))}. "
        f"The goal is to explain {', '.join(recommendations)} based on these factors and product features. "
        f"If feedback is provided, acknowledge it and suggest future improvement.]"
    )

    return "\n".join(explanation_parts) + (llm_thought_process if 'user_A' in user_id else '') # Add thought process for user_A to show model-agnostic interpretation idea

# --- 4. Web Interface (Gradio) ---

def recommend_and_explain(user_id, user_feedback=""):
    if user_id not in user_profiles:
        return "", "Invalid User ID. Please use 'user_A' or 'user_B'."

    recommendations = get_recommendations(user_id)
    explanation = generate_llm_explanation(user_id, recommendations, user_feedback)

    rec_display = []
    for prod_id in recommendations:
        info = get_product_info(prod_id)
        rec_display.append(f"- {info.get('name', prod_id)} ({info.get('category', 'N/A')}) - ${info.get('price', 'N/A')}")

    return "\n".join(rec_display), explanation


iface = gr.Interface(
    fn=recommend_and_explain,
    inputs=[
        gr.Textbox(label="User ID (e.g., user_A, user_B)"),
        gr.Textbox(label="Your Feedback (Optional)", placeholder="e.g., 'I prefer cheaper options' or 'Tell me more about the features'")
    ],
    outputs=[
        gr.Textbox(label="Recommended Products"),
        gr.Textbox(label="Explanation by LLM")
    ],
    title="E-commerce Product Recommendations with LLM Explanations",
    description="Enter a User ID to get personalized product recommendations and understand why they were chosen. Provide feedback to refine explanations."
)

iface.launch(share=False)
