import os
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# Set your OpenAI API key as an environment variable or directly here
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# --- Simulated Data ---
product_catalog = {
    "SKU001": {"name": "Laptop Pro", "category": "Electronics", "price": 1200},
    "SKU002": {"name": "Gaming Mouse X", "category": "Electronics", "price": 75},
    "SKU003": {"name": "Ergonomic Keyboard", "category": "Electronics", "price": 100},
    "SKU004": {"name": "Smartwatch Series 5", "category": "Wearables", "price": 350},
    "SKU005": {"name": "Noise-Cancelling Headphones", "category": "Audio", "price": 250},
    "SKU006": {"name": "Webcam HD", "category": "Electronics", "price": 60},
    "SKU007": {"name": "Monitor 4K", "category": "Electronics", "price": 400},
    "SKU008": {"name": "External SSD 1TB", "category": "Storage", "price": 150},
    "SKU009": {"name": "Wireless Charger", "category": "Accessories", "price": 30},
    "SKU010": {"name": "Backpack Laptop", "category": "Bags", "price": 80},
}

user_data = {
    "user123": {
        "browsing_history": ["SKU001", "SKU007", "SKU002"],  # Laptop Pro, Monitor 4K, Gaming Mouse X
        "purchase_history": ["SKU001", "SKU005"],  # Laptop Pro, Noise-Cancelling Headphones
        "loyalty_status": "Gold"
    }
}

# --- LLM Initialization ---
llm = ChatOpenAI(temperature=0.7, model_name="gpt-4o") # or gpt-3.5-turbo

# --- Prompt Templates ---

# Prompt 1: Initial Product Recommendation
initial_recommendation_template = PromptTemplate(
    input_variables=["user_browsing_history", "user_purchase_history"],
    template=(
        "Based on the user's browsing history ({user_browsing_history}) "
        "and purchase history ({user_purchase_history}), suggest 3-5 initial product recommendations from the product catalog. "
        "List the SKUs and a brief reason. Format as: SKU: <Reason>\nSKU: <Reason>\n...\nRecommendations:"
    )
)

# Prompt 2: Refined Recommendations
refined_recommendation_template = PromptTemplate(
    input_variables=["initial_recommendations", "realtime_context"],
    template=(
        "Given these initial product recommendations:\n{initial_recommendations}\n\n" 
        "And the real-time context (e.g., items currently in cart or recently viewed): {realtime_context}.\n\n" 
        "Refine these recommendations to be more relevant. Prioritize items that complement the real-time context. "
        "Provide 3 refined recommendations with SKUs and a brief reason. Format as: SKU: <Reason>\nSKU: <Reason>\n...\nRefined Recommendations:"
    )
)

# Prompt 3: Upsell/Cross-sell Opportunities
upsell_cross_sell_template = PromptTemplate(
    input_variables=["refined_recommendations", "product_catalog"],
    template=(
        "Considering the refined product recommendations:\n{refined_recommendations}\n\n" 
        "And the available product catalog: {product_catalog}.\n\n" 
        "Suggest 2-3 upsell or cross-sell opportunities for the user based on the refined recommendations. "
        "For each suggestion, state if it's an upsell or cross-sell and why. Format as: <SKU> (<Type>): <Reason>\n<SKU> (<Type>): <Reason>\n...\nUpsell/Cross-sell Suggestions:"
    )
)

# Prompt 4: Personalized Offers Generation
personalized_offers_template = PromptTemplate(
    input_variables=["upsell_cross_sell_suggestions", "user_loyalty_status"],
    template=(
        "Based on the following upsell/cross-sell suggestions:\n{upsell_cross_sell_suggestions}\n\n" 
        "And the user's loyalty status: {user_loyalty_status}.\n\n" 
        "Generate a short, personalized promotional offer or discount for the user. "
        "Keep it enticing and relevant to the suggestions and their loyalty status.\nPersonalized Offer:"
    )
)

# --- Create LLM Chains ---

chain_1 = LLMChain(llm=llm, prompt=initial_recommendation_template, output_key="initial_recommendations", verbose=True)
chain_2 = LLMChain(llm=llm, prompt=refined_recommendation_template, output_key="refined_recommendations", verbose=True)
chain_3 = LLMChain(llm=llm, prompt=upsell_cross_sell_template, output_key="upsell_cross_sell_suggestions", verbose=True)
chain_4 = LLMChain(llm=llm, prompt=personalized_offers_template, output_key="personalized_offer", verbose=True)

# --- Combine Chains into a Sequential Chain ---

overall_chain = SequentialChain(
    chains=[chain_1, chain_2, chain_3, chain_4],
    input_variables=["user_browsing_history", "user_purchase_history", "realtime_context", "product_catalog", "user_loyalty_status"],
    output_variables=["initial_recommendations", "refined_recommendations", "upsell_cross_sell_suggestions", "personalized_offer"],
    verbose=True
)

# --- Main Execution --- 
if __name__ == "__main__":
    # Example User Input
    current_user_id = "user123"
    current_user_data = user_data[current_user_id]
    realtime_context = "SKU006 in cart, recently viewed SKU008" # Webcam HD, External SSD 1TB

    inputs = {
        "user_browsing_history": ", ".join(current_user_data["browsing_history"]),
        "user_purchase_history": ", ".join(current_user_data["purchase_history"]),
        "realtime_context": realtime_context,
        "product_catalog": str(product_catalog), # Pass as string for easier prompt injection
        "user_loyalty_status": current_user_data["loyalty_status"]
    }

    print(f"\n--- Running Recommendation System for {current_user_id} ---")
    response = overall_chain.invoke(inputs)

    print("\n--- Final Output ---")
    print(f"Initial Recommendations: {response['initial_recommendations']}")
    print(f"Refined Recommendations: {response['refined_recommendations']}")
    print(f"Upsell/Cross-sell Suggestions: {response['upsell_cross_sell_suggestions']}")
    print(f"Personalized Offer: {response['personalized_offer']}")
