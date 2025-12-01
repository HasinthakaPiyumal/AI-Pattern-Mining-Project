from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.chains import LLMChain, SequentialChain
from langchain_core.language_models import LLM
from typing import Any, List, Dict
import json

# --- 1. Mock Product Database ---
mock_products_db = [
    {
        "id": "p1",
        "name": "Stylish Blue Denim Jeans",
        "category": "Apparel",
        "attributes": {"color": "blue", "size": ["S", "M", "L"], "material": "denim"},
        "description": "Classic blue denim jeans, perfect for everyday wear. Made from high-quality stretch denim for comfort.",
        "price": 49.99
    },
    {
        "id": "p2",
        "name": "Red Cotton T-Shirt",
        "category": "Apparel",
        "attributes": {"color": "red", "size": ["M", "L", "XL"], "material": "cotton"},
        "description": "Comfortable red cotton t-shirt, breathable and soft. Ideal for casual outings.",
        "price": 19.99
    },
    {
        "id": "p3",
        "name": "Wireless Bluetooth Headphones",
        "category": "Electronics",
        "attributes": {"color": "black", "connectivity": "bluetooth", "feature": "noise-cancelling"},
        "description": "Immersive audio experience with these wireless noise-cancelling headphones. Long battery life and comfortable fit.",
        "price": 129.99
    },
    {
        "id": "p4",
        "name": "Ergonomic Office Chair",
        "category": "Furniture",
        "attributes": {"color": "grey", "material": "mesh", "feature": "adjustable lumbar"},
        "description": "Boost your productivity with this ergonomic office chair featuring adjustable lumbar support and breathable mesh.",
        "price": 249.00
    },
    {
        "id": "p5",
        "name": "Vintage Leather Wallet",
        "category": "Accessories",
        "attributes": {"color": "brown", "material": "leather", "style": "vintage"},
        "description": "A timeless vintage leather wallet with multiple card slots and a coin pocket. Durable and stylish.",
        "price": 39.50
    },
]

# --- 2. LLM Wrapper (Mock Implementation) ---
class MockLLM(LLM):
    response_map: Dict[str, str]

    @property
    def _llm_type(self) -> str:
        return "mock_llm"

    def _call(self, prompt: str, stop: List[str] = None) -> str:
        # Simple mock logic: look for keywords in prompt and return pre-defined response
        if "EXTRACT_PREFERENCES" in prompt:
            return json.dumps({"category": "Apparel", "keywords": ["jeans", "blue", "denim"], "intent": "browse"})
        elif "FILTER_PRODUCTS" in prompt:
            # Simulate filtering based on extracted preferences
            return json.dumps([
                {"id": "p1", "name": "Stylish Blue Denim Jeans", "price": 49.99, "description": "Classic blue denim jeans, perfect for everyday wear."},
                {"id": "p2", "name": "Red Cotton T-Shirt", "price": 19.99, "description": "Comfortable red cotton t-shirt, breathable and soft."}
            ])
        elif "GENERATE_RECOMMENDATION" in prompt:
            return "Based on your interest in apparel, we think you'll love these: Stylish Blue Denim Jeans and Red Cotton T-Shirt. They are comfortable and perfect for your style!"
        elif "ENHANCE_DESCRIPTION" in prompt:
            if "Stylish Blue Denim Jeans" in prompt:
                return "Elevate your casual wardrobe with our premium **Stylish Blue Denim Jeans**. Crafted from high-quality stretch denim, these jeans offer unparalleled comfort and a flattering fit. Their versatile blue hue makes them a perfect match for any top, ensuring you look effortlessly chic all day long. A must-have staple for your everyday adventures!"
            elif "Red Cotton T-Shirt" in prompt:
                return "Experience ultimate comfort with our vibrant **Red Cotton T-Shirt**. Made from 100% breathable cotton, it's incredibly soft against your skin, making it ideal for relaxing at home or stepping out in style. Its classic cut and bold red color add a pop of personality to your casual ensemble. Discover your new favorite go-to tee!"
        return "Mock LLM response."

    def _identifying_params(self) -> Dict[str, Any]:
        return {"response_map": self.response_map}

# Initialize the mock LLM
mock_llm = MockLLM(response_map={})

# --- 3. Prompt Templates ---

# Prompt 1: Intent & Preference Extraction
preference_extraction_template = """
As an e-commerce assistant, analyze the user's input to extract their preferences and intent.
User Input: {user_input}

Extract the following as a JSON object:
{{ "category": "string", "keywords": ["string"], "intent": "string" (e.g., "browse", "buy", "compare") }}

Example:
User Input: I'm looking for blue denim jeans.
Output: {{ "category": "Apparel", "keywords": ["blue", "denim", "jeans"], "intent": "browse" }}

Now, for the given User Input, extract the preferences. Always include the 'EXTRACT_PREFERENCES' tag.
EXTRACT_PREFERENCES: {{user_input}}
"""
preference_extraction_prompt = PromptTemplate(
    template=preference_extraction_template,
    input_variables=["user_input"],
)
preference_extraction_chain = LLMChain(
    llm=mock_llm,
    prompt=preference_extraction_prompt,
    output_parser=JsonOutputParser(),
    output_key="extracted_preferences",
)


# Prompt 2: Product Filtering & Selection
product_filtering_template = """
Given the following extracted preferences and available products, filter and select the most relevant products.
Extracted Preferences: {extracted_preferences}
Available Products: {available_products}

Return a JSON array of up to 2 product objects (id, name, price, description) that best match the preferences.

Example:
Extracted Preferences: {{ "category": "Apparel", "keywords": ["blue", "jeans"], "intent": "browse" }}
Available Products: [
  {{"id": "p1", "name": "Blue Jeans"}},
  {{"id": "p2", "name": "Red Shirt"}}
]
Output: [
  {{"id": "p1", "name": "Blue Jeans", "price": 50.00, "description": "Comfy blue jeans"}}
]

Now, filter the products. Always include the 'FILTER_PRODUCTS' tag.
FILTER_PRODUCTS: Extracted Preferences: {extracted_preferences}, Available Products: {available_products}
"""
product_filtering_prompt = PromptTemplate(
    template=product_filtering_template,
    input_variables=["extracted_preferences", "available_products"],
)
product_filtering_chain = LLMChain(
    llm=mock_llm,
    prompt=product_filtering_prompt,
    output_parser=JsonOutputParser(),
    output_key="filtered_products",
)


# Prompt 3: Personalized Recommendation Message Generation
recommendation_message_template = """
Based on the user's preferences and the selected products, generate a friendly and personalized recommendation message.
User Preferences: {extracted_preferences}
Recommended Products: {filtered_products}

Generate a recommendation message. Always include the 'GENERATE_RECOMMENDATION' tag.
GENERATE_RECOMMENDATION: User Preferences: {extracted_preferences}, Recommended Products: {filtered_products}
"""
recommendation_message_prompt = PromptTemplate(
    template=recommendation_message_template,
    input_variables=["extracted_preferences", "filtered_products"],
)
recommendation_message_chain = LLMChain(
    llm=mock_llm,
    prompt=recommendation_message_prompt,
    output_key="recommendation_message",
)


# Prompt 4: Dynamic Description Enhancement
description_enhancement_template = """
Enhance the following product description based on the user's context/preferences.
Original Description: {original_description}
User Context/Preferences: {user_context}

Return the enhanced description.

Enhance the description. Always include the 'ENHANCE_DESCRIPTION' tag.
ENHANCE_DESCRIPTION: Original Description: {original_description}, User Context/Preferences: {user_context}
"""
description_enhancement_prompt = PromptTemplate(
    template=description_enhancement_template,
    input_variables=["original_description", "user_context"],
)
description_enhancement_chain = LLMChain(
    llm=mock_llm,
    prompt=description_enhancement_prompt,
    output_key="enhanced_description",
)


# --- 4. Prompt Chain Orchestration (LangChain SequentialChain) ---

# Recommendation Generation Chain
recommendation_generation_chain = SequentialChain(
    chains=[
        preference_extraction_chain,
        product_filtering_chain,
        recommendation_message_chain
    ],
    input_variables=["user_input", "available_products"],
    output_variables=["recommendation_message", "filtered_products"],
    verbose=True
)

# --- Flow Demonstration ---
def run_recommender(user_query: str):
    print(f"\n--- Running Recommender for: '{user_query}' ---")
    
    # Convert mock_products_db to a string for the LLM prompt
    products_str = json.dumps(mock_products_db)
    
    # Invoke the recommendation chain
    recommendation_output = recommendation_generation_chain.invoke({
        "user_input": user_query,
        "available_products": products_str
    })
    
    print("\nRecommendation Output:")
    print(f"Message: {recommendation_output['recommendation_message']}")
    print(f"Filtered Products: {json.dumps(recommendation_output['filtered_products'], indent=2)}")

    print("\n--- Enhancing Descriptions for Recommended Products ---")
    enhanced_products = []
    for product in recommendation_output['filtered_products']:
        user_context_for_enhancement = recommendation_output['extracted_preferences'] # Reusing extracted preferences as context
        enhanced_desc_output = description_enhancement_chain.invoke({
            "original_description": product['description'],
            "user_context": json.dumps(user_context_for_enhancement)
        })
        product['enhanced_description'] = enhanced_desc_output['enhanced_description']
        enhanced_products.append(product)
    
    print("\nEnhanced Products:")
    for product in enhanced_products:
        print(f"Product: {product['name']}")
        print(f"  Original Description: {product['description']}")
        print(f"  Enhanced Description: {product['enhanced_description']}")
        print("---")

if __name__ == "__main__":
    run_recommender("I need some new clothes, especially blue jeans.")
    print("\n" + "="*50 + "\n")
    run_recommender("Looking for a comfortable red t-shirt.")
