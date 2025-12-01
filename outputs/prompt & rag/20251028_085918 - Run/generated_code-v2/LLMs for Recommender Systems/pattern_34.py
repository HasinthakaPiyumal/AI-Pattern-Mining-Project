import os
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# Mocking external libraries for the code generation context
# In a real scenario, these would be imported from langchain, openai etc.
class MockLLM:
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model_name = model_name

    def invoke(self, prompt: str) -> str:
        # Simulate LLM response based on prompt content
        if "few-shot" in prompt.lower() and "running shoes" in prompt.lower():
            return "Recommended products: Nike Air Zoom, Adidas Ultraboost, Brooks Ghost"
        elif "few-shot" in prompt.lower() and "novel" in prompt.lower():
            return "Recommended products: Project Hail Mary, The Midnight Library, Where the Crawdads Sing"
        elif "chain-of-thought" in prompt.lower() and "laptop" in prompt.lower():
            if "durable" in prompt.lower() and "video editing" in prompt.lower() and "$1500" in prompt.lower() and "no apple" in prompt.lower():
                return (
                    "Step 1: Identified product type: laptop.\n"
                    "Step 2: Extracted criteria: durable, video editing, under $1500, lightweight, good battery life. Exclusions: no Apple.\n"
                    "Step 3: Candidates filtered based on criteria and exclusions.\n"
                    "Recommended products: Dell XPS 15, HP Spectre x360, Lenovo ThinkPad X1 Extreme"
                )
            elif "gaming laptop" in prompt.lower() and "$2000" in prompt.lower():
                 return (
                    "Step 1: Identified product type: gaming laptop.\n"
                    "Step 2: Extracted criteria: under $2000, good graphics card, high refresh rate screen.\n"
                    "Step 3: Candidates filtered based on criteria.\n"
                    "Recommended products: Asus ROG Zephyrus G14, MSI Katana 15, Acer Predator Helios 300"
                 )
            else:
                return "Recommended products: Generic Laptop A, Generic Laptop B"
        elif "recommendation" in prompt.lower():
            return "Recommended products: Item X, Item Y, Item Z"
        return "Sorry, I couldn't generate recommendations for that."

class Product(BaseModel):
    id: str
    name: str
    category: str
    description: str
    price: float
    attributes: Dict[str, str] = Field(default_factory=dict)

class ProductCatalog:
    def __init__(self, products: List[Product]):
        self.products = products
        self._product_map = {p.id: p for p in products}

    def get_product_by_id(self, product_id: str) -> Optional[Product]:
        return self._product_map.get(product_id)

    def get_candidates(self, query: str, top_n: int = 20) -> List[Product]:
        """
        Simulates a candidate generation module.
        In a real system, this would involve a search engine (e.g., Elasticsearch),
        vector similarity search (e.g., Chroma, Faiss), or a collaborative filtering model.
        For this example, it's a simple keyword-based filter.
        """
        query_lower = query.lower()
        matching_products = []
        for product in self.products:
            if (
                query_lower in product.name.lower() or
                query_lower in product.description.lower() or
                query_lower in product.category.lower()
            ):
                matching_products.append(product)
            # Add simple attribute matching
            for attr_val in product.attributes.values():
                if query_lower in attr_val.lower():
                    matching_products.append(product)
                    break
        # Remove duplicates and return top_n
        unique_products = list({p.id: p for p in matching_products}.values())
        return unique_products[:top_n]

class IntelliShopRecommender:
    def __init__(self, llm: Any, product_catalog: ProductCatalog):
        self.llm = llm
        self.product_catalog = product_catalog

    def _format_product_for_prompt(self, product: Product) -> str:
        attrs = ", ".join([f"{k}: {v}" for k, v in product.attributes.items()])
        return f"Product Name: {product.name}, Category: {product.category}, Price: ${product.price:.2f}, Description: {product.description}, Attributes: {{{attrs}}}"

    def recommend_simple_query(self, user_query: str, num_recommendations: int = 3) -> List[Product]:
        """
        Uses zero/few-shot prompting for direct recommendations based on a simple query.
        """
        # Few-shot examples (hardcoded for simplicity, could be dynamic)
        few_shot_examples = [
            {
                "input": "I need a new pair of running shoes.",
                "output": "Recommended: Nike Air Zoom, Adidas Ultraboost, Brooks Ghost"
            },
            {
                "input": "Suggest a good novel to read.",
                "output": "Recommended: Project Hail Mary, The Midnight Library, Where the Crawdads Sing"
            },
        ]

        examples_str = "\n".join([f"User: {ex['input']}\nAssistant: {ex['output']}" for ex in few_shot_examples])

        prompt = f"""
        You are an AI product recommender for an e-commerce store.
        Provide personalized product recommendations based on user queries.
        Use the following examples for in-context learning to understand the recommendation style.

        {examples_str}

        User: {user_query}
        Assistant: Recommended:
        """

        llm_response = self.llm.invoke(prompt)
        # Parse the response to extract product names
        recommended_names_str = llm_response.split("Recommended:")[1].strip()
        recommended_names = [name.strip() for name in recommended_names_str.split(',') if name.strip()]

        # Try to map names back to actual products from the catalog (simplified)
        actual_recommendations = []
        for name in recommended_names:
            # Simple substring matching for demonstration
            for product in self.product_catalog.products:
                if name.lower() in product.name.lower():
                    actual_recommendations.append(product)
                    break
            if len(actual_recommendations) >= num_recommendations:
                break
        return actual_recommendations

    def recommend_complex_query(self, user_query: str, num_recommendations: int = 3) -> List[Product]:
        """
        Uses Chain-of-Thought prompting and candidate generation for complex queries.
        """
        # Step 1: Candidate Generation
        # Use the user_query for initial candidate fetching
        candidates = self.product_catalog.get_candidates(user_query, top_n=50)

        if not candidates:
            return []

        # Format candidates for the LLM prompt
        formatted_candidates = "\n".join([self._format_product_for_prompt(p) for p in candidates])

        # Step 2: Chain-of-Thought Prompting for Filtering and Ranking
        cot_prompt = f"""
        You are an advanced AI product recommender. Your task is to analyze a user's complex preferences and recommend the best products from a provided list of candidates.
        Follow these steps to arrive at the final recommendation:

        <CoT>
        Step 1: Understand the User's Core Need.
        Carefully read the user's query and identify the primary product category or type they are looking for.

        Step 2: Extract Key Positive Criteria.
        From the user's query, list all explicit and implicit positive attributes or features they desire (e.g., "durable", "under $1500", "good battery life").

        Step 3: Extract Key Exclusion Criteria.
        From the user's query, list any products, brands, or attributes that the user explicitly wants to avoid (e.g., "no Apple products").

        Step 4: Filter Candidates Based on Criteria.
        Review the provided list of candidate products. For each candidate, determine if it meets the positive criteria and does not violate any exclusion criteria. Explain your reasoning for including or excluding a candidate.

        Step 5: Rank Remaining Candidates.
        Based on the degree to which they satisfy the positive criteria and their overall suitability, rank the filtered candidates from most to least recommended. Justify the ranking.

        Step 6: Formulate Final Recommendation.
        Present the top {num_recommendations} recommended products clearly.
        </CoT>

        User Query: "{user_query}"

        Available Candidate Products:
        {formatted_candidates}

        Think step-by-step according to the <CoT> instructions to provide the best recommendations.
        """

        llm_response = self.llm.invoke(cot_prompt)

        # Simplified parsing for demonstration: look for "Recommended products:"
        # In a real application, a more robust parsing (e.g., regex, structured output from LLM)
        # would be needed, or the LLM could be instructed to output JSON.
        if "Recommended products:" in llm_response:
            recommendation_part = llm_response.split("Recommended products:", 1)[1].strip()
            # Attempt to extract product names, assuming they are comma-separated or newline-separated
            extracted_names = [name.strip() for name in recommendation_part.split(',') if name.strip()]
            if not extracted_names:
                 extracted_names = [name.strip() for name in recommendation_part.split('\n') if name.strip()]

            actual_recommendations = []
            for name in extracted_names:
                # Simple substring matching to map back to products
                for product in candidates:
                    if name.lower() in product.name.lower():
                        actual_recommendations.append(product)
                        break
                if len(actual_recommendations) >= num_recommendations:
                    break
            return actual_recommendations
        else:
            print("LLM did not provide a 'Recommended products:' section in CoT response. Returning top candidates directly.")
            # Fallback: if LLM doesn't format as expected, return top candidates
            return candidates[:num_recommendations]

# Example Usage
# Mock Product Data
products_data = [
    Product(id="p1", name="Nike Air Zoom Pegasus 40", category="Running Shoes", price=130.00, description="Comfortable and responsive for daily runs.", attributes={"brand": "Nike", "type": "running", "color": "black"}),
    Product(id="p2", name="Adidas Ultraboost Light", category="Running Shoes", price=190.00, description="Lightweight and energy-returning for performance.", attributes={"brand": "Adidas", "type": "running", "color": "white"}),
    Product(id="p3", name="Brooks Ghost 15", category="Running Shoes", price=140.00, description="Smooth ride and soft cushioning.", attributes={"brand": "Brooks", "type": "running", "color": "blue"}),
    Product(id="p4", name="Dell XPS 15", category="Laptop", price=1499.00, description="Powerful laptop for creative work and productivity.", attributes={"brand": "Dell", "processor": "Intel i7", "ram": "16GB", "storage": "512GB SSD", "screen_size": "15.6 inch", "weight": "1.9kg", "battery_life": "10 hours"}),
    Product(id="p5", name="HP Spectre x360 14", category="Laptop", price=1399.00, description="Convertible laptop with premium design and features.", attributes={"brand": "HP", "processor": "Intel i7", "ram": "16GB", "storage": "1TB SSD", "screen_size": "13.5 inch", "weight": "1.3kg", "battery_life": "12 hours"}),
    Product(id="p6", name="Lenovo ThinkPad X1 Extreme Gen 5", category="Laptop", price=1799.00, description="High-performance workstation laptop for demanding tasks.", attributes={"brand": "Lenovo", "processor": "Intel i9", "ram": "32GB", "storage": "1TB SSD", "screen_size": "16 inch", "weight": "1.8kg", "battery_life": "8 hours"}),
    Product(id="p7", name="Apple MacBook Air M2", category="Laptop", price=1199.00, description="Ultra-thin and powerful laptop with M2 chip.", attributes={"brand": "Apple", "processor": "Apple M2", "ram": "8GB", "storage": "256GB SSD", "screen_size": "13.6 inch", "weight": "1.24kg", "battery_life": "18 hours"}),
    Product(id="p8", name="Project Hail Mary", category="Book", price=15.00, description="Sci-fi novel by Andy Weir.", attributes={"author": "Andy Weir", "genre": "Sci-Fi"}),
    Product(id="p9", name="The Midnight Library", category="Book", price=14.00, description="Novel by Matt Haig about alternate lives.", attributes={"author": "Matt Haig", "genre": "Fiction"}),
    Product(id="p10", name="Where the Crawdads Sing", category="Book", price=13.00, description="Mystery and coming-of-age novel by Delia Owens.", attributes={"author": "Delia Owens", "genre": "Fiction"}),
    Product(id="p11", name="Asus ROG Zephyrus G14", category="Gaming Laptop", price=1699.00, description="Compact gaming laptop with powerful AMD processor and NVIDIA GPU.", attributes={"brand": "Asus", "processor": "AMD Ryzen 9", "graphics": "NVIDIA GeForce RTX 4060", "ram": "16GB", "storage": "1TB SSD", "screen_size": "14 inch", "refresh_rate": "165Hz"}),
    Product(id="p12", name="MSI Katana 15", category="Gaming Laptop", price=1399.00, description="Mid-range gaming laptop with Intel CPU and NVIDIA graphics.", attributes={"brand": "MSI", "processor": "Intel Core i7", "graphics": "NVIDIA GeForce RTX 4050", "ram": "16GB", "storage": "512GB SSD", "screen_size": "15.6 inch", "refresh_rate": "144Hz"}),
    Product(id="p13", name="Acer Predator Helios 300", category="Gaming Laptop", price=1899.00, description="High-performance gaming laptop with advanced cooling.", attributes={"brand": "Acer", "processor": "Intel Core i9", "graphics": "NVIDIA GeForce RTX 4070", "ram": "32GB", "storage": "1TB SSD", "screen_size": "17.3 inch", "refresh_rate": "240Hz"})

]
product_catalog = ProductCatalog(products_data)
mock_llm = MockLLM()
recommender = IntelliShopRecommender(llm=mock_llm, product_catalog=product_catalog)

def main():
    print("--- IntelliShop AI Product Recommender ---")

    # Simple Query Recommendation (Few-shot)
    print("\n--- Simple Query: Running Shoes ---")
    simple_query = "I need a new pair of running shoes."
    recs_simple = recommender.recommend_simple_query(simple_query)
    print(f"User Query: \"{simple_query}\"")
    if recs_simple:
        print("Recommended Products:")
        for r in recs_simple:
            print(f"- {r.name} ({r.category}) - ${r.price:.2f}")
    else:
        print("No recommendations found for this query.")

    print("\n--- Simple Query: Good Novel ---")
    simple_query_book = "Suggest a good novel to read."
    recs_simple_book = recommender.recommend_simple_query(simple_query_book)
    print(f"User Query: \"{simple_query_book}\"")
    if recs_simple_book:
        print("Recommended Products:")
        for r in recs_simple_book:
            print(f"- {r.name} ({r.category}) - ${r.price:.2f}")
    else:
        print("No recommendations found for this query.")

    # Complex Query Recommendation (Chain-of-Thought with Candidate Generation)
    print("\n--- Complex Query: Durable laptop for video editing, under $1500, no Apple ---")
    complex_query = "I need a durable laptop for video editing under $1500, preferably lightweight and with good battery life, but I already have an Apple laptop, so no more Apple products."
    recs_complex = recommender.recommend_complex_query(complex_query)
    print(f"User Query: \"{complex_query}\"")
    if recs_complex:
        print("Recommended Products:")
        for r in recs_complex:
            print(f"- {r.name} ({r.category}) - ${r.price:.2f}")
    else:
        print("No recommendations found for this complex query.")

    print("\n--- Complex Query: Gaming laptop under $2000 ---")
    complex_query_gaming = "I need a gaming laptop under $2000 with a good graphics card and high refresh rate screen."
    recs_complex_gaming = recommender.recommend_complex_query(complex_query_gaming)
    print(f"User Query: \"{complex_query_gaming}\"")
    if recs_complex_gaming:
        print("Recommended Products:")
        for r in recs_complex_gaming:
            print(f"- {r.name} ({r.category}) - ${r.price:.2f}")
    else:
        print("No recommendations found for this complex query.")

if __name__ == "__main__":
    main()