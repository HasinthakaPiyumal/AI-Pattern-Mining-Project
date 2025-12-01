from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import random

# --- 1. Data Management Layer (Mock In-memory Data) ---

PRODUCT_CATALOG = {
    "p101": {"name": "Vintage Leather Messenger Bag", "category": "Bags", "description": "A classic, durable leather bag for everyday use.", "style": "classic", "material": "leather", "price": 120},
    "p102": {"name": "Organic Cotton T-Shirt", "category": "Apparel", "description": "Soft and breathable t-shirt made from 100% organic cotton.", "style": "casual", "material": "organic cotton", "sustainable": True, "price": 30},
    "p103": {"name": "Recycled Plastic Water Bottle", "category": "Accessories", "description": "Eco-friendly water bottle made from recycled materials.", "sustainable": True, "price": 15},
    "p104": {"name": "Athletic Shorts", "category": "Apparel", "description": "Lightweight shorts perfect for running and workouts.", "style": "sporty", "price": 45},
    "p105": {"name": "Performance Socks", "category": "Apparel", "description": "Moisture-wicking socks for athletic activities.", "style": "sporty", "price": 10},
    "p106": {"name": "Graphic Tee", "category": "Apparel", "description": "Comfortable cotton tee with a unique graphic print.", "style": "casual", "price": 25},
    "p107": {"name": "Classic Blue Jeans", "category": "Apparel", "description": "Timeless denim jeans, versatile for any occasion.", "style": "casual", "price": 60},
    "p108": {"name": "Stylish Sneakers", "category": "Footwear", "description": "Fashionable and comfortable sneakers for daily wear.", "style": "casual", "price": 80},
    "p109": {"name": "Linen Short-Sleeve Button-Down Shirt", "category": "Apparel", "description": "Light and airy shirt, ideal for warm weather.", "style": "classic", "material": "linen", "price": 70},
    "p110": {"name": "Beige Chinos", "category": "Apparel", "description": "Versatile and comfortable chinos, suitable for smart-casual looks.", "style": "classic", "price": 55},
    "p111": {"name": "White Espadrilles", "category": "Footwear", "description": "Comfortable and stylish summer shoes with a rope sole.", "style": "classic", "price": 65},
    "p112": {"name": "Smartwatch for Outdoors", "category": "Electronics", "description": "GPS-enabled smartwatch with long battery life, perfect for hiking and outdoor adventures.", "tech_savvy": True, "outdoor": True, "price": 300},
    "p113": {"name": "Waterproof Hiking Boots", "category": "Footwear", "description": "Durable and waterproof boots for challenging trails.", "outdoor": True, "price": 150},
    "p114": {"name": "Portable Solar Charger", "category": "Electronics", "description": "Lightweight solar charger for phones and gadgets on the go.", "tech_savvy": True, "outdoor": True, "price": 80},
    "p115": {"name": "Ergonomic Office Chair", "category": "Furniture", "description": "Comfortable chair designed for long hours of work.", "price": 200},
    "p116": {"name": "Noise-Cancelling Headphones", "category": "Electronics", "description": "Premium headphones for immersive audio experience.", "price": 250},
}

USER_PROFILES = {
    "userA": {"preferences": ["likes sporty"], "past_purchases": ["p104", "p105"]},
    "userB": {"preferences": ["likes casual"], "past_purchases": ["p107", "p106"]},
    "newUser": {"preferences": ["likes classic styles", "prefers sustainable materials"], "past_purchases": ["p102", "p103"]},
}

# --- 2. LLM Integration Layer (Mock LLM Client) ---

class LLMClient:
    def __init__(self, model_name: str = "mock-llm-model"):
        self.model_name = model_name

    def _generate_mock_recommendations(self, prompt_type: str, query: str, num_recs: int = 3) -> List[Dict[str, Any]]:
        # This is a mock function simulating LLM output
        all_products = list(PRODUCT_CATALOG.values())
        random.shuffle(all_products)
        selected_products = []

        if "messenger bag" in query.lower():
            selected_products.extend([PRODUCT_CATALOG["p101"], PRODUCT_CATALOG["p109"], PRODUCT_CATALOG["p110"]])
        elif "sustainable" in query.lower() or "eco-friendly" in query.lower():
            selected_products.extend([p for p in all_products if p.get("sustainable")])
        elif "sporty" in query.lower():
            selected_products.extend([p for p in all_products if p.get("style") == "sporty"])
        elif "casual" in query.lower():
            selected_products.extend([p for p in all_products if p.get("style") == "casual"])
        elif "classic" in query.lower():
            selected_products.extend([p for p in all_products if p.get("style") == "classic"])
        elif "outfit for a casual summer evening" in query.lower():
             selected_products.extend([PRODUCT_CATALOG["p109"], PRODUCT_CATALOG["p110"], PRODUCT_CATALOG["p111"]]) # Linen shirt, Chinos, Espadrilles
        elif "gift for a tech-savvy friend who loves the outdoors" in query.lower():
            selected_products.extend([PRODUCT_CATALOG["p112"], PRODUCT_CATALOG["p114"], PRODUCT_CATALOG["p113"]]) # Smartwatch, Solar Charger, Hiking Boots
        elif "running shoes" in query.lower():
            # For reranking, this mock assumes candidates are passed in, so it just picks some
            return [PRODUCT_CATALOG["p104"], PRODUCT_CATALOG["p105"]]
        
        if not selected_products:
            selected_products = random.sample(all_products, min(len(all_products), num_recs))

        # Ensure unique products and limit to num_recs
        unique_products = []
        seen_names = set()
        for p in selected_products:
            if p["name"] not in seen_names:
                unique_products.append(p)
                seen_names.add(p["name"])
                if len(unique_products) == num_recs:
                    break
        
        # Fill up if not enough unique recommendations were found based on keywords
        while len(unique_products) < num_recs and len(unique_products) < len(all_products):
            new_product = random.choice(all_products)
            if new_product["name"] not in seen_names:
                unique_products.append(new_product)
                seen_names.add(new_product["name"])

        return unique_products

    def get_zero_shot_recommendations(self, prompt: str) -> List[Dict[str, Any]]:
        print(f"[LLM] Zero-shot request: {prompt}")
        # In a real scenario, call LLM API here
        # For mock, we'll parse the prompt to guess recommendations
        return self._generate_mock_recommendations("zero-shot", prompt, num_recs=5)

    def get_few_shot_recommendations(self, prompt: str) -> List[Dict[str, Any]]:
        print(f"[LLM] Few-shot request: {prompt}")
        # In a real scenario, call LLM API here
        return self._generate_mock_recommendations("few-shot", prompt, num_recs=3)

    def get_cot_recommendations(self, prompt: str) -> Dict[str, Any]:
        print(f"[LLM] CoT request: {prompt}")
        # In a real scenario, call LLM API here, which returns reasoning and then final recs
        if "casual summer evening event" in prompt.lower():
            reasoning = "A casual summer evening suggests comfort, breathability, and a touch of style. Avoid heavy fabrics. Focus on lightweight materials like linen or cotton. Colors can be bright or muted pastels. Top: A breathable, stylish top. Bottom: Comfortable chinos or tailored shorts. Shoes: Espadrilles, loafers, or stylish sneakers."
            recommendations = [PRODUCT_CATALOG["p109"], PRODUCT_CATALOG["p110"], PRODUCT_CATALOG["p111"]]
        elif "tech-savvy friend who loves the outdoors" in prompt.lower():
            reasoning = "Tech-savvy implies gadgets, durable gear, perhaps smart features. Loves outdoors suggests camping gear, hiking essentials, portable power. Combine these."
            recommendations = [PRODUCT_CATALOG["p112"], PRODUCT_CATALOG["p114"], PRODUCT_CATALOG["p113"]]
        else:
            reasoning = "Based on the complex query, breaking it down into attributes and categories leads to..."
            recommendations = self._generate_mock_recommendations("cot", prompt, num_recs=3)
        
        return {"reasoning": reasoning, "recommendations": recommendations}

# --- 3. Candidate Generation Module ---

class CandidateGenerator:
    def get_candidates(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        print(f"[CandidateGenerator] Generating candidates for: {query}")
        query_lower = query.lower()
        candidates = []
        for product_id, product_info in PRODUCT_CATALOG.items():
            if query_lower in product_info["name"].lower() or \
               query_lower in product_info["description"].lower() or \
               query_lower in product_info["category"].lower():
                candidates.append({"id": product_id, **product_info})
            if len(candidates) >= limit:
                break
        return candidates

# --- 4. Recommendation Orchestration Layer ---

class RecommenderService:
    def __init__(self, llm_client: LLMClient, candidate_generator: CandidateGenerator):
        self.llm_client = llm_client
        self.candidate_generator = candidate_generator

    def _get_user_context(self, user_id: str) -> Dict[str, Any]:
        return USER_PROFILES.get(user_id, {"preferences": [], "past_purchases": []})

    def recommend_for_new_user_or_product(self, user_id: str, product_description: Optional[str] = None) -> List[Dict[str, Any]]:
        user_context = self._get_user_context(user_id)
        
        if user_context["past_purchases"] or user_context["preferences"]:
            # Few-shot scenario for existing user or inferred preferences
            past_purchase_names = [PRODUCT_CATALOG[pid]["name"] for pid in user_context["past_purchases"] if pid in PRODUCT_CATALOG]
            prompt = f"Given these user preferences: {user_context['preferences']}, and past purchases: {past_purchase_names}, recommend 3 new items. " \
                     "\nUser A preferences: ['likes sporty'], past purchases: ['Athletic Shorts', 'Performance Socks'] -> Recommendations: ['Running Shoes', 'Sport Water Bottle']" \
                     "\nUser B preferences: ['likes casual'], past purchases: ['Classic Blue Jeans', 'Graphic Tee'] -> Recommendations: ['Stylish Sneakers', 'Denim Jacket']" \
                     f"\n[{user_id}] preferences: {user_context['preferences']}, past purchases: {past_purchase_names} -> Recommendations: "
            return self.llm_client.get_few_shot_recommendations(prompt)
        else:
            # Zero-shot for truly new users or a generic product exploration
            if product_description:
                prompt = f"Based on the description of '{product_description}' and typical related items, suggest 5 complementary items a user might be interested in. Output: [Item 1, Item 2, Item 3, Item 4, Item 5]"
            else:
                prompt = "Suggest 5 popular and versatile e-commerce products for a general user. Output: [Item 1, Item 2, Item 3, Item 4, Item 5]"
            return self.llm_client.get_zero_shot_recommendations(prompt)

    def recommend_complex_query(self, user_id: str, query: str) -> Dict[str, Any]:
        user_context = self._get_user_context(user_id)
        user_prefs_str = ", ".join(user_context["preferences"])
        past_purchases_str = ", ".join([PRODUCT_CATALOG[pid]["name"] for pid in user_context["past_purchases"] if pid in PRODUCT_CATALOG])

        prompt = f"User: {user_id}. Preferences: {user_prefs_str}. Past Purchases: {past_purchases_str}. " \
                 f"User wants: '{query}'.\nReasoning: "
        return self.llm_client.get_cot_recommendations(prompt)

    def rerank_candidates(self, user_id: str, query: str, candidate_product_ids: List[str]) -> List[Dict[str, Any]]:
        user_context = self._get_user_context(user_id)
        candidates = [{**PRODUCT_CATALOG[pid], "id": pid} for pid in candidate_product_ids if pid in PRODUCT_CATALOG]
        
        if not candidates:
            return []

        # For reranking, the LLM takes the initial candidates and refines their order
        # The prompt needs to list the candidates and the user's nuanced query
        candidate_list_str = "; ".join([f"{c['name']} ({c['description']})" for c in candidates])
        
        prompt = f"Given the user's query: '{query}', and their preferences: {user_context['preferences']}, " \
                 f"and a list of candidate products: [{candidate_list_str}]. " \
                 "Rerank these candidates to prioritize the most relevant ones. Output the reranked list as [Item 1, Item 2, ...]."
        
        print(f"[LLM] Reranking request: {prompt}")
        # In a real scenario, the LLM would return an ordered list of product names/IDs
        # For mock, we'll just randomly reorder or apply simple logic
        reranked_names = [p["name"] for p in self.llm_client._generate_mock_recommendations("rerank", query, num_recs=len(candidates))]
        
        # Map reranked names back to full product info, preserving original candidates where possible
        name_to_product = {p["name"]: p for p in candidates}
        final_reranked_products = []
        for name in reranked_names:
            if name in name_to_product:
                final_reranked_products.append(name_to_product[name])
                del name_to_product[name] # Ensure uniqueness if mock generates duplicates
        
        # Add any remaining candidates not picked by mock LLM (if list was shorter)
        for product_id in candidate_product_ids:
            if PRODUCT_CATALOG[product_id]["name"] not in [p["name"] for p in final_reranked_products]:
                 final_reranked_products.append({**PRODUCT_CATALOG[product_id], "id": product_id})
        
        return final_reranked_products

# --- FastAPI Application ---

app = FastAPI(
    title="SPReE: Smart Product Recommender for E-commerce",
    description="Leveraging LLMs for direct, in-context, and Chain-of-Thought recommendations."
)

# Initialize services
llm_client_instance = LLMClient()
candidate_generator_instance = CandidateGenerator()
recommender_service_instance = RecommenderService(llm_client_instance, candidate_generator_instance)

# Pydantic models for request bodies
class NewUserRecommendationRequest(BaseModel):
    user_id: str
    product_description: Optional[str] = None

class ComplexQueryRecommendationRequest(BaseModel):
    user_id: str
    query: str

class RerankCandidatesRequest(BaseModel):
    user_id: str
    query: str
    candidate_product_ids: List[str]

class ProductResponse(BaseModel):
    id: str
    name: str
    category: str
    description: str
    # Add other fields as necessary from PRODUCT_CATALOG
    style: Optional[str] = None
    material: Optional[str] = None
    sustainable: Optional[bool] = None
    price: Optional[int] = None
    tech_savvy: Optional[bool] = None
    outdoor: Optional[bool] = None

class CoTResponse(BaseModel):
    reasoning: str
    recommendations: List[ProductResponse]

@app.post("/recommend/new_user", response_model=List[ProductResponse], summary="Zero/Few-shot recommendations for new users or products")
async def recommend_for_new_user_endpoint(request: NewUserRecommendationRequest):
    """Provides initial product recommendations for new users or based on a new product description, utilizing zero-shot or few-shot prompting strategies."""
    recommendations = recommender_service_instance.recommend_for_new_user_or_product(request.user_id, request.product_description)
    return recommendations

@app.post("/recommend/complex_query", response_model=CoTResponse, summary="Chain-of-Thought recommendations for complex user queries")
async def recommend_complex_query_endpoint(request: ComplexQueryRecommendationRequest):
    """Generates recommendations for complex queries (e.g., outfit building, gift suggestions) by employing Chain-of-Thought reasoning."""
    recommendation_data = recommender_service_instance.recommend_complex_query(request.user_id, request.query)
    return recommendation_data

@app.post("/recommend/rerank", response_model=List[ProductResponse], summary="LLM-based reranking of candidate products")
async def rerank_candidates_endpoint(request: RerankCandidatesRequest):
    """Takes a list of candidate product IDs and a nuanced user query, then uses an LLM to rerank them based on detailed preferences."""
    if not request.candidate_product_ids:
        raise HTTPException(status_code=400, detail="Candidate product IDs cannot be empty.")
    
    # In a real system, candidate_generator_instance.get_candidates might be called here
    # if the candidates were not provided directly.
    # For this example, we assume candidate_product_ids are already available.
    
    reranked_products = recommender_service_instance.rerank_candidates(request.user_id, request.query, request.candidate_product_ids)
    return reranked_products
