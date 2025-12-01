from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn

# --- 1. Data Models ---
class User(BaseModel):
    id: str
    name: str
    preferences: List[str]
    browsing_history: List[str]
    past_purchases: List[str]

class Product(BaseModel):
    id: str
    name: str
    category: str
    description: str
    specs: Dict[str, Any]
    reviews: List[str]

class RecommendationRequest(BaseModel):
    user_id: str

class RecommendedProduct(BaseModel):
    product_id: str
    product_name: str
    explanation: str

class RecommendationResponse(BaseModel):
    user_id: str
    recommendations: List[RecommendedProduct]

class ExplanationRequest(BaseModel):
    user_id: str
    product_id: str
    current_explanation: str
    user_query: str

class ExplanationResponse(BaseModel):
    product_id: str
    updated_explanation: str

# --- 2. Placeholder Data Storage ---
USERS_DB = {
    "user123": User(
        id="user123",
        name="Alice",
        preferences=["photography", "landscape", "travel"],
        browsing_history=["camera_lenses", "tripods", "wide_aperture_lenses"],
        past_purchases=["prime_lens_A", "backpack_B"]
    ),
    "user456": User(
        id="user456",
        name="Bob",
        preferences=["gaming", "action", "console_accessories"],
        browsing_history=["gaming_mouse", "mechanical_keyboard"],
        past_purchases=["gaming_headset_C"]
    )
}

PRODUCTS_DB = {
    "prod_XYZ_cam": Product(
        id="prod_XYZ_cam",
        name="XYZ Pro Camera",
        category="Electronics",
        description="A professional-grade mirrorless camera with exceptional low-light capabilities.",
        specs={
            "megapixels": 45,
            "sensor_type": "Full-Frame",
            "video_resolution": "8K",
            "image_stabilization": "Optical",
            "lens_mount": "XYZ-mount"
        },
        reviews=["Amazing for landscapes!", "Low light performance is unreal.", "Pairs well with prime lenses."]
    ),
    "prod_ABC_game": Product(
        id="prod_ABC_game",
        name="Adventure Quest",
        category="Video Games",
        description="An open-world RPG with stunning graphics and a gripping storyline.",
        specs={
            "platform": "PC, PS5, Xbox Series X",
            "genre": "RPG",
            "multiplayer": False
        },
        reviews=["Highly addictive!", "Great story.", "Graphics are next-gen."]
    )
}

# Simulating feature importance scores from a core recommender
# In a real system, this would come from the actual recommender model.
RECOMMENDER_SIGNALS_DB = {
    "user123": {
        "prod_XYZ_cam": {
            "high_megapixels": 0.8,
            "low_light_performance": 0.9,
            "prime_lens_compatibility": 0.7,
            "category_match": 0.85
        }
    },
    "user456": {
        "prod_ABC_game": {
            "genre_match": 0.9,
            "high_ratings": 0.8,
            "platform_match": 0.75
        }
    }
}

# --- 3. Core Recommendation Engine (Mock) ---
class MockRecommendationEngine:
    def get_recommendations(self, user_id: str) -> List[Dict[str, Any]]:
        if user_id == "user123":
            return [
                {"product_id": "prod_XYZ_cam", "signals": RECOMMENDER_SIGNALS_DB["user123"]["prod_XYZ_cam"]}
            ]
        elif user_id == "user456":
            return [
                {"product_id": "prod_ABC_game", "signals": RECOMMENDER_SIGNALS_DB["user456"]["prod_ABC_game"]}
            ]
        return []

# --- 4. LLM Connector (Mock) ---
class MockLLMConnector:
    def generate_text(self, prompt: str) -> str:
        if "XYZ Pro Camera" in prompt and "landscape photographer" in prompt:
            if "video" in prompt.lower():
                return "The XYZ Pro Camera excels in still photography, particularly for landscapes and low light. While it records stunning 8K video, its primary strengths lie in its sensor and optics for stills. For professional video production, you might consider models optimized specifically for cinema."
            return "We noticed you're an avid landscape photographer and have been looking at lenses with wide apertures. This camera's 45-megapixel sensor excels in low-light conditions, perfect for those dawn and dusk shots you love, and it pairs beautifully with the prime lens you recently purchased for crisp, detailed landscapes. Its optical image stabilization ensures sharpness even in challenging environments."
        elif "Adventure Quest" in prompt and "gaming" in prompt:
            if "multiplayer" in prompt.lower():
                return "Adventure Quest is a single-player RPG, focusing on an immersive narrative and world exploration. If you're looking for multiplayer experiences, we can suggest other titles."
            return "Given your interest in gaming, we highly recommend Adventure Quest. It's an acclaimed open-world RPG known for its stunning graphics and deep, engaging storyline, providing countless hours of immersive gameplay."
        return "I'm sorry, I cannot generate an explanation for this specific query at the moment."

# --- 5. Explanation Generation Module ---
class ExplanationGenerator:
    def __init__(self, llm_connector: MockLLMConnector):
        self.llm_connector = llm_connector

    def _get_user_context(self, user_id: str) -> Optional[User]:
        return USERS_DB.get(user_id)

    def _get_product_details(self, product_id: str) -> Optional[Product]:
        return PRODUCTS_DB.get(product_id)

    def _format_signals_for_llm(self, signals: Dict[str, float]) -> str:
        signal_phrases = []
        if signals.get("high_megapixels", 0) > 0.7: signal_phrases.append("high megapixels")
        if signals.get("low_light_performance", 0) > 0.7: signal_phrases.append("excellent low-light performance")
        if signals.get("prime_lens_compatibility", 0) > 0.6: signal_phrases.append("compatibility with prime lenses")
        if signals.get("category_match", 0) > 0.7: signal_phrases.append("a strong match to your category interests")
        if signals.get("genre_match", 0) > 0.7: signal_phrases.append("its genre matching your preferences")
        if signals.get("high_ratings", 0) > 0.7: signal_phrases.append("high user ratings")
        if signals.get("platform_match", 0) > 0.7: signal_phrases.append("platform compatibility")

        if signal_phrases:
            return f"Key signals for this recommendation include: {', '.join(signal_phrases)}."
        return ""

    def _construct_initial_prompt(self, user: User, product: Product, signals_str: str) -> str:
        user_pref_str = ", ".join(user.preferences)
        user_history_str = ", ".join(user.past_purchases + user.browsing_history)
        
        prompt = f"""
You are an intelligent e-commerce assistant providing personalized product explanations.
User Profile:
  ID: {user.id}
  Name: {user.name}
  Preferences: {user_pref_str}
  History: {user_history_str}

Product Recommended:
  ID: {product.id}
  Name: {product.name}
  Category: {product.category}
  Description: {product.description}
  Key Specs: {product.specs}
  Reviews Summary: {'. '.join(product.reviews[:2])}.

Underlying Recommendation Signals: {signals_str}

Generate a concise, persuasive explanation for why the '{product.name}' is a good fit for this user, emphasizing natural language and addressing their likely interests. Focus on the benefits that align with the user's profile and the recommendation signals.

Explanation:
"""
        return prompt

    def _construct_interactive_prompt(self, user: User, product: Product, signals_str: str, current_explanation: str, user_query: str) -> str:
        user_pref_str = ", ".join(user.preferences)
        user_history_str = ", ".join(user.past_purchases + user.browsing_history)

        prompt = f"""
You are an intelligent e-commerce assistant, continuing a conversation about a product recommendation.
User Profile:
  ID: {user.id}
  Name: {user.name}
  Preferences: {user_pref_str}
  History: {user_history_str}

Product Discussed:
  ID: {product.id}
  Name: {product.name}
  Category: {product.category}
  Description: {product.description}
  Key Specs: {product.specs}
  Reviews Summary: {'. '.join(product.reviews[:2])}.

Underlying Recommendation Signals: {signals_str}

Previous Explanation: {current_explanation}

User's Follow-up Question: {user_query}

Based on the previous explanation, product details, user profile, and the new query, provide a dynamic and personalized answer or refined explanation. Be concise and directly address the user's question.

Response:
"""
        return prompt

    def generate_initial_explanation(self, user_id: str, product_id: str, signals: Dict[str, float]) -> str:
        user = self._get_user_context(user_id)
        product = self._get_product_details(product_id)
        if not user or not product:
            return "Cannot generate explanation: User or product not found."
        
        signals_str = self._format_signals_for_llm(signals)
        prompt = self._construct_initial_prompt(user, product, signals_str)
        return self.llm_connector.generate_text(prompt)

    def generate_interactive_explanation(self, user_id: str, product_id: str, signals: Dict[str, float], current_explanation: str, user_query: str) -> str:
        user = self._get_user_context(user_id)
        product = self._get_product_details(product_id)
        if not user or not product:
            return "Cannot generate interactive explanation: User or product not found."
        
        signals_str = self._format_signals_for_llm(signals)
        prompt = self._construct_interactive_prompt(user, product, signals_str, current_explanation, user_query)
        return self.llm_connector.generate_text(prompt)

# --- 6. FastAPI Application ---
app = FastAPI(title="IntelliExplain Product Recommender API")

mock_llm = MockLLMConnector()
explanation_generator = ExplanationGenerator(llm_connector=mock_llm)
mock_recommender = MockRecommendationEngine()

@app.post("/recommend", response_model=RecommendationResponse)
async def get_product_recommendations(request: RecommendationRequest):
    user_id = request.user_id
    if user_id not in USERS_DB:
        raise HTTPException(status_code=404, detail="User not found")
    
    raw_recommendations = mock_recommender.get_recommendations(user_id)
    if not raw_recommendations:
        return RecommendationResponse(user_id=user_id, recommendations=[])
    
    processed_recommendations = []
    for rec in raw_recommendations:
        product_id = rec["product_id"]
        signals = rec["signals"]
        product_details = PRODUCTS_DB.get(product_id)
        if product_details:
            explanation = explanation_generator.generate_initial_explanation(user_id, product_id, signals)
            processed_recommendations.append(RecommendedProduct(
                product_id=product_id,
                product_name=product_details.name,
                explanation=explanation
            ))
    
    return RecommendationResponse(user_id=user_id, recommendations=processed_recommendations)

@app.post("/explain", response_model=ExplanationResponse)
async def get_interactive_explanation(request: ExplanationRequest):
    user_id = request.user_id
    product_id = request.product_id
    user_query = request.user_query
    current_explanation = request.current_explanation

    if user_id not in USERS_DB:
        raise HTTPException(status_code=404, detail="User not found")
    if product_id not in PRODUCTS_DB:
        raise HTTPException(status_code=404, detail="Product not found")
    
    # In a real system, we'd retrieve the *original* signals for this product for the user
    # For this mock, we'll assume a lookup or store them with the initial recommendation.
    # Using a simplified lookup here.
    signals = RECOMMENDER_SIGNALS_DB.get(user_id, {}).get(product_id, {})

    updated_explanation = explanation_generator.generate_interactive_explanation(
        user_id, product_id, signals, current_explanation, user_query
    )
    
    return ExplanationResponse(product_id=product_id, updated_explanation=updated_explanation)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
