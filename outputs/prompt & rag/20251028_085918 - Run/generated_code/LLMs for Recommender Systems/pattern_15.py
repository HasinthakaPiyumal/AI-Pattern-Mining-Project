
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import asyncio

# --- 1. Data Layer (Simulated In-Memory Databases) ---

products_db: Dict[str, Dict[str, Any]] = {
    "p101": {"id": "p101", "name": "Hiking Boots", "category": "Footwear", "price": 120.0, "attributes": ["waterproof", "durable", "grip"], "description": "Sturdy waterproof hiking boots for all terrains."},
    "p102": {"id": "p102", "name": "Lightweight Tent", "category": "Camping Gear", "price": 250.0, "attributes": ["ultralight", "easy setup", "3-person"], "description": "An ultralight tent perfect for backpacking."},
    "p103": {"id": "p103", "name": "Running Shoes", "category": "Footwear", "price": 90.0, "attributes": ["cushioned", "breathable", "road running"], "description": "Comfortable running shoes for daily jogs."},
    "p104": {"id": "p104", "name": "Sleeping Bag", "category": "Camping Gear", "price": 150.0, "attributes": ["warm", "compact", "winter"], "description": "Warm sleeping bag for cold weather camping."},
    "p105": {"id": "p105", "name": "Water Bottle", "category": "Accessories", "price": 25.0, "attributes": ["insulated", "stainless steel"], "description": "Insulated stainless steel water bottle."},
    "p106": {"id": "p106", "name": "Trekking Poles", "category": "Hiking Gear", "price": 70.0, "attributes": ["adjustable", "lightweight", "carbon fiber"], "description": "Adjustable carbon fiber trekking poles."},
    "p107": {"id": "p107", "name": "Smartwatch", "category": "Electronics", "price": 299.0, "attributes": ["heart rate monitor", "gps", "water resistant"], "description": "Advanced smartwatch with health tracking and GPS."},
    "p108": {"id": "p108", "name": "Yoga Mat", "category": "Fitness", "price": 45.0, "attributes": ["non-slip", "eco-friendly", "thick"], "description": "Premium non-slip yoga mat for all practices."}
}

user_interactions_db: Dict[str, Dict[str, Any]] = {
    "user1": {"history": ["p101", "p102"], "likes": ["Hiking", "Camping"], "preferences": {"budget": "medium", "style": "rugged"}},
    "user2": {"history": ["p103", "p105"], "likes": ["Running", "Fitness"], "preferences": {"budget": "low", "style": "modern"}},
    "user3": {"history": ["p107", "p108"], "likes": ["Tech", "Wellness"], "preferences": {"budget": "high", "style": "sleek"}}
}

# --- Pydantic Models ---

class Product(BaseModel):
    id: str
    name: str
    category: str
    price: float
    attributes: List[str]
    description: str

class RecommendationRequest(BaseModel):
    user_id: str
    limit: int = 5

class RecommendationResponse(BaseModel):
    user_id: str
    recommendations: List[Product]
    explanation: Optional[str] = None

class ProductEnrichmentRequest(BaseModel):
    product_id: str
    name: str
    attributes: List[str]
    current_description: Optional[str] = None

class ProductEnrichmentResponse(BaseModel):
    product_id: str
    original_description: str
    llm_generated_description: str
    llm_categorization: str

class ChatRequest(BaseModel):
    user_id: str
    message: str
    conversation_history: Optional[List[Dict[str, str]]] = None

class ChatResponse(BaseModel):
    user_id: str
    response: str
    updated_conversation_history: List[Dict[str, str]]

class ExplainRecommendationRequest(BaseModel):
    user_id: str
    product_id: str

class ExplainRecommendationResponse(BaseModel):
    user_id: str
    product_id: str
    explanation: str

# --- 2. Core Recommender System (Simple Rule-Based) ---

class RecommendationEngine:
    def get_recommendations(self, user_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        user_data = user_interactions_db.get(user_id)
        if not user_data:
            # Default to popular items if user has no history
            return list(products_db.values())[:limit]

        user_likes = user_data.get("likes", [])
        user_history_ids = set(user_data.get("history", []))
        user_preferences = user_data.get("preferences", {})

        candidate_products = []
        for product_id, product in products_db.items():
            if product_id not in user_history_ids:
                # Simple content-based filtering based on categories and attributes
                score = 0
                for like in user_likes:
                    if like.lower() in product["category"].lower():
                        score += 2
                    if any(like.lower() in attr.lower() for attr in product["attributes"]):
                        score += 1
                
                # Consider preferences (e.g., budget)
                if "budget" in user_preferences:
                    if user_preferences["budget"] == "low" and product["price"] <= 100:
                        score += 1
                    elif user_preferences["budget"] == "medium" and 50 < product["price"] <= 200:
                        score += 1
                    elif user_preferences["budget"] == "high" and product["price"] > 150:
                        score += 1

                if score > 0:
                    candidate_products.append((score, product))

        candidate_products.sort(key=lambda x: x[0], reverse=True)
        return [prod for _, prod in candidate_products[:limit]]

# --- 3. LLM Integration Layer (Simulated with placeholder functions) ---

class LLMClient:
    async def _call_llm(self, prompt: str) -> str:
        # Simulate an asynchronous API call to an LLM
        await asyncio.sleep(0.5) # Simulate network latency
        # In a real application, this would call OpenAI, Gemini, Llama, etc.
        print(f"[LLM CALL SIMULATED] Prompt: {prompt[:100]}...")
        return f"LLM_RESPONSE: {prompt}"

    async def generate_description(self, name: str, attributes: List[str], current_description: Optional[str] = None) -> str:
        base_prompt = f"Generate a compelling product description for '{name}' with attributes: {', '.join(attributes)}. "
        if current_description:
            base_prompt += f"The current description is: '{current_description}'. Improve upon it." 
        else:
            base_prompt += "Make it engaging and highlight key features."
        response = await self._call_llm(base_prompt)
        return response.replace(f"LLM_RESPONSE: {base_prompt}", f"A wonderfully crafted description for {name} highlighting {', '.join(attributes)}. This is an improvement based on '{current_description}' if provided.")

    async def categorize_product(self, name: str, description: str) -> str:
        prompt = f"Categorize the product '{name}' with description '{description}' into a single, appropriate e-commerce category (e.g., 'Electronics', 'Footwear', 'Camping Gear')."
        response = await self._call_llm(prompt)
        return response.replace(f"LLM_RESPONSE: {prompt}", "Simulated Category: " + description.split()[0].capitalize())

    async def explain_recommendation(self, user_id: str, product_name: str, user_preferences: Dict[str, Any]) -> str:
        prompt = f"Given user preferences {user_preferences}, explain why '{product_name}' was recommended. Focus on alignment with user's likes and budget. Keep it concise and personalized."
        response = await self._call_llm(prompt)
        return response.replace(f"LLM_RESPONSE: {prompt}", f"Based on your interest in {', '.join(user_preferences.get('likes', ['outdoor activities']))} and your {user_preferences.get('budget', 'medium')} budget, we think you'll love the {product_name} because it aligns perfectly with those values.")

    async def process_complex_query(self, user_query: str, user_id: str) -> List[Dict[str, Any]]:
        # In a real scenario, this would parse the query into filters/actions
        prompt = f"Translate the user query '{user_query}' for user '{user_id}' into a list of product filters. Example: 'Show me shoes good for hiking in wet weather and budget-friendly' -> [['category', 'Footwear'], ['attribute', 'waterproof'], ['attribute', 'hiking'], ['price_max', 100]]. Return as JSON list of lists."
        response = await self._call_llm(prompt)
        
        # Simulate parsing the LLM response into filters
        mock_filters = []
        if "shoes" in user_query.lower():
            mock_filters.append(["category", "Footwear"])
        if "hiking" in user_query.lower():
            mock_filters.append(["attribute", "hiking"])
        if "wet weather" in user_query.lower() or "waterproof" in user_query.lower():
            mock_filters.append(["attribute", "waterproof"])
        if "budget-friendly" in user_query.lower() or "cheap" in user_query.lower():
            mock_filters.append(["price_max", 100])
        
        # Apply filters to products_db (simple simulation)
        filtered_products = []
        for product in products_db.values():
            match = True
            for filter_type, filter_value in mock_filters:
                if filter_type == "category" and product["category"].lower() != filter_value.lower():
                    match = False
                    break
                if filter_type == "attribute" and filter_value.lower() not in [attr.lower() for attr in product["attributes"]]:
                    match = False
                    break
                if filter_type == "price_max" and product["price"] > filter_value:
                    match = False
                    break
            if match:
                filtered_products.append(product)
        return filtered_products


    async def get_chatbot_response(self, user_id: str, message: str, conversation_history: List[Dict[str, str]]) -> Dict[str, Any]:
        history_str = "\n".join([f"{h['role']}: {h['content']}" for h in conversation_history])
        prompt = f"The following is a conversation between a user and an e-commerce assistant. User ID: {user_id}.\n{history_str}\nUser: {message}\nAssistant:"
        response = await self._call_llm(prompt)
        
        # Simulate generating a conversational response
        simulated_llm_response = f"Hello {user_id}! I received your message about '{message}'. How can I help you find something specific today?"
        if "recommend" in message.lower():
            simulated_llm_response = f"I can help with recommendations! What kind of products are you looking for, {user_id}?"
        elif "price" in message.lower():
            simulated_llm_response = f"Prices vary, but I can help you find budget-friendly options. What product are you interested in?"

        new_history = conversation_history + [{"role": "user", "content": message}, {"role": "assistant", "content": simulated_llm_response}]
        return {"response": simulated_llm_response, "updated_conversation_history": new_history}


# --- FastAPI Application --- 

app = FastAPI(title="LLM-Enhanced E-commerce Recommender")

recommender_engine = RecommendationEngine()
llm_client = LLMClient()

@app.get("/", summary="Root")
async def read_root():
    return {"message": "Welcome to the LLM-Enhanced E-commerce Recommender API!"}

@app.post("/recommendations/{user_id}", response_model=RecommendationResponse, summary="Get Personalized Product Recommendations")
async def get_personalized_recommendations(user_id: str, request: RecommendationRequest):
    if user_id not in user_interactions_db:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found.")
    
    raw_recommendations = recommender_engine.get_recommendations(user_id, request.limit)
    recommended_products = [Product(**p) for p in raw_recommendations]

    # Simulate LLM-generated explanation for the top recommendation
    explanation = None
    if recommended_products:
        user_data = user_interactions_db.get(user_id, {})
        explanation = await llm_client.explain_recommendation(
            user_id,
            recommended_products[0].name,
            user_data.get("preferences", {})
        )

    return RecommendationResponse(
        user_id=user_id,
        recommendations=recommended_products,
        explanation=explanation
    )

@app.post("/product/enrich", response_model=ProductEnrichmentResponse, summary="Enrich Product Data using LLM")
async def enrich_product_data(request: ProductEnrichmentRequest):
    product_id = request.product_id
    original_product = products_db.get(product_id)

    if not original_product:
        raise HTTPException(status_code=404, detail=f"Product {product_id} not found.")

    # Simulate description generation and categorization
    llm_generated_description = await llm_client.generate_description(
        request.name, request.attributes, request.current_description
    )
    llm_categorization = await llm_client.categorize_product(
        request.name, llm_generated_description
    )

    # Update the product in the database (for demonstration)
    products_db[product_id]["description"] = llm_generated_description
    products_db[product_id]["category"] = llm_categorization.replace("Simulated Category: ", "") # Clean up simulated prefix

    return ProductEnrichmentResponse(
        product_id=product_id,
        original_description=original_product["description"],
        llm_generated_description=llm_generated_description,
        llm_categorization=llm_categorization
    )

@app.post("/chat", response_model=ChatResponse, summary="Engage in Conversational Product Discovery/Support")
async def chat_with_assistant(request: ChatRequest):
    llm_response_data = await llm_client.get_chatbot_response(
        request.user_id, request.message, request.conversation_history or []
    )
    return ChatResponse(
        user_id=request.user_id,
        response=llm_response_data["response"],
        updated_conversation_history=llm_response_data["updated_conversation_history"]
    )

@app.post("/product/{product_id}/explain/{user_id}", response_model=ExplainRecommendationResponse, summary="Get Explanation for a Specific Product Recommendation")
async def explain_specific_recommendation(product_id: str, user_id: str):
    product = products_db.get(product_id)
    user_data = user_interactions_db.get(user_id)

    if not product:
        raise HTTPException(status_code=404, detail=f"Product {product_id} not found.")
    if not user_data:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found.")
    
    explanation = await llm_client.explain_recommendation(
        user_id, product["name"], user_data.get("preferences", {})
    )
    return ExplainRecommendationResponse(
        user_id=user_id,
        product_id=product_id,
        explanation=explanation
    )

@app.post("/query-products/{user_id}", response_model=RecommendationResponse, summary="Process Complex Natural Language Product Queries")
async def query_products_with_llm(user_id: str, query: str):
    if user_id not in user_interactions_db:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found.")
    
    # LLM processes query and returns filtered products
    filtered_products_raw = await llm_client.process_complex_query(query, user_id)
    filtered_products = [Product(**p) for p in filtered_products_raw]
    
    explanation = f"Products matching your complex query: '{query}'. The LLM processed your request to find these items."

    return RecommendationResponse(
        user_id=user_id,
        recommendations=filtered_products,
        explanation=explanation
    )


# --- Run the FastAPI application ---

if __name__ == "__main__":
    # To run this file, execute: uvicorn ecommerce_llm_recommender:app --reload
    # Or simply run this script as: python ecommerce_llm_recommender.py
    print("\n\nTo run this FastAPI application, use the command: uvicorn ecommerce_llm_recommender:app --reload")
    print("You can also run this script directly for a basic start.\n\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
