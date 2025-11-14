import base64
import uuid
from typing import Dict, List, Any, Optional, Tuple

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI(
    title="Multi-modal E-commerce Personal Shopper AI",
    description="An AI assistant that understands diverse shopping queries via text, voice, and images."
)

# In-memory session storage for dialogue management and user preferences
sessions: Dict[str, Dict[str, Any]] = {}

class ChatRequestText(BaseModel):
    session_id: Optional[str] = None
    text_input: str

class ChatResponse(BaseModel):
    session_id: str
    response: str
    products_found: List[Dict[str, Any]] = []
    user_preferences: Dict[str, Any]

# --- Mock External Services (ASR, Image Analysis, Translation, Product Search) ---

async def _speech_to_text(audio_data: bytes) -> str:
    """Mocks a Speech-to-Text service. In a real app, this would use a library like SpeechRecognition or a cloud API."""
    # Simulate some processing time and return a hardcoded text for demonstration
    # For a real implementation, you'd decode audio and use an ASR model
    print(f"[MOCK] Processing audio data of size: {len(audio_data)} bytes")
    if b"red dress" in audio_data.lower(): # Very basic keyword check for demo
        return "Show me a red summer dress under $50"
    elif b"hiking gift" in audio_data.lower():
        return "Find a gift for my sister who likes hiking"
    return "User said something about shopping."

async def _analyze_image(image_data: bytes, text_query: Optional[str] = None) -> Dict[str, Any]:
    """Mocks an Image Analysis service (e.g., CLIP, object detection)."""
    # Simulate image processing and feature extraction
    print(f"[MOCK] Analyzing image data of size: {len(image_data)} bytes with query: {text_query}")
    # In a real scenario, this would return extracted features, objects, brands, etc.
    if b"shoe" in image_data.lower() or (text_query and "shoe" in text_query.lower()):
        return {"visual_features": "shoe_like", "category": "footwear", "brand": "unknown"}
    if b"dress" in image_data.lower() or (text_query and "dress" in text_query.lower()):
        return {"visual_features": "dress_like", "category": "apparel", "color": "blue"}
    return {"visual_features": "generic", "category": "misc", "description": "an item"}

async def _translate_text(text: str, target_language: str = "en", detected_language: str = "es") -> str:
    """Mocks a Machine Translation service."""
    # For simplicity, we'll only mock Spanish to English translation
    if detected_language.lower() == "es" and target_language.lower() == "en":
        if "hola" in text.lower():
            return "hello, I'm looking for a product."
        if "vestido rojo" in text.lower():
            return "red dress"
    return text # Return original if no specific translation rule

async def _product_search(query_params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Mocks an E-commerce product search and recommendation engine."""
    print(f"[MOCK] Searching products with params: {query_params}")
    products = [
        {"id": "p1", "name": "Summer Red Dress", "category": "apparel", "price": 45.00, "color": "red", "description": "Lightweight summer dress.", "image_url": "http://example.com/red_dress.jpg"},
        {"id": "p2", "name": "Hiking Boots Pro", "category": "footwear", "price": 120.00, "color": "brown", "description": "Durable boots for hiking.", "image_url": "http://example.com/hiking_boots.jpg"},
        {"id": "p3", "name": "Casual Blue Dress", "category": "apparel", "price": 30.00, "color": "blue", "description": "Comfortable everyday dress.", "image_url": "http://example.com/blue_dress.jpg"},
        {"id": "p4", "name": "Smart Home Hub", "category": "electronics", "price": 80.00, "color": "black", "description": "Control your smart home.", "image_url": "http://example.com/smarthub.jpg"},
        {"id": "p5", "name": "Red Casual Tee", "category": "apparel", "price": 20.00, "color": "red", "description": "Basic red t-shirt.", "image_url": "http://example.com/red_tee.jpg"},
    ]

    results = []
    # Simple filtering logic for demonstration
    for p in products:
        match = True
        if query_params.get("category") and query_params["category"].lower() not in p["category"].lower():
            match = False
        if query_params.get("color") and query_params["color"].lower() not in p["color"].lower():
            match = False
        if query_params.get("max_price") and p["price"] > query_params["max_price"]:
            match = False
        if query_params.get("min_price") and p["price"] < query_params["min_price"]:
            match = False
        if query_params.get("keywords"):
            found_keyword = False
            for keyword in query_params["keywords"]:
                if keyword.lower() in p["name"].lower() or keyword.lower() in p["description"].lower():
                    found_keyword = True
                    break
            if not found_keyword:
                match = False
        if query_params.get("target_audience") == "sister_hiking" and "hiking" not in p["name"].lower() and "hiking" not in p["description"].lower():
             match = False

        if match:
            results.append(p)
            
    return results

# --- Core AI Layer (LLM & Intent Comprehension) ---

async def _llm_process_query(
    session_id: str,
    current_query: str,
    history: List[Dict[str, str]], # Represents chat history { "role": "user"/"assistant", "content": "..." }
    user_preferences: Dict[str, Any]
) -> Tuple[str, Dict[str, Any], List[Dict[str, Any]]]:
    """Mocks the LLM's intent comprehension, entity extraction, ambiguity resolution, and response generation."""
    print(f"[LLM_MOCK] Processing query: '{current_query}' with history: {history} and prefs: {user_preferences}")
    response_text = "I'm sorry, I didn't quite understand that. Can you please rephrase?"
    products_found = []
    search_params = {}
    updated_preferences = user_preferences.copy()

    lower_query = current_query.lower()

    # Simulate intent recognition and entity extraction
    if "red summer dress" in lower_query and "under $50" in lower_query:
        search_params = {"category": "apparel", "color": "red", "max_price": 50.00, "keywords": ["summer", "dress"]}
        response_text = "Certainly! Looking for red summer dresses under $50. One moment please..."
    elif "gift for my sister who likes hiking" in lower_query:
        updated_preferences["gift_recipient"] = "sister"
        updated_preferences["recipient_interests"] = "hiking"
        search_params = {"target_audience": "sister_hiking", "category": "outdoor", "keywords": ["hiking", "gift"]}
        response_text = "A thoughtful gift for your sister who enjoys hiking! I'll find some great options for you."
    elif "something similar to this but in a different color" in lower_query and user_preferences.get("last_image_category"):
        current_category = user_preferences["last_image_category"]
        current_color = user_preferences.get("last_image_color")
        new_color = "blue" if current_color != "blue" else "green"
        search_params = {"category": current_category, "color": new_color}
        response_text = f"Okay, finding {current_category} in {new_color} for you."
    elif "what brand is this shoe" in lower_query and user_preferences.get("last_image_category") == "footwear":
        response_text = "Based on the image, identifying the exact brand can be tricky, but it looks like a popular athletic shoe style. Would you like me to find similar shoes?"
    elif "comfortable for a casual evening out" in lower_query:
        search_params = {"category": "apparel", "keywords": ["comfortable", "casual", "evening"]}
        # Simulate ambiguity resolution: Ask for more details
        response_text = "I can certainly help with that! To narrow it down, are you thinking of a dress, pants, or something else? Do you have a color preference?"
    elif "unique gadget" in lower_query or "surprise me with a gadget" in lower_query:
        search_params = {"category": "electronics", "keywords": ["unique", "gadget"], "min_price": 50.00, "max_price": 150.00}
        response_text = "A unique gadget, excellent choice! I'm searching for innovative tech that might surprise you."
    elif "show me all red products" in lower_query:
        search_params = {"color": "red"}
        response_text = "Here are some red products I found:"
    else:
        # Default intent: general search based on keywords
        keywords = [word for word in lower_query.split() if len(word) > 2 and word not in ["a", "an", "the", "for", "me", "show", "find", "i", "want"]]
        if keywords:
            search_params["keywords"] = keywords
            response_text = f"Searching for items related to {' '.join(keywords)}. "

    # Integrate personalized learning (mock: apply preferences to search)
    if updated_preferences.get("last_category_searched") and not search_params.get("category"):
        # If no category in current query, lean on last searched category
        pass # For this simple mock, we'll let direct query intent override this more complex logic.

    if search_params:
        products_found = await _product_search(search_params)
        if products_found:
            product_names = ", ".join([p["name"] for p in products_found[:3]])
            response_text += f"I found {len(products_found)} items. For example: {product_names}. "
            if len(products_found) > 3:
                response_text += "Would you like to see more details?"
        else:
            response_text += "I couldn't find any products matching your specific request."
    
    # Update last relevant query/category in preferences if a search was performed
    if search_params.get("category"):
        updated_preferences["last_category_searched"] = search_params["category"]
    if products_found:
        updated_preferences["last_products_found"] = [p["id"] for p in products_found]

    return response_text, updated_preferences, products_found

# --- FastAPI Endpoints ---

@app.post("/chat/text", response_model=ChatResponse)
async def chat_text(request: ChatRequestText):
    session_id = request.session_id if request.session_id else str(uuid.uuid4())
    
    if session_id not in sessions:
        sessions[session_id] = {"history": [], "user_preferences": {}}
    
    session = sessions[session_id]
    history = session["history"]
    user_preferences = session["user_preferences"]

    # Append user message to history
    history.append({"role": "user", "content": request.text_input})

    # LLM Processing
    ai_response, updated_preferences, products_found = await _llm_process_query(session_id, request.text_input, history, user_preferences)

    # Update session
    session["user_preferences"] = updated_preferences
    history.append({"role": "assistant", "content": ai_response})
    sessions[session_id] = session

    return ChatResponse(
        session_id=session_id,
        response=ai_response,
        products_found=products_found,
        user_preferences=updated_preferences
    )

@app.post("/chat/voice", response_model=ChatResponse)
async def chat_voice(
    session_id: Optional[str] = Form(None),
    audio_file: UploadFile = File(...)
):
    session_id = session_id if session_id else str(uuid.uuid4())

    if session_id not in sessions:
        sessions[session_id] = {"history": [], "user_preferences": {}}

    session = sessions[session_id]
    history = session["history"]
    user_preferences = session["user_preferences"]

    audio_data = await audio_file.read()
    text_input = await _speech_to_text(audio_data)

    history.append({"role": "user_voice", "content": text_input})

    ai_response, updated_preferences, products_found = await _llm_process_query(session_id, text_input, history, user_preferences)

    session["user_preferences"] = updated_preferences
    history.append({"role": "assistant", "content": ai_response})
    sessions[session_id] = session

    return ChatResponse(
        session_id=session_id,
        response=ai_response,
        products_found=products_found,
        user_preferences=updated_preferences
    )

@app.post("/chat/image", response_model=ChatResponse)
async def chat_image(
    session_id: Optional[str] = Form(None),
    image_file: UploadFile = File(...),
    text_query: Optional[str] = Form(None)
):
    session_id = session_id if session_id else str(uuid.uuid4())

    if session_id not in sessions:
        sessions[session_id] = {"history": [], "user_preferences": {}}

    session = sessions[session_id]
    history = session["history"]
    user_preferences = session["user_preferences"]

    image_data = await image_file.read()
    image_analysis_results = await _analyze_image(image_data, text_query)

    # Update preferences with insights from image analysis
    if image_analysis_results.get("category"):
        user_preferences["last_image_category"] = image_analysis_results["category"]
    if image_analysis_results.get("color"):
        user_preferences["last_image_color"] = image_analysis_results["color"]

    # Construct query for LLM based on image analysis and optional text
    llm_query = f"User uploaded an image. Image analysis suggests: {image_analysis_results}. "
    if text_query:
        llm_query += f"User also asked: {text_query}"
    else:
        llm_query += "User is asking for something related to the image."
    
    history.append({"role": "user_image", "content": llm_query})

    ai_response, updated_preferences, products_found = await _llm_process_query(session_id, llm_query, history, user_preferences)

    session["user_preferences"] = updated_preferences
    history.append({"role": "assistant", "content": ai_response})
    sessions[session_id] = session

    return ChatResponse(
        session_id=session_id,
        response=ai_response,
        products_found=products_found,
        user_preferences=updated_preferences
    )

@app.get("/status")
async def get_status():
    return {"status": "running", "message": "E-commerce Personal Shopper AI is operational."}

# To run this application:
# 1. Save the code as `main.py`.
# 2. Make sure you have FastAPI and Uvicorn installed: `pip install fastapi "uvicorn[standard]" pydantic`
# 3. Run from your terminal: `uvicorn main:app --reload`
# 4. Access the API documentation at http://127.0.0.1:8000/docs
