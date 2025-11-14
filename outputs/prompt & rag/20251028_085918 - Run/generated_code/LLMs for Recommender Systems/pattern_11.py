import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# LLM and Embedding Libraries
from sentence_transformers import SentenceTransformer
import chromadb
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.agents import AgentExecutor, Tool, create_react_agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

# Recommender System Libraries
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- Configuration and Initialization ---

# Set your OpenAI API key (ensure it's in your environment variables or replace this line)
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# Initialize LLM (using a mock if API key is not set, otherwise OpenAI)
llm = ChatOpenAI(temperature=0.7, model="gpt-4-turbo-preview")

# Initialize Sentence Transformer model for embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize ChromaDB client and collection
chroma_client = chromadb.Client()
products_collection = chroma_client.get_or_create_collection(name="ecommerce_products")

# --- Data Simulation (In-memory for demonstration) ---

mock_products = [
    {"id": "P001", "name": "Smartwatch Pro X", "description": "Advanced smartwatch with health tracking, GPS, and long battery life. Perfect for fitness enthusiasts.", "category": "Electronics", "price": 299.99},
    {"id": "P002", "name": "Organic Coffee Beans", "description": "Premium organic Arabica coffee beans, medium roast, rich flavor profile with hints of chocolate and nuts. Ethically sourced.", "category": "Groceries", "price": 15.50},
    {"id": "P003", "name": "Noise-Cancelling Headphones", "description": "Over-ear headphones with superior noise cancellation and immersive audio. Ideal for travel and focus.", "category": "Electronics", "price": 199.00},
    {"id": "P004", "name": "Ergonomic Office Chair", "description": "Adjustable office chair designed for maximum comfort and support during long working hours. Breathable mesh back.", "category": "Furniture", "price": 350.00},
    {"id": "P005", "name": "Vegan Protein Powder", "description": "Plant-based protein powder with pea and rice protein, vanilla flavor. Great for post-workout recovery.", "category": "Supplements", "price": 45.00},
    {"id": "P006", "name": "Portable Bluetooth Speaker", "description": "Compact and waterproof Bluetooth speaker with 24-hour battery life. Perfect for outdoor adventures.", "category": "Electronics", "price": 75.00},
    {"id": "P007", "name": "Yoga Mat Deluxe", "description": "Extra-thick, non-slip yoga mat made from eco-friendly materials. Provides excellent cushioning and grip.", "category": "Fitness", "price": 50.00},
]

mock_user_interactions = {
    "U001": {"purchases": ["P001", "P003"], "views": ["P006", "P004"], "preferences": "interested in tech gadgets and travel gear"},
    "U002": {"purchases": ["P002", "P005"], "views": ["P007", "P004"], "preferences": "enjoys healthy food, fitness, and sustainable products"},
    "U003": {"purchases": ["P004"], "views": ["P001", "P003", "P006"], "preferences": "looking for home office improvements and sometimes tech"}
}

# --- LLM-based Feature Engineering & Embedding Generation ---

def generate_and_store_embeddings(products: List[Dict[str, Any]]):
    """Generates embeddings for products and stores them in ChromaDB."""
    product_ids = [p["id"] for p in products]
    product_descriptions = [f"{p['name']}. {p['description']} Category: {p['category']}. Price: ${p['price']}" for p in products]
    
    embeddings = embedding_model.encode(product_descriptions).tolist()
    
    # Store in ChromaDB
    products_collection.upsert(
        documents=product_descriptions,
        metadatas=[{"id": p["id"], "name": p["name"], "category": p["category"], "price": p["price"]} for p in products],
        ids=product_ids,
        embeddings=embeddings
    )
    print(f"Generated and stored embeddings for {len(products)} products.")

# Generate embeddings on startup
generate_and_store_embeddings(mock_products)

# --- Core Recommender Engine ---

def get_product_embedding(product_id: str) -> Optional[List[float]]:
    """Retrieves the embedding for a given product ID from ChromaDB."""
    try:
        results = products_collection.get(ids=[product_id], include=['embeddings'])
        if results and results['embeddings']:
            return results['embeddings'][0]
    except Exception as e:
        print(f"Error retrieving embedding for {product_id}: {e}")
    return None

def get_user_profile_embedding(user_id: str) -> Optional[List[float]]:
    """Generates a user profile embedding based on their preferences and past interactions."""
    user_data = mock_user_interactions.get(user_id)
    if not user_data: 
        return None
    
    # Combine preferences and descriptions of purchased/viewed items
    user_text = user_data.get("preferences", "")
    for pid in user_data.get("purchases", []) + user_data.get("views", []):
        product = next((p for p in mock_products if p["id"] == pid), None)
        if product:
            user_text += f" {product['name']}: {product['description']}."
            
    if user_text.strip():
        return embedding_model.encode([user_text.strip()]).tolist()[0]
    return None

def get_recommendations_hybrid(user_id: str, num_recommendations: int = 5) -> List[Dict[str, Any]]:
    """Generates hybrid recommendations using user profile and product embeddings."""
    user_embedding = get_user_profile_embedding(user_id)
    if user_embedding is None:
        return []

    # Retrieve all product embeddings from ChromaDB
    all_product_data = products_collection.get(ids=[p["id"] for p in mock_products], include=['embeddings', 'metadatas'])
    
    if not all_product_data or not all_product_data['embeddings']:
        return []
        
    product_embeddings = np.array(all_product_data['embeddings'])
    product_ids = all_product_data['ids']
    product_metadatas = all_product_data['metadatas']
    
    # Calculate cosine similarity between user embedding and all product embeddings
    similarities = cosine_similarity([user_embedding], product_embeddings)[0]
    
    # Sort by similarity and exclude already purchased/viewed items (simple approach)
    user_interactions = mock_user_interactions.get(user_id, {})
    already_interacted_ids = set(user_interactions.get("purchases", []) + user_interactions.get("views", []))
    
    recommended_products = []
    for idx in np.argsort(similarities)[::-1]: # Sort in descending order
        product_id = product_ids[idx]
        if product_id not in already_interacted_ids:
            product_info = next((p for p in mock_products if p["id"] == product_id), None)
            if product_info:
                recommended_products.append(product_info)
        if len(recommended_products) >= num_recommendations:
            break
            
    return recommended_products

# --- LLM Orchestration & Interaction Layer (LangChain) ---

# 1. Explanation Generator Agent
explanation_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an intelligent e-commerce assistant. Your task is to explain why a specific product was recommended to a user, based on their profile and the product's attributes. Be concise, friendly, and highlight key reasons."),
    ("user", "User Profile: {user_profile}\nProduct: {product_name} - {product_description}\nWhy was this recommended?")
])
explanation_chain = LLMChain(llm=llm, prompt=explanation_prompt)

def explain_recommendation(user_id: str, product_id: str) -> str:
    user_profile = mock_user_interactions.get(user_id, {}).get("preferences", "")
    product = next((p for p in mock_products if p["id"] == product_id), None)
    
    if not product:
        return f"Product {product_id} not found."
    
    # Augment user profile with recent interactions for better explanation context
    user_interactions_text = f"Recent purchases: {', '.join(mock_user_interactions.get(user_id,{}).get('purchases',[]))}. Recent views: {', '.join(mock_user_interactions.get(user_id,{}).get('views',[]))}."
    full_user_profile = f"{user_profile}. {user_interactions_text}"
    
    explanation = explanation_chain.run(
        user_profile=full_user_profile,
        product_name=product["name"],
        product_description=product["description"]
    )
    return explanation

# 2. Conversational Agent
def get_product_details_by_name(product_name_query: str) -> Optional[Dict[str, Any]]:
    """Finds product details by a partial or full name match."""
    product_name_query_lower = product_name_query.lower()
    for p in mock_products:
        if product_name_query_lower in p["name"].lower():
            return p
    return None

def get_products_by_category(category_query: str) -> List[Dict[str, Any]]:
    """Finds products by category."""
    category_query_lower = category_query.lower()
    return [p for p in mock_products if category_query_lower in p["category"].lower()]

tools = [
    Tool(
        name="GetRecommendations",
        func=lambda user_id: str(get_recommendations_hybrid(user_id)),
        description="Useful for getting personalized product recommendations for a given user_id."
    ),
    Tool(
        name="GetProductDetailsByName",
        func=lambda product_name: str(get_product_details_by_name(product_name)),
        description="Useful for finding details of a product by its name. Input should be the product name."
    ),
    Tool(
        name="GetProductsByCategory",
        func=lambda category: str(get_products_by_category(category)),
        description="Useful for finding products within a specific category. Input should be the category name."
    ),
    # Add more tools as needed, e.g., for filtering by price, brand, etc.
]

conversational_prompt = PromptTemplate.from_template(
    """You are a helpful and friendly e-commerce assistant. Answer the user's questions about products and recommendations using the provided tools.
    
    Chat History: {chat_history}
    Human: {input}
    {agent_scratchpad}"""
)

conversational_agent = create_react_agent(llm, tools, conversational_prompt)
conversational_agent_executor = AgentExecutor(agent=conversational_agent, tools=tools, verbose=True, handle_parsing_errors=True)

def chat_with_recommender(user_id: str, query: str, chat_history: List[Dict[str, str]]) -> str:
    """Handles conversational queries with the recommender system."""
    # Convert chat_history to Langchain's message format
    formatted_chat_history = []
    for msg in chat_history:
        if msg["role"] == "user":
            formatted_chat_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            formatted_chat_history.append(AIMessage(content=msg["content"]))

    try:
        response = conversational_agent_executor.invoke({"input": query, "chat_history": formatted_chat_history, "user_id": user_id})
        return response.get("output", "I'm sorry, I couldn't process that request.")
    except Exception as e:
        print(f"Error in conversational agent: {e}")
        return "I apologize, I encountered an error trying to understand your request."

# 3. Workflow Automation Agent (Simplified)
marketing_copy_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a creative marketing copywriter for an e-commerce platform. Generate compelling, concise marketing copy for the given product. Highlight its key benefits and target audience. Keep it under 50 words."),
    ("user", "Product Name: {product_name}\nProduct Description: {product_description}\nTarget Audience: {target_audience}\nGenerate marketing copy:")
])
marketing_copy_chain = LLMChain(llm=llm, prompt=marketing_copy_prompt)

def generate_marketing_copy(product_id: str) -> str:
    product = next((p for p in mock_products if p["id"] == product_id), None)
    if not product:
        return f"Product {product_id} not found."
    
    # Infer target audience or use a generic one
    target_audience = "general consumers" # Can be more sophisticated
    if "fitness" in product["description"].lower() or "health" in product["description"].lower():
        target_audience = "health-conscious individuals and fitness enthusiasts"
    elif "office" in product["description"].lower() or "work" in product["description"].lower():
        target_audience = "professionals and remote workers"

    copy = marketing_copy_chain.run(
        product_name=product["name"],
        product_description=product["description"],
        target_audience=target_audience
    )
    return copy

# --- FastAPI Application ---

app = FastAPI(title="LLM-Enhanced E-commerce Recommender")

class RecommendationRequest(BaseModel):
    user_id: str
    num_recommendations: int = 5

class Product(BaseModel):
    id: str
    name: str
    description: str
    category: str
    price: float

class RecommendationResponse(BaseModel):
    user_id: str
    recommendations: List[Product]

class ExplainRequest(BaseModel):
    user_id: str
    product_id: str

class ExplanationResponse(BaseModel):
    user_id: str
    product_id: str
    explanation: str

class ChatRequest(BaseModel):
    user_id: str
    query: str
    chat_history: List[Dict[str, str]] = [] # [{'role': 'user', 'content': 'hi'}, {'role': 'assistant', 'content': 'hello'}]

class ChatResponse(BaseModel):
    user_id: str
    response: str
    new_chat_history: List[Dict[str, str]]

class MarketingCopyRequest(BaseModel):
    product_id: str

class MarketingCopyResponse(BaseModel):
    product_id: str
    marketing_copy: str

@app.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    if request.user_id not in mock_user_interactions:
        raise HTTPException(status_code=404, detail="User not found")
    
    recommendations = get_recommendations_hybrid(request.user_id, request.num_recommendations)
    return {"user_id": request.user_id, "recommendations": recommendations}

@app.post("/explain_recommendation", response_model=ExplanationResponse)
async def get_explanation(request: ExplainRequest):
    if request.user_id not in mock_user_interactions:
        raise HTTPException(status_code=404, detail="User not found")
    product = next((p for p in mock_products if p["id"] == request.product_id), None)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
        
    explanation = explain_recommendation(request.user_id, request.product_id)
    return {"user_id": request.user_id, "product_id": request.product_id, "explanation": explanation}

@app.post("/chat_recommendation", response_model=ChatResponse)
async def chat_with_recsys(request: ChatRequest):
    # Ensure user_id exists if you want to use it within the agent
    if request.user_id not in mock_user_interactions and request.user_id != "guest": # Allow guest for general queries
        raise HTTPException(status_code=404, detail="User not found")
        
    response_text = chat_with_recommender(request.user_id, request.query, request.chat_history)
    
    # Update chat history for the next turn
    new_chat_history = request.chat_history + [
        {"role": "user", "content": request.query},
        {"role": "assistant", "content": response_text}
    ]
    
    return {"user_id": request.user_id, "response": response_text, "new_chat_history": new_chat_history}

@app.post("/generate_marketing_copy", response_model=MarketingCopyResponse)
async def create_marketing_copy(request: MarketingCopyRequest):
    product = next((p for p in mock_products if p["id"] == request.product_id), None)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
        
    copy = generate_marketing_copy(request.product_id)
    return {"product_id": request.product_id, "marketing_copy": copy}

# --- Streamlit UI (for demonstration) ---

import streamlit as st
import requests # For making requests to FastAPI backend

FASTAPI_BASE_URL = "http://127.0.0.1:8000" # Or wherever your FastAPI app is running

st.set_page_config(layout="wide")
st.title("🛒 LLM-Enhanced E-commerce Recommender")
st.markdown("This demo showcases an e-commerce recommender system powered by Large Language Models.")

# Sidebar for User Selection
selected_user_id = st.sidebar.selectbox(
    "Select User ID",
    list(mock_user_interactions.keys()) + ["guest"],
    index=0
)

st.sidebar.markdown(f"### User Profile for {selected_user_id}")
if selected_user_id != "guest":
    user_data = mock_user_interactions.get(selected_user_id)
    if user_data:
        st.sidebar.json(user_data)

# --- Tabs for different functionalities ---

tab1, tab2, tab3, tab4 = st.tabs(["Recommendations", "Explain Recommendation", "Chat with Recommender", "Marketing Automation"])

with tab1:
    st.header("Get Personalized Recommendations")
    if st.button("Get Recommendations"):
        if selected_user_id == "guest":
            st.warning("Please select a specific user to get personalized recommendations.")
        else:
            with st.spinner("Fetching recommendations..."):
                try:
                    response = requests.post(f"{FASTAPI_BASE_URL}/recommendations", json={"user_id": selected_user_id, "num_recommendations": 5})
                    response.raise_for_status() # Raise an exception for bad status codes
                    recommendations_data = response.json()
                    st.subheader(f"Recommendations for User {selected_user_id}:")
                    for rec in recommendations_data["recommendations"]:
                        st.write(f"- **{rec['name']}** (Category: {rec['category']}, Price: ${rec['price']:.2f})\n  *{rec['description']}*")
                except requests.exceptions.RequestException as e:
                    st.error(f"Error fetching recommendations: {e}")
                    st.warning("Is the FastAPI backend running? Run: `uvicorn recommender_system_llm_enhanced:app --reload`")

with tab2:
    st.header("Explain a Recommendation")
    product_to_explain_id = st.selectbox(
        "Select a product to explain",
        [p["id"] for p in mock_products],
        format_func=lambda pid: next((p["name"] for p in mock_products if p["id"] == pid), pid)
    )
    if st.button("Explain Recommendation"):
        if selected_user_id == "guest":
            st.warning("Please select a specific user to get an explanation relevant to their profile.")
        else:
            with st.spinner("Generating explanation..."):
                try:
                    response = requests.post(f"{FASTAPI_BASE_URL}/explain_recommendation", json={"user_id": selected_user_id, "product_id": product_to_explain_id})
                    response.raise_for_status()
                    explanation_data = response.json()
                    st.subheader(f"Explanation for {product_to_explain_id} to User {selected_user_id}:")
                    st.info(explanation_data["explanation"])
                except requests.exceptions.RequestException as e:
                    st.error(f"Error generating explanation: {e}")
                    st.warning("Is the FastAPI backend running? Run: `uvicorn recommender_system_llm_enhanced:app --reload`")

with tab3:
    st.header("Chat with the Recommender")
    st.markdown("Ask questions about products, categories, or get general recommendations. You can use 'guest' user for general queries.")

    # Initialize chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat messages from history on app rerun
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What would you like to know?"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Add user message to chat history
        # (The full history sent to API will include this and previous, then returned new_chat_history will update this)
        current_chat_history_for_api = st.session_state.chat_history + [{'role': 'user', 'content': prompt}]

        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{FASTAPI_BASE_URL}/chat_recommendation", 
                    json={"user_id": selected_user_id, "query": prompt, "chat_history": st.session_state.chat_history}
                )
                response.raise_for_status()
                chat_response_data = response.json()
                assistant_response = chat_response_data["response"]
                st.session_state.chat_history = chat_response_data["new_chat_history"]

                # Display assistant response in chat message container
                with st.chat_message("assistant"):
                    st.markdown(assistant_response)

            except requests.exceptions.RequestException as e:
                st.error(f"Error during chat: {e}")
                st.warning("Is the FastAPI backend running? Run: `uvicorn recommender_system_llm_enhanced:app --reload`")
                with st.chat_message("assistant"):
                    st.markdown("I apologize, I encountered an error trying to respond.")

with tab4:
    st.header("Marketing Content Automation")
    product_for_marketing_id = st.selectbox(
        "Select a product for marketing copy",
        [p["id"] for p in mock_products],
        format_func=lambda pid: next((p["name"] for p in mock_products if p["id"] == pid), pid),
        key="marketing_product_select"
    )
    if st.button("Generate Marketing Copy"):
        with st.spinner("Generating copy..."):
            try:
                response = requests.post(f"{FASTAPI_BASE_URL}/generate_marketing_copy", json={"product_id": product_for_marketing_id})
                response.raise_for_status()
                marketing_copy_data = response.json()
                st.subheader(f"Marketing Copy for {product_for_marketing_id}:")
                st.success(marketing_copy_data["marketing_copy"])
            except requests.exceptions.RequestException as e:
                st.error(f"Error generating marketing copy: {e}")
                st.warning("Is the FastAPI backend running? Run: `uvicorn recommender_system_llm_enhanced:app --reload`")


# How to run instructions for the user
st.sidebar.markdown("""
---
**How to Run:**

1.  **Install Dependencies:**
    `pip install fastapi uvicorn "python-multipart" streamlit sentence-transformers chromadb-client langchain openai numpy scikit-learn`

2.  **Set OpenAI API Key:**
    Ensure your `OPENAI_API_KEY` environment variable is set or uncomment and replace `"YOUR_OPENAI_API_KEY"` in the script.

3.  **Start FastAPI Backend:**
    Open a terminal and run:
    `uvicorn recommender_system_llm_enhanced:app --reload`

4.  **Start Streamlit Frontend:**
    Open *another* terminal and run:
    `streamlit run recommender_system_llm_enhanced.py`

""")

# Entry point for uvicorn (only if running this file directly with uvicorn)
if __name__ == '__main__':
    # This block is typically not run when using `uvicorn app:app` but useful for direct execution if needed
    # It's better to run uvicorn as described in the instructions.
    pass

