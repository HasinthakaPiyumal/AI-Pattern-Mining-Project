from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
import os
from typing import List, Dict, Any
import uuid

# Mocking external libraries and LLM interactions
# In a real application, these would be actual imports and API calls

class MockSentenceTransformer:
    def encode(self, texts: List[str]) -> List[List[float]]:
        # Simulate embedding generation with dummy vectors
        return [[float(i) * 0.01 for i in range(768)] for _ in texts]

class MockChromaDBCollection:
    def __init__(self, name):
        self.name = name
        self.documents = []
        self.metadatas = []
        self.embeddings = []
        self.ids = []

    def add(self, documents: List[str], metadatas: List[Dict], embeddings: List[List[float]], ids: List[str]):
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)
        self.embeddings.extend(embeddings)
        self.ids.extend(ids)

    def query(self, query_embeddings: List[List[float]], n_results: int = 10) -> Dict:
        # Simple cosine similarity simulation (not actual cosine sim, just a mock ranking)
        if not self.embeddings:
            return {"documents": [[]], "metadatas": [[]], "ids": [[]]}

        # A very simplistic mock query: just return the first n_results documents
        # In a real scenario, this would involve actual vector similarity search
        results = {
            "documents": [self.documents[:n_results]],
            "metadatas": [self.metadatas[:n_results]],
            "ids": [self.ids[:n_results]],
            "distances": [[0.1 * i for i in range(n_results)]] # Mock distances
        }
        return results

class MockChromaDBClient:
    def __init__(self):
        self.collections = {}

    def get_or_create_collection(self, name: str) -> MockChromaDBCollection:
        if name not in self.collections:
            self.collections[name] = MockChromaDBCollection(name)
        return self.collections[name]

# --- Product Data Manager ---
class ProductManager:
    def __init__(self, embedding_model, vector_db_client, collection_name: str = "products"):
        self.embedding_model = embedding_model
        self.vector_db_client = vector_db_client
        self.collection = self.vector_db_client.get_or_create_collection(collection_name)
        self.products_data: Dict[str, Any] = {}

    def load_products(self, file_path: str):
        if not os.path.exists(file_path):
            # Create a dummy products.json if it doesn't exist for demonstration
            dummy_products = [
                {
                    "id": "prod1",
                    "name": "Wireless Bluetooth Headphones",
                    "category": "Electronics",
                    "price": 79.99,
                    "raw_description": "High-quality sound, comfortable ear cups, long battery life, noise cancellation."
                },
                {
                    "id": "prod2",
                    "name": "Ergonomic Office Chair",
                    "category": "Home Office",
                    "price": 199.99,
                    "raw_description": "Adjustable lumbar support, breathable mesh, smooth rolling casters, easy assembly."
                }
            ]
            with open(file_path, "w") as f:
                json.dump(dummy_products, f, indent=4)
            print(f"Created dummy {file_path}")

        with open(file_path, "r") as f:
            products = json.load(f)
            for product in products:
                self.products_data[product["id"]] = product
            print(f"Loaded {len(self.products_data)} products.")

    def enrich_product_description(self, product_id: str, description: str) -> str:
        # Simulate LLM enriching the description
        # In a real scenario, this would call an LLM API
        enriched_desc = f"Enhanced description for {product_id}: {description}. This product is highly recommended for {{user_context}} and features {{detailed_features}}."
        return enriched_desc

    def generate_embeddings(self, text: str) -> List[float]:
        return self.embedding_model.encode([text])[0]

    def add_products_to_vector_db(self):
        documents = []
        metadatas = []
        embeddings = []
        ids = []

        for prod_id, product in self.products_data.items():
            # Ensure enriched_description exists, or use raw_description
            description_to_embed = product.get("enriched_description", product["raw_description"])
            documents.append(description_to_embed)
            metadatas.append({"id": product["id"], "name": product["name"], "category": product["category"]})
            embeddings.append(self.generate_embeddings(description_to_embed))
            ids.append(product["id"])

        if ids:
            self.collection.add(documents=documents, metadatas=metadatas, embeddings=embeddings, ids=ids)
            print(f"Added {len(ids)} products to vector DB.")

    def search_products(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_embedding = self.generate_embeddings(query)
        results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)

        found_products = []
        if results and results["ids"]: # ChromaDB returns list of lists
            for prod_id in results["ids"][0]:
                if prod_id in self.products_data:
                    found_products.append(self.products_data[prod_id])
        return found_products

# --- Recommendation Engine ---
class Recommender:
    def __init__(self, product_manager: ProductManager):
        self.product_manager = product_manager

    def get_personalized_recommendations(self, user_profile: Dict[str, Any], query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        # Simulate LLM-based personalized recommendations
        # In a real scenario, this would involve more sophisticated LLM prompting
        # combined with semantic search results and user profile analysis.
        print(f"User profile: {user_profile}, Query: {query}")
        # For demonstration, we'll just use semantic search results
        recommended_products = self.product_manager.search_products(query, top_k=top_k)

        # Further refine or re-rank based on user_profile (simulated)
        if "preferred_category" in user_profile:
            recommended_products = [p for p in recommended_products if p.get("category") == user_profile["preferred_category"]] or recommended_products
        
        return recommended_products

    def generate_explanation(self, recommended_product: Dict[str, Any], user_profile: Dict[str, Any], reason_context: str) -> str:
        # Simulate LLM generating a natural language explanation
        # This would use an LLM with a carefully crafted prompt.
        user_name = user_profile.get("name", "valued customer")
        explanation = f"Hello {user_name}! We recommend the {recommended_product['name']} because {reason_context}. " \
                      f"It matches your interest in '{recommended_product['category']}' products and is known for its {recommended_product.get('enriched_description', recommended_product['raw_description']).split('.')[0]} "\
                      f"based on our advanced analysis."
        return explanation

# --- Conversational Shopping Assistant ---
class ChatBot:
    def __init__(self, product_manager: ProductManager, recommender: Recommender):
        self.product_manager = product_manager
        self.recommender = recommender
        self.agent = None # Placeholder for a langchain agent

    def initialize_agent(self):
        # In a real scenario, this would initialize a langchain agent with tools.
        # For this mock, we'll just indicate it's ready.
        print("Mock Langchain agent initialized.")
        # Example of a mock tool callable by the 'agent'
        self.agent_tools = {
            "search_products": self.product_manager.search_products,
            "get_recommendations": self.recommender.get_personalized_recommendations
        }

    def chat_with_assistant(self, user_message: str, user_profile: Dict[str, Any]) -> str:
        # Simulate agent reasoning and response generation
        user_message_lower = user_message.lower()

        if "hello" in user_message_lower or "hi" in user_message_lower:
            return "Hello! How can I help you find something today?"
        elif "recommend" in user_message_lower or "looking for" in user_message_lower:
            # Try to extract a query from the message
            query = user_message_lower.replace("recommend", "").replace("looking for", "").strip()
            query = query if query else "general products"
            recs = self.recommender.get_personalized_recommendations(user_profile, query, top_k=2)
            if recs:
                rec_names = ", ".join([p["name"] for p in recs])
                return f"Based on your request, I recommend: {rec_names}. Would you like more details on any of these?"
            else:
                return "I couldn't find any specific recommendations for that right now. Could you be more specific?"
        elif "details on" in user_message_lower or "tell me about" in user_message_lower:
            product_name_query = user_message_lower.split("details on")[-1].split("tell me about")[-1].strip()
            # A simplistic lookup
            found_product = next((p for p in self.product_manager.products_data.values() if product_name_query in p["name"].lower()), None)
            if found_product:
                return self.recommender.generate_explanation(found_product, user_profile, f"you asked about {found_product['name']}")
            else:
                return "I couldn't find details for that product. Please try again."
        else:
            return "I'm not sure how to respond to that. Can you ask in a different way?"

# --- FastAPI Application ---
app = FastAPI()

# Global instances (initialized once at startup)
embedding_model = MockSentenceTransformer()
vector_db_client = MockChromaDBClient()
product_manager = ProductManager(embedding_model, vector_db_client)
recommender = Recommender(product_manager)
chat_bot = ChatBot(product_manager, recommender)

# Load initial products and add to DB on startup
@app.on_event("startup")
async def startup_event():
    product_file = "data/products.json"
    os.makedirs(os.path.dirname(product_file), exist_ok=True)
    product_manager.load_products(product_file)
    product_manager.add_products_to_vector_db()
    chat_bot.initialize_agent()

class ProductEnrichRequest(BaseModel):
    product_id: str

class RecommendRequest(BaseModel):
    user_profile: Dict[str, Any]
    query: str
    top_k: int = 5

class ChatRequest(BaseModel):
    user_message: str
    user_profile: Dict[str, Any]

@app.post("/enrich-products")
async def enrich_products_endpoint():
    # In a real app, this would iterate through products needing enrichment
    # For this example, we'll enrich all loaded products and update the DB
    for prod_id, product in product_manager.products_data.items():
        original_desc = product["raw_description"]
        enriched_desc = product_manager.enrich_product_description(prod_id, original_desc)
        product_manager.products_data[prod_id]["enriched_description"] = enriched_desc
    product_manager.add_products_to_vector_db() # Re-add with enriched descriptions
    return {"message": "Products enriched and updated in vector DB successfully."}

@app.post("/recommend")
async def get_recommendations(request: RecommendRequest):
    recommendations = recommender.get_personalized_recommendations(
        request.user_profile, request.query, request.top_k
    )
    if not recommendations:
        return {"recommendations": [], "explanation": "No recommendations found for your query."}

    # For simplicity, generate explanation for the first recommendation
    explanation = recommender.generate_explanation(
        recommended_product=recommendations[0],
        user_profile=request.user_profile,
        reason_context="it highly matches your search criteria and preferences"
    )
    return {"recommendations": recommendations, "explanation": explanation}

@app.post("/chat")
async def chat_with_shopping_assistant(request: ChatRequest):
    response = chat_bot.chat_with_assistant(request.user_message, request.user_profile)
    return {"response": response}
