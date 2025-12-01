import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_chroma import Chroma

# --- 1. Mock Product Data ---
# A small, synthetic product catalog
PRODUCTS_DATA = [
    {"id": "p1", "name": "UltraBoost 23 Running Shoes", "description": "High-performance running shoes with responsive cushioning and a breathable upper. Ideal for long-distance runners.", "category": "Footwear"},
    {"id": "p2", "name": "Smart Fitness Tracker X1", "description": "Waterproof fitness tracker with heart rate monitoring, sleep tracking, and GPS. Connects to your smartphone.", "category": "Wearables"},
    {"id": "p3", "name": "Noise-Cancelling Headphones Pro", "description": "Premium over-ear headphones with industry-leading noise cancellation and crystal-clear audio. Long battery life.", "category": "Audio"},
    {"id": "p4", "name": "Ergonomic Office Chair 5000", "description": "Adjustable ergonomic chair designed for maximum comfort and posture support during long working hours. High-quality mesh fabric.", "category": "Furniture"},
    {"id": "p5", "name": "Portable Bluetooth Speaker Mini", "description": "Compact and powerful Bluetooth speaker with rich bass and 10 hours of playtime. Perfect for outdoor adventures.", "category": "Audio"},
    {"id": "p6", "name": "Yoga Mat Eco-Friendly", "description": "Thick, non-slip yoga mat made from sustainable, non-toxic materials. Provides excellent grip for all yoga styles.", "category": "Fitness"},
    {"id": "p7", "name": "4K Smart TV 65-inch", "description": "Stunning 4K UHD display with smart features, built-in streaming apps, and voice control. Immersive home entertainment.", "category": "Electronics"},
    {"id": "p8", "name": "Electric Kettle Rapid Boil", "description": "Fast-boiling electric kettle with a 1.7L capacity, stainless steel interior, and automatic shut-off.", "category": "Kitchen Appliances"},
    {"id": "p9", "name": "Weighted Blanket Therapy", "description": "Soft, breathable weighted blanket designed to promote relaxation and improve sleep quality. Available in various weights.", "category": "Home Goods"},
    {"id": "p10", "name": "Gaming Laptop Beast Pro", "description": "High-end gaming laptop with a powerful processor, dedicated graphics card, and high-refresh-rate display. Ready for intense gaming.", "category": "Electronics"},
]

# --- 2. Embedding Model Initialization ---
# Using a SentenceTransformer for generating embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# --- 3. ChromaDB Setup and Product Ingestion ---
# Initialize ChromaDB client (in-memory for simplicity)
chroma_client = chromadb.Client() # Can use chromadb.PersistentClient(path="/path/to/db") for persistence

# Create or get a collection
collection_name = "ecommerce_products"
try:
    vector_store = chroma_client.get_or_create_collection(name=collection_name)
except Exception: # Handle case if collection exists but Chroma has issues
    chroma_client.delete_collection(name=collection_name)
    vector_store = chroma_client.get_or_create_collection(name=collection_name)


# Ingest products into ChromaDB
product_ids = [p["id"] for p in PRODUCTS_DATA]
product_descriptions = [p["description"] for p in PRODUCTS_DATA]
product_metadata = PRODUCTS_DATA # Store full product dict as metadata

# Generate embeddings
embeddings = embedding_model.encode(product_descriptions).tolist()

# Add to ChromaDB
vector_store.add(
    documents=product_descriptions,
    metadatas=product_metadata,
    embeddings=embeddings,
    ids=product_ids
)

print(f"Ingested {len(PRODUCTS_DATA)} products into ChromaDB.")

# Create a Langchain retriever from the Chroma collection
# We use a custom embedding function compatible with SentenceTransformer
class SentenceTransformerEmbeddings:
    def __init__(self, model):
        self.model = model
    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()
    def embed_query(self, text):
        return self.model.encode(text).tolist()

langchain_embeddings_function = SentenceTransformerEmbeddings(embedding_model)
chroma_retriever = Chroma(
    client=chroma_client,
    collection_name=collection_name,
    embedding_function=langchain_embeddings_function # Pass the custom embedding function
).as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 relevant products

# --- 4. Mock LLM Function ---
# This function simulates an LLM response. In a real application, this would
# be an API call to a fine-tuned LLM (e.g., via vLLM or a local transformers pipeline).
def mock_llm(prompt: str) -> str:
    prompt_lower = prompt.lower()
    
    if "rank" in prompt_lower or "best for" in prompt_lower or "recommend" in prompt_lower:
        # Simulate ranking response
        if "running shoes" in prompt_lower or "shoes" in prompt_lower:
            return "Based on your interest in running shoes, I recommend the 'UltraBoost 23 Running Shoes' due to their high-performance cushioning and breathable design, ideal for long distances. Other good options include general sports footwear."
        elif "fitness tracker" in prompt_lower:
            return "For fitness tracking, the 'Smart Fitness Tracker X1' is highly recommended for its waterproof design, heart rate monitoring, and GPS features."
        elif "headphones" in prompt_lower or "audio" in prompt_lower:
            return "Considering audio quality and noise cancellation, the 'Noise-Cancelling Headphones Pro' are a top choice. If portability is key, the 'Portable Bluetooth Speaker Mini' is great."
        else:
            return "To recommend the best product, please specify what you are looking for. For example, 'rank these items for comfort' or 'which is best for gaming?'"
    
    # Simulate QA/generation response
    if "ultraboost 23" in prompt_lower:
        return "The UltraBoost 23 Running Shoes offer responsive cushioning and a breathable upper, making them perfect for long-distance running. They focus on comfort and performance."
    elif "smart fitness tracker" in prompt_lower:
        return "The Smart Fitness Tracker X1 is waterproof and features heart rate monitoring, sleep tracking, and GPS. It's designed to help you monitor your health and activities."
    elif "noise-cancelling headphones" in prompt_lower:
        return "The Noise-Cancelling Headphones Pro provide premium over-ear audio with industry-leading noise cancellation for an immersive listening experience."
    elif "office chair" in prompt_lower:
        return "The Ergonomic Office Chair 5000 is built for comfort and posture support during long working hours, featuring adjustable components and high-quality mesh."
    elif "portable bluetooth speaker" in prompt_lower:
        return "The Portable Bluetooth Speaker Mini is compact, delivers rich bass, and offers 10 hours of playtime, ideal for music on the go."
    elif "yoga mat" in prompt_lower:
        return "The Yoga Mat Eco-Friendly is a thick, non-slip mat made from sustainable, non-toxic materials, ensuring excellent grip for all yoga styles."
    elif "4k smart tv" in prompt_lower:
        return "The 4K Smart TV 65-inch features a stunning UHD display, built-in streaming apps, and voice control for a complete home entertainment system."
    elif "electric kettle" in prompt_lower:
        return "The Electric Kettle Rapid Boil offers fast boiling with a 1.7L capacity, a stainless steel interior, and an automatic shut-off feature for safety."
    elif "weighted blanket" in prompt_lower:
        return "The Weighted Blanket Therapy is a soft, breathable blanket designed to promote relaxation and improve sleep quality through gentle pressure."
    elif "gaming laptop" in prompt_lower:
        return "The Gaming Laptop Beast Pro is a high-end machine equipped with a powerful processor and dedicated graphics, optimized for demanding gaming sessions."
    elif "hello" in prompt_lower or "hi" in prompt_lower:
        return "Hello! How can I assist you with our products today?"
    else:
        return "I'm an e-commerce assistant. How can I help you with product information or recommendations?"

# --- 5. Langchain Prompt Templates ---
# Unified instruction format for both QA and ranking
template = """
You are an intelligent e-commerce assistant. Your task is to answer user questions or rank products based on the provided context.

Context:
{context}

Question: {question}

If the question asks for a ranking or recommendation, provide a ranked list with brief explanations. Otherwise, answer the question directly.
"""
prompt = ChatPromptTemplate.from_template(template)

# --- 6. Langchain RAG Chain Setup ---
# The RAG chain combines retrieval and LLM response generation
# Using a simple StringOutputParser for the mock LLM
rag_chain = (
    {"context": chroma_retriever, "question": RunnablePassthrough()}
    | prompt
    | RunnableLambda(mock_llm) # Integrate the mock LLM
    | StrOutputParser()
)

# --- 7. FastAPI App Definition ---
app = FastAPI(
    title="Intelligent E-commerce Assistant API",
    description="API for an AI assistant that answers product questions and ranks products efficiently.",
    version="1.0.0",
)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response: str

@app.post("/query", response_model=QueryResponse, summary="Query the e-commerce assistant")
async def query_assistant(request: QueryRequest):
    """
    Send a query to the e-commerce assistant.
    The assistant can answer questions about products or provide ranked recommendations.
    """
    print(f"Received query: {request.query}")
    llm_response = rag_chain.invoke(request.query)
    print(f"Assistant response: {llm_response}")
    return QueryResponse(response=llm_response)

# --- 8. Uvicorn Run Command ---
if __name__ == "__main__":
    print("Starting FastAPI application...")
    print("Access the API at http://127.0.0.1:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
