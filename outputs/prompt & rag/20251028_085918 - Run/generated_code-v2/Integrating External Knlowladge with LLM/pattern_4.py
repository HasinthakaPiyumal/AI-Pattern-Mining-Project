from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import uvicorn

# Placeholder for a medical knowledge graph (in a real system, this would be a database)
MEDICAL_KNOWLEDGE_GRAPH = [
    "Aspirin is a nonsteroidal anti-inflammatory drug (NSAID) used to reduce pain, fever, and inflammation.",
    "Diabetes mellitus is a metabolic disease that causes high blood sugar. Insulin resistance or insufficient insulin production are causes.",
    "Hypertension, or high blood pressure, is a common condition in which the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease.",
    "Paracetamol (Acetaminophen) is a pain reliever and a fever reducer. It's often used for mild to moderate pain.",
    "COVID-19 is an infectious disease caused by the SARS-CoV-2 virus. Most people infected with the virus will experience mild to moderate respiratory illness.",
    "The human heart has four chambers: two atria and two ventricles."
]

# Initialize the sentence transformer model for embeddings
# In a production environment, this might be loaded once at startup
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Pre-compute embeddings for the knowledge graph (for faster retrieval)
knowledge_graph_embeddings = embedding_model.encode(MEDICAL_KNOWLEDGE_GRAPH, convert_to_tensor=True)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    retrieved_facts: list[str]

app = FastAPI()

def retrieve_facts(query: str, top_k: int = 2) -> list[str]:
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    # Compute cosine similarity between query and knowledge graph facts
    similarities = util.cos_sim(query_embedding, knowledge_graph_embeddings)[0]
    # Get top_k most similar facts
    top_k_indices = similarities.topk(top_k).indices
    retrieved_facts = [MEDICAL_KNOWLEDGE_GRAPH[i] for i in top_k_indices]
    return retrieved_facts

def generate_response_with_llm(query: str, context: list[str]) -> str:
    # This is a placeholder for actual LLM interaction.
    # In a real system, you would call an LLM (e.g., via transformers library or an API).
    # For this example, we'll simply combine the context and query.
    
    prompt = f"Given the following medical facts: {'; '.join(context)}. Answer the question: {query}"
    # Mock LLM generation
    mock_llm_response = f"Based on the provided medical information and your question: \"{query}\", here's a potential answer: {prompt}. (This is a simulated LLM response)"
    return mock_llm_response

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    retrieved_facts = retrieve_facts(request.query)
    llm_answer = generate_response_with_llm(request.query, retrieved_facts)
    return QueryResponse(answer=llm_answer, retrieved_facts=retrieved_facts)

if __name__ == "__main__":
    # To run this, save it as medical_rag_system.py and run:
    # uvicorn medical_rag_system:app --reload
    # Then you can interact with it via an API client or a Streamlit frontend.
    print("FastAPI application started. Access at http://127.0.0.1:8000")
    print("Example usage with curl:\ncurl -X POST -H \"Content-Type: application/json\" -d '{\"query\": \"What is hypertension?\"}' http://127.0.0.1:8000/query")
    import sys
    if "streamlit" not in sys.modules:
        # Only run uvicorn if Streamlit isn't already running this script indirectly
        uvicorn.run(app, host="0.0.0.0", port=8000)
