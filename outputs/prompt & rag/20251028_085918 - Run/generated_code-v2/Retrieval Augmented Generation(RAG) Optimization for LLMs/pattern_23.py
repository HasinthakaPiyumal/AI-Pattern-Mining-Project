from sentence_transformers import SentenceTransformer, util
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
import gradio as gr
import os

# Configuration
TOP_N_RETRIEVE = 50
TOP_K_RERANK = 5
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Placeholder Knowledge Base (can be loaded from files)
KNOWLEDGE_BASE_DOCS = [
    "Our return policy for electronics states that items can be returned within 30 days of purchase if unopened. For opened items, a 15% restocking fee applies after 14 days.",
    "To initiate a refund for a damaged item, please contact our customer support with your order number and photos of the damage within 7 days of delivery.",
    "All apparel items have a 60-day return window, provided they are unworn and have original tags attached. Refunds will be processed within 5-7 business days.",
    "Warranty claims for appliances typically require proof of purchase and a description of the defect. Our warranty covers manufacturing defects for one year.",
    "For digital products, all sales are final. No refunds or returns are accepted once the product has been downloaded or accessed.",
    "Shipping costs for returns are the responsibility of the customer unless the item received was incorrect or damaged by us.",
    "Refunds are issued to the original payment method. Please allow up to 10 business days for the refund to reflect on your statement.",
    "Items purchased during a promotional sale or with a discount code might have different return conditions. Refer to the specific promotion terms.",
    "Our customer service chat logs indicate common issues with product installation. Please check the product manual for troubleshooting tips before requesting a return.",
    "If you received an incorrect item, please notify us within 48 hours of delivery for a free return label and a replacement."
]

# Initialize Components
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
    add_start_index=True,
)
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# ChromaDB Setup
client = chromadb.PersistentClient(path="./chroma_db")
collection_name = "ecommerce_support_knowledge_base"
try:
    collection = client.get_or_create_collection(name=collection_name)
except Exception as e:
    print(f"Error getting/creating collection: {e}")
    print("Attempting to delete and recreate collection...")
    client.delete_collection(name=collection_name)
    collection = client.get_or_create_collection(name=collection_name)

# Load and Embed Knowledge Base
def load_and_embed_knowledge_base(docs):
    if collection.count() == 0:
        print("Loading and embedding knowledge base...")
        chunks = []
        ids = []
        for i, doc in enumerate(docs):
            split_docs = text_splitter.split_text(doc)
            for j, chunk in enumerate(split_docs):
                chunks.append(chunk)
                ids.append(f"doc{i}_chunk{j}")
        
        chunk_embeddings = embedding_model.encode(chunks, show_progress_bar=True).tolist()
        collection.add(embeddings=chunk_embeddings, documents=chunks, ids=ids)
        print(f"Added {len(chunks)} chunks to ChromaDB.")
    else:
        print("Knowledge base already loaded in ChromaDB.")

load_and_embed_knowledge_base(KNOWLEDGE_BASE_DOCS)

# Retrieval Module
def retrieve_contexts(query: str, top_n: int = TOP_N_RETRIEVE) -> list[str]:
    query_embedding = embedding_model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_n,
        include=['documents', 'distances']
    )
    # Sort by distance (lower is better similarity for cosine distance)
    sorted_results = sorted(zip(results['documents'][0], results['distances'][0]), key=lambda x: x[1])
    return [doc for doc, dist in sorted_results]

# Reranking Module (simulated using embedding similarity)
def rerank_contexts(query: str, contexts: list[str], top_k: int = TOP_K_RERANK) -> list[str]:
    if not contexts:
        return []
    query_embedding = embedding_model.encode(query)
    context_embeddings = embedding_model.encode(contexts)
    
    # Calculate cosine similarity between query and each context
    similarities = util.cos_sim(query_embedding, context_embeddings)[0].tolist()
    
    # Pair contexts with their similarities and sort in descending order
    reranked_pairs = sorted(zip(contexts, similarities), key=lambda x: x[1], reverse=True)
    
    return [doc for doc, score in reranked_pairs[:top_k]]

# Generation Module (simulated LLM generation)
def generate_answer(query: str, contexts: list[str]) -> str:
    if not contexts:
        return "I am sorry, but I could not find relevant information to answer your question. Please try rephrasing or contact customer support."
    
    combined_context = "\n\n".join(contexts)
    prompt = f"Question: {query}\n\nRelevant Information:\n{combined_context}\n\nBased on the relevant information provided, please provide a concise and helpful answer to the question."
    
    # Simulate LLM response - in a real application, this would call a real LLM
    # For this demo, we'll give a structured response based on the contexts.
    simulated_answer = (
        f"Hello! Regarding your query about '{query}', here's what I found from our policies:\n\n"
        f"{combined_context[:500]}... (truncated for brevity)\n\n"
        f"Please refer to the specific policy details for full information or contact our support team for further assistance."
    )
    return simulated_answer

# Orchestration
def process_customer_query(query: str) -> str:
    retrieved_contexts = retrieve_contexts(query, TOP_N_RETRIEVE)
    reranked_contexts = rerank_contexts(query, retrieved_contexts, TOP_K_RERANK)
    final_answer = generate_answer(query, reranked_contexts)
    return final_answer

# Gradio Interface
if __name__ == "__main__":
    print("Starting Gradio interface...")
    demo = gr.Interface(
        fn=process_customer_query,
        inputs=gr.Textbox(lines=2, placeholder="Ask a question about returns, refunds, or warranty claims..."),
        outputs="text",
        title="E-commerce Smart Customer Support Assistant",
        description="Ask me anything about product returns, refunds, or warranty policies. I use a Retrieve-Rerank-Generate pipeline for accurate answers."
    )
    demo.launch()
