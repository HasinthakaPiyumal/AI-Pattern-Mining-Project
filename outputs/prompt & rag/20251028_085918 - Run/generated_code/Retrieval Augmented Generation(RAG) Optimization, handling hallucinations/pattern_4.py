import os
import asyncio
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from transformers import pipeline
import spacy

# Guardrails for controlled generation
from guardrails.hub import FactCheck
from guardrails import Guard

# --- Configuration --- #

# Environment variables for API keys (e.g., OpenAI API key)
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# Initialize LLM
# For demonstration, using OpenAI. In a real scenario, could use Llama 2/Mistral via Hugging Face/vLLM
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2)

# Initialize Embedding Model
# You can choose a different model if needed, e.g., "all-MiniLM-L6-v2"
embeddings_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize Vector Store (ChromaDB - in-memory for this example)
# In a production setup, this would be persistent or a client connection to a running Chroma instance
vectorstore = Chroma(embedding_function=embeddings_model, persist_directory="./chroma_db")

# Initialize Summarization Pipeline
# Using a small, fast model for demonstration
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# Initialize SpaCy for NER and keyword extraction
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading en_core_web_sm model for SpaCy...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Simple in-memory cache for demonstration (replace with Redis in production)
cache: Dict[str, Any] = {}

# Guardrails setup
fact_check_guard = Guard.from_string(validators=[FactCheck(llm=llm, on_fail="fix")])

# --- Knowledge Base Management (Simplified Ingestion) --- #

def ingest_documents(documents: List[Dict[str, str]]):
    """Simulates ingesting documents into the vector store."""
    print(f"Ingesting {len(documents)} documents...")
    texts = [doc["content"] for doc in documents]
    metadatas = [{"source": doc["source"], "title": doc["title"]} for doc in documents]
    vectorstore.add_texts(texts=texts, metadatas=metadatas)
    print("Ingestion complete.")

# Example usage: Pre-populate with some mock medical data
mock_medical_data = [
    {
        "content": "Ibuprofen is a nonsteroidal anti-inflammatory drug (NSAID) used for pain relief, fever reduction, and anti-inflammatory effects. Common side effects include nausea, vomiting, dyspepsia, and gastrointestinal bleeding. It should be used with caution in patients with kidney disease or asthma.",
        "source": "Medical Journal A",
        "title": "Ibuprofen Guidelines"
    },
    {
        "content": "Paracetamol (Acetaminophen) is a widely used over-the-counter analgesic and antipyretic. It is generally safe at recommended doses, but overdose can cause severe liver damage. It works by inhibiting prostaglandin synthesis in the central nervous system.",
        "source": "Drug Database B",
        "title": "Paracetamol Overview"
    },
    {
        "content": "Diabetes Mellitus Type 2 is a chronic condition characterized by high blood sugar levels. Management includes lifestyle changes, oral medications like metformin, and sometimes insulin. Regular monitoring of blood glucose and HbA1c is crucial.",
        "source": "Clinical Guideline C",
        "title": "Type 2 Diabetes Management"
    },
    {
        "content": "Hypertension (high blood pressure) is a major risk factor for cardiovascular diseases. Treatment involves lifestyle modifications (diet, exercise) and antihypertensive medications such as ACE inhibitors, ARBs, beta-blockers, and diuretics. Regular blood pressure checks are essential.",
        "source": "Cardiology Journal D",
        "title": "Hypertension Treatment Protocols"
    }
]
ingest_documents(mock_medical_data)

# --- Retrieval Module --- #

async def _get_embeddings(text: str) -> List[float]:
    """Generates embeddings for a given text."""
    return embeddings_model.embed_query(text)

async def _perform_semantic_search(query_embedding: List[float], k: int = 5) -> List[Dict[str, Any]]:
    """Performs semantic search on the vector store."""
    # Chroma's similarity_search_by_vector returns documents and their scores
    docs_with_scores = vectorstore.similarity_search_by_vector(query_embedding, k=k)
    return [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs_with_scores]

async def _perform_keyword_search(query: str, k: int = 3) -> List[Dict[str, Any]]:
    """Simulates keyword search (e.g., what Elasticsearch would do)."""
    # For this example, we'll do a simple substring match for demonstration
    results = []
    for doc in mock_medical_data:
        if query.lower() in doc["content"].lower() or query.lower() in doc["title"].lower():
            results.append({"content": doc["content"], "metadata": {"source": doc["source"], "title": doc["title"]}})
        if len(results) >= k: # Limit results
            break
    return results

async def _adaptive_decision_making(query: str) -> str:
    """Decides whether to retrieve external knowledge, rely on LLM, or abstain."""
    doc = nlp(query)
    medical_entities = [ent.text for ent in doc.ents if ent.label_ in ["MED_DRUG", "DISEASE", "SYMPTOM"]]

    if "latest research" in query.lower() or "new guidelines" in query.lower() or medical_entities: # Simplified logic
        return "retrieve"
    elif len(query.split()) < 5: # Very short queries might be simple facts LLM knows
        return "llm_only"
    elif "what is" in query.lower() or "explain" in query.lower(): # General knowledge
        return "retrieve"
    else:
        return "llm_only" # Default to LLM if no strong retrieval signal

async def retrieve_information(query: str) -> List[Dict[str, Any]]:
    """Orchestrates retrieval based on adaptive decision-making."""
    decision = await _adaptive_decision_making(query)
    print(f"Adaptive decision: {decision}")

    if decision == "llm_only":
        return [] # No external retrieval needed
    elif decision == "retrieve":
        query_embedding = await _get_embeddings(query)
        semantic_results = await _perform_semantic_search(query_embedding)
        keyword_results = await _perform_keyword_search(query)

        # Combine and deduplicate results
        combined_results_map = {doc["content"]: doc for doc in semantic_results}
        for doc in keyword_results:
            combined_results_map[doc["content"]] = doc
        
        return list(combined_results_map.values())
    else: # abstain or other complex decision
        return [] # For now, no retrieval

# --- Context Conditioning Module --- #

async def _rerank_documents(query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Simulates reranking documents. In a real system, use a cross-encoder model."""
    # For simplicity, we'll just sort by content length here as a placeholder for relevance
    # A true reranker would use a model to score query-document pairs
    return sorted(documents, key=lambda x: len(x["content"]), reverse=True)

async def _summarize_context(context_text: str, max_length: int = 150, min_length: int = 50) -> str:
    """Summarizes the retrieved context to fit LLM's context window."""
    if not context_text:
        return ""
    try:
        summary = summarizer(context_text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]["summary_text"]
    except Exception as e:
        print(f"Error summarizing: {e}")
        return context_text[:max_length * 2] # Fallback to truncation if summarization fails

async def _build_prompt(query: str, context: str) -> str:
    """Constructs the prompt for the LLM."""
    if context:
        prompt_template = """You are a highly accurate and trustworthy Medical Information Assistant. 
        Answer the following medical question based *strictly* on the provided context. 
        If the answer is not in the context, state that you don't have enough information.

        Context: 
        {context}

        Question: {query}

        Answer:"""
        return prompt_template.format(context=context, query=query)
    else:
        prompt_template = """You are a highly accurate and trustworthy Medical Information Assistant. 
        Answer the following medical question to the best of your knowledge. 
        If you are unsure or the question is complex, state that for critical medical decisions, a healthcare professional should be consulted.

        Question: {query}

        Answer:"""
        return prompt_template.format(query=query)

# --- Generation Module --- #

async def generate_response(query: str, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generates a response using the LLM, potentially augmented with retrieved context."""
    
    # Context Conditioning
    reranked_docs = await _rerank_documents(query, retrieved_docs)
    
    context_texts = [doc["content"] for doc in reranked_docs]
    full_context = "\n\n".join(context_texts)
    
    # Summarize if context is too long (simplified check)
    summarized_context = full_context
    if len(full_context) > 1000: # Arbitrary threshold for summarization
        print("Context too long, summarizing...")
        summarized_context = await _summarize_context(full_context)

    prompt = await _build_prompt(query, summarized_context)

    print(f"Generated Prompt:\n{prompt[:500]}...")
    
    # LLM Chain execution
    llm_chain = LLMChain(prompt=PromptTemplate.from_template(prompt), llm=llm)
    
    # Controlled Generation with Guardrails
    try:
        # Here FactCheck will try to ensure the generated response aligns with the context
        # If on_fail="fix", Guardrails will attempt to regenerate/modify the output
        raw_llm_output, validated_output = await asyncio.to_thread(fact_check_guard.llm_call, 
                                                             prompt,
                                                             llm_chain.run,
                                                             full_context=summarized_context) # Pass full context to FactCheck
        
        response_text = validated_output.output
        # Guardrails might return more than just the text, extract what's needed
        if isinstance(response_text, dict) and "response" in response_text:
            response_text = response_text["response"]
        elif not isinstance(response_text, str):
            response_text = str(response_text) # Ensure it's a string
            
    except Exception as e:
        print(f"Guardrails failed: {e}. Falling back to raw LLM output.")
        response_text = await asyncio.to_thread(llm_chain.run, prompt) # Fallback
    
    sources = []
    for doc in reranked_docs:
        sources.append(f"{doc['metadata'].get('title', 'N/A')} (Source: {doc['metadata'].get('source', 'N/A')})")

    return {"response": response_text, "sources": list(set(sources))}

# --- FastAPI Application --- #

app = FastAPI(
    title="Medical RAG Assistant",
    description="An AI assistant for healthcare professionals leveraging Retrieval-Augmented Generation."
)

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def medical_query(request: QueryRequest):
    """Endpoint to query the medical RAG assistant."""
    query = request.query
    
    # Check cache first
    if query in cache:
        print("Cache hit!")
        return cache[query]

    try:
        # 1. Retrieval
        retrieved_docs = await retrieve_information(query)
        print(f"Retrieved {len(retrieved_docs)} documents.")

        # 2. Generation (with context conditioning and guardrails)
        response_data = await generate_response(query, retrieved_docs)
        
        # Cache the response
        cache[query] = response_data

        return response_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "Medical RAG Assistant is running."}

# To run the application:
# 1. Save this code as `medical_rag_assistant.py`
# 2. Install necessary packages: `pip install fastapi "uvicorn[standard]" langchain_community langchain-openai sentence-transformers transformers spacy guardrails-ai`
#    (Also `pip install "chromadb>=0.4.18"` if not included with langchain_community)
# 3. Download spacy model: `python -m spacy download en_core_web_sm`
# 4. Set your OpenAI API key in the environment or directly in the code: `export OPENAI_API_KEY="YOUR_KEY"`
# 5. Run with Uvicorn: `uvicorn medical_rag_assistant:app --reload --port 8000`
# 6. Access the API at http://localhost:8000/docs for Swagger UI. 

# Example Query:
# POST /query
# { "query": "What are the side effects of Ibuprofen?" }
# { "query": "How is Type 2 Diabetes managed?" }
# { "query": "What is the latest research on hypertension treatment?" } 
# { "query": "Tell me a bedtime story." } # Test adaptive decision/LLM-only path
