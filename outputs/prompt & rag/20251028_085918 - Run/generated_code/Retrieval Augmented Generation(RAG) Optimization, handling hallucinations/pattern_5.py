
import streamlit as st
import functools
import asyncio
import logging
import os
from dotenv import load_dotenv
import numpy as np
from typing import List, Dict, Any

# --- Mocking external libraries and services for demonstration ---
# In a real application, you would install and use these libraries:
# pip install scikit-learn numpy sentence-transformers faiss-cpu openai beautifulsoup4 python-dotenv streamlit

# Mock SentenceTransformer
class MockSentenceTransformer:
    def encode(self, sentences, convert_to_tensor=False):
        # Simulate embeddings as random numpy arrays
        return np.random.rand(len(sentences), 768).astype(np.float32)

# Mock FAISS (in-memory for simplicity)
class MockFAISSIndex:
    def __init__(self, dimension):
        self.dimension = dimension
        self.index = None
        self.texts = []

    def add_vectors(self, vectors: np.ndarray, texts: List[str]):
        if self.index is None:
            # Simulate FAISS index creation
            # In a real scenario, you'd use faiss.IndexFlatL2 or similar
            self.index = vectors # Simplified: store vectors directly
            self.texts = texts
        else:
            self.index = np.vstack((self.index, vectors))
            self.texts.extend(texts)

    def search(self, query_vector: np.ndarray, k: int) -> List[Dict[str, Any]]:
        if self.index is None or len(self.index) == 0:
            return []

        # Simulate vector search by finding closest vectors
        # In a real FAISS, this would be `D, I = self.index.search(query_vector, k)`
        
        # For simplicity, calculate Euclidean distance to all vectors
        distances = np.linalg.norm(self.index - query_vector, axis=1)
        
        # Get indices of the k smallest distances
        top_k_indices = np.argsort(distances)[:k]

        results = []
        for idx in top_k_indices:
            results.append({"text": self.texts[idx], "similarity": 1 - distances[idx] / np.max(distances)})
            # Simplified similarity score
        return results

# Mock OpenAI API Client
class MockOpenAI:
    def chat(self):
        return self

    def completions(self, model, messages, **kwargs):
        class MockCompletion:
            def __init__(self, content):
                self.content = content

            @property
            def message(self):
                class MockMessage:
                    def __init__(self, content):
                        self.content = content
                    
                    @property
                    def content(self):
                        return self.content

                return MockMessage(self.content)

        # Simulate LLM response
        user_message = next((m["content"] for m in messages if m["role"] == "user"), "")
        context_message = next((m["content"] for m in messages if m["role"] == "system" and "context" in m["content"]), "")

        if "COVID-19 treatment" in user_message and "Remdesivir" in context_message:
            response_content = "Based on the provided context, Remdesivir is an antiviral drug approved for the treatment of COVID-19 in certain hospitalized patients."
        elif "diabetes management" in user_message and "metformin" in context_message:
            response_content = "According to the retrieved information, Metformin is a first-line medication for type 2 diabetes, primarily working by decreasing glucose production in the liver."
        elif "cancer diagnosis" in user_message:
            response_content = "For cancer diagnosis, the context suggests a combination of imaging (MRI, CT), biopsies, and laboratory tests are crucial."
        else:
            response_content = f"I'm an AI medical assistant. Based on your query '{user_message}' and the provided context, I can provide information. Please note that this is a simulated response and not real medical advice. The context was: {context_message[:100]}..."

        class MockResponse:
            def __init__(self, content):
                self.choices = [MockCompletion(content)]

        return MockResponse(response_content)

# --- Configuration and Setup ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Global Components (instantiated once) ---
@st.cache_resource
def get_embedding_model():
    # In a real app: return SentenceTransformer('all-MiniLM-L6-v2') or a BioBERT model
    return MockSentenceTransformer()

@st.cache_resource
def get_vector_db():
    # In a real app: return Pinecone(api_key=..., environment=...) or Chroma.Client()
    # Using a fixed dimension for mock embeddings
    return MockFAISSIndex(dimension=768)

# --- Data Ingestion and Indexing --- 
class MedicalDocument:
    def __init__(self, id: str, text: str, source: str):
        self.id = id
        self.text = text
        self.source = source

class MedicalLiteratureScraper:
    def fetch_documents(self) -> List[MedicalDocument]:
        # Simulate fetching medical literature
        sample_docs = [
            MedicalDocument("doc1", "Remdesivir is an antiviral medication developed by Gilead Sciences. It was approved for use in COVID-19 treatment for certain hospitalized patients to shorten recovery time. Clinical trials have shown its efficacy.", "ClinicalTrials.gov"),
            MedicalDocument("doc2", "Metformin is a medication commonly used to treat type 2 diabetes. It helps control high blood sugar by decreasing glucose production in the liver and improving insulin sensitivity. It is often the first-line pharmacotherapy.", "Diabetes Care Journal"),
            MedicalDocument("doc3", "Early diagnosis of cancer significantly improves prognosis. Diagnostic methods include imaging techniques like MRI and CT scans, biopsies for histological examination, and blood tests for tumor markers.", "Oncology Today"),
            MedicalDocument("doc4", "The recommended dosage for adult patients with moderate to severe COVID-19 requiring supplemental oxygen is a 200 mg loading dose of Remdesivir on Day 1, followed by 100 mg once daily for 5 days. Renal function must be monitored.", "FDA Guidelines"),
            MedicalDocument("doc5", "Lifestyle modifications, including diet and exercise, are crucial in managing type 2 diabetes alongside medications like Metformin. Regular monitoring of blood glucose levels is essential to prevent complications.", "Endocrine Society"),
        ]
        logging.info(f"Fetched {len(sample_docs)} sample medical documents.")
        return sample_docs

class DocumentProcessor:
    def __init__(self, chunk_size: int = 250, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str) -> List[str]:
        # A simple character-based chunking for demonstration
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunks.append(text[i : i + self.chunk_size])
        return chunks

    def process_documents(self, documents: List[MedicalDocument]) -> List[Dict[str, str]]:
        processed_chunks = []
        for doc in documents:
            chunks = self.chunk_text(doc.text)
            for i, chunk in enumerate(chunks):
                processed_chunks.append({
                    "id": f"{doc.id}_{i}",
                    "text": chunk,
                    "source": doc.source,
                    "original_doc_id": doc.id
                })
        logging.info(f"Processed {len(documents)} documents into {len(processed_chunks)} chunks.")
        return processed_chunks

class EmbeddingGenerator:
    def __init__(self, model):
        self.model = model

    @functools.lru_cache(maxsize=1000) # Cache embeddings for efficiency
    def generate_embedding(self, text: str) -> np.ndarray:
        return self.model.encode([text])[0]

def index_medical_knowledge(scraper: MedicalLiteratureScraper, processor: DocumentProcessor, 
                            embedder: EmbeddingGenerator, vector_db: MockFAISSIndex):
    st.session_state["indexing_status"] = "Starting indexing..."
    raw_documents = scraper.fetch_documents()
    processed_chunks = processor.process_documents(raw_documents)
    
    texts_to_embed = [chunk["text"] for chunk in processed_chunks]
    embeddings = np.array([embedder.generate_embedding(text) for text in texts_to_embed])
    
    vector_db.add_vectors(embeddings, texts_to_embed)
    st.session_state["indexing_status"] = f"Indexing complete. Added {len(processed_chunks)} chunks to vector DB."
    logging.info(st.session_state["indexing_status"])

# --- Retrieval-Augmented Generation (RAG) Core ---
class IntelligentRetriever:
    def __init__(self, embedder: EmbeddingGenerator, vector_db: MockFAISSIndex, top_k: int = 5):
        self.embedder = embedder
        self.vector_db = vector_db
        self.top_k = top_k

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        query_embedding = self.embedder.generate_embedding(query)
        retrieved_docs = self.vector_db.search(query_embedding, k=self.top_k)
        logging.info(f"Retrieved {len(retrieved_docs)} documents for query: '{query[:50]}...'")
        return retrieved_docs

    def rerank(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Simplified re-ranking: just return as is for demonstration
        # In a real system, a cross-encoder or a more sophisticated model would re-order based on relevance.
        logging.info("Skipping re-ranking (simplified for demo).")
        return documents

class ContextConditioner:
    def format_context(self, retrieved_documents: List[Dict[str, Any]], max_tokens: int = 2000) -> str:
        context_parts = []
        current_tokens = 0
        for i, doc in enumerate(retrieved_documents):
            doc_text = doc.get("text", "")
            # Estimate tokens by character count / 4
            doc_tokens = len(doc_text) // 4 
            if current_tokens + doc_tokens < max_tokens:
                context_parts.append(f"Document {i+1} (Source: {doc.get('source', 'N/A')}): {doc_text}")
                current_tokens += doc_tokens
            else:
                break
        formatted_context = "\n\n".join(context_parts)
        logging.info(f"Formatted context with approx {current_tokens} tokens from {len(context_parts)} documents.")
        return formatted_context

class AdaptiveDecisionMaker:
    def decide_action(self, query: str, retrieved_context: str) -> str:
        # Simplified decision logic for demonstration
        if not retrieved_context or len(retrieved_context.strip()) < 50:
            return "clarify"
        elif "complex" in query.lower() or "detailed explanation" in query.lower():
            return "retrieve_and_generate"
        else:
            return "generate_directly" # Simulate LLM could answer directly if context not strictly needed

class LLMService:
    def __init__(self, api_client):
        self.client = api_client # MockOpenAI instance
        self.model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

    @functools.lru_cache(maxsize=500) # Cache LLM responses
    async def generate_response(self, prompt: str, context: str = "") -> str:
        messages = []
        if context:
            messages.append({"role": "system", "content": f"You are a medical assistant. Use the following context to answer the user's question accurately and thoroughly:\n\nContext: {context}"})
        messages.append({"role": "user", "content": prompt})

        logging.info(f"Sending prompt to LLM (model: {self.model}, context_len: {len(context)})...")
        
        # Simulate async call
        await asyncio.sleep(0.1) # Non-blocking sleep
        response = self.client.chat.completions(model=self.model, messages=messages, temperature=0.7, max_tokens=500)
        
        return response.choices[0].message.content

class MedicalKnowledgeAssistant:
    def __init__(self, retriever: IntelligentRetriever, conditioner: ContextConditioner, 
                 decision_maker: AdaptiveDecisionMaker, llm_service: LLMService):
        self.retriever = retriever
        self.conditioner = conditioner
        self.decision_maker = decision_maker
        self.llm_service = llm_service

    async def answer_query(self, query: str) -> Dict[str, Any]:
        logging.info(f"Answering query: {query}")
        retrieved_docs = self.retriever.retrieve(query)
        reranked_docs = self.retriever.rerank(query, retrieved_docs)
        
        context = self.conditioner.format_context(reranked_docs)
        
        action = self.decision_maker.decide_action(query, context)
        
        final_answer = ""
        if action == "clarify":
            final_answer = "I need more information to provide a precise answer. Could you please elaborate on your query?"
        elif action == "retrieve_and_generate" or action == "generate_directly":
            final_answer = await self.llm_service.generate_response(query, context)
        
        return {"answer": final_answer, "retrieved_context": context, "action_taken": action}

# --- Streamlit UI --- 
st.set_page_config(layout="wide", page_title="Medical Knowledge Assistant")

def main():
    st.title("🩺 Medical Knowledge Assistant")
    st.markdown("--- This RAG-powered system provides accurate and up-to-date medical information. --- A demonstration using mocked services. ---")

    # Initialize components
    embedder = get_embedding_model()
    vector_db = get_vector_db()
    scraper = MedicalLiteratureScraper()
    processor = DocumentProcessor()
    mock_openai_client = MockOpenAI()
    llm_service = LLMService(mock_openai_client)

    retriever = IntelligentRetriever(embedder, vector_db)
    conditioner = ContextConditioner()
    decision_maker = AdaptiveDecisionMaker()
    assistant = MedicalKnowledgeAssistant(retriever, conditioner, decision_maker, llm_service)

    # Indexing status
    if "indexing_status" not in st.session_state:
        st.session_state["indexing_status"] = "Not started"
    
    st.sidebar.header("System Status")
    st.sidebar.text(f"Indexing: {st.session_state['indexing_status']}")

    if st.sidebar.button("Re-index Medical Knowledge (Demo Data)"):
        with st.spinner("Indexing documents and generating embeddings..."):
            index_medical_knowledge(scraper, processor, embedder, vector_db)
        st.sidebar.success("Indexing complete!")

    st.sidebar.markdown("--- Queries will use cached embeddings and LLM responses where possible. ---")

    query = st.text_area("Enter your medical query here:", "What is Remdesivir used for in COVID-19 treatment?")

    if st.button("Get Medical Information"):
        if not query.strip():
            st.warning("Please enter a query.")
            return
        
        # Ensure indexing has happened at least once
        if vector_db.index is None or len(vector_db.texts) == 0:
            st.error("Medical knowledge not indexed. Please click 'Re-index Medical Knowledge'.")
            return

        with st.spinner("Retrieving and generating answer..."):
            # Run async function in a synchronous Streamlit context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(assistant.answer_query(query))
            loop.close()

        st.subheader("Generated Answer:")
        st.write(result["answer"])

        st.subheader("Action Taken:")
        st.info(result["action_taken"])

        st.subheader("Retrieved Context (Used by LLM):")
        if result["retrieved_context"]:
            st.expander("View Context").markdown(result["retrieved_context"])
        else:
            st.info("No specific context was retrieved or used for this query.")

if __name__ == "__main__":
    main()
