import PyPDF2
from sentence_transformers import SentenceTransformer
import chromadb
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from typing import List, Dict

# Ensure NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

class Document:
    def __init__(self, content: str, metadata: Dict = None):
        self.content = content
        self.metadata = metadata if metadata is not None else {}

class DataProcessor:
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
    def _text_splitter(self, text: str, chunk_size: int = 256, chunk_overlap: int = 50) -> List[str]:
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_words = word_tokenize(sentence)
            sentence_length = len(sentence_words)

            if current_length + sentence_length <= chunk_size:
                current_chunk.extend(sentence_words)
                current_length += sentence_length
            else:
                chunks.append(" ".join(current_chunk))
                overlap_start = max(0, len(current_chunk) - chunk_overlap)
                current_chunk = current_chunk[overlap_start:] + sentence_words
                current_length = len(current_chunk)

        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks

    def load_and_process_documents(self, raw_documents: List[Document]) -> List[Dict]:
        processed_data = []
        for i, doc in enumerate(raw_documents):
            if doc.metadata.get("type") == "pdf" and not doc.content.startswith("This is some"):
                text_content = doc.content 
            else:
                text_content = doc.content

            chunks = self._text_splitter(text_content)
            for j, chunk_text in enumerate(chunks):
                embedding = self.embedding_model.encode(chunk_text).tolist()
                processed_data.append({
                    "id": f"{doc.metadata.get('source', 'doc')}_{i}_chunk_{j}",
                    "text": chunk_text,
                    "embedding": embedding,
                    "metadata": {**doc.metadata, "chunk_index": j}
                })
        return processed_data

class KnowledgeBase:
    def __init__(self, collection_name: str = "medical_knowledge_base"):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_documents(self, processed_documents: List[Dict]):
        ids = [d["id"] for d in processed_documents]
        documents = [d["text"] for d in processed_documents]
        embeddings = [d["embedding"] for d in processed_documents]
        metadatas = [d["metadata"] for d in processed_documents]
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Added {len(processed_documents)} chunks to the knowledge base.")

    def retrieve(self, query_embedding: List[float], k: int = 5) -> List[Dict]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=['documents', 'metadatas', 'distances']
        )
        retrieved_contexts = []
        if results and results['documents']:
            for i in range(len(results['documents'][0])):
                retrieved_contexts.append({
                    "text": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i]
                })
        return retrieved_contexts

class RankRAGModel:
    def __init__(self, llm_model_name: str = "distilgpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(llm_model_name)
        self.generator = pipeline(
            "text-generation", 
            model=self.model, 
            tokenizer=self.tokenizer,
            device=-1 
        )
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    def re_rank_contexts(self, query: str, contexts: List[Dict], top_k: int = 3) -> List[Dict]:
        query_embedding = self.embedding_model.encode(query).tolist()
        
        scored_contexts = []
        for context in contexts:
            context_embedding = self.embedding_model.encode(context["text"]).tolist()
            score = sum(q * c for q, c in zip(query_embedding, context_embedding))
            scored_contexts.append({"context": context, "score": score})
        
        scored_contexts.sort(key=lambda x: x["score"], reverse=True)
        
        return [item["context"] for item in scored_contexts[:top_k]]

    def generate_answer(self, query: str, re_ranked_contexts: List[Dict]) -> str:
        context_texts = "\n".join([ctx["text"] for ctx in re_ranked_contexts])
        
        prompt = (
            f"Given the following medical contexts, answer the question accurately and concisely.\n\n"
            f"Contexts:\n{context_texts}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )

        generated_output = self.generator(
            prompt,
            max_new_tokens=150,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        return generated_output[0]['generated_text'].replace(prompt, "").strip()

def main():
    print("--- RankRAG Medical Information Assistant ---")

    print("\n1. Data Ingestion & Preprocessing...")
    medical_documents = [
        Document("This is some dummy content about hypertension guidelines. Hypertension, or high blood pressure, is a common condition.", {"source": "guideline", "title": "Hypertension Guidelines 2023"}),
        Document("Aspirin is a nonsteroidal anti-inflammatory drug (NSAID) used to reduce pain, fever, or inflammation.", {"source": "drug_info", "drug_name": "Aspirin"}),
        Document("Patient X presented with chest pain and shortness of breath. Diagnosis: unstable angina.", {"source": "patient_record", "patient_id": "PX001"}),
        Document("New research suggests a link between diet and cardiovascular disease. Regular exercise is also crucial.", {"source": "research_paper", "title": "Diet and CVD Study"}),
        Document("The normal range for blood pressure is generally considered to be less than 120/80 mmHg. Lifestyle modifications are often the first line of treatment.", {"source": "guideline", "title": "Hypertension Guidelines 2023_Part2"}),
        Document("Ibuprofen, another NSAID, is used for pain relief and reducing inflammation. It works by inhibiting prostaglandin synthesis.", {"source": "drug_info", "drug_name": "Ibuprofen"}),
    ]

    data_processor = DataProcessor()
    processed_docs = data_processor.load_and_process_documents(medical_documents)
    print(f"Processed {len(processed_docs)} document chunks.")

    print("\n2. Initializing Knowledge Base...")
    knowledge_base = KnowledgeBase()
    knowledge_base.add_documents(processed_docs)
    
    print("\n3. Initializing RankRAG Model (simulated)...")
    rankrag_model = RankRAGModel()

    user_query = "What are the guidelines for managing high blood pressure and common medications?"
    print(f"\nUser Query: {user_query}")

    print("\n--- Initial Retrieval ---")
    query_embedding = data_processor.embedding_model.encode(user_query).tolist()
    initial_retrieved_contexts = knowledge_base.retrieve(query_embedding, k=10)
    print(f"Retrieved {len(initial_retrieved_contexts)} initial contexts.")

    print("\n--- Context Re-ranking ---")
    re_ranked_contexts = rankrag_model.re_rank_contexts(user_query, initial_retrieved_contexts, top_k=3)
    print(f"Re-ranked to {len(re_ranked_contexts)} contexts:")
    for i, ctx in enumerate(re_ranked_contexts):
        print(f"  Rank {i+1} (Source: {ctx['metadata'].get('source')}): {ctx['text']}")

    print("\n--- Answer Generation ---")
    generated_answer = rankrag_model.generate_answer(user_query, re_ranked_contexts)
    print("\nGenerated Answer:")
    print(generated_answer)
    print("\n--- End of Demonstration ---")

if __name__ == "__main__":
    main()