import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
from functools import lru_cache
import torch

class MedicalKnowledgeBase:
    def __init__(self, documents: List[str], embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.documents = documents
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embeddings = self.embedding_model.encode(documents, convert_to_tensor=True).cpu().numpy()
        self.dimension = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(self.dimension)
        self.faiss_index.add(self.embeddings)

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        distances, indices = self.faiss_index.search(query_embedding, k)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                results.append({"text": self.documents[idx], "distance": distances[0][i]})
        return results

class QueryAnalyzer:
    def analyze(self, query: str) -> Dict[str, str]:
        if "diagnosis" in query.lower() or "symptoms" in query.lower():
            return {"type": "diagnostic", "complexity": "complex"}
        elif "drug interaction" in query.lower() or "medication" in query.lower():
            return {"type": "drug_interaction", "complexity": "complex"}
        elif "treatment" in query.lower() or "therapy" in query.lower():
            return {"type": "treatment", "complexity": "complex"}
        else:
            return {"type": "factual", "complexity": "simple"}

class AdaptiveRAGEngine:
    def __init__(
        self, 
        knowledge_base: MedicalKnowledgeBase,
        reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        llm_model_name: str = "gpt2"
    ):
        self.knowledge_base = knowledge_base
        self.query_analyzer = QueryAnalyzer()

        self.reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_model_name)
        self.reranker_model = AutoModelForSequenceClassification.from_pretrained(reranker_model_name)
        self.reranker_pipeline = pipeline("text-classification", model=self.reranker_model, tokenizer=self.reranker_tokenizer)

        self.llm_pipeline = pipeline("text-generation", model=llm_model_name, device=0 if torch.cuda.is_available() else -1)

    @lru_cache(maxsize=128)
    def _get_embedding(self, text: str) -> np.ndarray:
        return self.knowledge_base.embedding_model.encode(text, convert_to_tensor=True).cpu().numpy().reshape(1, -1)

    def _retrieve_documents(self, query_embedding: np.ndarray, k: int) -> List[Dict[str, Any]]:
        return self.knowledge_base.search(query_embedding, k=k)

    def _rerank_documents(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not documents:
            return []
        
        pairs = [[query, doc["text"]] for doc in documents]
        scores = [res["score"] for res in self.reranker_pipeline(pairs)]
        
        for i, doc in enumerate(documents):
            doc["relevance_score"] = scores[i]
        
        return sorted(documents, key=lambda x: x["relevance_score"], reverse=True)

    def _self_reflect(self, retrieved_contexts: List[str], generated_answer: str) -> float:
        if not retrieved_contexts or not generated_answer:
            return 0.1
        confidence = 0.5 + 0.1 * len(retrieved_contexts) + 0.1 * (len(generated_answer.split()) / 50)
        return min(confidence, 0.95)

    def _generate_answer(self, query: str, contexts: List[str]) -> str:
        prompt = f"Given the following medical contexts, answer the query accurately and concisely:\n\nContexts:\n"
        for i, context in enumerate(contexts):
            prompt += f"{i+1}. {context}\n"
        prompt += f"\nQuery: {query}\nAnswer:"

        try:
            response = self.llm_pipeline(prompt, max_new_tokens=200, num_return_sequences=1, do_sample=False)
            generated_text = response[0]["generated_text"].replace(prompt, "").strip()
            return generated_text if generated_text else "Could not generate a conclusive answer based on the provided information."
        except Exception as e:
            return f"An error occurred during answer generation: {e}"

    def process_query(self, query: str, max_iterations: int = 2) -> Dict[str, Any]:
        query_analysis = self.query_analyzer.analyze(query)
        query_embedding = self._get_embedding(query)
        
        all_retrieved_contexts = []
        final_answer = ""
        confidence_score = 0.0
        
        for iteration in range(max_iterations):
            k = 5 if query_analysis["complexity"] == "simple" else 10
            
            if query_analysis["type"] == "drug_interaction":
                retrieved_docs = self._retrieve_documents(query_embedding, k=k+2)
            else:
                retrieved_docs = self._retrieve_documents(query_embedding, k=k)
            
            reranked_docs = self._rerank_documents(query, retrieved_docs)
            
            current_contexts = [doc["text"] for doc in reranked_docs if doc["relevance_score"] > 0.5]
            all_retrieved_contexts.extend(current_contexts)

            if not current_contexts and iteration == 0:
                final_answer = "No relevant information found in the knowledge base."
                confidence_score = 0.1
                break

            generated_answer = self._generate_answer(query, list(set(all_retrieved_contexts)))

            confidence_score = self._self_reflect(list(set(all_retrieved_contexts)), generated_answer)

            if confidence_score > 0.7 or iteration == max_iterations - 1:
                final_answer = generated_answer
                break
            
        if confidence_score < 0.5:
            final_answer = "I am not confident enough to provide a definitive answer. More information might be needed."
            
        return {
            "query": query,
            "query_analysis": query_analysis,
            "answer": final_answer,
            "confidence": confidence_score,
            "sources": [doc["text"] for doc in reranked_docs[:3]]
        }

app = FastAPI()

dummy_medical_docs = [
    "Acute myocardial infarction (heart attack) is a serious condition where blood flow to the heart muscle is blocked.",
    "Symptoms of a heart attack include chest pain, shortness of breath, pain in the left arm, and sweating.",
    "Common treatments for myocardial infarction involve aspirin, beta-blockers, ACE inhibitors, and statins.",
    "Aspirin helps prevent blood clots. Beta-blockers reduce heart rate and blood pressure.",
    "Drug interactions between Warfarin and NSAIDs can increase the risk of bleeding.",
    "Metformin is a common medication for type 2 diabetes. Side effects can include nausea and diarrhea.",
    "Diagnosing appendicitis often involves physical examination, blood tests (elevated white blood cell count), and imaging like CT scans or ultrasound.",
    "Pneumonia is an infection that inflames air sacs in one or both lungs, which may fill with fluid or pus.",
    "Treatment for bacterial pneumonia typically includes antibiotics. Viral pneumonia usually resolves on its own.",
    "Hypertension, or high blood pressure, increases the risk of heart disease and stroke. Lifestyle changes and medication can manage it."
]

kb = MedicalKnowledgeBase(dummy_medical_docs)
rag_engine = AdaptiveRAGEngine(kb)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query: str
    query_analysis: Dict[str, str]
    answer: str
    confidence: float
    sources: List[str]

@app.post("/query", response_model=QueryResponse)
async def process_medical_query(request: QueryRequest):
    result = rag_engine.process_query(request.query)
    return QueryResponse(**result)