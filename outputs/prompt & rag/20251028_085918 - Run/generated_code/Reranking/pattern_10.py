import random
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

# --- 1. Dummy Data and In-Memory Document Store ---
medical_documents = [
    "Aspirin is commonly used as an analgesic for pain relief and an antipyretic for fever reduction. It also has anti-inflammatory effects and is used as an antiplatelet agent to prevent blood clots.",
    "Type 2 diabetes is a chronic condition that affects the way the body processes blood sugar (glucose). It is characterized by insulin resistance or insufficient insulin production.",
    "Hypertension, or high blood pressure, is a common condition in which the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease.",
    "The COVID-19 pandemic caused by SARS-CoV-2 has led to a global health crisis. Symptoms include fever, cough, fatigue, and loss of taste or smell. Vaccination is a key preventive measure.",
    "Chemotherapy is a type of cancer treatment that uses drugs to kill cancer cells. It works by stopping or slowing the growth of cancer cells, which grow and divide quickly.",
    "Migraine is a severe type of headache characterized by throbbing pain on one side of the head, sensitivity to light and sound, and sometimes nausea or vomiting.",
    "Common side effects of antibiotics include nausea, diarrhea, and allergic reactions. It's important to complete the full course of treatment to prevent antibiotic resistance.",
    "Asthma is a chronic respiratory condition characterized by inflammation and narrowing of the airways, leading to symptoms like wheezing, shortness of breath, chest tightness, and coughing.",
    "Influenza, commonly known as the flu, is a contagious respiratory illness caused by influenza viruses. Symptoms are similar to the common cold but are usually more severe and can include high fever, body aches, and extreme fatigue.",
    "Magnetic Resonance Imaging (MRI) is a medical imaging technique used in radiology to form pictures of the anatomy and the physiological processes of the body in both health and disease. MRI scanners use strong magnetic fields and radio waves."
]

class InMemoryVectorStore:
    def __init__(self, documents, embedding_model):
        self.documents = documents
        self.embedding_model = embedding_model
        self.document_embeddings = self.embedding_model.encode(documents, convert_to_tensor=False)

    def retrieve(self, query, top_k=3):
        query_embedding = self.embedding_model.encode(query, convert_to_tensor=False)
        # Calculate cosine similarity
        similarities = np.dot(self.document_embeddings, query_embedding) / \
                       (np.linalg.norm(self.document_embeddings, axis=1) * np.linalg.norm(query_embedding))
        
        # Get top_k indices
        top_k_indices = np.argsort(similarities)[::-1][:top_k]
        return [(self.documents[i], similarities[i]) for i in top_k_indices]

# --- 2. Embedding Model ---
# Using a SentenceTransformer for general purpose embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
vector_store = InMemoryVectorStore(medical_documents, embedding_model)

# --- 3. Main Language Model (Conceptual) ---
# For this demonstration, the "main LM" will simulate generating a response based on the context.
# In a real application, this would be a powerful generative LM (e.g., GPT-3.5, Llama2, Mistral).
class ConceptualMainLM:
    def __init__(self):
        # For a real application, initialize a transformers pipeline for text generation here.
        # E.g., self.generator = pipeline("text-generation", model="distilgpt2")
        pass

    def generate_response(self, prompt):
        # Simulate LM understanding and response generation.
        if "Aspirin" in prompt and "pain relief" in prompt:
            return "Based on the provided information, Aspirin is indicated for pain relief, fever reduction, and has anti-inflammatory and antiplatelet effects."
        elif "Type 2 diabetes" in prompt and "blood sugar" in prompt:
            return "The context indicates that Type 2 diabetes is a chronic condition affecting blood sugar processing, often due to insulin resistance."
        elif "Hypertension" in prompt or "blood pressure" in prompt:
            return "According to the documents, hypertension refers to high blood pressure which can lead to heart disease."
        elif "COVID-19" in prompt or "SARS-CoV-2" in prompt:
            return "The provided text discusses COVID-19 as a global health crisis caused by SARS-CoV-2, mentioning symptoms and vaccination."
        elif "Chemotherapy" in prompt and "cancer" in prompt:
            return "Chemotherapy is a cancer treatment using drugs to target and kill fast-growing cancer cells."
        elif "Migraine" in prompt and "headache" in prompt:
            return "Migraine is described as a severe headache, often with throbbing pain, light/sound sensitivity, and potential nausea/vomiting."
        elif "antibiotics" in prompt and "side effects" in prompt:
            return "Common antibiotic side effects include nausea, diarrhea, and allergic reactions. Completing the course prevents resistance."
        elif "Asthma" in prompt and "airways" in prompt:
            return "Asthma is a chronic condition involving inflamed and narrowed airways, causing wheezing, shortness of breath, and coughing."
        elif "Influenza" in prompt or "flu" in prompt:
            return "Influenza, or the flu, is a contagious respiratory illness caused by viruses, with more severe symptoms than a common cold."
        elif "MRI" in prompt and "imaging" in prompt:
            return "MRI is a medical imaging technique that uses magnetic fields and radio waves to visualize body anatomy and processes."
        else:
            return f"I processed your query based on the context provided. Key information includes: {prompt[:150]}..."


# --- 4. Zero-Shot LM Reranking ---
# Using a Cross-Encoder for reranking based on semantic similarity
# This model is specifically trained to score pairs of (query, document) for relevance.
reranker_model = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2")

class ZeroShotReranker:
    def __init__(self, reranker_model):
        self.reranker_model = reranker_model

    def rerank(self, query, documents_with_scores):
        # documents_with_scores is a list of (document_text, similarity_score)
        sentences = [[query, doc_text] for doc_text, _ in documents_with_scores]
        rerank_scores = self.reranker_model.predict(sentences)
        
        # Combine original documents with new rerank scores and sort
        reranked_docs = sorted(zip(documents_with_scores, rerank_scores), key=lambda x: x[1], reverse=True)
        return [(doc_text, new_score) for (doc_text, original_score), new_score in reranked_docs]

# --- 5. Predictive Reranking (Conceptual Trained LM-Dedicated Reranker) ---
# This is a conceptual implementation. In a real scenario, this would be a trained model.
class PredictiveReranker:
    def __init__(self):
        # In a real system, load a trained model here (e.g., from scikit-learn, PyTorch, TensorFlow)
        # For conceptual demo, we'll just simulate a reranking effect.
        pass

    def rerank(self, query, documents_with_scores):
        # Simulate a learned reranking strategy. For simplicity, we'll just slightly perturb
        # the zero-shot scores or apply a heuristic based on query length/keywords.
        # This would ideally be a model trained to maximize downstream LM performance.
        print("\n[Predictive Reranker Applied - conceptually adjusting ranks]")
        # Example conceptual reranking: prioritize documents with more query keywords
        reranked = []
        query_keywords = set(query.lower().split())
        for doc_text, score in documents_with_scores:
            doc_keywords = set(doc_text.lower().split())
            overlap = len(query_keywords.intersection(doc_keywords))
            # A simplistic way to 