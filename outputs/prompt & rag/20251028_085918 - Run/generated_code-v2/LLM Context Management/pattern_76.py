import torch
import numpy as np
from transformers import pipeline
from sentence_transformers import SentenceTransformer

class KnowledgeBase:
    def __init__(self, documents):
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.documents = documents
        self.embeddings = self._embed_documents(documents)

    def _embed_documents(self, documents):
        texts = [doc["text"] for doc in documents]
        return self.embedding_model.encode(texts, convert_to_tensor=True)

    def retrieve(self, query_embedding, k=3):
        query_embedding_np = query_embedding.cpu().numpy()
        document_embeddings_np = self.embeddings.cpu().numpy()

        similarities = np.dot(document_embeddings_np, query_embedding_np.T) / \
                       (np.linalg.norm(document_embeddings_np, axis=1) * np.linalg.norm(query_embedding_np))

        top_k_indices = np.argsort(similarities)[::-1][:k]
        
        retrieved_docs = []
        for i in top_k_indices:
            doc = self.documents[i]
            retrieved_docs.append({"text": doc["text"], "source": doc["source"]})
        return retrieved_docs

class MedicalResearchAssistant:
    def __init__(self, llm_model_name="google/flan-t5-small", embedding_model_name="all-MiniLM-L6-v2", knowledge_base_documents=None):
        if knowledge_base_documents is None:
            knowledge_base_documents = []
        self.knowledge_base = KnowledgeBase(knowledge_base_documents)
        self.llm = pipeline("text2text-generation", model=llm_model_name)
        self.embedding_model = self.knowledge_base.embedding_model

    def answer_query(self, query):
        query_embedding = self.embedding_model.encode(query, convert_to_tensor=True)
        retrieved_docs = self.knowledge_base.retrieve(query_embedding)

        context = "\n".join([doc["text"] for doc in retrieved_docs])
        sources = [doc["source"] for doc in retrieved_docs]

        prompt = f"""Context:
{context}

Question: {query}

Answer:"""

        generated_response = self.llm(prompt, max_new_tokens=200, do_sample=False)[0]["generated_text"]
        
        return generated_response, list(set(sources))

if __name__ == "__main__":
    sample_medical_documents = [
        {"text": "Aspirin is a nonsteroidal anti-inflammatory drug (NSAID) used to reduce fever and relieve mild to moderate pain. It is also used as an antiplatelet agent to prevent blood clots.", "source": "Mayo Clinic - Aspirin"},
        {"text": "Paracetamol (acetaminophen) is a common painkiller used to treat aches and pain and reduce high temperature. It's available in many forms, including tablets, capsules, and syrup.", "source": "NHS - Paracetamol"},
        {"text": "Diabetes mellitus, commonly known as diabetes, is a metabolic disease that causes high blood sugar. The hormone insulin moves sugar from the blood into your cells to be stored for energy.", "source": "WHO - Diabetes"},
        {"text": "Type 1 diabetes is a chronic condition in which the pancreas produces little or no insulin. It typically appears in childhood or adolescence.", "source": "CDC - Type 1 Diabetes"},
        {"text": "Hypertension, or high blood pressure, is a common condition in which the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease.", "source": "American Heart Association - Hypertension"},
        {"text": "The recommended daily allowance of Vitamin D for adults is 600-800 IU. It plays a crucial role in bone health and immune function.", "source": "NIH - Vitamin D Fact Sheet"},
        {"text": "Common symptoms of a myocardial infarction (heart attack) include chest pain, shortness of breath, pain in the left arm, and lightheadedness.", "source": "World Heart Federation - Heart Attack"}
    ]

    assistant = MedicalResearchAssistant(knowledge_base_documents=sample_medical_documents)

    queries = [
        "What is aspirin used for?",
        "What are the symptoms of a heart attack?",
        "What is diabetes?",
        "Recommended daily intake of Vitamin D?",
        "What is paracetamol?"
    ]

    print("\n--- Medical Research Assistant Demo ---\n")
    for i, query in enumerate(queries):
        print(f"Query {i+1}: {query}")
        answer, sources = assistant.answer_query(query)
        print(f"Answer: {answer}")
        print(f"Sources: {', '.join(sources)}\n")