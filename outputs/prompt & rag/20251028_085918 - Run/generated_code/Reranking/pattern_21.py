import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

class MedInfoBot:
    def __init__(self, document_paths=None, embedding_model_name='all-MiniLM-L6-v2', top_k_retrieval=5):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.documents = []
        self.document_embeddings = None
        self.faiss_index = None
        self.top_k_retrieval = top_k_retrieval

        if document_paths:
            self._load_documents(document_paths)
            self._build_knowledge_base()

    def _load_documents(self, document_paths):
        for path in document_paths:
            with open(path, 'r', encoding='utf-8') as f:
                self.documents.append(f.read())
        print(f"Loaded {len(self.documents)} documents.")

    def _build_knowledge_base(self):
        print("Building knowledge base embeddings...")
        self.document_embeddings = self.embedding_model.encode(self.documents, convert_to_tensor=False)
        dimension = self.document_embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(np.array(self.document_embeddings).astype('float32'))
        print("Knowledge base built successfully.")

    def _retrieve_documents(self, query: str) -> list:
        query_embedding = self.embedding_model.encode([query], convert_to_tensor=False).astype('float32')
        distances, indices = self.faiss_index.search(query_embedding, self.top_k_retrieval)
        
        retrieved_docs_with_scores = []
        for i, idx in enumerate(indices[0]):
            retrieved_docs_with_scores.append({
                "content": self.documents[idx],
                "score": 1 - distances[0][i] / (2 * np.max(distances[0]) + 1e-6) # Normalize distance to a similarity score
            })
        return retrieved_docs_with_scores

    def _rerank_documents(self, query: str, retrieved_docs: list) -> list:
        # Simulate zero-shot LM-based reranking. In a real scenario, this would involve
        # prompting an LM to score the relevance of each retrieved document to the query.
        # For this prototype, we'll assume the initial retrieval is good enough or
        # a simple heuristic like prioritizing documents with higher similarity scores (already implicit).
        # A more advanced reranker would involve a cross-encoder model.
        
        # Here, we'll just sort by the existing similarity score from retrieval
        reranked_docs = sorted(retrieved_docs, key=lambda x: x['score'], reverse=True)
        print(f"Reranked {len(reranked_docs)} documents.")
        return reranked_docs

    def _construct_prompt(self, query: str, grounded_docs: list) -> str:
        context_str = "\n\n".join([f"Document {i+1}:\n{doc['content']}" for i, doc in enumerate(grounded_docs)])
        prompt = (
            f"You are a medical information assistant. Use the provided medical context "
            f"to answer the user's query accurately and comprehensively. "
            f"If the context does not contain enough information, state that you cannot provide a complete answer based on the given context.\n\n"
            f"Medical Context:\n{context_str}\n\n"
            f"User Query: {query}\n\n"
            f"Answer:"
        )
        return prompt

    def _call_language_model(self, prompt: str) -> str:
        # This is a placeholder for an actual Language Model API call or local inference.
        # In a real application, you would integrate with an LM like Google's Gemini, OpenAI's GPT, etc.
        # For demonstration, we'll simulate a response.
        print("Calling Language Model...")
        if "rare disease" in prompt.lower() and "Document 1" in prompt:
            return "Based on the provided medical context (Document 1), a rare disease like 'XYZ Syndrome' presents with symptoms such as fever, rash, and joint pain. Treatment typically involves supportive care and specific medications depending on the severity. Further consultation with a specialist is recommended."
        elif "drug interaction" in prompt.lower() and "Document 1" in prompt:
            return "According to the context (Document 1), a significant interaction exists between Drug A and Drug B, potentially leading to increased risk of cardiac arrhythmias. Close monitoring of ECG and electrolyte levels is advised when co-administering these medications."
        elif not any(doc in prompt for doc in [f"Document {i+1}" for i in range(self.top_k_retrieval)]):
            return "I cannot provide a complete answer to your query as the necessary medical context was not retrieved."
        else:
            return "Based on the provided medical context, the answer to your query is: [Simulated comprehensive answer based on retrieved documents]."

    def _should_retrieve(self, query: str) -> bool:
        # Implement conditional retrieval logic.
        # For simplicity, always retrieve for now, but in a real system,
        # this could use a small LM or keyword matching to decide if external knowledge is needed.
        # E.g., if query is a simple factual lookup vs. complex medical scenario.
        if any(keyword in query.lower() for keyword in ["rare disease", "drug interaction", "treatment protocol", "patient case"]):
            return True
        return False # Default to not retrieve if query seems simple, but for prototype always retrieve

    def query(self, user_query: str) -> str:
        print(f"User Query: {user_query}")

        if self._should_retrieve(user_query):
            print("Triggering knowledge retrieval...")
            retrieved_docs = self._retrieve_documents(user_query)
            
            if not retrieved_docs:
                return "I could not find relevant medical information in the knowledge base for your query."
            
            reranked_docs = self._rerank_documents(user_query, retrieved_docs)
            grounded_prompt = self._construct_prompt(user_query, reranked_docs)
        else:
            print("Answering directly without knowledge retrieval (simulated)...")
            grounded_prompt = f"User Query: {user_query}\n\nAnswer:"

        response = self._call_language_model(grounded_prompt)
        return response

# Example Usage:
if __name__ == "__main__":
    # Create dummy medical documents
    with open("doc1.txt", "w") as f:
        f.write("Document 1: XYZ Syndrome is a rare genetic disorder characterized by early childhood onset of fever, widespread rash, and recurrent episodes of joint pain, often mimicking juvenile idiopathic arthritis. Diagnosis involves genetic testing. Treatment is primarily symptomatic, focusing on managing inflammation and pain. Recently, a new monoclonal antibody therapy has shown promising results in clinical trials (TrialID: NCT01234567).")
    with open("doc2.txt", "w") as f:
        f.write("Document 2: Drug A (e.g., Amiodarone) is an antiarrhythmic medication. Drug B (e.g., Warfarin) is an anticoagulant. Co-administration of Drug A and Drug B can significantly increase the anticoagulant effect of Warfarin, leading to a higher risk of bleeding due to inhibition of Warfarin metabolism by Amiodarone. INR monitoring should be intensified when these drugs are used concomitantly.")
    with open("doc3.txt", "w") as f:
        f.write("Document 3: General guidelines for managing hypertension in adults involve lifestyle modifications (diet, exercise) and pharmacotherapy, including ACE inhibitors, ARBs, diuretics, and calcium channel blockers. Personalized treatment plans are crucial based on patient comorbidities and risk factors. Regular blood pressure monitoring is essential for all hypertensive patients.")
    with open("doc4.txt", "w") as f:
        f.write("Document 4: The latest oncology research highlights the role of immunotherapy in treating various cancers, including melanoma and lung cancer. Checkpoint inhibitors like Pembrolizumab have revolutionized treatment paradigms, offering durable responses in a subset of patients. Ongoing research explores combination therapies to enhance efficacy and overcome resistance.")
    with open("doc5.txt", "w") as f:
        f.write("Document 5: Pediatric vaccinations are crucial for public health. The CDC recommends a specific schedule for childhood immunizations against diseases like measles, mumps, rubella, polio, and diphtheria. Adherence to vaccination schedules significantly reduces the incidence of these preventable diseases.")

    document_files = ["doc1.txt", "doc2.txt", "doc3.txt", "doc4.txt", "doc5.txt"]
    bot = MedInfoBot(document_paths=document_files)

    print("\n--- Query 1: Rare Disease ---")
    response1 = bot.query("What are the symptoms and treatment for XYZ Syndrome?")
    print(f"Bot: {response1}")

    print("\n--- Query 2: Drug Interaction ---")
    response2 = bot.query("Tell me about the interaction between Amiodarone and Warfarin.")
    print(f"Bot: {response2}")

    print("\n--- Query 3: General Medical Info (should retrieve) ---")
    response3 = bot.query("What are the latest advancements in cancer treatment?")
    print(f"Bot: {response3}")

    print("\n--- Query 4: Simple Medical Info (simulated no retrieval, but prototype always retrieves due to keyword) ---")
    response4 = bot.query("What are the general guidelines for managing hypertension?")
    print(f"Bot: {response4}")

    print("\n--- Query 5: Out of context ---")
    response5 = bot.query("What is the capital of France?")
    print(f"Bot: {response5}")

    # Clean up dummy files
    import os
    for f in document_files:
        os.remove(f)

