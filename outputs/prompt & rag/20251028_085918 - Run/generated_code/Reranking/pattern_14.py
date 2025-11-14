
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import random

class SmartCustomerSupportAssistant:
    def __init__(self):
        # 1. Knowledge Base and Embedding
        self.knowledge_base_documents = [
            "Our refund policy states that returns are accepted within 30 days of purchase with a valid receipt.",
            "Shipping usually takes 5-7 business days for standard delivery within the country.",
            "Expedited shipping options are available at an additional cost for 2-3 day delivery.",
            "To track your order, please visit the 'My Orders' section on our website and enter your order number.",
            "We accept major credit cards, PayPal, and Apple Pay for online purchases.",
            "Our customer support team is available Monday to Friday, 9 AM to 5 PM EST.",
            "You can reach customer support via live chat on our website or by calling 1-800-XXX-XXXX.",
            "Product warranties vary by item; please check the product page for specific details.",
            "To apply a discount code, enter it in the promo code box at checkout.",
            "Our privacy policy details how we collect, use, and protect your personal information."
        ]

        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.document_embeddings = self.embedding_model.encode(self.knowledge_base_documents, convert_to_tensor=True)

        self.faiss_index = faiss.IndexFlatL2(self.document_embeddings.shape[1])
        self.faiss_index.add(self.document_embeddings.cpu().numpy())

        # 2. Retrieval and Reranking
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        # 3. Conditional Retrieval (Simplified with a dummy trained model)
        self.conditional_retrieval_model = self._train_conditional_retrieval_model()

        # 4. InContext Retrieval-Augmented Language Modeling (InContext RALM)
        self.lm_pipeline = pipeline("text-generation", model="distilgpt2")

    def _train_conditional_retrieval_model(self):
        # Dummy training data for conditional retrieval model
        # Features: query length, target: 1 if retrieval needed, 0 otherwise
        dummy_queries = [
            "What is my order status?", # Retrieval needed
            "Hello", # No retrieval
            "Where is my package right now?", # Retrieval needed
            "Thanks!", # No retrieval
            "Can I return this item and what is the policy?", # Retrieval needed
            "How are you?", # No retrieval
            "Tell me about your shipping costs and delivery times.", # Retrieval needed
            "Good day", # No retrieval
            "I need assistance with a product warranty.", # Retrieval needed
            "Great service!" # No retrieval
        ]
        # Simple feature: query length
        X = np.array([len(q.split()) for q in dummy_queries]).reshape(-1, 1)
        y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0]) # Manual labels

        model = LogisticRegression()
        model.fit(X, y)
        return model

    def _predict_retrieval_necessity(self, query: str) -> bool:
        query_features = np.array([len(query.split())]).reshape(-1, 1)
        prediction = self.conditional_retrieval_model.predict(query_features)
        return bool(prediction[0])

    def _retrieve_documents(self, query: str, top_k: int = 5) -> list:
        query_embedding = self.embedding_model.encode(query, convert_to_tensor=True).cpu().numpy()
        D, I = self.faiss_index.search(np.expand_dims(query_embedding, axis=0), top_k)
        retrieved_docs = [self.knowledge_base_documents[i] for i in I[0]]
        return retrieved_docs

    def _rerank_documents(self, query: str, candidate_documents: list, top_n: int = 3) -> list:
        if not candidate_documents:
            return []
        sentence_pairs = [[query, doc] for doc in candidate_documents]
        scores = self.cross_encoder.predict(sentence_pairs)
        reranked_indices = np.argsort(scores)[::-1]
        top_reranked_docs = [candidate_documents[i] for i in reranked_indices[:top_n]]
        return top_reranked_docs

    def _generate_lm_response(self, prompt: str) -> str:
        # Limiting max_new_tokens to avoid overly long generations for demonstration
        response = self.lm_pipeline(prompt, max_new_tokens=100, num_return_sequences=1, truncation=True)[0]['generated_text']
        # Remove the input prompt from the generated text
        return response[len(prompt):].strip()

    def get_answer(self, customer_query: str) -> dict:
        use_retrieval = self._predict_retrieval_necessity(customer_query)
        augmented_prompt = customer_query
        attributed_sources = []

        if use_retrieval:
            print(f"[Debug] Conditional Retrieval: Retrieval needed for query: '{customer_query}'")
            candidate_docs = self._retrieve_documents(customer_query, top_k=10)
            if candidate_docs:
                reranked_docs = self._rerank_documents(customer_query, candidate_docs, top_n=3)
                if reranked_docs:
                    context = " ".join(reranked_docs)
                    augmented_prompt = f"Context: {context}\n\nQuestion: {customer_query}\nAnswer:"
                    attributed_sources = reranked_docs
                else:
                    print("[Debug] Reranking returned no documents. Proceeding without augmentation.")
            else:
                print("[Debug] Initial retrieval found no candidate documents. Proceeding without augmentation.")
        else:
            print(f"[Debug] Conditional Retrieval: No retrieval needed for query: '{customer_query}'")

        # Generate LM response
        lm_raw_response = self._generate_lm_response(augmented_prompt)
        
        # Simple post-processing to clean up the LM's output if it repeats the prompt
        if augmented_prompt in lm_raw_response:
            final_answer = lm_raw_response.replace(augmented_prompt, '').strip()
        else:
            final_answer = lm_raw_response.strip()

        return {
            "answer": final_answer,
            "sources": attributed_sources,
            "retrieval_used": use_retrieval
        }

if __name__ == "__main__":
    assistant = SmartCustomerSupportAssistant()

    print("\n--- Smart Customer Support Assistant Demo ---")
    print("Type 'exit' to quit.")

    while True:
        query = input("\nEnter your query: ")
        if query.lower() == 'exit':
            break

        response = assistant.get_answer(query)

        print("\nAssistant: " + response["answer"])
        if response["sources"]:
            print("Sources:")
            for i, source in enumerate(response["sources"]):
                print(f"  {i+1}. {source}")
        print(f"Retrieval Used: {response['retrieval_used']}")
