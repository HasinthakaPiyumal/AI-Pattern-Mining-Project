import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline # For a conceptual LM
import random

# --- 1. Knowledge Base Simulation ---
KNOWLEDGE_BASE = [
    {"id": "doc_1", "content": "Patient presents with fever, cough, and fatigue. Common cold is a frequent diagnosis. Treat with rest and fluids.", "disease_type": "common"},
    {"id": "doc_2", "content": "Symptoms: persistent fatigue, muscle weakness, dry eyes, difficulty swallowing. Suspect Myasthenia Gravis (rare autoimmune). Diagnosis involves antibody tests. Treatment: immunosuppressants, pyridostigmine.", "disease_type": "rare"},
    {"id": "doc_3", "content": "Headache, nausea, stiff neck. Meningitis. Immediate medical attention required. Lumbar puncture for diagnosis.", "disease_type": "common"},
    {"id": "doc_4", "content": "Progressive difficulty walking, numbness, tingling, vision problems. Could be Multiple Sclerosis. MRI scans are crucial. Immunomodulatory therapies available.", "disease_type": "rare"},
    {"id": "doc_5", "content": "Seasonal allergies cause sneezing, runny nose, itchy eyes. Antihistamines are effective.", "disease_type": "common"},
    {"id": "doc_6", "content": "Unexplained weight loss, fever, night sweats, localized pain. Consider Lymphoma (rare cancer). Biopsy for definitive diagnosis.", "disease_type": "rare"},
    {"id": "doc_7", "content": "Sore throat, difficulty breathing, rash. Strep throat common. Antibiotics needed.", "disease_type": "common"},
    {"id": "doc_8", "content": "Severe muscle cramps, weakness, dark urine after intense exercise. Rhabdomyolysis (rare, but can be induced). Hydration is key.", "disease_type": "rare"},
]

# --- 2. Conditional Retrieval Model (Simulated/Rule-based/Trained Placeholder) ---
class ConditionalRetrievalModel:
    def __init__(self):
        # In a real scenario, this would be a trained model (e.g., LogisticRegression, RandomForest)
        # trained on features extracted from patient symptoms to predict if a case is 'common' or 'rare'.
        # For this demo, we'll use a simple keyword-based heuristic.
        self.rare_disease_keywords = ["persistent fatigue", "muscle weakness", "difficulty swallowing", "numbness", "tingling", "unexplained weight loss", "dark urine"]
        # Simulate a trained model for structure, but its prediction logic will be simple.
        self.vectorizer = TfidfVectorizer()
        self.model = LogisticRegression()
        self._is_trained = False

    def _train_dummy_model(self):
        # Dummy training for demonstration purposes
        X = [doc["content"] for doc in KNOWLEDGE_BASE]
        y = [1 if doc["disease_type"] == "rare" else 0 for doc in KNOWLEDGE_BASE] # 1 for rare, 0 for common

        # Only train if there's enough data and not already trained
        if len(X) > 1 and not self._is_trained:
            X_vec = self.vectorizer.fit_transform(X)
            self.model.fit(X_vec, y)
            self._is_trained = True
            print("Conditional Retrieval Model: Dummy training complete.")

    def predict_needs_external_retrieval(self, query: str) -> bool:
        """
        Predicts whether external knowledge retrieval is necessary based on the query.
        For this demo, we'll use a simple heuristic and a placeholder trained model.
        """
        self._train_dummy_model() # Ensure dummy model is trained

        # Heuristic 1: Check for rare disease keywords
        for keyword in self.rare_disease_keywords:
            if keyword in query.lower():
                print(f"Conditional Retrieval: Keyword '{keyword}' detected. Triggering external retrieval.")
                return True

        # Heuristic 2: Use the dummy trained model
        if self._is_trained:
            query_vec = self.vectorizer.transform([query])
            prediction = self.model.predict(query_vec)[0]
            if prediction == 1: # Predicted as potentially rare
                print("Conditional Retrieval: Dummy trained model suggests external retrieval for potential rare case.")
                return True

        print("Conditional Retrieval: Query appears common. Relying on internal knowledge (or limited retrieval).")
        return False

# --- 3. Retrieval and Reranking Functions ---

def retrieve_documents(query: str, top_k: int = 5) -> list[dict]:
    """
    Simulates initial document retrieval based on keyword matching.
    In a real system, this would involve vector embeddings and similarity search (e.g., FAISS, Chroma).
    """
    relevant_docs = []
    query_terms = query.lower().split()

    for doc in KNOWLEDGE_BASE:
        score = 0
        doc_content_lower = doc["content"].lower()
        for term in query_terms:
            if term in doc_content_lower:
                score += doc_content_lower.count(term)
        if score > 0:
            relevant_docs.append({"doc": doc, "score": score})

    # Sort by score and take top_k
    relevant_docs.sort(key=lambda x: x["score"], reverse=True)
    return [d["doc"] for d in relevant_docs[:top_k]]

class ZeroShotLMReranker:
    def __init__(self):
        # In a real scenario, this would load a pre-trained LM for reranking.
        # For demonstration, we'll simulate its scoring capability.
        # e.g., self.reranker_lm = pipeline("text-classification", model="some/cross-encoder-model")
        print("Zero-Shot LM Reranker initialized (simulated).")

    def rerank_documents(self, query: str, documents: list[dict]) -> list[dict]:
        """
        Simulates zero-shot LM reranking. It scores documents based on
        how "likely" they are to explain the query, without explicit training.
        Here, we'll use a heuristic that boosts scores for documents that are very relevant
        to the query, simulating semantic understanding.
        """
        if not documents:
            return []

        reranked_scores = []
        for doc in documents:
            # Simulate LM's "probability of upcoming text"
            # A more sophisticated simulation would involve checking for coherence or direct answers
            # For simplicity, we'll enhance existing keyword match score and add a random factor
            # to simulate semantic understanding beyond simple keyword match.

            # Basic relevance based on query terms
            relevance_score = 0
            query_terms = query.lower().split()
            doc_content_lower = doc["content"].lower()
            for term in query_terms:
                if term in doc_content_lower:
                    relevance_score += doc_content_lower.count(term)

            # Simulate "semantic" boost: if document content is very similar to query, boost more
            # In a real system, this would be cosine similarity on embeddings or LM's entailment score
            semantic_boost = random.uniform(0.5, 1.5) if relevance_score > 0 else 0.1 # Boost slightly if relevant

            # Combined simulated score
            lm_rerank_score = (relevance_score * semantic_boost) + random.uniform(0, 0.5) # Add some randomness for variation

            reranked_scores.append({"doc": doc, "score": lm_rerank_score})

        reranked_scores.sort(key=lambda x: x["score"], reverse=True)
        print(f"Zero-Shot LM Reranker: Reranked {len(documents)} documents.")
        return [d["doc"] for d in reranked_scores]

# --- 4. InContext Retrieval-Augmented Language Modeling (RALM) ---
class InContextRALMLanguageModel:
    def __init__(self):
        # In a real system, this would be an actual LLM client.
        # e.g., self.llm = openai.OpenAI() or HuggingFace pipeline for text generation.
        self.lm_pipeline = pipeline("text-generation", model="gpt2") # Placeholder for conceptual LM

    def generate_response(self, prompt: str) -> str:
        """
        Generates a response using the LM, prepending retrieved context.
        """
        print(f"\n--- LM Input Prompt ---\n{prompt}\n-----------------------\n")
        # Simulate LM response generation
        # In a real scenario, you'd call self.llm.chat.completions.create(...)
        # or self.lm_pipeline(prompt, max_new_tokens=...)
        # For simplicity, we'll just acknowledge the prompt and give a mock response.
        mock_response = (
            "Based on the provided information, the potential diagnostic considerations are: "
            f"'{prompt[:100]}...' [Mock LM Output for diagnosis]. "
            "Further investigation via [Mock LM Output for investigation] is recommended."
        )
        # return self.lm_pipeline(prompt, max_new_tokens=150, num_return_sequences=1)[0][\'generated_text\']
        return mock_response

    def formulate_ralm_prompt(self, query: str, retrieved_docs: list[dict]) -> str:
        """
        Constructs the prompt by prepending retrieved documents to the user query.
        """
        context = "\n\n".join([f"Document {i+1} (ID: {doc['id']}): {doc['content']}" for i, doc in enumerate(retrieved_docs)])
        if context:
            prompt = (
                f"Contextual Information:\n{context}\n\n"
                f"Patient Symptoms: {query}\n\n"
                "Based on the provided context and symptoms, please provide potential diagnoses, "
                "relevant medical insights, and recommended next steps for investigation. "
                "Cite the document IDs if possible."
            )
        else:
            prompt = (
                f"Patient Symptoms: {query}\n\n"
                "Based on general medical knowledge, please provide potential diagnoses, "
                "relevant medical insights, and recommended next steps for investigation."
            )
        return prompt

# --- Main Application Logic ---
class ClinicalDecisionSupportSystem:
    def __init__(self):
        self.conditional_retrieval_model = ConditionalRetrievalModel()
        self.zero_shot_reranker = ZeroShotLMReranker()
        self.ralm_lm = InContextRALMLanguageModel()

    def query_system(self, patient_symptoms: str) -> str:
        print(f"\n--- System Query for Patient Symptoms: '{patient_symptoms}' ---")

        # 1. Conditional Retrieval
        needs_external_retrieval = self.conditional_retrieval_model.predict_needs_external_retrieval(patient_symptoms)

        retrieved_documents = []
        if needs_external_retrieval:
            print("Performing comprehensive external knowledge retrieval...")
            # 2. Initial Retrieval
            initial_retrieved = retrieve_documents(patient_symptoms, top_k=10) # Get more for reranking
            print(f"Initial retrieval found {len(initial_retrieved)} documents.")

            # 3. Zero-Shot LM Reranking
            retrieved_documents = self.zero_shot_reranker.rerank_documents(patient_symptoms, initial_retrieved)[:5] # Take top 5 after reranking
            print(f"Reranked documents selected: {len(retrieved_documents)}")
            for i, doc in enumerate(retrieved_documents):
                print(f"  Doc {i+1} (ID: {doc['id']}): {doc['content'][:100]}...")
        else:
            print("Skipping comprehensive external retrieval. Using limited internal knowledge/basic retrieval.")
            # For common cases, maybe just a few most relevant internal docs or direct LM knowledge
            retrieved_documents = retrieve_documents(patient_symptoms, top_k=2) # Fewer docs for common cases
            for i, doc in enumerate(retrieved_documents):
                print(f"  Doc {i+1} (ID: {doc['id']}): {doc['content'][:100]}...")


        # 4. InContext RALM
        ralm_prompt = self.ralm_lm.formulate_ralm_prompt(patient_symptoms, retrieved_documents)
        diagnosis_recommendations = self.ralm_lm.generate_response(ralm_prompt)

        return diagnosis_recommendations

# --- Example Usage ---
def main():
    system = ClinicalDecisionSupportSystem()

    print("\n--- Scenario 1: Common Cold Symptoms ---")
    response1 = system.query_system("Patient has fever, cough, and general fatigue for two days.")
    print("\nSystem Response (Common Cold):")
    print(response1)
    print("\n" + "="*80 + "\n")

    print("\n--- Scenario 2: Suspected Rare Disease (Myasthenia Gravis) ---")
    response2 = system.query_system("Patient reports persistent fatigue, muscle weakness, and occasional difficulty swallowing for several months.")
    print("\nSystem Response (Myasthenia Gravis):")
    print(response2)
    print("\n" + "="*80 + "\n")

    print("\n--- Scenario 3: Another Rare Disease (Lymphoma) ---")
    response3 = system.query_system("Patient experiences unexplained weight loss, night sweats, and localized pain in the neck.")
    print("\nSystem Response (Lymphoma):")
    print(response3)
    print("\n" + "="*80 + "\n")

    print("\n--- Scenario 4: Another Common Issue (Headache) ---")
    response4 = system.query_system("Severe headache and nausea.")
    print("\nSystem Response (Headache):")
    print(response4)
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()