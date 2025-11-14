"""
medical_diagnostic_assistant.py

This script implements a medical diagnostic assistant leveraging various AI design patterns:
- InContext Retrieval-Augmented Language Modeling (InContext RALM)
- Zero-Shot LM Reranking
- Predictive Reranking (Trained LM-Dedicated Reranker)
- Conditional Retrieval

The assistant provides evidence-based diagnoses and treatment recommendations by dynamically retrieving and reranking relevant medical literature, integrating it into a Language Model's context, and conditionally deciding when external retrieval is necessary to optimize efficiency.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from transformers import pipeline, set_seed
import torch

# Set random seed for reproducibility
set_seed(42)
np.random.seed(42)
torch.manual_seed(42)

class MedicalDiagnosticAssistant:
    def __init__(self,
                 embedding_model_name: str = 'all-MiniLM-L6-v2',
                 zero_shot_lm_name: str = 'distilbert-base-uncased',
                 generation_lm_name: str = 'gpt2'):
        
        print("Initializing Medical Diagnostic Assistant...")

        # 1. Data Layer (Simulated/Mock)
        self.medical_documents = [
            "Aspirin is a nonsteroidal anti-inflammatory drug (NSAID) used to reduce fever and relieve mild to moderate pain.",
            "Diabetes mellitus is a chronic condition that affects the way the body processes blood sugar (glucose).",
            "Hypertension, or high blood pressure, is a common condition in which the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease.",
            "Common symptoms of the flu include fever, chills, muscle aches, cough, congestion, and runny nose.",
            "For a broken bone, immediate medical attention is necessary, often involving splinting or casting.",
            "Migraines are severe headaches often accompanied by throbbing pain, sensitivity to light and sound, and nausea.",
            "Asthma is a chronic respiratory condition characterized by inflammation and narrowing of the airways, leading to difficulty breathing.",
            "The liver plays a vital role in detoxification, protein synthesis, and the production of biochemicals necessary for digestion.",
            "Kidney stones are hard deposits made of minerals and salts that form inside your kidneys.",
            "Symptoms of a heart attack can include chest pain, shortness of breath, pain in the arm, and lightheadedness."
        ]
        self.document_ids = [f"doc_{i}" for i in range(len(self.medical_documents))]

        # 2. Embedding and Retrieval Layer
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.document_embeddings = self.embedding_model.encode(self.medical_documents, convert_to_tensor=True)

        # 3. Reranking Layer
        # Zero-Shot LM Reranker (using a small LM for feature extraction or direct scoring)
        self.zero_shot_lm_reranker_pipeline = pipeline(
            "sentiment-analysis", 
            model=zero_shot_lm_name, 
            tokenizer=zero_shot_lm_name,
            return_all_scores=True
        )
        
        # Predictive Reranker (Trained LM-Dedicated Reranker)
        self.predictive_reranker = self._train_predictive_reranker()

        # 4. Conditional Retrieval Layer
        self.conditional_retrieval_model = self._train_conditional_retrieval_model()

        # 5. Language Model (LM) Integration Layer
        self.generation_lm_pipeline = pipeline(
            "text-generation", 
            model=generation_lm_name,
            tokenizer=generation_lm_name
        )

        print("Medical Diagnostic Assistant initialized successfully.")

    def _train_predictive_reranker(self):
        """
        Trains a simple Logistic Regression model for predictive reranking.
        In a real scenario, this would be trained on a dataset of query-document pairs
        with human relevance judgments or downstream LM performance metrics.
        Features could include initial similarity, zero-shot LM score, keyword overlap, etc.
        """
        print("Training Predictive Reranker...")
        # Mock data for training: (embedding_similarity, zero_shot_score, relevance_label)
        # A more sophisticated model would have more features.
        mock_features = []
        mock_labels = []

        for _ in range(200):
            # Simulate a query-document pair
            query_embedding = torch.randn(self.document_embeddings.shape[1])
            doc_embedding = self.document_embeddings[np.random.randint(len(self.medical_documents))]
            
            # Simulate initial similarity
            sim_score = torch.nn.functional.cosine_similarity(query_embedding, doc_embedding, dim=0).item()
            
            # Simulate zero-shot score (e.g., from -1 to 1 for sentiment)
            zero_shot_score = np.random.uniform(-0.8, 0.8)
            
            # Simulate a relevance label (0 or 1)
            # Make it somewhat correlated with scores
            if (sim_score + zero_shot_score) / 2 + np.random.uniform(-0.5, 0.5) > 0.4:
                label = 1 # Relevant
            else:
                label = 0 # Not relevant
            
            mock_features.append([sim_score, zero_shot_score])
            mock_labels.append(label)
        
        X_train, X_test, y_train, y_test = train_test_split(
            np.array(mock_features), np.array(mock_labels), test_size=0.2, random_state=42
        )

        model = LogisticRegression()
        model.fit(X_train, y_train)
        print(f"Predictive Reranker trained. Accuracy on mock test set: {model.score(X_test, y_test):.2f}")
        return model

    def _train_conditional_retrieval_model(self):
        """
        Trains a simple Logistic Regression model for conditional retrieval.
        This model decides if external knowledge is needed for a given query.
        Features could include query length, complexity, presence of specific keywords, etc.
        """
        print("Training Conditional Retrieval Model...")
        # Mock data for training: (query_length, contains_critical_keyword, need_retrieval_label)
        mock_features = []
        mock_labels = []
        critical_keywords = ["diagnose", "treatment", "severe", "emergency"]

        for _ in range(100):
            query = "This is a mock patient query. " * np.random.randint(2, 10)
            if np.random.rand() < 0.3:
                query += f" {np.random.choice(critical_keywords)} "
            
            query_length = len(query.split())
            contains_critical = 1 if any(k in query.lower() for k in critical_keywords) else 0
            
            # Simulate label: more complex queries or those with critical keywords often need retrieval
            label = 1 if (query_length > 10 or contains_critical == 1) and np.random.rand() > 0.2 else 0
            
            mock_features.append([query_length, contains_critical])
            mock_labels.append(label)
        
        X_train, X_test, y_train, y_test = train_test_split(
            np.array(mock_features), np.array(mock_labels), test_size=0.2, random_state=42
        )

        model = LogisticRegression()
        model.fit(X_train, y_train)
        print(f"Conditional Retrieval Model trained. Accuracy on mock test set: {model.score(X_test, y_test):.2f}")
        return model

    def _retrieve_documents(self, query_embedding: torch.Tensor, top_k: int = 5):
        """
        Performs initial cosine similarity search to retrieve top-k documents.
        """
        # Compute cosine similarity between query and all document embeddings
        similarities = torch.nn.functional.cosine_similarity(query_embedding, self.document_embeddings)
        
        # Get top-k indices
        top_k_values, top_k_indices = torch.topk(similarities, top_k)
        
        retrieved_docs_with_scores = [
            (self.medical_documents[idx], self.document_ids[idx], score.item())
            for idx, score in zip(top_k_indices, top_k_values)
        ]
        return retrieved_docs_with_scores

    def _zero_shot_rerank(self, query: str, candidate_docs_with_scores: list):
        """
        Uses a zero-shot LM to rerank candidate documents.
        For this example, we'll use a sentiment analysis model to give a 'relevance' score.
        A more advanced approach would involve a specialized prompt for relevance.
        """
        reranked_docs = []
        for doc_text, doc_id, initial_score in candidate_docs_with_scores:
            # Construct a prompt for the zero-shot LM to assess relevance
            # Using sentiment here as a proxy for relevance for simplicity
            # In a real scenario, you'd prompt like: "Does the following document answer the question: [query]? Document: [doc_text]"
            # And interpret the LM's output (e.g., probability of 'yes', or a generated relevance score).
            # Here, we'll just check if the LM finds the query+doc 'positive' as a sign of relevance.
            
            try:
                # Use the sentiment analysis pipeline to get scores
                # This is a simplification; a true zero-shot reranker would need a custom prompt.
                results = self.zero_shot_lm_reranker_pipeline(f"{query} [SEP] {doc_text}")
                # Look for 'positive' score
                positive_score = next((item['score'] for item in results[0] if item['label'] == 'POSITIVE'), 0.5)
                zero_shot_relevance_score = positive_score
            except Exception as e:
                # Fallback in case of LM issues
                print(f"Warning: Zero-shot LM reranking failed for a document. Error: {e}. Using default score.")
                zero_shot_relevance_score = 0.5 # Neutral score

            reranked_docs.append({
                "text": doc_text,
                "id": doc_id,
                "initial_score": initial_score,
                "zero_shot_relevance": zero_shot_relevance_score
            })
        # Sort by zero-shot relevance for immediate effect, but predictive reranker will refine
        return sorted(reranked_docs, key=lambda x: x['zero_shot_relevance'], reverse=True)

    def _predictive_rerank(self, reranked_docs_from_zero_shot: list, query_embedding: torch.Tensor):
        """
        Applies the trained predictive reranker to refine document relevance scores.
        """
        predictive_reranked_docs = []
        for doc_info in reranked_docs_from_zero_shot:
            # Re-calculate similarity for consistency or use stored initial_score
            doc_text = doc_info["text"]
            doc_embedding = self.embedding_model.encode(doc_text, convert_to_tensor=True)
            sim_score = torch.nn.functional.cosine_similarity(query_embedding, doc_embedding, dim=0).item()
            
            features = np.array([[sim_score, doc_info["zero_shot_relevance"]]])
            predictive_score = self.predictive_reranker.predict_proba(features)[:, 1][0]
            
            doc_info["predictive_relevance"] = predictive_score
            predictive_reranked_docs.append(doc_info)
            
        return sorted(predictive_reranked_docs, key=lambda x: x['predictive_relevance'], reverse=True)

    def _should_retrieve_external_knowledge(self, query: str) -> bool:
        """
        Uses the conditional retrieval model to decide if external knowledge is needed.
        """
        query_length = len(query.split())
        critical_keywords = ["diagnose", "treatment", "severe", "emergency"]
        contains_critical = 1 if any(k in query.lower() for k in critical_keywords) else 0
        
        features = np.array([[query_length, contains_critical]])
        prediction = self.conditional_retrieval_model.predict(features)[0]
        return bool(prediction)

    def diagnose(self, patient_query: str, top_k_retrieval: int = 5, top_k_rerank: int = 3):
        """
        Orchestrates the diagnostic process.
        1. Conditionally determines if external retrieval is needed.
        2. If needed, retrieves and reranks documents.
        3. Augments the LM input with relevant documents (InContext RALM).
        4. Generates a diagnosis/recommendation using the augmented LM.
        """
        print(f"\nProcessing patient query: '{patient_query}'")
        
        retrieved_documents = []
        final_context = patient_query
        
        # 4. Conditional Retrieval
        if self._should_retrieve_external_knowledge(patient_query):
            print("Conditional Retrieval: External knowledge deemed necessary.")
            query_embedding = self.embedding_model.encode(patient_query, convert_to_tensor=True)
            
            # 2. Embedding and Retrieval Layer
            initial_candidates = self._retrieve_documents(query_embedding, top_k=top_k_retrieval)
            print(f"Initial retrieval found {len(initial_candidates)} candidates.")

            # 3. Reranking Layer - Zero-Shot LM Reranking
            zero_shot_reranked = self._zero_shot_rerank(patient_query, initial_candidates)
            print(f"Zero-shot reranking completed. Top candidate by zero-shot: {zero_shot_reranked[0]['id']} (score: {zero_shot_reranked[0]['zero_shot_relevance']:.2f})")

            # 3. Reranking Layer - Predictive Reranking
            predictive_reranked = self._predictive_rerank(zero_shot_reranked, query_embedding)
            
            # Select top_k_rerank documents for context
            retrieved_documents = predictive_reranked[:top_k_rerank]
            
            # 5. InContext Retrieval-Augmented Language Modeling (InContext RALM)
            context_docs_str = "\n\nRelevant Medical Information:\n" + "\n".join(
                [f"- [{doc['id']}] {doc['text']}" for doc in retrieved_documents]
            )
            final_context = f"{context_docs_str}\n\nPatient Query: {patient_query}\n\nBased on the provided information and the patient query, provide a diagnosis and treatment recommendation:"
            print(f"InContext RALM: Augmented LM input with {len(retrieved_documents)} documents.")
        else:
            print("Conditional Retrieval: External knowledge not deemed necessary.")
            final_context = f"Patient Query: {patient_query}\n\nProvide a diagnosis and treatment recommendation:"

        # 5. Core Language Model Generation
        try:
            # Use a smaller max_new_tokens for a quick demo
            response = self.generation_lm_pipeline(final_context, max_new_tokens=150, num_return_sequences=1)[0]['generated_text']
            # Extract only the newly generated part after the prompt
            generated_text = response[len(final_context):].strip()
        except Exception as e:
            print(f"Error during LM generation: {e}. Returning a generic response.")
            generated_text = "Could not generate a specific diagnosis or recommendation at this time due to an internal error."

        diagnosis_recommendation = generated_text

        return {
            "query": patient_query,
            "diagnosis_recommendation": diagnosis_recommendation,
            "retrieved_sources": [(doc['id'], doc['text']) for doc in retrieved_documents],
            "full_lm_input": final_context # For debugging/inspection
        }

# --- Example Usage ---
if __name__ == "__main__":
    # Note: Loading models can take some time and memory.
    # For a lighter demo, you can reduce model sizes or use mock pipelines.
    assistant = MedicalDiagnosticAssistant(
        embedding_model_name='all-MiniLM-L6-v2', # Smaller, faster embedding model
        zero_shot_lm_name='distilbert-base-uncased', # Smaller LM for zero-shot reranking
        generation_lm_name='gpt2' # Smaller LM for text generation
    )

    patient_queries = [
        "I have a terrible headache, sensitivity to light, and feel nauseous.",
        "My blood pressure readings have been consistently high recently.",
        "I'm feeling generally unwell, with a cough and muscle aches, but no severe symptoms.",
        "I think I broke my arm playing sports. It's swollen and very painful.",
        "I need general information about healthy eating."
    ]

    for query in patient_queries:
        result = assistant.diagnose(query, top_k_retrieval=5, top_k_rerank=2)
        print("\n" + "="*80)
        print(f"Patient Query: {result['query']}")
        print("-"*80)
        print(f"Diagnosis & Recommendation: {result['diagnosis_recommendation']}")
        print("-"*80)
        if result['retrieved_sources']:
            print("Retrieved Sources (ID and Text Snippet):")
            for doc_id, doc_text in result['retrieved_sources']:
                print(f"  - [{doc_id}] {doc_text[:100]}...")
        else:
            print("No external sources retrieved for this query.")
        print("="*80)

    # Example to demonstrate conditional retrieval not always triggering
    print("\n\nDemonstrating Conditional Retrieval for a simple query:")
    simple_query = "What is the main function of the liver?"
    simple_result = assistant.diagnose(simple_query, top_k_retrieval=5, top_k_rerank=2)
    print("\n" + "="*80)
    print(f"Patient Query: {simple_result['query']}")
    print("-"*80)
    print(f"Diagnosis & Recommendation: {simple_result['diagnosis_recommendation']}")
    print("-"*80)
    if simple_result['retrieved_sources']:
        print("Retrieved Sources (ID and Text Snippet):")
        for doc_id, doc_text in simple_result['retrieved_sources']:
            print(f"  - [{doc_id}] {doc_text[:100]}...")
    else:
        print("No external sources retrieved for this query.")
    print("="*80)
