
from typing import List, Dict, Any
from conditional_retriever import ConditionalRetriever
from document_retriever import DocumentRetriever
from reranker import Reranker
from lm_generator import LMGenerator

class MedAssistRAGSystem:
    """
    Orchestrates the MedAssist AI system for retrieval-augmented medical knowledge.
    """
    def __init__(self, 
                 conditional_model_path: str = "./conditional_model",
                 embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 vector_db_path: str = "./medical_vector_db",
                 zero_shot_reranker_model: str = "cross-encoder/ms-marco-TinyBERT-L-2-v2",
                 trained_reranker_model_path: str = "./trained_medical_reranker",
                 lm_model_name: str = "gpt2"): # Placeholder for a more capable LM
        
        self.conditional_retriever = ConditionalRetriever(model_path=conditional_model_path)
        self.document_retriever = DocumentRetriever(embedding_model_name=embedding_model_name,
                                                    vector_db_path=vector_db_path)
        self.reranker = Reranker(zero_shot_model_name=zero_shot_reranker_model,
                                 trained_model_path=trained_reranker_model_path)
        self.lm_generator = LMGenerator(lm_model_name=lm_model_name)

        # Initialize document retriever with some dummy data for demonstration
        self.document_retriever.add_documents([
            {"id": "doc1", "text": "Aspirin is a nonsteroidal anti-inflammatory drug (NSAID) used to reduce fever and relieve mild to moderate pain. It can also be used as an antiplatelet."},
            {"id": "doc2", "text": "Type 2 diabetes is a chronic condition that affects the way the body processes blood sugar (glucose). The body either doesn't produce enough insulin, or it resists insulin's effects."},
            {"id": "doc3", "text": "Hypertension, or high blood pressure, is a common condition in which the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease."},
            {"id": "doc4", "text": "Paracetamol (acetaminophen) is a pain reliever and a fever reducer. It is commonly used for headaches, muscle aches, arthritis, backache, toothaches, colds, and fevers."},
            {"id": "doc5", "text": "Insulin is a hormone produced by the pancreas that helps regulate blood glucose levels. It allows glucose to enter cells, providing them with energy."}
        ])

    def ask_medassist(self, query: str, top_k_retrieval: int = 10, top_k_rerank: int = 3) -> Dict[str, Any]:
        """
        Processes a medical query using the RAG system.
        """
        print(f"\nUser Query: {query}")

        # 1. Conditionally Retrieve
        needs_retrieval = self.conditional_retriever.should_retrieve(query)
        print(f"Conditional Retrieval: {'Retrieval needed' if needs_retrieval else 'No retrieval needed'}")

        grounding_documents = []
        if needs_retrieval:
            # 2. Retrieve Grounding Documents
            retrieved_docs = self.document_retriever.retrieve(query, top_k=top_k_retrieval)
            print(f"Retrieved {len(retrieved_docs)} initial documents.")

            if retrieved_docs:
                # 3. Rerank Documents (Zero-Shot and Predictive)
                zero_shot_reranked_docs = self.reranker.zero_shot_rerank(query, retrieved_docs)
                print(f"Zero-shot reranked {len(zero_shot_reranked_docs)} documents.")
                
                # Further refine with a trained reranker if available and necessary
                # For this example, we'll just take the top_k_rerank from zero-shot if trained is not fully implemented
                if self.reranker.has_trained_model:
                    final_reranked_docs = self.reranker.predictive_rerank(query, zero_shot_reranked_docs, top_k=top_k_rerank)
                    print(f"Predictive reranked to {len(final_reranked_docs)} documents.")
                else:
                    final_reranked_docs = zero_shot_reranked_docs[:top_k_rerank]
                    print(f"Using top {top_k_rerank} documents from zero-shot reranking (trained model not fully initialized).")

                grounding_documents = [doc['text'] for doc in final_reranked_docs]

        # 4. InContext Retrieval-Augmented Language Modeling (InContext RALM)
        # 5. Generate Answer with Attribution
        response, attribution = self.lm_generator.generate_response(query, grounding_documents)
        print(f"\nGenerated Response:\n{response}")
        if attribution:
            print(f"\nAttribution:\n{attribution}")
        else:
            print("\nNo specific attribution provided (likely no retrieval occurred or LM generated without direct reference).")

        return {"query": query, "response": response, "attribution": attribution, "retrieved_docs": grounding_documents}

if __name__ == "__main__":
    # Instantiate the system (models and DBs will be loaded or initialized)
    medassist = MedAssistRAGSystem()

    # Example queries
    queries = [
        "What is aspirin used for?",
        "Explain type 2 diabetes.",
        "What are the symptoms of common cold?", # Query where retrieval might not be strictly necessary for a general LM
        "What is the primary function of insulin?",
        "What is hypertension?"
    ]

    for q in queries:
        result = medassist.ask_medassist(q)
        print("="*80)
