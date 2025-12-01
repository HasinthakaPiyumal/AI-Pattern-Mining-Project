"""medical_retrieval_module.py"""
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class MedicalKnowledgeRetrievalModule:
    """
    A plug-and-play module for retrieving relevant medical knowledge from a corpus.
    Uses Sentence Transformers for embeddings and FAISS for efficient similarity search.
    """
    def __init__(self, medical_documents: list[str], model_name: str = "all-MiniLM-L6-v2"):
        self.medical_documents = medical_documents
        self.model = SentenceTransformer(model_name)
        self.document_embeddings = self.model.encode(medical_documents, show_progress_bar=False)
        self.dimension = self.document_embeddings.shape[1]
        self.index = faiss.IndexFlatL2(self.dimension)  # L2 distance for similarity
        self.index.add(np.array(self.document_embeddings).astype('float32'))
        print(f"MedicalKnowledgeRetrievalModule initialized with {len(medical_documents)} documents.")

    def retrieve_knowledge(self, query: str, top_k: int = 3) -> list[str]:
        """
        Retrieves the most relevant medical documents for a given query.
        """
        query_embedding = self.model.encode([query], show_progress_bar=False)
        distances, indices = self.index.search(np.array(query_embedding).astype('float32'), top_k)

        retrieved_documents = [
            self.medical_documents[idx] for idx in indices[0]
        ]
        return retrieved_documents

# Example Usage (for testing the module independently)
if __name__ == "__main__":
    sample_medical_data = [
        "Aspirin is a nonsteroidal anti-inflammatory drug (NSAID) used to reduce fever and relieve mild to moderate pain.",
        "Diabetes mellitus is a metabolic disease that causes high blood sugar. The hormone insulin moves sugar from the blood into your cells to be stored for energy.",
        "Hypertension, also known as high blood pressure, is a long-term medical condition in which the blood pressure in the arteries is persistently elevated.",
        "Common cold is a viral infectious disease of the upper respiratory tract that primarily affects the nose, throat, sinuses, and larynx.",
        "Antibiotics are medicines that fight bacterial infections in people and animals. They work by killing the bacteria or by making it difficult for the bacteria to grow and multiply.",
        "The heart is a muscular organ in most animals, which pumps blood through the blood vessels of the circulatory system.",
        "Symptoms of a heart attack include chest pain, shortness of breath, and pain in the left arm.",
        "Vaccines help develop immunity by imitating an infection. This type of infection rarely causes illness, but it does cause the immune system to produce T-lymphocytes and antibodies."
    ]

    retrieval_module = MedicalKnowledgeRetrievalModule(sample_medical_data)

    test_query = "What are the symptoms of a heart problem?"
    relevant_docs = retrieval_module.retrieve_knowledge(test_query)
    print(f"\nQuery: '{test_query}'")
    print("Retrieved Knowledge:")
    for doc in relevant_docs:
        print(f"- {doc}")

    test_query_2 = "How do medicines fight infections?"
    relevant_docs_2 = retrieval_module.retrieve_knowledge(test_query_2)
    print(f"\nQuery: '{test_query_2}'")
    print("Retrieved Knowledge:")
    for doc in relevant_docs_2:
        print(f"- {doc}")

"""llm_qa_system.py"""
from transformers import pipeline
# from medical_retrieval_module import MedicalKnowledgeRetrievalModule # Assuming it's in the same directory

class LLMQueryAnsweringSystem:
    """
    Integrates a general-purpose LLM with the MedicalKnowledgeRetrievalModule
    to provide augmented, context-aware answers to medical queries.
    """
    def __init__(self, retrieval_module: MedicalKnowledgeRetrievalModule, llm_model_name: str = "distilgpt2"):
        self.retrieval_module = retrieval_module
        # Initialize a text-generation pipeline with a pre-trained LLM
        # For production, consider larger models and more robust inference setups (e.g., vLLM, custom serving)
        try:
            self.llm_pipeline = pipeline("text-generation", model=llm_model_name)
            print(f"LLMQueryAnsweringSystem initialized with LLM: {llm_model_name}")
        except Exception as e:
            print(f"Error loading LLM model {llm_model_name}: {e}")
            print("Please ensure the model name is correct and you have an internet connection for download.")
            # Fallback to a dummy function or raise error
            self.llm_pipeline = self._dummy_llm_pipeline

    def _dummy_llm_pipeline(self, prompt, **kwargs):
        """
        A dummy LLM pipeline for when the actual model fails to load.
        """
        print("Using dummy LLM pipeline.")
        return [{
            "generated_text": prompt + " (Dummy LLM response: I couldn't process this query without a proper medical context.)"
        }]

    def answer_query(self, user_query: str, max_new_tokens: int = 100) -> str:
        """
        Retrieves relevant medical knowledge and uses it to augment the LLM's response to a user query.
        """
        print(f"\nProcessing query: '{user_query}'")
        # Step 1: Retrieve relevant medical knowledge
        retrieved_knowledge = self.retrieval_module.retrieve_knowledge(user_query)
        context = " ".join(retrieved_knowledge)
        print(f"Retrieved Context: {context[:200]}...") # Show first 200 chars of context

        # Step 2: Construct the prompt for the LLM with the augmented context
        prompt = (
            f"Given the following medical context, answer the user's question accurately and concisely:\n\n"
            f"Context: {context}\n\n"
            f"User Question: {user_query}\n\n"
            f"Answer:"
        )

        # Step 3: Get the LLM to generate a response based on the augmented prompt
        try:
            llm_response = self.llm_pipeline(prompt, max_new_tokens=max_new_tokens, num_return_sequences=1, truncation=True)
            generated_text = llm_response[0]['generated_text']

            # The LLM might repeat the prompt or add introductory phrases. Extract the actual answer.
            # A more robust parsing mechanism might be needed for complex outputs.
            answer_prefix = f"Answer:"
            if answer_prefix in generated_text:
                final_answer = generated_text.split(answer_prefix, 1)[1].strip()
            else:
                final_answer = generated_text.strip()

            return final_answer
        except Exception as e:
            print(f"Error during LLM generation: {e}")
            return "I am currently unable to provide an answer due to an internal error or missing LLM model. Please try again later."

"""main.py"""
# from medical_retrieval_module import MedicalKnowledgeRetrievalModule
# from llm_qa_system import LLMQueryAnsweringSystem
import warnings

# Suppress specific future warnings from Sentence Transformers/Transformers library
warnings.filterwarnings("ignore", category=FutureWarning)

def main():
    print("Initializing Medical Query Answering System...")

    # 1. Prepare sample medical data (in a real application, this would come from a database)
    sample_medical_data = [
        "Aspirin is a nonsteroidal anti-inflammatory drug (NSAID) used to reduce fever and relieve mild to moderate pain.",
        "Diabetes mellitus is a metabolic disease that causes high blood sugar. The hormone insulin moves sugar from the blood into your cells to be stored for energy.",
        "Hypertension, also known as high blood pressure, is a long-term medical condition in which the blood pressure in the arteries is persistently elevated.",
        "Common cold is a viral infectious disease of the upper respiratory tract that primarily affects the nose, throat, sinuses, and larynx.",
        "Antibiotics are medicines that fight bacterial infections in people and animals. They work by killing the bacteria or by making it difficult for the bacteria to grow and multiply.",
        "The heart is a muscular organ in most animals, which pumps blood through the blood vessels of the circulatory system.",
        "Symptoms of a heart attack include chest pain, shortness of breath, and pain in the left arm.",
        "Vaccines help develop immunity by imitating an infection. This type of infection rarely causes illness, but it does cause the immune system to produce T-lymphocytes and antibodies.",
        "Insulin is a hormone produced by the pancreas that helps regulate blood sugar levels. It's crucial for treating diabetes.",
        "Cholesterol is a waxy, fat-like substance found in all your body's cells. Your body needs some cholesterol to make hormones, vitamin D, and substances that help you digest foods.",
        "High cholesterol can increase the risk of heart disease and stroke."
    ]

    # 2. Initialize the Plug-and-Play Medical Knowledge Retrieval Module
    try:
        retrieval_module = MedicalKnowledgeRetrievalModule(sample_medical_data)
    except ImportError:
        print("Error: 'sentence_transformers' or 'faiss' not installed. Please install with `pip install sentence-transformers faiss-cpu`.")
        return
    except Exception as e:
        print(f"Error initializing MedicalKnowledgeRetrievalModule: {e}")
        return

    # 3. Initialize the LLM-based Query Answering System
    # Using a small LLM (distilgpt2) for demonstration. For better results, use larger models.
    try:
        qa_system = LLMQueryAnsweringSystem(retrieval_module=retrieval_module, llm_model_name="distilgpt2")
    except ImportError:
        print("Error: 'transformers' not installed. Please install with `pip install transformers`.")
        return
    except Exception as e:
        print(f"Error initializing LLMQueryAnsweringSystem: {e}")
        return

    print("System ready. You can now ask medical questions.")

    # 4. Demonstrate with example queries
    queries = [
        "What is aspirin used for?",
        "Tell me about diabetes.",
        "What are the signs of a heart attack?",
        "How do vaccines work?",
        "What is hypertension?",
        "What is the role of insulin in the body?",
        "Explain cholesterol and its risks."
    ]

    for i, query in enumerate(queries):
        print(f"\n--- Query {i+1} ---")
        answer = qa_system.answer_query(query)
        print(f"Final Answer: {answer}")
        print("--------------------")

if __name__ == "__main__":
    main()
