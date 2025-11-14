""" 
medical_information_assistant.py

A Medical Information Assistant that leverages various AI design patterns to provide accurate and attributable medical information.
It incorporates InContext Retrieval-Augmented Language Modeling (RALM), Zero-Shot LM Reranking,
Predictive Reranking (Trained LM-Dedicated Reranker), and Conditional Retrieval.
"""

import os
import random
from typing import List, Dict, Any

# Libraries for Knowledge Base
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Libraries for Core LM and Reranking
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Libraries for Predictive Models
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score


class MedicalKnowledgeBase:
    """Manages the medical document knowledge base and provides retrieval capabilities."""
    def __init__(self, documents: List[str], embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.documents = documents
        self.embedding_model_name = embedding_model_name
        self.vectorstore = None
        self._load_and_embed_documents()

    def _load_and_embed_documents(self):
        print("Loading and embedding medical documents...")
        # Convert raw strings to Document objects
        docs = [Document(page_content=d) for d in self.documents]

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        split_docs = text_splitter.split_documents(docs)

        # Create embeddings
        # Using device="cuda" if available for faster embeddings
        embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name,
                                           model_kwargs={'device': 'cuda'} if torch.cuda.is_available() else {})

        # Create FAISS vectorstore
        self.vectorstore = FAISS.from_documents(split_docs, embeddings)
        print(f"Medical Knowledge Base loaded with {len(split_docs)} chunks.")

    def get_retriever(self, k: int = 5):
        """Returns a retriever configured for the vector store."""
        return self.vectorstore.as_retriever(search_kwargs={"k": k})


class CoreLanguageModel:
    """Manages the core pre-trained language model for text generation."""
    def __init__(self, model_name: str = "distilgpt2"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Set padding token if not already present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        # For simplicity, using a pipeline for generation
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
        print(f"Core Language Model '{model_name}' loaded.")

    def generate(self, prompt: str, max_new_tokens: int = 100) -> str:
        """Generates text based on the input prompt."""
        # Ensure the prompt is not too long for the model's context window
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        if input_ids.shape[1] > self.tokenizer.model_max_length:
            print(f"Warning: Input prompt length ({input_ids.shape[1]}) exceeds model max length ({self.tokenizer.model_max_length}). Truncating.")
            # Truncate from the beginning to keep the most recent context
            prompt = self.tokenizer.decode(input_ids[0, -self.tokenizer.model_max_length:].tolist(), skip_special_tokens=True)

        result = self.generator(
            prompt,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True, # Enable sampling for more varied outputs
            top_k=50,       # Top-k sampling
            top_p=0.95,     # Nucleus sampling
            temperature=0.7 # Generation temperature
        )
        generated_text = result[0]['generated_text']
        # Remove the input prompt from the generated text
        if generated_text.startswith(prompt):
            return generated_text[len(prompt):].strip()
        return generated_text.strip()


class ZeroShotReranker:
    """Reranks documents using a pre-trained language model in a zero-shot manner."""
    def __init__(self, lm: CoreLanguageModel):
        self.lm = lm

    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """Reranks a list of documents based on their relevance to the query using the LM."""
        print("Performing Zero-Shot LM Reranking...")
        if not documents:
            return []

        scored_documents = []
        for doc in documents:
            # Create a prompt for the LM to judge relevance
            # This is a simplified approach. More robust methods exist (e.g., cross-encoders).
            rerank_prompt = (
                f"Given the query: '{query}', rate the relevance of the following medical information "
                f"on a scale from 1 (Not Relevant) to 5 (Highly Relevant). Respond with only the number.\n\n"
                f"Information: {doc.page_content}\n\nRelevance Score:"
            )
            # Generate a short response to classify relevance
            relevance_prediction = self.lm.generate(rerank_prompt, max_new_tokens=2).strip()

            # Try to parse the numerical score
            score = 0
            try:
                score = int(relevance_prediction)
                score = max(1, min(5, score)) # Clamp score between 1 and 5
            except ValueError:
                # Fallback if LM doesn't return a number, e.g., default to low relevance
                score = 1

            scored_documents.append((doc, score))

        # Sort documents by score in descending order
        scored_documents.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored_documents]


class PredictiveReranker:
    """A trained reranker model to predict document relevance for maximizing LM performance."""
    def __init__(self, vectorizer=None):
        self.model = LogisticRegression(max_iter=1000)
        self.vectorizer = vectorizer if vectorizer else TfidfVectorizer()
        self.is_trained = False

    def train(self, queries: List[str], documents_list: List[List[Document]], labels: List[List[int]]):
        """
        Trains the predictive reranker.
        Args:
            queries: List of queries.
            documents_list: List of lists of candidate documents for each query.
            labels: List of lists of binary labels (1 if document is relevant to query, 0 otherwise).
                    Labels should correspond to the documents_list.
        """
        print("Training Predictive Reranker...")
        X_train, y_train = [], []
        for i, query in enumerate(queries):
            for j, doc in enumerate(documents_list[i]):
                # Create a feature by concatenating query and document content
                feature_text = query + " [SEP] " + doc.page_content
                X_train.append(feature_text)
                y_train.append(labels[i][j])

        if not X_train:
            print("No training data for Predictive Reranker.")
            return

        # Vectorize the training data. Vectorizer should be fitted globally if shared.
        X_train_vec = self.vectorizer.transform(X_train)

        self.model.fit(X_train_vec, y_train)
        self.is_trained = True
        print("Predictive Reranker trained.")

    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """Reranks a list of documents using the trained predictive model."""
        if not self.is_trained:
            print("Predictive Reranker not trained. Skipping reranking.")
            return documents
        if not documents:
            return []

        print("Performing Predictive Reranking...")
        X_test = [query + " [SEP] " + doc.page_content for doc in documents]
        X_test_vec = self.vectorizer.transform(X_test)

        # Get probabilities of being relevant (class 1)
        relevance_scores = self.model.predict_proba(X_test_vec)[:, 1]

        scored_documents = list(zip(documents, relevance_scores))
        scored_documents.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored_documents]


class ConditionalRetrievalPredictor:
    """A predictive model to determine when external knowledge retrieval is truly needed."""
    def __init__(self, vectorizer=None):
        self.model = LogisticRegression(max_iter=1000)
        self.vectorizer = vectorizer if vectorizer else TfidfVectorizer()
        self.is_trained = False

    def train(self, queries: List[str], should_retrieve_labels: List[int]):
        """
        Trains the conditional retrieval predictor.
        Args:
            queries: List of queries.
            should_retrieve_labels: List of binary labels (1 if retrieval is needed, 0 otherwise).
        """
        print("Training Conditional Retrieval Predictor...")
        if not queries:
            print("No training data for Conditional Retrieval Predictor.")
            return

        # Vectorize the training data. Vectorizer should be fitted globally if shared.
        X_train_vec = self.vectorizer.transform(queries)

        self.model.fit(X_train_vec, should_retrieve_labels)
        self.is_trained = True
        print("Conditional Retrieval Predictor trained.")

    def predict_retrieval_needed(self, query: str) -> bool:
        """Predicts whether retrieval is needed for a given query."""
        if not self.is_trained:
            print("Conditional Retrieval Predictor not trained. Always retrieving by default.")
            return True # Default to retrieval if not trained

        query_vec = self.vectorizer.transform([query])
        prediction = self.model.predict(query_vec)[0]
        return bool(prediction)


class MedicalInformationAssistant:
    """Integrates various AI patterns to provide medical information with enhanced accuracy and attribution."""
    def __init__(self, medical_documents: List[str], lm_model_name: str = "distilgpt2",
                 embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.knowledge_base = MedicalKnowledgeBase(medical_documents, embedding_model_name)
        self.lm = CoreLanguageModel(lm_model_name)
        self.zero_shot_reranker = ZeroShotReranker(self.lm)

        # Shared TfidfVectorizer for Predictive Reranker and Conditional Retrieval Predictor
        self.vectorizer_for_predictive_models = TfidfVectorizer()
        self.predictive_reranker = PredictiveReranker(self.vectorizer_for_predictive_models)
        self.conditional_retrieval_predictor = ConditionalRetrievalPredictor(self.vectorizer_for_predictive_models)

    def train_predictive_models(self,
                                retrieval_training_data: Dict[str, bool],
                                reranking_training_data: Dict[str, Dict[str, bool]]):
        """
        Trains the Conditional Retrieval Predictor and Predictive Reranker.

        Args:
            retrieval_training_data: A dictionary where keys are queries and values are booleans
                                     indicating if retrieval is needed (True) or not (False).
                                     Example: {"What is ibuprofen?": True, "Hello": False}
            reranking_training_data: A dictionary where keys are queries. Values are another dictionary
                                     where keys are document contents and values are booleans indicating
                                     if that document is relevant to the query.
                                     Example: {"What is diabetes?": {"doc1 content": True, "doc2 content": False}}
        """
        print("\n--- Starting Predictive Model Training ---")

        # Collect all text to fit the shared vectorizer
        all_text_for_vectorizer = []
        for query in retrieval_training_data.keys():
            all_text_for_vectorizer.append(query)
        for query, doc_relevance_map in reranking_training_data.items():
            all_text_for_vectorizer.append(query)
            for doc_content in doc_relevance_map.keys():
                all_text_for_vectorizer.append(query + " [SEP] " + doc_content)

        if all_text_for_vectorizer:
            self.vectorizer_for_predictive_models.fit(all_text_for_vectorizer)
            print(f"Shared TfidfVectorizer fitted with {len(self.vectorizer_for_predictive_models.vocabulary_)} terms.")

        # Prepare data for Conditional Retrieval Predictor
        cr_queries = list(retrieval_training_data.keys())
        cr_labels = [1 if v else 0 for v in retrieval_training_data.values()]
        self.conditional_retrieval_predictor.train(cr_queries, cr_labels)

        # Prepare data for Predictive Reranker
        pr_queries = []
        pr_documents_list = []
        pr_labels_list = []

        for query, doc_relevance_map in reranking_training_data.items():
            current_docs = [Document(page_content=d_content) for d_content in doc_relevance_map.keys()]
            current_labels = [1 if is_relevant else 0 for is_relevant in doc_relevance_map.values()]

            if current_docs: # Ensure there are documents for this query
                pr_queries.append(query)
                pr_documents_list.append(current_docs)
                pr_labels_list.append(current_labels)

        self.predictive_reranker.train(pr_queries, pr_documents_list, pr_labels_list)
        print("--- Predictive Model Training Complete ---\n")

    def query(self, user_query: str, use_predictive_reranker: bool = False, top_k_retrieval: int = 3) -> str:
        """Processes a user query and returns a medical information response with attribution."""
        final_context = ""
        retrieved_docs_contents = []
        attribution_sources = []

        # 1. Conditional Retrieval Check
        if self.conditional_retrieval_predictor.predict_retrieval_needed(user_query):
            print(f"Conditional Retrieval: Retrieval needed for query '{user_query}'.")

            # 2. Document Retrieval
            # Retrieve more documents initially to allow rerankers to select the best ones
            retriever = self.knowledge_base.get_retriever(k=10)
            candidate_documents = retriever.invoke(user_query)
            print(f"Initial retrieval found {len(candidate_documents)} documents.")

            if candidate_documents:
                # 3. Zero-Shot Reranking
                reranked_docs_zero_shot = self.zero_shot_reranker.rerank(user_query, candidate_documents)
                if reranked_docs_zero_shot:
                    print(f"Zero-shot reranking completed. Top document: {reranked_docs_zero_shot[0].page_content[:50]}...")

                # 4. Predictive Reranking (Optional)
                if use_predictive_reranker and self.predictive_reranker.is_trained:
                    reranked_docs_final = self.predictive_reranker.rerank(user_query, reranked_docs_zero_shot)
                    if reranked_docs_final:
                        print(f"Predictive reranking applied. Top document: {reranked_docs_final[0].page_content[:50]}...")
                else:
                    reranked_docs_final = reranked_docs_zero_shot
                    if use_predictive_reranker and not self.predictive_reranker.is_trained:
                        print("Predictive Reranker requested but not trained, skipping.")

                # 5. InContext RALM Integration - Context Construction
                selected_docs = reranked_docs_final[:top_k_retrieval]
                retrieved_docs_contents = [doc.page_content for doc in selected_docs]
                final_context = "\n\n".join(retrieved_docs_contents)
                attribution_sources = [doc.page_content.split('.')[0] + '.' for doc in selected_docs] # First sentence as source hint
                print(f"Using {len(selected_docs)} documents for context.")
            else:
                print("No relevant documents retrieved for context.")
        else:
            print(f"Conditional Retrieval: Retrieval NOT needed for query '{user_query}'.")

        # 6. Core LM Inference
        # Prepare the prompt for the LM
        if final_context:
            lm_prompt = f"Based on the following medical information, answer the question accurately and concisely. If the information is not sufficient, state that you cannot answer from the provided context.\n\nMedical Information:\n{final_context}\n\nQuestion: {user_query}\nAnswer:"
        else:
            lm_prompt = f"Answer the following question accurately and concisely. If it is a medical question and you do not have sufficient internal knowledge, state that you cannot provide an answer.\n\nQuestion: {user_query}\nAnswer:"

        response = self.lm.generate(lm_prompt)

        # 7. Response Generation & Attribution
        attribution = ""
        if attribution_sources:
            attribution = "\n\nSources:\n" + "\n".join([f"- {src}" for src in attribution_sources])

        return response + attribution


if __name__ == "__main__":
    # Mock Medical Documents
    medical_docs = [
        "Ibuprofen is a nonsteroidal anti-inflammatory drug (NSAID) used for pain relief, fever reduction, and to reduce inflammation. It works by inhibiting the production of prostaglandins.",
        "Diabetes mellitus is a chronic metabolic disease characterized by high blood sugar levels. Type 1 diabetes is an autoimmune condition, while Type 2 diabetes is often associated with insulin resistance. Common symptoms include increased thirst, frequent urination, and unexplained weight loss.",
        "Hypertension, also known as high blood pressure, is a serious medical condition that significantly increases the risks of heart, brain, kidney and other diseases. It is diagnosed when blood pressure readings are consistently 140/90 mmHg or higher. Lifestyle changes can help manage hypertension.",
        "Symptoms of a common cold include a runny nose, sneezing, sore throat, and coughing. It is typically caused by a viral infection and usually resolves within a week with rest and fluids.",
        "Vaccines work by training the immune system to recognize and fight off specific pathogens, like viruses or bacteria, without exposing the body to the full disease. This creates immunity.",
        "Aspirin is often used as a pain reliever and fever reducer. It also has anti-inflammatory effects and is used as an anti-platelet agent to prevent blood clots in certain cardiovascular conditions, but should be used under medical supervision.",
        "Asthma is a chronic respiratory condition characterized by inflammation and narrowing of the airways, leading to symptoms like wheezing, shortness of breath, chest tightness, and coughing. Triggers can include allergens, exercise, and cold air.",
        "The human heart is a muscular organ that pumps blood throughout the body. It has four chambers: two atria and two ventricles. It is located slightly to the left of your breastbone.",
        "Vitamin D is essential for bone health and immune function. It can be obtained from sunlight exposure, certain foods like fatty fish, and supplements. Deficiency can lead to bone problems.",
        "The liver performs many essential functions, including detoxification, protein synthesis, and the production of biochemicals necessary for digestion. It is the largest internal organ.",
        "A balanced diet rich in fruits, vegetables, whole grains, and lean proteins is crucial for maintaining overall health and preventing chronic diseases, such as heart disease and diabetes.",
        "Regular physical activity, such as walking, jogging, or swimming, helps improve cardiovascular health, strengthen muscles, manage weight, and boost mood. Aim for at least 30 minutes most days of the week."
    ]

    # Initialize the assistant
    print("Initializing Medical Information Assistant...")
    assistant = MedicalInformationAssistant(medical_docs)

    # --- Train Predictive Models with Synthetic Data ---
    # Synthetic data for Conditional Retrieval Predictor
    retrieval_training_data = {
        "What is ibuprofen?": True,
        "What are the symptoms of diabetes?": True,
        "Tell me about the human heart.": True,
        "What is a common cold?": True,
        "What is the capital of France?": False, # Common knowledge, no retrieval needed
        "What color is the sky?": False,        # Common knowledge, no retrieval needed
        "Who wrote Hamlet?": False,            # Common knowledge, no retrieval needed
        "How do vaccines work?": True,
        "What is hypertension?": True,
        "What is vitamin D?": True,
        "Tell me about healthy eating.": True,
        "What is the liver's function?": True,
        "Who is the current US president?": False # Common knowledge, no retrieval needed
    }

    # Synthetic data for Predictive Reranker
    # Documents here should ideally be the actual chunked documents, but using content strings for simplicity.
    reranking_training_data = {
        "What is ibuprofen?": {
            "Ibuprofen is a nonsteroidal anti-inflammatory drug (NSAID) used for pain relief, fever reduction, and to reduce inflammation. It works by inhibiting the production of prostaglandins.": True,
            "Aspirin is often used as a pain reliever and fever reducer. It also has anti-inflammatory effects and is used as an anti-platelet agent to prevent blood clots in certain cardiovascular conditions, but should be used under medical supervision.": False,
            "Diabetes mellitus is a chronic metabolic disease characterized by high blood sugar levels. Type 1 diabetes is an autoimmune condition, while Type 2 diabetes is often associated with insulin resistance. Common symptoms include increased thirst, frequent urination, and unexplained weight loss.": False
        },
        "What are the symptoms of diabetes?": {
            "Diabetes mellitus is a chronic metabolic disease characterized by high blood sugar levels. Type 1 diabetes is an autoimmune condition, while Type 2 diabetes is often associated with insulin resistance. Common symptoms include increased thirst, frequent urination, and unexplained weight loss.": True,
            "Symptoms of a common cold include a runny nose, sneezing, sore throat, and coughing. It is typically caused by a viral infection and usually resolves within a week with rest and fluids.": False,
            "Hypertension, also known as high blood pressure, is a serious medical condition that significantly increases the risks of heart, brain, kidney and other diseases. It is diagnosed when blood pressure readings are consistently 140/90 mmHg or higher. Lifestyle changes can help manage hypertension.": False
        },
        "How do vaccines work?": {
            "Vaccines work by training the immune system to recognize and fight off specific pathogens, like viruses or bacteria, without exposing the body to the full disease. This creates immunity.": True,
            "The human heart is a muscular organ that pumps blood throughout the body. It has four chambers: two atria and two ventricles. It is located slightly to the left of your breastbone.": False,
            "A balanced diet rich in fruits, vegetables, whole grains, and lean proteins is crucial for maintaining overall health and preventing chronic diseases, such as heart disease and diabetes.": False
        }
    }

    assistant.train_predictive_models(retrieval_training_data, reranking_training_data)


    # --- Test Queries ---
    print("\n--- Testing Medical Information Assistant ---")

    queries_to_test = [
        "What is ibuprofen used for?",
        "What are the symptoms of asthma?",
        "Who is the president of the moon?", # Should trigger Conditional Retrieval and find no relevant docs
        "What color is the sky?",          # Should trigger Conditional Retrieval and skip retrieval
        "How do vaccines work to protect us?",
        "Tell me about diabetes and its types.",
        "What are the main functions of the liver?",
        "Give me some advice for maintaining good health."
    ]

    for i, query in enumerate(queries_to_test):
        print(f"\n--- Query {i+1}: {query} ---")
        # Test with and without predictive reranker to show its potential
        response = assistant.query(query, use_predictive_reranker=True) # Set to True to use predictive reranker
        print(f"Assistant Response:\n{response}")
        print("-" * 50)

    # Example demonstrating Conditional Retrieval skipping
    print("\n--- Testing Conditional Retrieval Skipping ---")
    query_common_knowledge = "What is the capital of France?"
    print(f"\n--- Query: {query_common_knowledge} ---")
    response_common_knowledge = assistant.query(query_common_knowledge, use_predictive_reranker=False)
    print(f"Assistant Response:\n{response_common_knowledge}")
    print("-" * 50)

    query_medical_known = "What are NSAIDs?"
    print(f"\n--- Query: {query_medical_known} ---")
    response_medical_known = assistant.query(query_medical_known, use_predictive_reranker=False)
    print(f"Assistant Response:\n{response_medical_known}")
    print("-" * 50)