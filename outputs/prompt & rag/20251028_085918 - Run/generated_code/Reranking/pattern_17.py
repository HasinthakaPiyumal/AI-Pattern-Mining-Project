"""
medical_diagnostic_assistant.py

This script implements a Medical Diagnostic Assistant with Enhanced Knowledge Retrieval.
It leverages InContext Retrieval-Augmented Language Modeling (RALM), Zero-Shot LM Reranking,
Predictive Reranking, and Conditional Retrieval to provide accurate and attributed diagnostic support.

Key Components:
1.  **Data Ingestion and Indexing:** Loads medical documents, splits them, creates embeddings,
    and stores them in a Chroma vector database.
2.  **InContext Retrieval-Augmented Language Modeling (InContext RALM):** Integrates an LLM
    with retrieved context for generating diagnostic responses.
3.  **Zero-Shot LM Reranking:** Re-ranks initial retrieved documents based on semantic similarity
    using a pre-trained sentence transformer.
4.  **Predictive Reranking (Trained LM-Dedicated Reranker):** (Placeholder) Represents a trained
    model to predict optimal document relevance.
5.  **Conditional Retrieval Module:** Determines whether external knowledge retrieval is necessary
    for a given query.

Usage:
- Ensure necessary libraries are installed (`pip install langchain chromadb transformers sentence-transformers scikit-learn openai`).
- Replace placeholder API keys and model names as needed.
- Run the script and interact with the `MedicalDiagnosticAssistant`.
"""

import os
from typing import List, Dict, Any

# Langchain components
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI # Using OpenAI as an example LLM

# Transformers for LM-based reranking
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# For predictive reranking placeholder
import torch
import torch.nn as nn
import torch.nn.functional as F

# Scikit-learn for conditional retrieval
from sklearn.linear_model import LogisticRegression
import numpy as np

# --- 1. Data Ingestion and Indexing ---

class MedicalKnowledgeBase:
    def __init__(self, persist_directory: str = "./chroma_db", model_name: str = "all-MiniLM-L6-v2"):
        self.persist_directory = persist_directory
        self.embeddings = SentenceTransformerEmbeddings(model_name=model_name)
        self.vectorstore = self._initialize_vectorstore()

    def _initialize_vectorstore(self):
        # Initialize ChromaDB, creating it if it doesn't exist
        print(f"Initializing ChromaDB at {self.persist_directory}")
        try:
            return Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)
        except Exception as e:
            print(f"Error initializing ChromaDB from existing directory: {e}. Creating a new one.")
            return Chroma.from_documents(documents=[], embedding_function=self.embeddings, persist_directory=self.persist_directory)

    def ingest_documents(self, doc_paths: List[str]):
        documents = []
        for path in doc_paths:
            try:
                loader = TextLoader(path)
                documents.extend(loader.load())
            except Exception as e:
                print(f"Could not load {path}: {e}")
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(documents)
        print(f"Ingesting {len(split_docs)} document chunks...")
        
        if split_docs:
            # Add documents incrementally to the existing vectorstore
            self.vectorstore.add_documents(split_docs)
            self.vectorstore.persist()
            print("Documents ingested and persisted.")
        else:
            print("No documents to ingest.")

    def retrieve_documents(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        print(f"Retrieving top {k} documents for query: '{query}'")
        results = self.vectorstore.similarity_search(query, k=k)
        return [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in results]

# --- 2. InContext Retrieval-Augmented Language Modeling (InContext RALM) ---

class InContextRALM:
    def __init__(self, llm_model_name: str = "gpt-3.5-turbo", temperature: float = 0.7):
        # Ensure OPENAI_API_KEY is set in environment variables
        if "OPENAI_API_KEY" not in os.environ:
            print("WARNING: OPENAI_API_KEY environment variable not set. Using a placeholder LLM will fail or require local setup.")
            # Fallback for demonstration if no API key is set, though it won't work without a valid LLM
            self.llm = None # Placeholder
        else:
            self.llm = ChatOpenAI(model_name=llm_model_name, temperature=temperature)
        
        # RetrievalQA will be built dynamically with the retriever
        self.qa_chain = None

    def set_retriever(self, retriever):
        if self.llm:
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff", # Stuff all retrieved documents into the prompt
                retriever=retriever,
                return_source_documents=True
            )
        else:
            print("LLM not initialized. Cannot set up QA chain.")

    def generate_response(self, query: str) -> Dict[str, Any]:
        if not self.qa_chain:
            return {"answer": "LLM or QA chain not properly initialized.", "source_documents": []}
        
        print(f"Generating response using RALM for query: '{query}'")
        response = self.qa_chain.invoke({"query": query})
        
        sources = []
        if response.get("source_documents"):
            for doc in response["source_documents"]:
                sources.append({"content": doc.page_content, "metadata": doc.metadata})

        return {"answer": response["result"], "source_documents": sources}

# --- 3. Zero-Shot LM Reranking ---

class ZeroShotLMReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.reranker_pipeline = pipeline(
            "text-classification", 
            model=self.model, 
            tokenizer=self.tokenizer
        )

    def rerank_documents(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not documents:
            return []

        print("Performing Zero-Shot LM Reranking...")
        # Prepare pairs for cross-encoder
        sentence_pairs = [[query, doc["page_content"]] for doc in documents]
        
        # The cross-encoder outputs a score (logit) for relevance
        # We need to extract the score for the positive class (entailment/relevance)
        scores = []
        # Process in batches if sentence_pairs is very large
        for pair in sentence_pairs:
            result = self.reranker_pipeline(pair)
            # Assuming 'LABEL_1' is the relevant/positive class, which is common for cross-encoders
            # For some models, it might be the only label and its score is the relevance.
            # We'll take the score of the most relevant label, or directly the score if it's binary.
            if len(result) > 0 and 'score' in result[0]:
                 # For cross-encoders, the score often indicates similarity/relevance directly
                 scores.append(result[0]['score'] if result[0]['label'] == 'LABEL_1' else (1 - result[0]['score']))
            else:
                # Fallback if the pipeline output structure is unexpected
                scores.append(0.0) # Assign a low score if unable to process

        # Pair documents with their scores and sort
        scored_documents = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        reranked_docs = [doc for doc, score in scored_documents]
        print(f"Reranked {len(reranked_docs)} documents.")
        return reranked_docs

# --- 4. Predictive Reranking (Trained LM-Dedicated Reranker) ---
# This is a placeholder. In a real scenario, this would be a pre-trained model.

class PredictiveRerankerModel(nn.Module):
    def __init__(self, input_dim: int = 768): # Assuming embedding dimension
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x)) # Output a relevance score between 0 and 1

class PredictiveReranker:
    def __init__(self, model_path: str = None, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformerEmbeddings(model_name=embedding_model_name)
        self.model = PredictiveRerankerModel() # Initialize with default input_dim
        if model_path and os.path.exists(model_path):
            print(f"Loading Predictive Reranker model from {model_path}")
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            self.model.eval()
        else:
            print("WARNING: Predictive Reranker model path not provided or does not exist. Using a dummy model.")
            # For demonstration, we'll just use a random score if no model is loaded

    def _get_embedding(self, text: str) -> np.ndarray:
        # SentenceTransformerEmbeddings returns a list of embeddings
        return self.embedding_model.embed_query(text)

    def rerank_documents(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not documents:
            return []

        print("Performing Predictive Reranking...")
        scored_documents = []
        
        # In a real scenario, you'd feed query_embedding + document_embedding to the trained model
        # For this placeholder, we'll simulate scores.
        with torch.no_grad():
            for doc in documents:
                # For a real predictive reranker, you'd combine query and doc embeddings
                # For this placeholder, let's just use a dummy score or a random one
                # A more sophisticated placeholder might compare embeddings and use a simple model
                
                # Placeholder: Simulate a score. A real model would use actual embeddings
                # For demo, let's make it a simple random value or based on some heuristic
                score = np.random.rand() # Dummy score if no model loaded
                if self.model: # If a model was theoretically loaded
                    # This is a simplification. A real model would take concatenated embeddings
                    # For now, let's pass a dummy tensor with the expected input_dim
                    dummy_input = torch.randn(1, 768) # Assuming 768 is embedding dim
                    score = self.model(dummy_input).item()

                scored_documents.append((doc, score))
        
        # Sort by predicted relevance score
        scored_documents = sorted(scored_documents, key=lambda x: x[1], reverse=True)
        reranked_docs = [doc for doc, score in scored_documents]
        print(f"Reranked {len(reranked_docs)} documents using predictive reranker.")
        return reranked_docs

# --- 5. Conditional Retrieval Module ---

class ConditionalRetrieval:
    def __init__(self, model_path: str = None):
        self.classifier = LogisticRegression() # Placeholder classifier
        self.vectorizer = None # Placeholder for a text vectorizer (e.g., TfidfVectorizer)
        
        # In a real scenario, you'd load a trained model and its vectorizer
        if model_path and os.path.exists(model_path):
            print(f"Loading Conditional Retrieval model from {model_path}")
            # self.classifier = load_model(model_path) # Example: joblib.load
            # self.vectorizer = load_vectorizer(vectorizer_path) # Example
            # For demonstration, we'll use a dummy trained model
            self.is_trained = True
            # Dummy training for demonstration purposes - DO NOT USE IN PRODUCTION
            X_dummy = np.array([[0.1, 0.2, 0.3], [0.9, 0.8, 0.7], [0.4, 0.5, 0.6]])
            y_dummy = np.array([0, 1, 0])
            self.classifier.fit(X_dummy, y_dummy) # Fit with dummy data
            self.vectorizer = lambda x: np.array([[len(x.split()) / 10.0, 'medical' in x.lower(), 'symptoms' in x.lower()]])

        else:
            print("WARNING: Conditional Retrieval model path not provided or does not exist. Using rule-based or untrained logic.")
            self.is_trained = False

    def should_retrieve(self, query: str) -> bool:
        print(f"Assessing query for conditional retrieval: '{query}'")
        if self.is_trained:
            # In a real system, vectorize query and predict
            # query_features = self.vectorizer.transform([query])
            # prediction = self.classifier.predict(query_features)[0]
            # return prediction == 1 # 1 means retrieve

            # For dummy trained model, use its dummy vectorizer
            query_features = self.vectorizer(query)
            prediction = self.classifier.predict(query_features)[0]
            should_retrieve = bool(prediction)
            print(f"Conditional Retrieval (Trained): Query '{query}' -> Should retrieve: {should_retrieve}")
            return should_retrieve
        else:
            # Rule-based logic if no trained model
            # Retrieve if query contains complex medical terms or is a question
            medical_keywords = ["diagnosis", "symptoms", "treatment", "prognosis", "disease", "condition", "syndrome"]
            is_complex_query = any(keyword in query.lower() for keyword in medical_keywords) or "?" in query
            print(f"Conditional Retrieval (Rule-based): Query '{query}' -> Should retrieve: {is_complex_query}")
            return is_complex_query

# --- Main Medical Diagnostic Assistant ---

class MedicalDiagnosticAssistant:
    def __init__(
        self,
        knowledge_base_dir: str = "./chroma_db",
        llm_model_name: str = "gpt-3.5-turbo",
        zero_shot_reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        predictive_reranker_path: str = None,
        conditional_retrieval_model_path: str = None
    ):
        self.knowledge_base = MedicalKnowledgeBase(persist_directory=knowledge_base_dir)
        self.ralm = InContextRALM(llm_model_name=llm_model_name)
        self.zero_shot_reranker = ZeroShotLMReranker(model_name=zero_shot_reranker_model)
        self.predictive_reranker = PredictiveReranker(model_path=predictive_reranker_path)
        self.conditional_retrieval = ConditionalRetrieval(model_path=conditional_retrieval_model_path)

    def ingest_medical_documents(self, doc_paths: List[str]):
        self.knowledge_base.ingest_documents(doc_paths)
        # Update the RALM's retriever with the refreshed knowledge base
        self.ralm.set_retriever(self.knowledge_base.vectorstore.as_retriever())

    def diagnose(self, query: str) -> Dict[str, Any]:
        print("\n--- Starting Diagnostic Process ---")
        final_response = {"answer": "", "source_documents": []}

        # Step 5: Conditional Retrieval
        if self.conditional_retrieval.should_retrieve(query):
            print("External knowledge retrieval deemed necessary.")
            # Step 1 (partial): Initial Retrieval
            initial_docs = self.knowledge_base.retrieve_documents(query, k=10)
            
            if not initial_docs:
                final_response["answer"] = "Could not find relevant external documents. Trying to answer with base LLM if possible."
                # Fallback to base LLM if no docs are found, without context
                if self.ralm.llm:
                    try:
                        base_llm_response = self.ralm.llm.invoke(query)
                        final_response["answer"] = base_llm_response.content
                    except Exception as e:
                        final_response["answer"] += f" Failed to get response from base LLM: {e}"
                return final_response

            # Step 3: Zero-Shot LM Reranking
            zero_shot_reranked_docs = self.zero_shot_reranker.rerank_documents(query, initial_docs)
            
            # Step 4: Predictive Reranking (Trained LM-Dedicated Reranker)
            # Use a smaller subset for the predictive reranker to optimize if needed
            predictive_reranked_docs = self.predictive_reranker.rerank_documents(query, zero_shot_reranked_docs[:5])

            # Combine and use the top documents (e.g., top 3 from predictive, then top from zero-shot if predictive has fewer)
            context_docs_for_ralm = predictive_reranked_docs if predictive_reranked_docs else zero_shot_reranked_docs
            # Ensure the RALM retriever is set to use the reranked documents indirectly
            # In a real system, the retriever would be more dynamic to feed these specific docs.
            # For simplicity with RetrievalQA, we pass the original retriever and rely on its top-k, 
            # but ideally, we'd feed the specific reranked docs as context.
            # Here, we will manually construct the prompt with the top reranked docs.

            # Manually construct prompt with reranked context for the LLM
            context_text = "\n\n".join([doc["page_content"] for doc in context_docs_for_ralm[:3]]) # Use top 3 as context
            prompt_with_context = f"Context from medical documents:\n{context_text}\n\nPatient Query: {query}\n\nBased on the provided context and your medical knowledge, please provide a diagnostic assessment and potential next steps, citing sources where applicable.\n"
            
            if self.ralm.llm:
                try:
                    llm_response = self.ralm.llm.invoke(prompt_with_context)
                    final_response["answer"] = llm_response.content
                    final_response["source_documents"] = context_docs_for_ralm[:3]
                except Exception as e:
                    final_response["answer"] = f"Error generating response with context: {e}"
            else:
                final_response["answer"] = "LLM not initialized. Cannot generate response with context."

        else:
            print("External knowledge retrieval deemed NOT necessary. Answering with base LLM.")
            # Step 2 (partial): InContext RALM without external retrieval (base LM)
            if self.ralm.llm:
                try:
                    base_llm_response = self.ralm.llm.invoke(query)
                    final_response["answer"] = base_llm_response.content
                except Exception as e:
                    final_response["answer"] = f"Error generating response from base LLM: {e}"
            else:
                final_response["answer"] = "LLM not initialized. Cannot generate base response."

        print("--- Diagnostic Process Complete ---\n")
        return final_response

# --- Example Usage ---

if __name__ == "__main__":
    # Create dummy medical documents
    os.makedirs("medical_docs", exist_ok=True)
    with open("medical_docs/doc1.txt", "w") as f:
        f.write("\n".join([
            "Patient presented with severe headache, fever, and stiff neck. Lumbar puncture revealed increased white blood cell count and low glucose, suggestive of bacterial meningitis. Treatment typically involves broad-spectrum antibiotics like ceftriaxone and vancomycin. Early diagnosis is crucial.",
            "Source: Harrison's Principles of Internal Medicine, Chapter 161: Bacterial Meningitis."
        ]))
    with open("medical_docs/doc2.txt", "w") as f:
        f.write("\n".join([
            "Common symptoms of viral infection include fatigue, muscle aches, and mild fever. Unlike bacterial infections, viral infections often do not respond to antibiotics. Rest and hydration are key. Examples include influenza and common cold.",
            "Source: CDC Guidelines for Viral Illnesses."
        ]))
    with open("medical_docs/doc3.txt", "w") as f:
        f.write("\n".join([
            "Diabetes Mellitus Type 2 is characterized by insulin resistance and hyperglycemia. Management includes diet, exercise, and medications such as metformin. Regular monitoring of blood glucose levels is essential to prevent complications like neuropathy and retinopathy.",
            "Source: American Diabetes Association Clinical Practice Guidelines."
        ]))
    with open("medical_docs/doc4.txt", "w") as f:
        f.write("\n".join([
            "A patient experiencing chest pain, shortness of breath, and radiating pain to the left arm should be immediately evaluated for myocardial infarction. ECG and cardiac enzyme markers are vital for diagnosis. Aspirin and nitroglycerin are often administered acutely.",
            "Source: American Heart Association Emergency Cardiovascular Care."
        ]))
    with open("medical_docs/doc5.txt", "w") as f:
        f.write("\n".join([
            "Seasonal allergies, also known as allergic rhinitis, manifest with sneezing, runny nose, itchy eyes, and nasal congestion. They are triggered by allergens like pollen. Antihistamines and nasal corticosteroids are common treatments.",
            "Source: Mayo Clinic - Allergic Rhinitis."
        ]))

    # Initialize the assistant
    # Set OPENAI_API_KEY as an environment variable or replace ChatOpenAI with a local LLM setup.
    assistant = MedicalDiagnosticAssistant(
        knowledge_base_dir="./medical_chroma_db",
        llm_model_name="gpt-3.5-turbo", # Or a local LLM e.g., via HuggingFacePipeline
        # predictive_reranker_path="./my_trained_reranker.pt" # Uncomment and provide path if you have a trained model
    )

    # Ingest documents
    doc_paths = [f"medical_docs/doc{i}.txt" for i in range(1, 6)]
    assistant.ingest_medical_documents(doc_paths)

    # --- Test Queries ---
    queries = [
        "What are the symptoms and treatment for bacterial meningitis?", # Complex, should retrieve
        "What are common signs of a heart attack?", # Complex, should retrieve
        "What is the weather like today?", # Simple, should NOT retrieve (rule-based)
        "How do you treat type 2 diabetes?", # Complex, should retrieve
        "What causes a common cold?", # Simple, might retrieve based on keywords if not specific enough (rule-based)
        "Tell me a joke.", # Simple, should NOT retrieve
        "What are the symptoms of seasonal allergies and how are they treated?" # Complex, should retrieve
    ]

    for i, query in enumerate(queries):
        print(f"\n----- Processing Query {i+1}/{len(queries)} -----")
        response = assistant.diagnose(query)
        print(f"Query: {query}")
        print(f"Answer: {response['answer']}")
        if response['source_documents']:
            print("Sources:")
            for j, source in enumerate(response['source_documents']):
                print(f"  {j+1}. {source['content'][:100]}... (Metadata: {source['metadata']})")
        else:
            print("No specific external sources used.")
        print("-----------------------------------------")

    # Clean up dummy documents and chromadb
    import shutil
    if os.path.exists("medical_docs"): shutil.rmtree("medical_docs")
    if os.path.exists("medical_chroma_db"): shutil.rmtree("medical_chroma_db")

