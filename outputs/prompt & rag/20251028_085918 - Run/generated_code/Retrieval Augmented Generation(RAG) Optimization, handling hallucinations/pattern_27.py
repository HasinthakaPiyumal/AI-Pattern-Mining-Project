"""
rag_system.py: Implements the Adaptive RAG logic for the Clinical Decision Support System.
"""

import os
from typing import List, Dict, Any
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch

class AdaptiveRAGSystem:
    def __init__(self, 
                 model_name: str = "HuggingFaceH4/zephyr-7b-beta", 
                 embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 persist_directory: str = "./chroma_db"):
        
        self.persist_directory = persist_directory
        
        # 1. Initialize Embedding Model
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        
        # 2. Initialize LLM
        print(f"Loading LLM: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16, # Use bfloat16 for better performance on modern GPUs
            device_map="auto",
            trust_remote_code=True
        )
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.15,
            device_map="auto"
        )
        self.llm = HuggingFacePipeline(pipeline=self.pipe)
        print("LLM Loaded.")

        # 3. Initialize Vector Store (ChromaDB)
        # If the directory exists, load the existing collection, otherwise initialize.
        if os.path.exists(persist_directory):
            print(f"Loading existing ChromaDB from {persist_directory}")
            self.vectorstore = Chroma(persist_directory=persist_directory, embedding_function=self.embeddings)
        else:
            print(f"Initializing new ChromaDB at {persist_directory}")
            self.vectorstore = None # Will be initialized upon document loading

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )

    def load_documents(self, data_path: str = "./data"):
        print(f"Loading documents from {data_path}...")
        documents = []
        for root, _, files in os.walk(data_path):
            for file in files:
                filepath = os.path.join(root, file)
                if file.endswith(".txt"):
                    loader = TextLoader(filepath)
                elif file.endswith(".pdf"):
                    loader = PyPDFLoader(filepath)
                else:
                    continue
                try:
                    documents.extend(loader.load())
                    print(f"Loaded {filepath}")
                except Exception as e:
                    print(f"Error loading {filepath}: {e}")

        if not documents:
            print("No documents loaded. Please ensure './data' contains .txt or .pdf files.")
            return

        print(f"Splitting {len(documents)} documents into chunks...")
        chunks = self.text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks.")

        # Initialize or update vectorstore
        if self.vectorstore is None:
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            print(f"New ChromaDB created and persisted to {self.persist_directory}")
        else:
            # This would typically involve more complex upsert logic for production
            # For this example, we'll just add new documents if any, assuming no duplicate handling.
            # A full system would manage document IDs to prevent duplicates or update existing ones.
            print("Adding documents to existing ChromaDB. Note: This example does not handle complex upsert logic or duplicate detection.")
            self.vectorstore.add_documents(chunks)
            self.vectorstore.persist()
            print(f"Documents added to ChromaDB and persisted to {self.persist_directory}")
        
        print("Document loading and indexing complete.")

    def _determine_retrieval_strategy(self, query: str) -> str:
        # Simple heuristic for dynamic retrieval strategy
        # In a real system, this could involve a small classification model
        # or more sophisticated query analysis.
        if len(query.split()) < 5 and any(word in query.lower() for word in ["drug", "dosage", "diagnosis", "symptom"]):
            return "keyword_enhanced_vector"
        return "vector_similarity"

    def _retrieve_documents(self, query: str, k: int = 5) -> List[str]:
        strategy = self._determine_retrieval_strategy(query)
        print(f"Retrieval Strategy: {strategy}")

        retrieved_docs = []
        if strategy == "vector_similarity":
            docs = self.vectorstore.similarity_search(query, k=k)
            retrieved_docs = [doc.page_content for doc in docs]
        elif strategy == "keyword_enhanced_vector":
            # This is a simplified example. A real implementation might combine
            # keyword search results with vector search results, or re-rank.
            docs_vector = self.vectorstore.similarity_search(query, k=k-2) # Get some via vector
            # Simulate keyword search by trying to find chunks containing query terms
            # In a real system, you'd integrate a true keyword search engine (e.g., Elasticsearch)
            keyword_docs = self.vectorstore.similarity_search_with_score(query, k=k) # Simple re-use of vector for demo
            keyword_docs = [doc for doc, score in keyword_docs if score < 0.5] # Assume lower score is better for similarity
            
            retrieved_docs_combined = list(set([doc.page_content for doc in docs_vector] + 
                                               [doc.page_content for doc in keyword_docs]))
            retrieved_docs = retrieved_docs_combined[:k] # Trim to k
        
        # Iterative Refinement (simple re-ranking based on LLM's perceived relevance)
        if retrieved_docs:
            print("Performing iterative refinement (LLM re-ranking)...")
            reranked_docs = []
            for doc_content in retrieved_docs:
                prompt_for_relevance = f"Given the query: '{query}' and the document: '{doc_content}', is this document highly relevant? Answer with 'yes' or 'no'." 
                relevance_check = self.llm.invoke(prompt_for_relevance).strip().lower()
                if "yes" in relevance_check:
                    reranked_docs.append(doc_content)
            retrieved_docs = reranked_docs if reranked_docs else retrieved_docs # Use reranked if not empty
        
        return retrieved_docs

    def _generate_answer(self, query: str, context: List[str]) -> Dict[str, Any]:
        context_str = "\n\n".join(context)
        if not context_str:
            return {"answer": "I don't have enough information in my knowledge base to answer this query.", "confidence": 0.0, "needs_more_info": True}

        prompt = f"""You are a highly intelligent and helpful Clinical Decision Support System. 
        Based on the following context, provide a comprehensive and accurate answer to the query. 
        Also, provide a confidence score (0.0-1.0) for your answer and indicate if more information 
        would significantly improve the answer (true/false).

        Context:
        {context_str}

        Query: {query}

        Format your response as a JSON object with 'answer', 'confidence', and 'needs_more_info' keys.
        Example: {{\"answer\": \"The patient should be prescribed X.\", \"confidence\": 0.9, \"needs_more_info\": false}}
        """
        
        try:
            llm_response = self.llm.invoke(prompt)
            # Attempt to parse the JSON response from the LLM
            # The LLM might not always return perfect JSON, so we need robust parsing
            import json
            # Find the first and last curly brace to try and extract JSON
            json_start = llm_response.find('{')
            json_end = llm_response.rfind('}')
            if json_start != -1 and json_end != -1:
                json_str = llm_response[json_start : json_end + 1]
                response_dict = json.loads(json_str)
            else:
                # Fallback if JSON parsing fails
                print(f"Warning: LLM did not return perfect JSON. Raw response: {llm_response}")
                response_dict = {
                    "answer": llm_response, # Return raw response as answer
                    "confidence": 0.5, 
                    "needs_more_info": True
                }
            return response_dict
        except Exception as e:
            print(f"Error parsing LLM response or during generation: {e}")
            return {"answer": f"An error occurred while generating the answer. Error: {e}", "confidence": 0.0, "needs_more_info": True}

    def query(self, patient_data: str) -> Dict[str, Any]:
        print(f"Processing query: {patient_data}")
        
        # Initial Retrieval
        retrieved_context = self._retrieve_documents(patient_data)
        
        # Generate initial answer with self-reflection
        response = self._generate_answer(patient_data, retrieved_context)
        
        # Adaptive decision: Retrieve more if confidence is low or more info is needed
        if response.get("needs_more_info") or response.get("confidence", 0.0) < 0.7:
            print("Adaptive RAG: Low confidence or more info needed. Attempting to retrieve more context...")
            # Formulate a refined query based on the initial response or original query
            refined_query = f"Expand on: {patient_data}. Specifically looking for details related to {response.get('answer', '')[:50]}..."
            additional_context = self._retrieve_documents(refined_query, k=3) # Retrieve fewer additional docs
            
            if additional_context:
                # Combine original and new context, remove duplicates
                combined_context = list(set(retrieved_context + additional_context))
                print(f"Retrieved {len(additional_context)} additional context chunks. Total context: {len(combined_context)}")
                response = self._generate_answer(patient_data, combined_context) # Re-generate with more context
            else:
                print("No additional context found.")
        
        return response

if __name__ == "__main__":
    # Example Usage:
    # 1. Create a 'data' directory and put some .txt or .pdf medical documents inside.
    #    e.g., medical_guideline.txt, patient_case_study.pdf
    
    # Ensure data directory exists
    if not os.path.exists("data"):
        os.makedirs("data")
        with open("data/sample_medical_text.txt", "w") as f:
            f.write("\n".join([
                "Diabetes mellitus is a chronic metabolic disease characterized by elevated levels of blood glucose (or blood sugar), which leads over time to serious damage to the heart, blood vessels, eyes, kidneys, and nerves.",
                "Type 1 diabetes, once known as juvenile diabetes or insulin-dependent diabetes, is a chronic condition in which the pancreas produces little or no insulin. Insulin is a hormone needed to allow sugar (glucose) to enter cells to produce energy.",
                "Type 2 diabetes, once known as adult-onset diabetes, is a chronic condition that affects the way your body processes blood sugar (glucose). With type 2 diabetes, your body either doesn't produce enough insulin, or it resists insulin.",
                "Common symptoms of diabetes include increased thirst, frequent urination, extreme hunger, unexplained weight loss, fatigue, blurred vision, slow-healing sores, and frequent infections.",
                "Metformin is a first-line medication for the treatment of type 2 diabetes, particularly in people who are overweight or obese and have normal kidney function. It works by decreasing glucose production by the liver and increasing insulin sensitivity.",
                "The recommended dosage for Metformin usually starts at 500 mg once or twice daily, increasing gradually to a maximum of 2000-2550 mg per day. Dosage adjustments should always be made under medical supervision.",
                "Hypertension, or high blood pressure, is a common condition in which the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease. Normal blood pressure is typically below 120/80 mmHg.",
                "For a patient presenting with symptoms of extreme thirst and frequent urination, particularly if they have a family history of diabetes, initial diagnostic tests would include fasting plasma glucose, oral glucose tolerance test, and HbA1c."
            ]))
        print("Created 'data/sample_medical_text.txt' for demonstration.")

    rag_system = AdaptiveRAGSystem()
    rag_system.load_documents()

    # Test queries
    print("\n--- Testing Queries ---")

    query1 = "What are the symptoms of diabetes?"
    response1 = rag_system.query(query1)
    print(f"\nQuery: {query1}")
    print(f"Response: {response1['answer']}")
    print(f"Confidence: {response1['confidence']}")
    print(f"Needs More Info: {response1['needs_more_info']}")

    query2 = "What is the recommended dosage for Metformin?"
    response2 = rag_system.query(query2)
    print(f"\nQuery: {query2}")
    print(f"Response: {response2['answer']}")
    print(f"Confidence: {response2['confidence']}")
    print(f"Needs More Info: {response2['needs_more_info']}")

    query3 = "A patient presents with high blood pressure. What diagnostic tests should be considered?"
    response3 = rag_system.query(query3)
    print(f"\nQuery: {query3}")
    print(f"Response: {response3['answer']}")
    print(f"Confidence: {response3['confidence']}")
    print(f"Needs More Info: {response3['needs_more_info']}")

    query4 = "What are the latest treatments for glioblastoma?" # Out of context for sample data
    response4 = rag_system.query(query4)
    print(f"\nQuery: {query4}")
    print(f"Response: {response4['answer']}")
    print(f"Confidence: {response4['confidence']}")
    print(f"Needs More Info: {response4['needs_more_info']}")

    query5 = "Explain the difference between Type 1 and Type 2 diabetes."
    response5 = rag_system.query(query5)
    print(f"\nQuery: {query5}")
    print(f"Response: {response5['answer']}")
    print(f"Confidence: {response5['confidence']}")
    print(f"Needs More Info: {response5['needs_more_info']}")
