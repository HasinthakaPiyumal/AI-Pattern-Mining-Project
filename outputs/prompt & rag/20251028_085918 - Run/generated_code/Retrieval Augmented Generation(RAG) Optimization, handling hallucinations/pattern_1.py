import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

class MedicalRAGAssistant:
    def __init__(self, model_name='all-MiniLM-L6-v2', top_k=3):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []
        self.document_texts = [] # Store original texts
        self.top_k = top_k

    def ingest_medical_documents(self, new_documents):
        """Ingests new medical documents, chunks them, and updates the FAISS index."""
        print(f"Ingesting {len(new_documents)} new documents...")
        for doc_id, doc_text in enumerate(new_documents):
            # Simple chunking: Treat each document as a chunk for this example
            # In a real scenario, documents would be split into smaller, meaningful chunks
            self.document_texts.append(doc_text)
            self.documents.append({'id': len(self.documents), 'text': doc_text})

        # Generate embeddings for new documents
        new_embeddings = self.model.encode(new_documents, convert_to_tensor=False)

        if self.index is None:
            # Initialize FAISS index
            self.dimension = new_embeddings.shape[1]
            self.index = faiss.IndexFlatL2(self.dimension) # L2 distance for similarity
            print(f"FAISS index initialized with dimension: {self.dimension}")
        
        # Add embeddings to the index
        self.index.add(np.array(new_embeddings).astype('float32'))
        print(f"FAISS index updated. Total documents indexed: {self.index.ntotal}")

    def retrieve_relevant_info(self, query):
        """Retrieves top_k most relevant medical information chunks for a given query."""
        query_embedding = self.model.encode([query], convert_to_tensor=False).astype('float32')
        
        if self.index is None or self.index.ntotal == 0:
            return []
            
        distances, indices = self.index.search(query_embedding, self.top_k)
        
        retrieved_documents = []
        for i, idx in enumerate(indices[0]):
            if idx != -1: # Ensure the index is valid
                retrieved_documents.append({
                    'text': self.document_texts[idx],
                    'distance': distances[0][i]
                })
        return retrieved_documents

    def generate_response(self, query, retrieved_info):
        """Simulates LLM response generation based on query and retrieved information."""
        if not retrieved_info:
            return "I couldn't find relevant medical information for your query. Please rephrase or provide more details."

        context = "\n\n".join([doc['text'] for doc in retrieved_info])
        
        # --- This is where a real LLM would be integrated ---
        # Example with a hypothetical LLM API call:
        # from openai import OpenAI
        # client = OpenAI()
        # response = client.chat.completions.create(
        #     model="gpt-4",
        #     messages=[
        #         {"role": "system", "content": "You are a medical diagnostic assistant. Provide concise, evidence-based answers based on the provided medical context."}, 
        #         {"role": "user", "content": f"Based on the following medical information:\n\n{context}\n\nAnswer the question: {query}"}
        #     ]
        # )
        # return response.choices[0].message.content
        # ---------------------------------------------------

        # For demonstration, we'll simulate an LLM by combining query and context
        simulated_response = f"Based on the medical information retrieved, regarding '{query}':\n\n" \
                             f"Retrieved context: {context}\n\n" \
                             f"[LLM would synthesize an answer here, e.g., 'The documents suggest...', 'According to recent guidelines...']"
        return simulated_response

    def assist(self, query):
        """Main function to provide RAG-powered medical assistance."""
        print(f"\nUser Query: {query}")
        retrieved_info = self.retrieve_relevant_info(query)
        response = self.generate_response(query, retrieved_info)
        return response

if __name__ == "__main__":
    assistant = MedicalRAGAssistant()

    # Simulate ingesting some medical documents
    medical_docs = [
        "COVID-19 symptoms often include fever, cough, fatigue, and loss of taste or smell. Severe cases may lead to pneumonia and acute respiratory distress syndrome (ARDS).",
        "Treatment for Type 2 Diabetes typically involves lifestyle modifications (diet, exercise), and may include medications like Metformin, GLP-1 receptor agonists, or insulin. Regular blood glucose monitoring is crucial.",
        "Hypertension (high blood pressure) is a major risk factor for heart disease and stroke. Management includes dietary changes (low sodium), regular physical activity, and often antihypertensive medications such as ACE inhibitors or calcium channel blockers.",
        "The recommended vaccination schedule for infants includes DTaP, Hib, Polio, PCV13, Rotavirus, and Hepatitis B vaccines at various stages from birth to 18 months. Consult updated CDC guidelines for specifics."
    ]
    assistant.ingest_medical_documents(medical_docs)

    # Test the assistant with various queries
    queries = [
        "What are the common symptoms of COVID-19?",
        "How is Type 2 Diabetes treated?",
        "What causes high blood pressure?", # Expecting some relevant info on hypertension management
        "What are the recommended infant vaccinations?",
        "Tell me about blockchain technology." # Irrelevant query
    ]

    for q in queries:
        result = assistant.assist(q)
        print(f"Assistant Response: {result}\n{'-'*80}")
