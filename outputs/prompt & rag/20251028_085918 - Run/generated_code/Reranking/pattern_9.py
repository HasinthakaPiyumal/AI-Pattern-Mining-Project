import gradio as gr
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
import numpy as np
from sklearn.linear_model import LogisticRegression
import random

# 1. Medical Knowledge Base & Embeddings Setup
# --- Dummy Medical Documents ---
medical_documents = [
    {
        "id": "doc1",
        "text": "Type 2 diabetes is a chronic condition that affects the way your body processes blood sugar (glucose). With type 2 diabetes, your body either doesn't produce enough insulin, or it resists insulin. This can lead to high blood sugar levels, which can cause serious health problems over time.",
        "source": "Mayo Clinic - Type 2 Diabetes"
    },
    {
        "id": "doc2",
        "text": "Insulin resistance is a key feature of type 2 diabetes. It means your body's cells don't respond normally to insulin. Insulin is a hormone made by your pancreas that helps glucose get into your cells for energy. When cells are insulin resistant, glucose builds up in the blood instead of being used for energy.",
        "source": "NIH - Insulin Resistance"
    },
    {
        "id": "doc3",
        "text": "Hypertension, or high blood pressure, is a common condition in which the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease. Blood pressure is determined by the amount of blood your heart pumps and the amount of resistance to blood flow in your arteries.",
        "source": "American Heart Association - Hypertension"
    },
    {
        "id": "doc4",
        "text": "Common symptoms of a heart attack include chest pain, shortness of breath, pain in the left arm, and lightheadedness. Seek emergency medical attention if you suspect you are having a heart attack.",
        "source": "CDC - Heart Attack Symptoms"
    },
    {
        "id": "doc5",
        "text": "The influenza virus causes the flu, a contagious respiratory illness. Symptoms include fever, cough, sore throat, muscle aches, and fatigue. Vaccination is recommended annually to prevent the flu.",
        "source": "WHO - Influenza"
    },
    {
        "id": "doc6",
        "text": "Migraine is a severe type of headache often accompanied by symptoms such as throbbing pain on one side of the head, nausea, vomiting, and extreme sensitivity to light and sound. Triggers can vary widely among individuals.",
        "source": "National Institute of Neurological Disorders and Stroke - Migraine"
    },
    {
        "id": "doc7",
        "text": "Pneumonia is an infection that inflames the air sacs in one or both lungs. The air sacs may fill with fluid or pus, causing cough with phlegm or pus, fever, chills and difficulty breathing. Various organisms, including bacteria, viruses and fungi, can cause pneumonia.",
        "source": "Mayo Clinic - Pneumonia"
    },
    {
        "id": "doc8",
        "text": "Arthritis is an inflammation of one or more joints, causing pain and stiffness that can worsen with age. Different types exist, including osteoarthritis and rheumatoid arthritis, each with distinct causes and treatments.",
        "source": "CDC - Arthritis"
    }
]

# Initialize SentenceTransformer for document and query embeddings
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Initialize ChromaDB
chroma_client = chromadb.Client()
collection_name = "medical_knowledge"

try:
    collection = chroma_client.get_or_create_collection(name=collection_name)
    # Add documents to ChromaDB if not already populated
    if collection.count() == 0:
        print("Populating ChromaDB with medical documents...")
        document_texts = [doc["text"] for doc in medical_documents]
        document_ids = [doc["id"] for doc in medical_documents]
        document_metadatas = [{
            "source": doc["source"],
            "original_text": doc["text"]
        } for doc in medical_documents]

        embeddings = embedding_model.encode(document_texts).tolist()
        collection.add(embeddings=embeddings, documents=document_texts, metadatas=document_metadatas, ids=document_ids)
        print(f"Added {collection.count()} documents to ChromaDB.")
    else:
        print(f"ChromaDB collection '{collection_name}' already contains {collection.count()} documents.")
except Exception as e:
    print(f"Error initializing ChromaDB: {e}")
    print("Please ensure ChromaDB is running or re-initialize if needed.")
    # Fallback for demonstration if ChromaDB is not available for some reason
    # In a real app, this would be robust error handling.
    collection = None


# 2. Conditional Retrieval Model (Simulated)
class ConditionalRetrievalModel:
    def __init__(self):
        # In a real scenario, this would be a trained model.
        # For demonstration, we use a simple rule-based approach or dummy trained model.
        # This model will predict if a query *needs* external retrieval.
        # We'll simulate a simple Logistic Regression trained on dummy data.
        # True means retrieval is needed, False means it's a simple query.
        self.model = LogisticRegression()
        # Dummy training data: Features could be query length, complexity score, keyword presence, etc.
        # Here, we'll simplify: simple queries (short) -> no retrieval (0), complex (long) -> retrieval (1)
        dummy_X = np.array([
            [5],  # "What is flu?" -> 0 (no retrieval)
            [10], # "Symptoms of heart attack?" -> 0 (no retrieval)
            [25], # "Explain the mechanism of insulin resistance in type 2 diabetes." -> 1 (retrieval)
            [15], # "What causes hypertension?" -> 1 (retrieval)
            [8],  # "Define arthritis." -> 0 (no retrieval)
            [30], # "How does the influenza virus spread and what are prevention methods?" -> 1 (retrieval)
            [12]  # "Tell me about migraine headaches." -> 1 (retrieval)
        ])
        dummy_y = np.array([0, 0, 1, 1, 0, 1, 1])
        self.model.fit(dummy_X, dummy_y)

    def predict(self, query: str) -> bool:
        # Simulate features: here, just query length as a proxy for complexity
        query_length_feature = np.array([[len(query.split())]])
        prediction = self.model.predict(query_length_feature)[0]
        return bool(prediction)

conditional_retrieval_predictor = ConditionalRetrievalModel()

# 3. Predictive Reranking (Trained LM-Dedicated Reranker)
# Using a Cross-Encoder for reranking - it's designed to score pairs (query, document)
reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# 4. In-Context Retrieval-Augmented Language Modeling (InContext RALM)
# Using Flan-T5 for answer generation
ralm_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
ralm_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")


class MedicalQueryAnsweringSystem:
    def __init__(self, embedding_model, chroma_collection, conditional_retrieval_predictor, reranker_model, ralm_tokenizer, ralm_model):
        self.embedding_model = embedding_model
        self.chroma_collection = chroma_collection
        self.conditional_retrieval_predictor = conditional_retrieval_predictor
        self.reranker_model = reranker_model
        self.ralm_tokenizer = ralm_tokenizer
        self.ralm_model = ralm_model

    def _initial_retrieve(self, query: str, top_k: int = 5) -> list:
        if not self.chroma_collection:
            return [] # No collection, no retrieval
        query_embedding = self.embedding_model.encode([query]).tolist()
        results = self.chroma_collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=['documents', 'metadatas']
        )
        retrieved_docs = []
        if results and results['documents'] and results['metadatas']:
            for i in range(len(results['documents'][0])):
                doc_text = results['documents'][0][i]
                doc_metadata = results['metadatas'][0][i]
                retrieved_docs.append({
                    "text": doc_text,
                    "source": doc_metadata.get("source", "N/A"),
                    "id": doc_metadata.get("id", f"retrieved_doc_{i}")
                })
        return retrieved_docs

    def _rerank_documents(self, query: str, retrieved_documents: list, top_n: int = 3) -> list:
        if not retrieved_documents:
            return []
        
        # Prepare pairs for the cross-encoder: [(query, doc_text), ...]
        sentence_pairs = [[query, doc["text"]] for doc in retrieved_documents]
        
        # Get scores from the cross-encoder
        scores = self.reranker_model.predict(sentence_pairs)
        
        # Combine documents with their scores and sort
        scored_documents = sorted(
            zip(retrieved_documents, scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Return top_n reranked documents
        return [doc for doc, score in scored_documents[:top_n]]

    def _generate_answer_with_ralm(self, query: str, context_documents: list) -> tuple:
        context_text = "\n\n".join([doc["text"] for doc in context_documents])
        if context_text:
            prompt = f"Context: {context_text}\n\nQuestion: {query}\n\nAnswer:"
        else:
            prompt = f"Question: {query}\n\nAnswer:"

        inputs = self.ralm_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        output_tokens = self.ralm_model.generate(
            **inputs,
            max_new_tokens=200,
            num_beams=4,
            early_stopping=True
        )
        answer = self.ralm_tokenizer.decode(output_tokens[0], skip_special_tokens=True)

        attributions = []
        if context_documents:
            attributions = list(set([doc["source"] for doc in context_documents]))
        
        return answer, attributions

    def process_query(self, query: str) -> tuple:
        # 1. Conditional Retrieval Check
        needs_retrieval = self.conditional_retrieval_predictor.predict(query)

        if needs_retrieval and self.chroma_collection:
            # 2. Initial Retrieval
            retrieved_docs = self._initial_retrieve(query, top_k=10)
            
            if retrieved_docs:
                # 3. Predictive Reranking
                reranked_docs = self._rerank_documents(query, retrieved_docs, top_n=3)
                # 4. In-Context RALM for Answer Generation
                answer, sources = self._generate_answer_with_ralm(query, reranked_docs)
                status = "Answer generated with external knowledge (RALM, Reranking, Retrieval)"
            else:
                # Fallback if retrieval yields no documents
                answer, sources = self._generate_answer_with_ralm(query, []) # Answer with just base LM
                status = "Answer generated by base LM (retrieval attempted but no relevant documents found)"
        else:
            # Conditional Retrieval bypasses or ChromaDB not available
            answer, sources = self._generate_answer_with_ralm(query, []) # Answer with just base LM
            status = "Answer generated by base LM (conditional retrieval decided no external knowledge needed or ChromaDB not active)"
            if not self.chroma_collection:
                 status = "Answer generated by base LM (ChromaDB not active)"

        return answer, sources, status


# Initialize the Medical Query Answering System
medical_system = MedicalQueryAnsweringSystem(
    embedding_model=embedding_model,
    chroma_collection=collection,
    conditional_retrieval_predictor=conditional_retrieval_predictor,
    reranker_model=reranker_model,
    ralm_tokenizer=ralm_tokenizer,
    ralm_model=ralm_model
)


# Gradio Interface
def medical_qa_interface(query: str) -> tuple:
    answer, sources, status = medical_system.process_query(query)
    
    source_str = ""
    if sources:
        source_str = "\n\n**Sources:**\n" + "\n".join([f"- {s}" for s in sources])
    
    full_response = f"**Status:** {status}\n\n**Answer:**\n{answer}{source_str}"
    
    return full_response


# Launch Gradio App
if __name__ == "__main__":
    if collection is None:
        print("Warning: ChromaDB collection is not active. The system will operate in base LM mode only for most queries.")
        print("To enable full functionality, ensure ChromaDB is properly initialized.")
    
    gr.Interface(
        fn=medical_qa_interface,
        inputs=gr.Textbox(lines=2, placeholder="Enter your medical query here..."),
        outputs=gr.Markdown(),
        title="Medical Query Answering System for Clinicians",
        description=(
            "This system answers medical queries using a Retrieval-Augmented Language Model (RALM). "
            "It incorporates Conditional Retrieval to decide if external knowledge is needed and "
            "Predictive Reranking to select the most relevant documents." 
            "Note: This is a demo with simulated data and models. Real-world performance requires extensive data and fine-tuning."
        )
    ).launch()
