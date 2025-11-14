
import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- 1. Simulate Medical Documents (Data Ingestion and Knowledge Base) ---
# In a real application, these would come from actual files or databases.
medical_documents = [
    "Clinical guideline for Type 2 Diabetes management: Metformin is the first-line treatment. Lifestyle modifications are crucial.",
    "Research paper on Hypertension: ACE inhibitors are effective for blood pressure control. Regular exercise and low sodium diet are recommended.",
    "Patient case study on Asthma: Bronchodilators for acute attacks, inhaled corticosteroids for long-term control. Avoid triggers like pollen.",
    "Latest research on Alzheimer's Disease: New drug candidates show promise in early-stage trials by targeting amyloid plaques.",
    "Pediatric vaccination schedule: Diphtheria, Tetanus, Pertussis (DTaP) at 2, 4, 6 months. MMR at 12-15 months.",
    "Management of Chronic Kidney Disease: Focus on blood pressure control, glycemic control, and dietary protein restriction.",
    "Emergency protocol for Anaphylaxis: Administer epinephrine immediately. Follow with antihistamines and corticosteroids if needed."
]

# Save simulated documents to temporary files for TextLoader
for i, doc_content in enumerate(medical_documents):
    with open(f"doc_{i}.txt", "w") as f:
        f.write(doc_content)

# Load documents
loaders = [TextLoader(f"doc_{i}.txt") for i in range(len(medical_documents))]
docs = []
for loader in loaders:
    docs.extend(loader.load())

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)

# Generate embeddings
# Using a local model, ensure 'sentence-transformers' is installed
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create Chroma vector store
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
vectorstore.persist()

# --- 2. Intelligent Retrieval Module ---
retriever = vectorstore.as_retriever()

# --- 3. Context Conditioning and LLM Integration ---
# Set up OpenAI API key (replace with your actual key or use environment variable)
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0) # Consider more powerful models for production

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a highly accurate medical diagnostic and treatment recommendation assistant. Your task is to provide concise, evidence-based responses to medical queries. ONLY use the provided context to answer the question. If the context does not contain enough information, state that you cannot provide a definitive answer and recommend consulting a human medical professional."),
    ("human", "Context: {context}\n\nQuestion: {input}")
])

# This chain combines the retrieved documents with the prompt and sends to the LLM
document_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, document_chain)

# --- 6. System-Level Optimizations (Basic Caching) ---
cached_responses = {}

# --- Main Function for Medical RAG System ---
def get_medical_recommendation(query: str):
    """
    Provides medical diagnostic and treatment recommendations based on retrieved information.
    Includes adaptive decision-making for uncertainty.
    """
    if query in cached_responses:
        print("\n[INFO] Response retrieved from cache.")
        return cached_responses[query]

    print(f"\n[INFO] Processing query: {query}")

    # Retrieve relevant documents
    retrieved_docs = retriever.invoke(query)

    # --- 5. Adaptive Decision-Making Layer (Simplified) ---
    # If no or very few relevant documents, suggest human specialist
    if not retrieved_docs or len(retrieved_docs) < 2: # Threshold can be adjusted
        print("[WARNING] Insufficient relevant information found. Recommending human consultation.")
        return {
            "query": query,
            "recommendation": "I cannot provide a definitive answer based on the available information. Please consult a human medical professional for accurate diagnosis and treatment.",
            "source_documents": []
        }

    # Generate response using RAG chain
    response = rag_chain.invoke({"input": query})

    # Extract relevant parts from the response
    llm_output = response["answer"]
    sources = [doc.metadata.get("source", "N/A") for doc in response["context"]]

    final_recommendation = {
        "query": query,
        "recommendation": llm_output,
        "source_documents": list(set(sources)) # Unique sources
    }
    
    # Cache the response
    cached_responses[query] = final_recommendation

    return final_recommendation

# --- Example Usage ---
if __name__ == "__main__":
    print("Initializing Medical RAG System...")
    
    # Example Queries
    query1 = "What is the recommended first-line treatment for Type 2 Diabetes?"
    query2 = "How to manage severe allergic reactions?"
    query3 = "Latest advancements in cancer treatment?" # Expecting less direct answer or human consult
    query4 = "What are the common vaccinations for infants?"
    query5 = "What are the dietary restrictions for Chronic Kidney Disease?"
    query6 = "How to cure common cold?" # Expecting human consult due to limited context

    # Get recommendations
    rec1 = get_medical_recommendation(query1)
    print("\nRecommendation 1:", rec1["recommendation"])
    print("Sources:", rec1["source_documents"])
    
    rec2 = get_medical_recommendation(query2)
    print("\nRecommendation 2:", rec2["recommendation"])
    print("Sources:", rec2["source_documents"])

    rec3 = get_medical_recommendation(query3)
    print("\nRecommendation 3:", rec3["recommendation"])
    print("Sources:", rec3["source_documents"])

    rec4 = get_medical_recommendation(query4)
    print("\nRecommendation 4:", rec4["recommendation"])
    print("Sources:", rec4["source_documents"])

    rec5 = get_medical_recommendation(query5)
    print("\nRecommendation 5:", rec5["recommendation"])
    print("Sources:", rec5["source_documents"])

    rec6 = get_medical_recommendation(query6) # This should trigger the adaptive decision-making
    print("\nRecommendation 6:", rec6["recommendation"])
    print("Sources:", rec6["source_documents"])

    # Test caching
    rec1_cached = get_medical_recommendation(query1)
    print("\nRecommendation 1 (cached test):", rec1_cached["recommendation"])
    
    # Clean up temporary files
    for i in range(len(medical_documents)):
        os.remove(f"doc_{i}.txt")
    print("\nCleaned up temporary document files.")


