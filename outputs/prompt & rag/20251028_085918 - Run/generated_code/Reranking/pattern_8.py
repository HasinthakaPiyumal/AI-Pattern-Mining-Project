
import os
from typing import List

# Langchain components
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_core.documents import Document

# Gradio for UI
import gradio as gr

# --- Configuration --- 
# Set your OpenAI API key as an environment variable or replace 'os.getenv("OPENAI_API_KEY")'
# For example: os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set. Please set it to use OpenAI models.")

# --- 1. Data Ingestion and Embedding Module --- 
# Dummy medical documents for demonstration
dummy_medical_docs = [
    "Aspirin is commonly used as an analgesic to relieve minor aches and pains, as an antipyretic to reduce fever, and as an anti-inflammatory medication. It can also be used as an antiplatelet to prevent blood clots. Common side effects include gastrointestinal upset and increased bleeding risk.",
    "Type 2 diabetes mellitus is a chronic metabolic disorder characterized by high blood sugar levels due to insulin resistance or insufficient insulin production. Management often involves lifestyle changes, oral medications like metformin, and sometimes insulin therapy. Regular monitoring of blood glucose is crucial.",
    "Hypertension, or high blood pressure, is a condition in which the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease. Treatments include diuretics, ACE inhibitors, beta-blockers, and lifestyle modifications such as diet and exercise.",
    "The COVID-19 pandemic is caused by the SARS-CoV-2 virus. Symptoms range from mild to severe and can include fever, cough, fatigue, and loss of taste or smell. Vaccination is a key preventative measure, alongside mask-wearing and social distancing. Antiviral treatments are available for severe cases.",
    "Migraine is a severe headache accompanied by symptoms such as throbbing pain, sensitivity to light and sound, and nausea. Triggers can include stress, certain foods, and hormonal changes. Treatments range from over-the-counter pain relievers to prescription triptans and CGRP inhibitors. Preventative medications are also available."
]

# Create a temporary file to simulate document loading
with open("medical_docs.txt", "w") as f:
    for doc in dummy_medical_docs:
        f.write(doc + "\n---\n") # Separate documents with a delimiter

# Load documents
loader = TextLoader("medical_docs.txt")
raw_documents = loader.load()

# Split documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50, separators=["\n---\n", "\n\n", ".", " "]
)

documents = text_splitter.split_documents(raw_documents)

# Initialize embeddings
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize Chroma vector store and add documents
# In a real application, you might persist this to disk
vectorstore = Chroma.from_documents(documents, embeddings)

print("Data Ingestion and Embedding complete. Vectorstore initialized.")

# --- 2. Conditional Retrieval Module (Placeholder) --- 
def conditional_retrieval_needed(query: str) -> bool:
    """
    Placeholder function to determine if external document retrieval is necessary.
    In a real system, this would be a trained model.
    For now, it always returns True, meaning retrieval is always performed.
    """
    print(f"Conditional Retrieval: Always retrieving for query: '{query}'")
    return True

# --- 3. Document Retrieval Module --- 
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# --- 4. Zero-Shot LM Reranking Module (Placeholder) --- 
def zero_shot_lm_rerank(documents: List[Document], query: str) -> List[Document]:
    """
    Placeholder for Zero-Shot LM Reranking.
    In a real system, an LM would be used to re-rank documents.
    For now, it returns documents as is.
    """
    print(f"Zero-Shot Reranking: Skipping for query: '{query}' - returning original order.")
    # In a real implementation, you'd use an LM to score/rank documents.
    # Example (conceptual):
    # scores = []
    # for doc in documents:
    #     prompt = f"Given the query: '{query}', how relevant is this document: '{doc.page_content}'? Score from 0 to 10." 
    #     lm_response = llm.invoke(prompt) # Assuming 'llm' is available
    #     score = parse_score_from_lm_response(lm_response)
    #     scores.append((score, doc))
    # return [doc for score, doc in sorted(scores, key=lambda x: x[0], reverse=True)]
    return documents

# --- 5. Predictive Reranking Module (Placeholder) --- 
def predictive_rerank(documents: List[Document], query: str) -> List[Document]:
    """
    Placeholder for Predictive Reranking (Trained LM-Dedicated Reranker).
    In a real system, a specialized trained model would re-rank documents.
    For now, it returns documents as is.
    """
    print(f"Predictive Reranking: Skipping for query: '{query}' - returning original order.")
    # In a real implementation, you'd load and use a trained model for reranking.
    # Example (conceptual):
    # trained_reranker_model.predict(query, documents)
    return documents

# --- 6. InContext Retrieval-Augmented Language Modeling (RALM) Module --- 
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.1, openai_api_key=OPENAI_API_KEY)

# Create a retrieval QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# --- Main Inquiry Function --- 
def medical_inquiry(query: str) -> str:
    """
    Processes a medical inquiry using the RALM system.
    """
    response_text = ""
    sources_text = ""

    if conditional_retrieval_needed(query):
        # Retrieve documents
        initial_docs = retriever.invoke(query)
        print(f"Initial retrieved documents (first 200 chars): {[doc.page_content[:200] for doc in initial_docs]}")

        # Apply Zero-Shot LM Reranking (placeholder)
        reranked_docs_zero_shot = zero_shot_lm_rerank(initial_docs, query)

        # Apply Predictive Reranking (placeholder)
        final_reranked_docs = predictive_rerank(reranked_docs_zero_shot, query)
        print(f"Final reranked documents (first 200 chars): {[doc.page_content[:200] for doc in final_reranked_docs]}")

        # Prepare context for RALM
        # Langchain's RetrievalQA chain automatically handles passing documents to the LLM
        # We can simulate the 'prepended' part conceptually by knowing how QA chain works.

        result = qa_chain.invoke({"query": query})
        response_text = result["result"]
        source_documents = result["source_documents"]

        if source_documents:
            sources_text = "\n\nSources:\n"
            for i, doc in enumerate(source_documents):
                sources_text += f"[{i+1}] {doc.page_content[:200]}...\n"
        else:
            sources_text = "\n\nNo specific sources found for this query."

    else:
        # If no retrieval is needed (LM answers from its own knowledge)
        response_text = llm.invoke(query).content
        sources_text = "\n\nAnswer generated from LM's parametric knowledge (no external retrieval needed)."
    
    return response_text + sources_text

# --- 7. User Interface (UI) Module with Gradio --- 
if __name__ == "__main__":
    print("Starting Gradio UI...")
    if not OPENAI_API_KEY:
        print("WARNING: OPENAI_API_KEY is not set. The LLM will not function.")
        print("Please set the OPENAI_API_KEY environment variable.")

    demo = gr.Interface(
        fn=medical_inquiry,
        inputs=gr.Textbox(lines=2, placeholder="Enter your medical question here..."),
        outputs="text",
        title="Medical Inquiry Assistant for Healthcare Professionals",
        description="Get accurate, attributed answers to medical questions by leveraging retrieval-augmented language modeling."
    )
    demo.launch()

    # Clean up dummy file
    if os.path.exists("medical_docs.txt"):
        os.remove("medical_docs.txt")
    print("Gradio UI stopped. Cleaned up temporary files.")
