
import os
import functools
from typing import List, Dict, Any

import gradio as gr
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.documents import Document

# --- 1. Data Ingestion and Knowledge Base Management ---

# Simulated Medical Documents
medical_documents_raw = [
    "Aspirin is commonly used for pain relief and to reduce fever. It also has anti-inflammatory properties and can be used to prevent blood clots. Common side effects include stomach upset and increased bleeding risk.",
    "Diabetes Mellitus is a chronic condition that affects how your body turns food into energy. There are two main types: Type 1 (insulin-dependent) and Type 2 (insulin resistance). Management often involves diet, exercise, and medication like insulin or metformin.",
    "Hypertension, or high blood pressure, is a common condition where the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease. Lifestyle changes and medications (e.g., ACE inhibitors, diuretics) are primary treatments.",
    "The common cold is a viral infection of your nose and throat (upper respiratory tract). It's usually harmless, although it might not feel that way. Symptoms include a runny nose, sore throat, cough, congestion, and sometimes body aches or a mild headache. Antibiotics are not effective against viral infections.",
    "COVID-19 is an infectious disease caused by the SARS-CoV-2 virus. Most people infected with the virus will experience mild to moderate respiratory illness and recover without requiring special treatment. However, some will become seriously ill and require medical attention. Vaccination is a key preventive measure.",
    "Asthma is a chronic lung disease that inflames and narrows the airways. Asthma causes recurring periods of wheezing (a whistling sound when you breathe), chest tightness, shortness of breath, and coughing. Triggers can include allergens, exercise, or cold air. Inhalers are often used for quick relief and long-term control.",
    "Migraine is a type of headache characterized by recurrent attacks of moderate to severe throbbing pain, usually on one side of the head, often accompanied by nausea, vomiting, and sensitivity to light and sound. Treatment options include pain relievers and preventive medications.",
    "Influenza (flu) is a contagious respiratory illness caused by influenza viruses. It can cause mild to severe illness, and at times can lead to death. The flu is different from a cold. The flu usually comes on suddenly. People who have the flu often feel some or all of these symptoms: fever, cough, sore throat, muscle aches, headaches, and fatigue."
]

CHROMA_DB_DIR = "./chroma_db"

def load_and_split_docs(texts: List[str]) -> List[Document]:
    """Loads and splits raw text documents into smaller chunks."""
    documents = [Document(page_content=text) for text in texts]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_documents(documents)

def create_vector_store(documents: List[Document]) -> Chroma:
    """Creates or loads a Chroma vector store with medical document embeddings."""
    print("Initializing embedding model...")
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    if os.path.exists(CHROMA_DB_DIR) and os.listdir(CHROMA_DB_DIR):
        print(f"Loading existing Chroma DB from {CHROMA_DB_DIR}")
        vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
    else:
        print(f"Creating new Chroma DB at {CHROMA_DB_DIR}")
        vectorstore = Chroma.from_documents(documents=documents, embedding_function=embeddings, persist_directory=CHROMA_DB_DIR)
        vectorstore.persist()
        print("Chroma DB created and persisted.")
    return vectorstore

# --- 2. Adaptive Retrieval Module ---

class ContextAssessor:
    """Assesses query complexity to guide dynamic retrieval."""
    def assess_query_complexity(self, query: str) -> str:
        """Determines if a query is simple or complex based on heuristics."""
        query_words = query.lower().split()
        # Simple heuristic: longer queries or those with specific keywords are complex
        complex_keywords = ["mechanism", "differentiate", "treatment plan", "interaction", "prognosis"]
        
        if len(query_words) > 7 or any(keyword in query for keyword in complex_keywords):
            return "complex"
        else:
            return "simple"

class DynamicRetriever:
    """Adapts retrieval strategy based on query complexity."""
    def __init__(self, vectorstore: Chroma, assessor: ContextAssessor):
        self.vectorstore = vectorstore
        self.assessor = assessor

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Retrieves documents with an adaptative 'k' parameter."""
        complexity = self.assessor.assess_query_complexity(query)
        
        if complexity == "complex":
            print(f"Assessed query as '{complexity}', retrieving more documents (k=5).")
            return self.vectorstore.as_retriever(search_kwargs={"k": 5}).invoke(query)
        else:
            print(f"Assessed query as '{complexity}', retrieving fewer documents (k=2).")
            return self.vectorstore.as_retriever(search_kwargs={"k": 2}).invoke(query)

# --- 3. Retrieval-Augmented Language Model (RALM) Core ---

class FakeLLM:
    """A placeholder LLM that simulates generating responses."""
    def invoke(self, messages: List[Any]) -> AIMessage:
        # Extract the human message (query) and context from the Langchain messages format
        query = ""
        context_str = ""
        for msg in messages:
            if isinstance(msg, HumanMessage):
                query = msg.content
            elif isinstance(msg, SystemMessage) and "Context:" in msg.content:
                context_str = msg.content.replace("You are a helpful medical assistant. Context:", "").strip()

        if context_str:
            response_content = f"Based on the provided medical information:\n{context_str}\n\nRegarding your query about '{query}', I can provide the following insights. (This is a simulated response based on the retrieved context.)"
        else:
            response_content = f"I am a simulated medical assistant. For your query about '{query}', I don't have specific context, but I can tell you that this is a placeholder response."
        
        return AIMessage(content=response_content)

def format_docs(docs: List[Document]) -> str:
    """Formats retrieved documents into a single string."""
    return "\n\n".join([doc.page_content for doc in docs])

def build_rag_chain(retriever: DynamicRetriever, llm: FakeLLM):
    """Builds the RAG chain using Langchain LCEL."""
    # Create a simple prompt template for the LLM
    template = """You are a helpful medical assistant. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Context: {context}
    Question: {question}
    Answer:"""
    
    # Using ChatPromptTemplate for compatibility with FakeLLM's 'messages' input
    chat_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful medical assistant. Context: {context}"),
        ("human", "{question}")
    ])

    rag_chain = (
        {"context": retriever.get_relevant_documents | RunnableLambda(format_docs), "question": RunnablePassthrough()}
        | chat_template
        | llm
        | StrOutputParser()
    )
    return rag_chain

# --- 4. Inference Optimization & Caching ---
# The caching is applied directly to the main query function.

# --- 5. User Interface (Gradio) ---

# Global instances for the application
splitted_docs = load_and_split_docs(medical_documents_raw)
vectorstore_instance = create_vector_store(splitted_docs)
context_assessor_instance = ContextAssessor()
dynamic_retriever_instance = DynamicRetriever(vectorstore_instance, context_assessor_instance)
fake_llm_instance = FakeLLM()
rag_chain_instance = build_rag_chain(dynamic_retriever_instance, fake_llm_instance)

@functools.lru_cache(maxsize=128)
def query_smart_medical_assistant(query: str) -> str:
    """Processes a medical query using the RALM and returns the response."""
    print(f"\nProcessing query: '{query}'")
    try:
        # The RAG chain handles retrieval and LLM interaction
        response = rag_chain_instance.invoke(query)
        return response
    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == "__main__":
    print("\n--- Smart Medical Assistant Initialized ---\n")
    
    # Gradio interface setup
    iface = gr.Interface(
        fn=query_smart_medical_assistant,
        inputs=gr.Textbox(lines=2, placeholder="Ask a medical question here..."),
        outputs=gr.Textbox(),
        title="Smart Medical Assistant (RALM Demo)",
        description="Ask any medical question and get a simulated, context-aware response based on retrieved knowledge. This demo features adaptive retrieval and response caching."
    )

    # Launch the Gradio app
    print("\nLaunching Gradio interface...\n")
    iface.launch()
    print("\nGradio interface closed.\n")

