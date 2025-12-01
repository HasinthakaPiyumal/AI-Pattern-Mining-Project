import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import gradio as gr

# Load environment variables from .env file
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHROMA_DB_PATH = "./chroma_db"
MEDICAL_KNOWLEDGE = [
    "Aspirin is commonly used for pain relief and to reduce fever. It also has anti-inflammatory properties and can be used to prevent blood clots.",
    "Diabetes mellitus is a chronic condition that affects how your body turns food into energy. Most of the food you eat is broken down into sugar (glucose) and released into your bloodstream. Your pancreas makes insulin, a hormone that acts like a key to let blood sugar into your body’s cells for use as energy.",
    "Hypertension, or high blood pressure, is a common condition in which the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease. Regular exercise, a healthy diet, and medication are common treatments.",
    "The common cold is a viral infection of your nose and throat (upper respiratory tract). It's usually harmless, although it might not feel that way. Many types of viruses can cause a common cold. Rhinoviroses are the most common.",
    "Antibiotics are medicines that fight bacterial infections in people and animals. They work by killing the bacteria or by making it difficult for the bacteria to grow and multiply. They are not effective against viral infections like the common cold or flu.",
    "Vaccines stimulate your immune system to produce antibodies, just as it would if you were exposed to the disease. After getting vaccinated, you develop immunity to that disease, without having to get the illness first.",
    "MRI (Magnetic Resonance Imaging) is a medical imaging technique used in radiology to form pictures of the anatomy and the physiological processes of the body. MRI scanners use strong magnetic fields, magnetic field gradients, and radio waves to generate images of the organs in the body.",
    "Chemotherapy is a drug treatment that uses powerful chemicals to kill fast-growing cells in your body. It's most often used to treat cancer, as cancer cells grow and multiply much faster than most other cells in the body.",
    "Physical therapy (PT) is a type of treatment that helps individuals improve or restore physical function and fitness. It can be used for rehabilitation after injury, prevention of further injury, or to manage chronic conditions."
]

# 1. Data Ingestion & Knowledge Base Management
# Simulate loading documents - in a real app, this would load from files/APIs
# For simplicity, we'll create a list of Document objects from MEDICAL_KNOWLEDGE
from langchain_core.documents import Document
documents = [Document(page_content=text, metadata={"source": "medical_wiki"}) for text in MEDICAL_KNOWLEDGE]

# Text Splitting
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
splits = text_splitter.split_documents(documents)

# Embedding Model
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Vector Database (Chroma)
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=CHROMA_DB_PATH)
vectorstore.persist()

# 2. Retrieval Mechanism
retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 relevant chunks

# 3. Large Language Model (LLM) Integration
llm = ChatOpenAI(model_name="gpt-3.5-turbo", api_key=OPENAI_API_KEY, temperature=0.0)

# Prompt Engineering
# A function to format the retrieved documents into a string for the prompt
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful medical assistant. Use the following pieces of context to answer the user's question accurately and thoroughly. If you don't know the answer based on the provided context, politely state that you don't have enough information."),
    ("human", "Context: {context}\nQuestion: {question}"),
])

# 4. Orchestration & RAG Chain
rag_chain = (
    {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 5. User Interface (Gradio)
def ask_medical_assistant(question: str) -> str:
    if not question.strip():
        return "Please enter a medical question."
    try:
        response = rag_chain.invoke(question)
        return response
    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == "__main__":
    print("Initializing Medical Information Assistant...")
    print(f"ChromaDB will persist at: {CHROMA_DB_PATH}")

    # Gradio interface setup
    if OPENAI_API_KEY:
        if os.path.exists(CHROMA_DB_PATH):
            print("Existing ChromaDB found. Using it.")
        else:
            print("ChromaDB not found. Creating and persisting initial knowledge base.")

        interface = gr.Interface(
            fn=ask_medical_assistant,
            inputs=gr.Textbox(lines=2, placeholder="Ask a medical question here..."),
            outputs="text",
            title="Medical Information Assistant (RAG)",
            description="Ask questions about medical conditions, treatments, and drugs. This assistant uses Retrieval Augmented Generation (RAG) to provide grounded answers from a medical knowledge base.",
            examples=[
                ["What is hypertension?"],
                ["How does aspirin work?"],
                ["Are antibiotics effective against the common cold?"],
                ["What is the purpose of chemotherapy?"],
                ["Tell me about MRI."]
            ]
        )
        interface.launch(share=False)
    else:
        print("Error: OPENAI_API_KEY not found. Please set it in your .env file.")
        print("Example .env content: OPENAI_API_KEY='your_openai_api_key_here'")
