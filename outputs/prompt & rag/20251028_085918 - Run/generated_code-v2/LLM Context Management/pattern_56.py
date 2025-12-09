import os
import gradio as gr
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from transformers import pipeline

# --- Configuration Constants ---
MEDICAL_DOCS_DIR = "medical_docs"
FAISS_INDEX_PATH = "faiss_medical_index"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# --- Global Variables for RAG components ---
vectorstore = None
rag_chain = None
llm_pipeline = None

# --- Knowledge Base Management ---
def build_and_save_vector_store():
    global vectorstore

    if not os.path.exists(MEDICAL_DOCS_DIR):
        os.makedirs(MEDICAL_DOCS_DIR)
        print(f"Created directory: {MEDICAL_DOCS_DIR}. Please add .txt medical documents here.")
        return None

    documents = []
    for filename in os.listdir(MEDICAL_DOCS_DIR):
        if filename.endswith(".txt"):
            file_path = os.path.join(MEDICAL_DOCS_DIR, filename)
            try:
                loader = TextLoader(file_path)
                loaded_docs = loader.load()
                for doc in loaded_docs:
                    doc.metadata["source"] = filename  # Add source filename to metadata
                documents.extend(loaded_docs)
            except Exception as e:
                print(f"Error loading {filename}: {e}")

    if not documents:
        print("No medical documents found. Please add .txt files to medical_docs/.")
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(FAISS_INDEX_PATH)
    print(f"FAISS index built and saved to {FAISS_INDEX_PATH}")
    return vectorstore

def load_vector_store():
    global vectorstore
    if vectorstore is not None:
        return vectorstore # Already loaded

    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    if os.path.exists(FAISS_INDEX_PATH):
        try:
            vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
            print(f"FAISS index loaded from {FAISS_INDEX_PATH}")
        except Exception as e:
            print(f"Error loading FAISS index: {e}. Rebuilding index.")
            vectorstore = build_and_save_vector_store()
    else:
        print("FAISS index not found. Building new index.")
        vectorstore = build_and_save_vector_store()
    return vectorstore

# --- LLM Setup ---
def load_llm_pipeline():
    global llm_pipeline
    if llm_pipeline is None:
        # Using a smaller model for demonstration; replace with a larger model like
        # 'tiiuae/falcon-7b-instruct' or 'HuggingFaceH4/zephyr-7b-beta' for better performance
        # Ensure you have sufficient RAM/GPU for larger models.
        try:
            llm_pipeline = pipeline(
                "text2text-generation",
                model="google/flan-t5-small",
                max_new_tokens=256
            )
            print("LLM pipeline (Flan-T5-Small) loaded.")
        except Exception as e:
            print(f"Error loading LLM (Flan-T5-Small): {e}. Please ensure the model is downloaded or try again.")
            llm_pipeline = None # Set to None if loading fails
    return llm_pipeline

# LangChain custom LLM wrapper for Hugging Face pipeline
class CustomHFText2TextLLM:
    def __init__(self, pipeline_instance):
        self.pipeline = pipeline_instance

    def invoke(self, prompt_value):
        response = self.pipeline(prompt_value.messages[-1].content)
        return response[0]["generated_text"]

# --- RAG Chain Setup ---
def setup_rag_chain():
    global rag_chain, vectorstore, llm_pipeline
    vectorstore = load_vector_store()
    llm_pipeline = load_llm_pipeline()

    if vectorstore is None or llm_pipeline is None:
        print("Vector store or LLM not initialized. RAG chain cannot be set up.")
        return None

    retriever = vectorstore.as_retriever()
    custom_llm = CustomHFText2TextLLM(llm_pipeline)

    template = """You are a medical assistant. Use the following pieces of context to answer the question. If you don't know the answer, just say that you don't know, don't try to make up an answer. Provide source filenames for retrieved facts.

Context: {context}

Question: {question}

Answer:"""
    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
        RunnableParallel({
            "context": lambda x: "\n\n".join([f"Source: {doc.metadata['source']}\n{doc.page_content}" for doc in retriever.invoke(x["question"])]) if x.get("question") else "No context provided",
            "question": RunnablePassthrough()
        })
        | prompt
        | custom_llm
        | StrOutputParser()
    )
    print("RAG chain set up.")
    return rag_chain

# --- Gradio Interface Functions ---
def answer_query(query):
    global rag_chain
    if rag_chain is None:
        return "System not fully initialized. Please try refreshing the knowledge base first or check for errors.", ""
    
    try:
        response = rag_chain.invoke({"question": query})
        
        # The RAG chain's 'context' part is not directly returned by invoke().
        # To display sources, we need to re-retrieve or modify the chain to return context.
        # For simplicity in this demo, let's assume the LLM incorporates source names.
        # A more robust solution would involve modifying the RAG chain to return sources explicitly.

        # For now, let's just make a placeholder for sources
        retrieved_docs = vectorstore.as_retriever().invoke(query) if vectorstore else []
        sources = "\n\nRetrieved Sources:\n" + "\n---\n".join([f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content[:200]}..." for doc in retrieved_docs])

        return response, sources
    except Exception as e:
        return f"An error occurred during query processing: {e}", ""

def refresh_knowledge_base_ui():
    global vectorstore, rag_chain
    print("Refreshing knowledge base...")
    vectorstore = build_and_save_vector_store()
    rag_chain = setup_rag_chain() # Re-setup RAG chain with new vector store
    if vectorstore and rag_chain:
        return "Knowledge base refreshed successfully!", ""
    else:
        return "Failed to refresh knowledge base. Check logs for errors.", ""

# --- Gradio UI Layout ---
with gr.Blocks() as demo:
    gr.Markdown("# Medical Knowledge and Treatment Recommendation System")
    gr.Markdown("This system provides evidence-based medical recommendations from a human-readable knowledge base.")

    with gr.Row():
        query_input = gr.Textbox(label="Enter your medical query:", placeholder="e.g., What are the treatment options for Type 2 Diabetes?")
    with gr.Row():
        submit_button = gr.Button("Get Recommendation")
        refresh_button = gr.Button("Refresh Knowledge Base")
    
    with gr.Row():
        output_recommendation = gr.Textbox(label="Recommendation:", interactive=False, lines=10)
    with gr.Row():
        output_sources = gr.Textbox(label="Retrieved Medical Sources:", interactive=False, lines=10)

    submit_button.click(fn=answer_query, inputs=query_input, outputs=[output_recommendation, output_sources])
    refresh_button.click(fn=refresh_knowledge_base_ui, inputs=[], outputs=[output_recommendation, output_sources])

    # Initialize vector store and RAG chain on startup
    demo.load(setup_rag_chain, inputs=[], outputs=[])

# --- Main Execution --- 
if __name__ == "__main__":
    print("Starting Medical Recommendation System...")
    print(f"Ensure medical documents are placed in the '{MEDICAL_DOCS_DIR}' directory.")
    print(f"Initial FAISS index will be saved to '{FAISS_INDEX_PATH}'.")
    demo.launch(share=False) # Set share=True for a public link (might take a while to generate))
