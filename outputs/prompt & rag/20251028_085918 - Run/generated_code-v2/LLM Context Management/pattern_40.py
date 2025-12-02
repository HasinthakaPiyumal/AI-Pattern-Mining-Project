import os
import PyPDF2
from bs4 import BeautifulSoup
import chromadb
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline, set_seed
import gradio as gr
import uuid

CHROMA_DB_PATH = "./chroma_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "distilgpt2"

client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
try:
    collection = client.get_or_create_collection(name="medical_knowledge_base")
except Exception:
    client = chromadb.Client()
    collection = client.get_or_create_collection(name="medical_knowledge_base")

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

try:
    set_seed(42)
    text_generator = pipeline("text-generation", model=LLM_MODEL_NAME)
except Exception:
    text_generator = None

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)

def _get_text_from_pdf(file_path):
    text = ""
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def _get_text_from_html(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")
        return soup.get_text()

def _get_text_from_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

def _clean_text(text):
    text = text.replace("\n", " ").replace("\t", " ")
    text = " ".join(text.split())
    return text

def _process_document(file_path, document_type, source_name="unknown"):
    raw_text = ""
    if document_type == "pdf":
        raw_text = _get_text_from_pdf(file_path)
    elif document_type == "html":
        raw_text = _get_text_from_html(file_path)
    elif document_type == "txt":
        raw_text = _get_text_from_txt(file_path)
    else:
        return [], [], []

    cleaned_text = _clean_text(raw_text)
    chunks = text_splitter.split_text(cleaned_text)

    documents = []
    metadatas = []
    ids = []
    for i, chunk in enumerate(chunks):
        documents.append(chunk)
        metadatas.append({"source": source_name, "document_type": document_type, "chunk_id": i})
        ids.append(str(uuid.uuid4()))
    return documents, metadatas, ids

def update_knowledge_base(file_obj, document_type, source_name):
    if file_obj is None:
        return "Please upload a file."

    file_path = file_obj.name
    try:
        documents, metadatas, ids = _process_document(file_path, document_type, source_name)
        if documents:
            embeddings = embedding_model.encode(documents).tolist()
            collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            return f"Successfully added {len(documents)} chunks from {source_name} to the knowledge base."
        else:
            return "No text extracted or chunks generated. Please check the document type and content."
    except Exception as e:
        return f"Error updating knowledge base: {e}"

def ask_medical_assistant(query):
    query_embedding = embedding_model.encode([query]).tolist()
    
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=5,
        include=["documents", "metadatas"]
    )

    retrieved_chunks = results["documents"][0]
    retrieved_metadatas = results["metadatas"][0]

    context = "\n".join(retrieved_chunks)
    
    if text_generator:
        prompt = f"Based on the following medical information, answer the question:\n\nInformation:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        
        max_input_length = text_generator.model.config.max_position_embeddings
        if len(prompt) > max_input_length:
            truncated_context_len = max_input_length - (len(prompt) - len(context))
            truncated_context = context[:truncated_context_len]
            prompt = f"Based on the following medical information, answer the question:\n\nInformation:\n{truncated_context}\n\nQuestion: {query}\n\nAnswer:"

        try:
            generated_response = text_generator(prompt, max_new_tokens=200, num_return_sequences=1, truncation=True)[0]["generated_text"]
            answer = generated_response.replace(prompt, "").strip()
        except Exception as e:
            answer = f"Error generating LLM response: {e}. Raw context: {context}"
    else:
        answer = f"LLM not initialized. Retrieved context: {context}"

    sources = [f"Source: {meta["source"]}, Type: {meta["document_type"]}" for meta in retrieved_metadatas]
    
    full_response = f"**Answer:** {answer}\n\n**Sources:**\n" + "\n".join(sources)
    return full_response

with gr.Blocks() as demo:
    gr.Markdown("# Medical Knowledge Assistant")
    
    with gr.Tab("Ask the Assistant"):
        query_input = gr.Textbox(label="Your Medical Query")
        answer_output = gr.Markdown(label="Assistant's Response and Sources")
        ask_button = gr.Button("Get Answer")
        ask_button.click(ask_medical_assistant, inputs=query_input, outputs=answer_output)
        
    with gr.Tab("Update Knowledge Base"):
        file_input = gr.File(label="Upload Document (PDF, HTML, or TXT)")
        doc_type_input = gr.Radio(["pdf", "html", "txt"], label="Document Type")
        source_name_input = gr.Textbox(label="Source Name (e.g., 'AHA Guidelines 2023', 'New England Journal of Medicine Article')")
        update_output = gr.Markdown(label="Update Status")
        update_button = gr.Button("Update Knowledge Base")
        update_button.click(update_knowledge_base, inputs=[file_input, doc_type_input, source_name_input], outputs=update_output)

demo.launch()