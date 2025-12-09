import os
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import gradio as gr

load_dotenv()

MEDICAL_DOCS_DIR = "./medical_docs"
CHROMA_DB_DIR = "./chroma_db"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def setup_knowledge_base():
    if not os.path.exists(MEDICAL_DOCS_DIR):
        os.makedirs(MEDICAL_DOCS_DIR)
        with open(os.path.join(MEDICAL_DOCS_DIR, "example_medical_guideline.txt"), "w") as f:
            f.write("""Example Medical Guideline for Hypertension:\n\n1. Diagnosis: Blood pressure consistently >= 140/90 mmHg.\n2. Treatment: Lifestyle modifications (diet, exercise) are first-line. If BP remains high, consider ACE inhibitors, ARBs, calcium channel blockers, or thiazide diuretics.\n3. Monitoring: Regular blood pressure checks and kidney function tests.\n\nExample Drug Information - Lisinopril (ACE Inhibitor):\n\nUses: Hypertension, heart failure, post-MI. \nSide Effects: Cough, dizziness, fatigue, hyperkalemia. \nContraindications: Pregnancy, angioedema history. \nDrug Interactions: Potassium-sparing diuretics, NSAIDs. \n""")
        with open(os.path.join(MEDICAL_DOCS_DIR, "diabetes_overview.txt"), "w") as f:
            f.write("""Diabetes Mellitus Overview:\n\nType 1 Diabetes: Autoimmune destruction of pancreatic beta cells, leading to absolute insulin deficiency. Requires insulin therapy.\nType 2 Diabetes: Insulin resistance and progressive insulin secretory defect. Managed with diet, exercise, oral medications, and sometimes insulin.\nGestational Diabetes: Glucose intolerance first recognized during pregnancy.\n\nCommon Symptoms: Polydipsia (increased thirst), polyuria (frequent urination), polyphagia (increased hunger), unexplained weight loss, fatigue.\nComplications: Nephropathy, retinopathy, neuropathy, cardiovascular disease.\n""")
    
    print(f"Loading documents from {MEDICAL_DOCS_DIR}...")
    loader_txt = DirectoryLoader(MEDICAL_DOCS_DIR, glob="**/*.txt", loader_cls=TextLoader)
    loader_pdf = DirectoryLoader(MEDICAL_DOCS_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
    
    docs = []
    try:
        docs.extend(loader_txt.load())
    except Exception as e:
        print(f"Error loading TXT documents: {e}")
    try:
        docs.extend(loader_pdf.load())
    except Exception as e:
        print(f"Error loading PDF documents: {e}")
    
    if not docs:
        print("No documents loaded. Please place .txt or .pdf files in the medical_docs directory.")
        return None

    print(f"Loaded {len(docs)} documents. Splitting into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    print(f"Creating embeddings and storing in Chroma DB at {CHROMA_DB_DIR}...")
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=CHROMA_DB_DIR)
    vectorstore.persist()
    print("Knowledge base setup complete.")
    return vectorstore

def create_rag_chain(vectorstore):
    if not vectorstore:
        return None

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY)
    
    prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
Cite the source documents by their filename.

{context}

Question: {question}
Helpful Answer:"""
    custom_prompt = PromptTemplate.from_template(prompt_template)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": custom_prompt},
        return_source_documents=True
    )
    print("RAG chain created.")
    return qa_chain

def query_knowledge_navigator(query, qa_chain):
    if not qa_chain:
        return "Error: Knowledge base not initialized. Please check the logs."

    try:
        result = qa_chain({"query": query})
        answer = result["result"]
        source_documents = result["source_documents"]

        sources_text = "\n\nSources:\n"
        if source_documents:
            unique_sources = set()
            for doc in source_documents:
                if doc.metadata and 'source' in doc.metadata:
                    unique_sources.add(os.path.basename(doc.metadata['source']))
            if unique_sources:
                sources_text += "\n".join(sorted(list(unique_sources)))
            else:
                sources_text += "No specific source filenames found."
        else:
            sources_text += "No source documents retrieved."
        
        return answer + sources_text
    except Exception as e:
        return f"An error occurred during query: {e}"

def main():
    if not OPENAI_API_KEY:
        print("OPENAI_API_KEY not found. Please set it in a .env file or as an environment variable.")
        return

    vectorstore = setup_knowledge_base()
    qa_chain = create_rag_chain(vectorstore)

    if not qa_chain:
        print("Failed to initialize RAG chain. Exiting.")
        return

    def chatbot_interface(query):
        return query_knowledge_navigator(query, qa_chain)

    iface = gr.Interface(
        fn=chatbot_interface,
        inputs=gr.Textbox(lines=2, placeholder="Enter your medical query here..."),
        outputs="text",
        title="Clinical Knowledge Navigator",
        description="Ask questions about medical guidelines, drug information, and conditions. The AI will provide answers and cite its sources."
    )
    print("Launching Gradio interface...")
    iface.launch(share=False)

if __name__ == "__main__":
    main()