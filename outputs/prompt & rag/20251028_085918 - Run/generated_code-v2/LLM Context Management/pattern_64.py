
from dotenv import load_dotenv
import os
import streamlit as st

from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda


load_dotenv()

# --- Configuration --- #
KNOWLEDGE_BASE_DIR = "./knowledge_base"
CHROMA_PERSIST_DIR = "./chroma_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Ensure knowledge base directory exists
os.makedirs(KNOWLEDGE_BASE_DIR, exist_ok=True)

def index_knowledge_base():
    if not os.path.exists(KNOWLEDGE_BASE_DIR) or not os.listdir(KNOWLEDGE_BASE_DIR):
        st.error(f"No documents found in {KNOWLEDGE_BASE_DIR}. Please add text files (e.g., .txt, .md) to this directory and rerun the indexing.")
        return

    st.info("Loading documents from knowledge base...")
    loader = DirectoryLoader(KNOWLEDGE_BASE_DIR, glob="**/*.txt", loader_cls=lambda path: open(path, encoding='utf-8').read())
    docs = loader.load()

    if not docs:
        st.error(f"No loadable text documents found in {KNOWLEDGE_BASE_DIR}. Please check file types and content.")
        return

    st.info(f"Loaded {len(docs)} documents. Splitting into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    st.info(f"Creating embeddings with {EMBEDDING_MODEL_NAME}...")
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    st.info("Persisting embeddings to ChromaDB. This might take a while...")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR
    )
    vectorstore.persist()
    st.success(f"Knowledge base indexed successfully! {len(splits)} chunks processed.")

def get_rag_chain():
    if not os.path.exists(CHROMA_PERSIST_DIR) or not os.listdir(CHROMA_PERSIST_DIR):
        st.error("ChromaDB not found or empty. Please index the knowledge base first by running the `index_knowledge_base()` function.")
        return None

    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)
    retriever = vectorstore.as_retriever()

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)

    template = """You are a clinical knowledge assistant. Use the following context to answer the user's question accurately and thoroughly. If you don't know the answer, state that you don't have enough information. Always cite the sources of your answer by including the 'source' filename where the information was found.

    Context: {context}

    Question: {question}

    Answer:"""
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        formatted = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "unknown")
            content = doc.page_content
            formatted.append(f"Source: {source}\nContent: {content}\n")
        return "\n---\n".join(formatted)

    rag_chain = (
        {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
        | prompt
        | llm
    )
    return rag_chain

st.set_page_config(page_title="Clinical Knowledge Assistant", layout="wide")
st.title("🧠 Clinical Knowledge Assistant")
st.markdown("--- Provide up-to-date medical information with transparency ---")

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

with st.sidebar:
    st.header("Knowledge Base Management")
    st.write("1. Add `.txt` or `.md` medical documents to the `./knowledge_base/` folder.")
    st.write("2. Click the button below to index the documents.")
    if st.button("Index Knowledge Base"): # Removed key
        with st.spinner("Indexing knowledge base..."): # Added spinner
            index_knowledge_base()
            st.session_state.rag_chain = get_rag_chain() # Re-initialize RAG chain after indexing

    st.header("Configuration")
    openai_api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY"))
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key

    if st.session_state.rag_chain is None:
        if openai_api_key and os.path.exists(CHROMA_PERSIST_DIR) and os.listdir(CHROMA_PERSIST_DIR):
            try:
                st.session_state.rag_chain = get_rag_chain()
                if st.session_state.rag_chain:
                    st.success("RAG Chain initialized!")
            except Exception as e:
                st.error(f"Error initializing RAG chain: {e}. Please ensure ChromaDB is indexed and API key is valid.")
        elif not openai_api_key:
            st.warning("Please enter your OpenAI API Key to start.")
        elif not (os.path.exists(CHROMA_PERSIST_DIR) and os.listdir(CHROMA_PERSIST_DIR)):
            st.warning("Knowledge base is not indexed. Please index it first.")


if st.session_state.rag_chain:
    question = st.text_area("Ask a medical question:", height=100)

    if st.button("Get Answer") and question:
        with st.spinner("Fetching answer..."): # Added spinner
            try:
                response = st.session_state.rag_chain.invoke(question)
                st.subheader("Answer:")
                st.write(response.content)

                st.subheader("Sources:")
                # Extract and display sources from the context used
                # The current LangChain RAG chain directly returns the LLM's response
                # and does not explicitly pass the retrieved docs back for direct display
                # To show sources, we need to modify the RAG chain slightly or re-retrieve and display
                # For now, relying on the LLM to cite sources in its answer as per the prompt.
                st.info("The answer above should include source citations as per the prompt instructions.")
                st.markdown("---Debugging Sources---")
                # To get actual source documents, the RAG chain needs to return both answer and docs.
                # A more advanced Langchain setup would involve returning a dictionary with both.
                # For simplicity with the current chain structure, we instruct the LLM to cite.
                # If explicit doc display is required, the chain needs restructuring to expose intermediate retrieval step.
                # For demonstration, let's show how to manually re-retrieve for display if needed.
                embeddings_display = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
                vectorstore_display = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings_display)
                retrieved_docs_display = vectorstore_display.as_retriever().invoke(question)
                for i, doc in enumerate(retrieved_docs_display):
                    source_name = doc.metadata.get("source", "Unknown Source")
                    st.markdown(f"**Source {i+1}:** `{os.path.basename(source_name)}`")
                    st.code(doc.page_content[:500] + "...", language="text")

            except Exception as e:
                st.error(f"An error occurred: {e}. Please check your OpenAI API key and ensure the knowledge base is indexed.")
elif not st.session_state.rag_chain:
    st.warning("Please configure your OpenAI API key and index the knowledge base to start.")
