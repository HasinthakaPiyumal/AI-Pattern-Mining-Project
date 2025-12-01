import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.docstore.document import Document
from typing import List, Dict

def scrape_medical_data(urls: List[str]) -> List[Document]:
    documents = []
    for url in urls:
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.content, "html.parser")
            paragraphs = soup.find_all("p")
            page_content = " ".join([p.get_text() for p in paragraphs])
            documents.append(Document(page_content=page_content, metadata={"source": url}))
        except Exception as e:
            st.warning(f"Could not scrape {url}: {e}")
    return documents

TRUSTED_MEDICAL_SOURCES = [
    "https://www.who.int/news-room/fact-sheets/detail/cancer",
    "https://www.cdc.gov/cancer/dcpc/about/index.htm",
    "https://www.cancer.org/cancer/what-is-cancer.html"
]

@st.cache_resource
def setup_vector_store(documents: List[Document]):
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory="./chroma_db")
    vectorstore.persist()
    return vectorstore, embeddings

def process_query(query: str) -> str:
    return query

def generate_answer_with_references(query: str, retrieved_docs: List[Document]) -> str:
    answer_parts = []
    references = {}
    ref_counter = 1

    answer_parts.append(f"Based on your query about '{query}', here's what we found:\n\n")

    if not retrieved_docs:
        answer_parts.append("We couldn't find specific supporting references for your query from our trusted sources.")
        return "".join(answer_parts)

    for i, doc in enumerate(retrieved_docs):
        source_url = doc.metadata.get("source", "Unknown Source")
        if source_url not in references.values():
            references[str(ref_counter)] = source_url
            current_ref_id = str(ref_counter)
            ref_counter += 1
        else:
            current_ref_id = [k for k, v in references.items() if v == source_url][0]

        snippet = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
        answer_parts.append(f"  - \"{snippet}\" [{current_ref_id}]\n")

    answer_parts.append("\n\n---\nReferences:\n")
    for ref_id, url in references.items():
        answer_parts.append(f"[{ref_id}] {url}\n")

    return "".join(answer_parts)

def main():
    st.set_page_config(page_title="Medical QA with References")
    st.title("Medical Query Answering System with Reference-Supported Explanations")
    st.markdown("Enter a medical query and get answers with verifiable sources from trusted medical websites.")

    st.sidebar.header("System Setup")
    if st.sidebar.button("Refresh Medical Data (Scrape & Embed)"):
        with st.spinner("Scraping and embedding medical data... This might take a moment."):
            scraped_docs = scrape_medical_data(TRUSTED_MEDICAL_SOURCES)
            if scraped_docs:
                st.session_state.vectorstore, st.session_state.embeddings = setup_vector_store(scraped_docs)
                st.sidebar.success("Medical data refreshed and embedded!")
            else:
                st.sidebar.error("Failed to scrape any medical data. Check URLs or network.")
    
    if "vectorstore" not in st.session_state or "embeddings" not in st.session_state:
        st.info("Please refresh medical data by clicking the button in the sidebar to initialize the system.")
        st.session_state.vectorstore = None
        st.session_state.embeddings = None

    user_query = st.text_input("Enter your medical query:", "What are the early symptoms of cancer?")

    if user_query and st.session_state.vectorstore and st.session_state.embeddings:
        if st.button("Get Answer"):
            with st.spinner("Searching for answers and references..."):
                processed_q = process_query(user_query)
                
                retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
                retrieved_documents = retriever.invoke(processed_q)
                
                final_answer = generate_answer_with_references(user_query, retrieved_documents)
                
                st.subheader("Generated Answer:")
                st.write(final_answer)
    elif user_query and (not st.session_state.vectorstore or not st.session_state.embeddings):
        st.warning("Please refresh medical data in the sidebar to use the system.")

if __name__ == "__main__":
    main()