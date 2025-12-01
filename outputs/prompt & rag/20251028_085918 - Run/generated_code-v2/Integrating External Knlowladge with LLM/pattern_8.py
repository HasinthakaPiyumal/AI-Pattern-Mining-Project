import os
import streamlit as st
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document

os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
os.environ["PINECONE_API_KEY"] = "YOUR_PINECONE_API_KEY"
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

INDEX_NAME = "medical-research-assistant"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "gpt-3.5-turbo"

@st.cache_resource
def get_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

@st.cache_resource
def initialize_pinecone():
    if not PINECONE_API_KEY:
        st.error("PINECONE_API_KEY is not set. Please set it in your environment variables.")
        st.stop()
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc

def upsert_documents_to_pinecone(pc_instance, embeddings_model, documents_data):
    if INDEX_NAME not in pc_instance.list_indexes():
        pc_instance.create_index(
            name=INDEX_NAME,
            dimension=embeddings_model.client.embed_query("test").shape[0], 
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-west-2")
        )
        st.info(f"Created Pinecone index: {INDEX_NAME}")
    else:
        st.info(f"Pinecone index '{INDEX_NAME}' already exists.")

    index = pc_instance.Index(INDEX_NAME)

    langchain_docs = []
    for i, doc_text in enumerate(documents_data):
        langchain_docs.append(Document(page_content=doc_text, metadata={"source": f"doc_{i+1}"}))

    if not langchain_docs:
        st.warning("No documents to upsert.")
        return

    st.write("Embedding documents and upserting to Pinecone (this might take a moment)...")
    
    if index.describe_index_stats().total_vector_count == 0:
        LangchainPinecone.from_documents(
            langchain_docs, embeddings_model, index_name=INDEX_NAME
        )
        st.success(f"Upserted {len(langchain_docs)} documents to Pinecone.")
    else:
        st.info("Index already contains data, skipping upsert for demo.")

sample_medical_docs = [
    "A study on the efficacy of Remdesivir for COVID-19 treatment showed reduced recovery time in hospitalized patients. However, its effect on mortality was not statistically significant. Source: NEJM, 2020.",
    "Metformin is a first-line medication for type 2 diabetes, primarily working by decreasing glucose production by the liver and improving insulin sensitivity. Common side effects include gastrointestinal upset. Source: ADA Guidelines, 2023.",
    "The use of mRNA vaccines has revolutionized immunology, demonstrating high effectiveness against various infectious diseases, including SARS-CoV-2. These vaccines work by instructing cells to produce a harmless piece of viral protein, triggering an immune response. Source: Nature, 2021.",
    "Hypertension, or high blood pressure, is a major risk factor for cardiovascular disease. Lifestyle modifications such as diet and exercise are crucial, alongside pharmacological treatments like ACE inhibitors or calcium channel blockers. Source: AHA, 2022.",
    "Alzheimer's disease is a progressive neurodegenerative disorder characterized by memory loss, cognitive decline, and behavioral changes. Current treatments focus on managing symptoms rather than curing the disease. Research into amyloid-beta plaques and tau tangles is ongoing. Source: Mayo Clinic, 2023."
]

@st.cache_resource
def get_rag_chain():
    if not OPENAI_API_KEY:
        st.error("OPENAI_API_KEY is not set. Please set it in your environment variables.")
        st.stop()

    pc_instance = initialize_pinecone()
    
    openai_embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-ada-002")

    try:
        vectorstore = LangchainPinecone.from_existing_index(INDEX_NAME, openai_embeddings)
    except Exception as e:
        st.error(f"Could not connect to Pinecone index '{INDEX_NAME}'. Please ensure it exists and is populated. Error: {e}")
        st.stop()
        
    retriever = vectorstore.as_retriever()

    template = """
You are a Medical Research Assistant. Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't have enough information.
Provide concise and factual answers, and always cite the source of information if available in the metadata (e.g., Source: [doc_ID]).

Context: {context}

Question: {question}

Answer:
"""
    qa_prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    llm = ChatOpenAI(model_name=LLM_MODEL_NAME, temperature=0, openai_api_key=OPENAI_API_KEY)

    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": qa_prompt}
    )
    return rag_chain

st.set_page_config(page_title="Medical Research Assistant")
st.title("👨‍⚕️ Medical Research Assistant")
st.markdown("Ask questions about medical literature, clinical trials, and drug information.")

openai_embeddings_for_upsert = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-ada-002")
pc_instance = initialize_pinecone()
upsert_documents_to_pinecone(pc_instance, openai_embeddings_for_upsert, sample_medical_docs)

rag_chain = get_rag_chain()

user_query = st.text_input("Enter your medical question:", "What are the primary treatments for type 2 diabetes?")

if st.button("Get Answer"):
    if user_query:
        with st.spinner("Searching and generating response..."):
            try:
                response = rag_chain({"query": user_query})
                st.subheader("Answer:")
                st.write(response["result"])

                if response.get("source_documents"):
                    st.subheader("Sources:")
                    for doc in response["source_documents"]:
                        st.write(f"- {doc.page_content} (Source: {doc.metadata.get('source', 'N/A')})")
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.warning("Please ensure your OpenAI and Pinecone API keys are correctly set and your Pinecone index is properly initialized and populated.")
    else:
        st.warning("Please enter a query.")
