import os
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient, models
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

import streamlit as st
import requests

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "saas_support_knowledge"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def setup_qdrant_db_function():
    if not os.path.exists("data"):
        os.makedirs("data")
        with open("data/product_feature_x.txt", "w") as f:
            f.write("Feature X allows users to integrate with external CRM systems. It supports Salesforce, HubSpot, and Zoho CRM. Configuration details can be found in the admin panel under 'Integrations'.")
        with open("data/faq_billing.txt", "w") as f:
            f.write("Billing cycles are monthly. You can upgrade or downgrade your plan anytime from the 'Subscription' section in your account settings. Refunds are processed within 5-7 business days.")
        with open("data/troubleshooting_login.txt", "w") as f:
            f.write("If you cannot log in, first try resetting your password. Ensure your email is correct. If the issue persists, contact support with your account details.")

    loader = DirectoryLoader("data/", glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    try:
        client.delete_collection(collection_name=COLLECTION_NAME)
    except Exception:
        pass

    Qdrant.from_documents(
        texts,
        embeddings,
        collection_name=COLLECTION_NAME,
        client=client,
        force_recreate=True
    )

app = FastAPI()

embeddings_model_fastapi = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
qdrant_client_fastapi = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
vectorstore_fastapi = Qdrant(
    client=qdrant_client_fastapi,
    collection_name=COLLECTION_NAME,
    embeddings=embeddings_model_fastapi
)
retriever_fastapi = vectorstore_fastapi.as_retriever()

llm_fastapi = ChatOpenAI(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY, temperature=0.0)

template_fastapi = """You are an AI customer support assistant for a SaaS company.\nUse the following context to answer the user's question.\nIf you don't know the answer, state that you don't have enough information, don't try to make up an answer.\n\nContext: {context}\n\nQuestion: {question}\n"""
prompt_fastapi = ChatPromptTemplate.from_template(template_fastapi)

def format_docs_fastapi(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain_fastapi = (
    {"context": retriever_fastapi | RunnableLambda(format_docs_fastapi), "question": RunnablePassthrough()}
    | prompt_fastapi
    | llm_fastapi
    | StrOutputParser()
)

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def get_response(request: QueryRequest):
    if not OPENAI_API_KEY:
        return {"error": "OPENAI_API_KEY is not set in environment variables."}
    try:
        response = rag_chain_fastapi.invoke(request.query)
        return {"response": response}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    st.title("SaaS Customer Support Assistant")
    st.markdown("Ask a question about our SaaS product and get answers from our knowledge base.")

    user_query = st.text_input("Your question:")

    if st.button("Get Answer"):
        if not user_query:
            st.warning("Please enter a question.")
        else:
            with st.spinner("Searching and generating response..."):
                try:
                    fastapi_url = "http://localhost:8000/query"
                    payload = {"query": user_query}
                    headers = {"Content-Type": "application/json"}
                    
                    response = requests.post(fastapi_url, json=payload, headers=headers)
                    response.raise_for_status()
                    
                    result = response.json()
                    if "response" in result:
                        st.success("Response:")
                        st.write(result["response"])
                    elif "error" in result:
                        st.error(f"Error from API: {result["error"]}")
                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to the FastAPI backend. Make sure it's running on http://localhost:8000.")
                except requests.exceptions.RequestException as e:
                    st.error(f"An error occurred during the API request: {e}")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")

    st.markdown("---")
    st.subheader("Database Setup")
    st.write("Click the button below to setup the Qdrant database with sample knowledge.")
    if st.button("Setup Qdrant Database"):
        with st.spinner("Setting up database... This might take a moment."):
            try:
                setup_qdrant_db_function()
                st.success("Qdrant database setup complete!")
            except Exception as e:
                st.error(f"Failed to setup Qdrant database: {e}. Ensure Qdrant is running.")