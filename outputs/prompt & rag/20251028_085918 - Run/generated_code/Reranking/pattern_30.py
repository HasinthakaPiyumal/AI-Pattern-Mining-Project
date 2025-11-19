import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain.schema import Document

load_dotenv()

# --- Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not found. Please set it in your .env file.")
    st.stop()

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_PERSIST_DIR = "./chroma_db"

# --- Knowledge Base Setup ---
def load_medical_docs_into_chroma():
    # In a real application, this would parse PDFs (using PyPDF2) and structured data (pandas)
    # For this example, we use some mock medical data.
    medical_data = [
        {"text": "Type 2 diabetes is a chronic condition that affects the way the body processes blood sugar (glucose). The body either doesn't produce enough insulin, or it resists insulin. Symptoms include increased thirst, frequent urination, and unexplained weight loss. Treatment often involves diet modification, exercise, and medication such as metformin or insulin.", "source": "WHO Guidelines on Diabetes"},
        {"text": "Hypertension, or high blood pressure, is a common condition in which the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease. Risk factors include obesity, lack of physical activity, and a high-sodium diet. Treatment typically includes lifestyle changes and medications like ACE inhibitors or diuretics.", "source": "AHA Blood Pressure Guidelines"},
        {"text": "The flu (influenza) is a contagious respiratory illness caused by influenza viruses. It can cause mild to severe illness. Serious outcomes of flu infection can result in hospitalization or death. Symptoms include fever, cough, sore throat, body aches, and fatigue. Annual vaccination is recommended, and antiviral drugs like oseltamivir can be used for treatment.", "source": "CDC Influenza Information"},
        {"text": "Appendicitis is an inflammation of the appendix, a finger-shaped pouch that projects from your colon on the lower right side of your abdomen. Appendicitis causes pain in your lower right abdomen. However, in most people, pain begins around the navel and then moves. Treatment almost always involves surgical removal of the appendix.", "source": "Mayo Clinic - Appendicitis"},
        {"text": "Metformin is an oral medication used to treat type 2 diabetes. It works by decreasing glucose production by the liver and improving insulin sensitivity. Common side effects include nausea, diarrhea, and abdominal discomfort. It should be used with caution in patients with kidney problems.", "source": "DrugBank - Metformin"},
    ]

    documents = [
        Document(page_content=data["text"], metadata={"source": data["source"]})
        for data in medical_data
    ]

    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    # Clear existing data for demonstration purposes if desired
    # if os.path.exists(CHROMA_PERSIST_DIR):
    #     import shutil
    #     shutil.rmtree(CHROMA_PERSIST_DIR)

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR
    )
    vectorstore.persist()
    return vectorstore

@st.cache_resource
def get_vectorstore():
    if not os.path.exists(CHROMA_PERSIST_DIR) or not os.listdir(CHROMA_PERSIST_DIR):
        st.info("Initializing medical knowledge base...")
        vectorstore = load_medical_docs_into_chroma()
        st.success("Medical knowledge base initialized.")
    else:
        embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        vectorstore = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)
        st.info("Loaded existing medical knowledge base.")
    return vectorstore

vectorstore = get_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# --- Reranking Module ---
def rerank_documents(query: str, documents: list[Document], llm_reranker: ChatOpenAI) -> list[Document]:
    if not documents:
        return []
    
    # Create a prompt for the LLM to rerank documents
    rerank_prompt_template = """You are an expert medical assistant. Rank the following medical documents based on their relevance to the user's query. Provide only the ranked list of document contents, from most relevant to least relevant.

User Query: {query}

Documents to rank:
{documents_str}

Ranked Documents (most relevant first):
"""

    documents_str = "\n---\n".join([f"Document {i+1}: {doc.page_content}" for i, doc in enumerate(documents)])
    
    rerank_prompt = PromptTemplate(
        template=rerank_prompt_template,
        input_variables=["query", "documents_str"]
    )
    
    chain = rerank_prompt | llm_reranker
    
    reranked_text = chain.invoke({"query": query, "documents_str": documents_str}).content
    
    # This parsing is simplistic; a more robust solution would be needed
    # For this example, we'll just try to match content roughly or assume LLM outputs clean content.
    # A better approach might be to have the LLM output document indices or a scoring.
    
    # For simplicity, we'll just return the original documents for now,
    # but in a real scenario, the `reranked_text` would be parsed
    # to reorder `documents`.
    return documents # Placeholder: Actual reranking logic to reorder based on reranked_text


# --- Conditional Retrieval Logic ---
def should_retrieve_knowledge(query: str) -> bool:
    # Simple heuristic: trigger retrieval if query contains medical keywords or asks for specific info
    medical_keywords = ["diagnosis", "treatment", "symptoms", "medication", "drug", "disease", "condition", "guidelines", "cause", "cure"]
    if any(keyword in query.lower() for keyword in medical_keywords):
        return True
    if "what is" in query.lower() or "tell me about" in query.lower() or "explain" in query.lower():
        return True
    return False

# --- Langchain Setup ---
llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY, temperature=0)

# Custom RAG chain to incorporate reranking and conditional retrieval
class MedicalAssistantChain:
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever
        self.qa_prompt = PromptTemplate(
            template="""You are a medical assistant. Answer the user's question truthfully and base your answer on the provided context. If the context does not contain enough information, state that you cannot provide a definitive answer but offer general medical knowledge if appropriate. Cite your sources from the context where possible.

Context: {context}

Question: {question}

Answer:
""",
            input_variables=["context", "question"],
        )

    def invoke(self, query: str):
        st.session_state.messages.append({"role": "user", "content": query})
        st.session_state.history.append((query, "")) # Add to history

        with st.spinner("Processing your query..."):
            if should_retrieve_knowledge(query):
                st.info("Retrieving relevant medical knowledge...")
                retrieved_docs = self.retriever.invoke(query)
                st.info(f"Found {len(retrieved_docs)} potential documents.")
                
                # Apply reranking (simplified for this example)
                # For a real application, the reranking output would be used to reorder retrieved_docs
                # reranked_docs = rerank_documents(query, retrieved_docs, self.llm)
                reranked_docs = retrieved_docs # Skipping actual reranking reordering for simplicity

                context = "\n\n".join([doc.page_content + f" (Source: {doc.metadata.get('source', 'N/A')})" for doc in reranked_docs])
            else:
                context = "No specific external medical knowledge retrieved for this query. Responding with general medical knowledge if applicable."
            
            final_prompt = self.qa_prompt.format(context=context, question=query)
            response = self.llm.invoke(final_prompt).content

            # Update history with full response
            last_user_query_index = len(st.session_state.history) - 1
            st.session_state.history[last_user_query_index] = (query, response)

            st.session_state.messages.append({"role": "assistant", "content": response})
            return response

medical_assistant_chain = MedicalAssistantChain(llm=llm, retriever=retriever)

# --- Streamlit UI ---
st.set_page_config(page_title="Medical Diagnosis and Treatment Assistant")
st.title("🩺 Medical Diagnosis and Treatment Assistant")
st.markdown("Ask me anything about medical conditions, diagnoses, or treatments.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Your medical query:"):
    _ = medical_assistant_chain.invoke(prompt)

st.sidebar.header("Chat History")
if st.session_state.history:
    for i, (q, a) in enumerate(reversed(st.session_state.history)):
        st.sidebar.subheader(f"Query {len(st.session_state.history) - i}")
        st.sidebar.write(f"**Q:** {q}")
        if a:
            st.sidebar.write(f"**A:** {a[:100]}...") # Show snippet of answer
        else:
            st.sidebar.write("*(Processing...)*")

