import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Configuration ---
# In a real application, replace with actual API keys and model names
# For demonstration, we'll use a placeholder for the LLM.

# Dummy Medical Literature (replace with actual document loading from PubMed, etc.)
dummy_medical_literature = [
    "Aspirin is commonly used as an anti-inflammatory and antiplatelet agent. Side effects can include gastrointestinal bleeding. It should be used with caution in patients with a history of ulcers.",
    "Type 2 diabetes mellitus is characterized by insulin resistance and relative insulin deficiency. Management often involves lifestyle modifications, metformin, and other oral hypoglycemic agents. Regular blood glucose monitoring is essential.",
    "Hypertension, or high blood pressure, is a major risk factor for cardiovascular disease. Treatment often includes ACE inhibitors, ARBs, calcium channel blockers, and diuretics. Lifestyle changes like diet and exercise are crucial.",
    "The COVID-19 pandemic significantly impacted global health. Symptoms vary but often include fever, cough, and fatigue. Vaccination and public health measures are key to control. New variants continue to emerge.",
    "Migraine is a severe headache often accompanied by nausea, vomiting, and sensitivity to light and sound. Triptans are a common class of abortive medications. Preventive therapies include beta-blockers and CGRP inhibitors."
]

# --- 1. Data Ingestion and Indexing Module (Simplified) ---
@st.cache_resource
def get_vectorstore():
    # Using a simple in-memory ChromaDB for demonstration
    # In a real system, this would persist to disk or connect to a cloud vector store
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = text_splitter.create_documents([doc for doc in dummy_medical_literature])

    # Embeddings: Using a common sentence transformer model
    # Consider 'bge-large-en-v1.5' or 'all-MiniLM-L6-v2' for production
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
    return vectorstore

vectorstore = get_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 relevant chunks

# --- Mock LLM for demonstration ---
# In a real application, you would integrate with OpenAI, Google Gemini, etc.
class MockLLM:
    def invoke(self, prompt_value):
        # Simulate LLM processing
        retrieved_context = prompt_value.messages[0].content.split("Context:\n")[1].split("\n\nQuestion:")[0]
        question = prompt_value.messages[0].content.split("Question:\n")[1]
        
        if "aspirin" in question.lower() and "gastrointestinal bleeding" in retrieved_context.lower():
            return "Based on the provided medical literature, Aspirin is used as an anti-inflammatory but can cause gastrointestinal bleeding. It should be used cautiously in patients with a history of ulcers."
        elif "type 2 diabetes" in question.lower() and "metformin" in retrieved_context.lower():
            return "According to the literature, Type 2 diabetes involves insulin resistance, and its management includes lifestyle changes, metformin, and other agents. Regular monitoring is vital."
        elif "hypertension" in question.lower() and "ACE inhibitors" in retrieved_context.lower():
            return "The retrieved information indicates that Hypertension is a cardiovascular risk factor, and treatment often involves ACE inhibitors, ARBs, and lifestyle adjustments."
        elif "COVID-19" in question.lower() and "vaccination" in retrieved_context.lower():
            return "The provided context states that COVID-19 has varied symptoms, and vaccination along with public health measures are crucial for control."
        elif "migraine" in question.lower() and "triptans" in retrieved_context.lower():
            return "Based on the literature, Migraine is a severe headache condition often treated with triptans for abortive relief and beta-blockers for prevention."
        else:
            return f"I've processed your question using the retrieved medical context. While I can't generate a sophisticated medical response with this mock LLM, the relevant information was: {retrieved_context}. Your question was: {question}."

mock_llm = MockLLM()

# --- 3. Context Conditioning Module ---
# Prompt template to instruct the LLM on how to use the retrieved context
rag_prompt = ChatPromptTemplate.from_template("""
Answer the question truthfully based only on the following context:
Context:
{context}

Question:
{question}
""")

# --- 5. Adaptive Decision-Making Module (Basic Heuristic) ---
def decide_strategy(query: str, retrieved_docs: list):
    # Simple heuristic: If the query directly asks about a known fact (without needing deep RAG),
    # or if retrieval yields no results, we can adapt.
    
    # For this basic example, if a specific keyword from our dummy data is in the query
    # and no documents are retrieved, we might indicate limited information.
    if not retrieved_docs:
        known_keywords = ["aspirin", "diabetes", "hypertension", "covid-19", "migraine"]
        if any(keyword in query.lower() for keyword in known_keywords):
            return "No specific, highly relevant literature found in the indexed database for your query. Consider refining your search or checking comprehensive medical databases."
        else:
            return "I could not find relevant information for your query. Please rephrase or provide more details."
    
    # In a real system, you might have a classification model here to decide
    # if the query can be answered directly by the LLM (if it's simple common knowledge)
    # or if it strictly requires retrieval.
    return "RAG"

# --- Overall RAG Chain using LangChain Expression Language (LCEL) ---
# Combines retrieval, context formatting, and LLM invocation

# Function to format documents for the prompt
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | mock_llm  # Using our mock LLM for demonstration
    | StrOutputParser()
)

# --- 7. User Interface Module (Streamlit) ---
st.set_page_config(page_title="Clinical Decision Support System (RAG Demo)")
st.title("🩺 Clinical Decision Support System (RAG Demo)")
st.markdown("This system demonstrates Retrieval-Augmented Generation (RAG) for medical queries using dummy literature.")
st.markdown("It aims to provide context-aware information by retrieving relevant snippets and using an LLM to formulate a response.")

query = st.text_area("Enter your clinical question here:", height=100, placeholder="e.g., What are the common uses and side effects of Aspirin?")

if st.button("Get Clinical Insight"):
    if query:
        with st.spinner("Retrieving and generating insight..."):
            # Simulate retrieval to pass to adaptive decision-making
            retrieved_documents = retriever.invoke(query)
            
            strategy = decide_strategy(query, retrieved_documents)

            if strategy == "RAG":
                response = rag_chain.invoke(query)
                st.success("Insight Generated (RAG-based):")
                st.write(response)
                with st.expander("See Retrieved Documents"):
                    for i, doc in enumerate(retrieved_documents):
                        st.write(f"**Document {i+1}:**")
                        st.write(doc.page_content)
                        st.write("---")
            else:
                st.warning(strategy) # Display the message from adaptive decision
    else:
        st.warning("Please enter a clinical question.")

st.markdown("""
### Architecture Overview (Simplified):
- **Data Ingestion/Indexing:** Dummy literature processed, chunked, and embedded into an in-memory ChromaDB.
- **Intelligent Retrieval:** Fetches top-k relevant chunks from ChromaDB based on query.
- **Context Conditioning:** Formats retrieved documents into a prompt for the LLM.
- **LLM Integration:** Uses a `MockLLM` for demonstration; real systems would use actual LLM APIs (e.g., OpenAI, Gemini).
- **Adaptive Decision-Making:** A basic heuristic to decide if retrieval is viable.
- **UI:** Powered by Streamlit for interactive querying.
""")
