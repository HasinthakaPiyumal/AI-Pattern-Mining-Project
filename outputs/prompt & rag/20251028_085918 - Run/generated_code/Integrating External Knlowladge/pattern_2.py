
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- Mock LLM for Demonstration Purposes ---
# In a real application, you would replace this with an actual LLM integration (e.g., OpenAI, HuggingFace)
class MockLLM:
    def __init__(self, medical_knowledge_base):
        self.medical_knowledge_base = medical_knowledge_base

    def __call__(self, prompt):
        # A very simplistic mock: try to find keywords from the prompt in the knowledge base
        # and construct a basic response. This is NOT an actual LLM.
        query = prompt.split("Context:\n")[0].replace("Question:", "").strip()
        context = prompt.split("Context:\n")[1].split("\n\nQuestion:")[0].strip()

        response_parts = [
            f"Based on the provided information, regarding '{query}':"
        ]

        if "medical condition" in query.lower() or "diagnosis" in query.lower():
            if "symptoms" in context.lower():
                response_parts.append(f"The context mentions symptoms related to a medical condition: {context}")
            else:
                response_parts.append(f"The context provides details about medical conditions: {context}")
        elif "drug interaction" in query.lower() or "medication" in query.lower():
            if "side effects" in context.lower():
                response_parts.append(f"Information on drug interactions and side effects is available: {context}")
            else:
                response_parts.append(f"The context discusses medications and their interactions: {context}")
        elif "treatment protocol" in query.lower():
            response_parts.append(f"Treatment protocols mentioned in the context include: {context}")
        elif "research findings" in query.lower() or "latest studies" in query.lower():
            response_parts.append(f"Recent research findings from the context indicate: {context}")
        else:
            response_parts.append(f"The relevant context is: {context}")

        return " ".join(response_parts) + " Please consult with a healthcare professional for accurate medical advice."


# --- Simulated Data Ingestion ---
def get_medical_documents():
    # In a real scenario, this would fetch data from PubMed, EHRs, news feeds, etc.
    # For this demo, we use a few sample medical texts.
    return [
        "COVID-19 is a highly contagious respiratory illness caused by the SARS-CoV-2 virus. Symptoms range from mild to severe, including fever, cough, fatigue, and loss of taste or smell. Vaccination is crucial for prevention.",
        "Type 2 diabetes is a chronic condition that affects the way the body processes blood sugar (glucose). It can lead to serious health complications if not managed properly. Lifestyle changes, such as diet and exercise, are often the first line of treatment, sometimes supplemented by medication like metformin.",
        "A recent study published in The Lancet suggests that a novel gene therapy shows promising results in treating a specific form of muscular dystrophy. Early phase clinical trials reported significant improvements in muscle function among participants.",
        "Drug interaction between Warfarin (an anticoagulant) and Ibuprofen (an NSAID) can increase the risk of bleeding. Patients on Warfarin should avoid Ibuprofen and consult their doctor for alternative pain relief.",
        "Hypertension, or high blood pressure, is a common condition where the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease. Regular monitoring and medication like ACE inhibitors are often prescribed.",
        "Alzheimer's disease is a progressive neurological disorder that causes the brain to shrink and brain cells to die. It's the most common cause of dementia, a continuous decline in thinking, behavioral and social skills that disrupts a person's ability to function independently. Current treatments focus on managing symptoms and slowing progression."
    ]

# --- RAG Setup ---
@st.cache_resource
def setup_rag_pipeline():
    documents = get_medical_documents()

    # 1. Text Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        add_start_index=True,
    )
    texts = text_splitter.create_documents(documents)

    # 2. Embedding Models
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # 3. Vector Database (Chroma)
    # Using a persistent client for Chroma to avoid re-embedding on every rerun if possible
    # For a real app, you'd manage persistence more robustly
    vectorstore = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory="./chroma_db")
    vectorstore.persist()

    # 4. LLM Core (Mock LLM)
    mock_llm_instance = MockLLM(medical_knowledge_base=documents)

    # 5. RAG Chain
    # Define a custom prompt template for better control over context integration
    custom_prompt_template = """Use the following pieces of context to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Always provide a disclaimer about consulting a healthcare professional for medical advice.

Context:
{context}

Question: {question}

Helpful Answer:"""
    custom_prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=mock_llm_instance,
        chain_type="stuff",  # "stuff" concatenates all retrieved docs into a single prompt
        retriever=vectorstore.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": custom_prompt}
    )
    return qa_chain

# --- Streamlit UI ---
st.set_page_config(page_title="Clinical Research Assistant LLM", layout="wide")
st.title("🩺 Clinical Research Assistant LLM")
st.markdown("Ask me about medical conditions, drug interactions, and recent research findings.")

qa_chain = setup_rag_pipeline()

user_query = st.text_input("Enter your medical query:", "What are the latest treatments for Alzheimer's disease?")

if user_query:
    with st.spinner("Searching and generating response..."):
        try:
            response = qa_chain.invoke({"query": user_query})
            st.subheader("Response:")
            st.write(response["result"])

            st.subheader("Source Documents:")
            for i, doc in enumerate(response["source_documents"]):
                st.markdown(f"**Document {i+1}:**")
                st.write(f"*Content:* {doc.page_content}")
                # st.write(f"*Source:* {doc.metadata.get('source', 'N/A')}") # if metadata was added
        except Exception as e:
            st.error(f"An error occurred: {e}. Please try again.")

st.markdown("__Disclaimer:__ This is a demonstration system. Always consult with qualified healthcare professionals for medical advice and treatment.")
