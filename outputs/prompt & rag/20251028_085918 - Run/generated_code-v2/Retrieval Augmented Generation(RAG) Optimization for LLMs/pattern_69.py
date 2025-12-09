import streamlit as st
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSequence

# --- Configuration (replace with actual API key and potentially a more robust setup)
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# --- Simulate a medical knowledge base
medical_documents = [
    "Aspirin is commonly used as an analgesic for pain relief, an antipyretic for fever reduction, and an anti-inflammatory agent. It also has antiplatelet effects, which are crucial in preventing cardiovascular events.",
    "Type 2 diabetes mellitus is characterized by insulin resistance and relative insulin deficiency. Management often involves lifestyle modifications, oral hypoglycemic agents, and sometimes insulin therapy.",
    "Hypertension, or high blood pressure, significantly increases the risk of heart disease, stroke, and kidney disease. Treatment typically includes lifestyle changes and antihypertensive medications such as ACE inhibitors, ARBs, beta-blockers, and diuretics.",
    "Pneumonia is an infection that inflames air sacs in one or both lungs, which may fill with fluid or pus. Symptoms include cough with phlegm or pus, fever, chills, and difficulty breathing. Antibiotics are generally prescribed for bacterial pneumonia.",
    "The COVID-19 pandemic, caused by the SARS-CoV-2 virus, led to widespread respiratory illness. Vaccination has proven highly effective in reducing severe disease and mortality. Common symptoms include fever, cough, fatigue, and loss of taste or smell.",
    "Migraine is a severe headache often accompanied by symptoms such as throbbing pain on one side of the head, nausea, vomiting, and extreme sensitivity to light and sound. Triptans are a class of drugs often used to treat acute migraine attacks.",
    "Cataracts are a clouding of the eye's natural lens, which can impair vision. They typically develop slowly and are common among older adults. Surgical removal of the cataract and replacement with an artificial lens is the primary treatment.",
    "Alzheimer's disease is a progressive neurodegenerative disorder that causes brain cells to waste away and die. It is the most common cause of dementia, leading to a continuous decline in thinking, behavioral and social skills, and eventually affects a person's ability to function independently.",
    "Asthma is a chronic respiratory condition characterized by inflammation and narrowing of the airways, leading to symptoms like wheezing, shortness of breath, chest tightness, and coughing. Inhalers are a common treatment to manage symptoms and prevent attacks.",
    "Cancer is a disease caused by an uncontrolled division of abnormal cells in a part of the body. Treatments vary widely depending on the type and stage of cancer, and can include surgery, chemotherapy, radiation therapy, immunotherapy, and targeted therapy."
]

# --- Embedding Model and Vector Store
@st.cache_resource
def get_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_texts(texts=medical_documents, embedding=embeddings)
    return vectorstore, embeddings

vectorstore, embeddings = get_vector_store()
retriever = vectorstore.as_retriever()

# --- LLM and RAG Chain
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0)

rag_prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a medical assistant. Use the following retrieved medical context to answer the user's question accurately and concisely. Cite the source of information if possible."),
    ("human", "Context: {context}\n\nQuestion: {question}")
])

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | rag_prompt_template
    | llm
)

# --- Streamlit UI
st.set_page_config(page_title="Medical Literature Q&A for Clinicians", layout="wide")
st.title("🧠 Medical Literature Q&A for Clinicians")
st.markdown("Ask questions about medical conditions, treatments, and research.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_query = st.chat_input("Ask a medical question...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.spinner("Searching and generating answer..."):
        try:
            response = rag_chain.invoke(user_query)
            llm_response = response.content
            st.session_state.messages.append({"role": "assistant", "content": llm_response})
            with st.chat_message("assistant"):
                st.markdown(llm_response)
        except Exception as e:
            error_message = f"An error occurred: {e}. Please ensure your OPENAI_API_KEY is correctly set and has access."
            st.session_state.messages.append({"role": "assistant", "content": error_message})
            with st.chat_message("assistant"):
                st.error(error_message)
