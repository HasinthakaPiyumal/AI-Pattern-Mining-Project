import streamlit as st
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# --- 1. Generative Language Model (Parametric Memory Component) ---
# Using a small, local model for demonstration purposes
# In a real application, replace with a more powerful, potentially fine-tuned medical LLM or an API-based LLM (e.g., OpenAI, Cohere)
@st.cache_resource
def load_llm():
    model_name = "google/gemma-2b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=500,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.1,
    )
    return HuggingFacePipeline(pipeline=pipe)

# --- 2. Retrieval Mechanism (Non-Parametric Memory Component) ---

# Sample Medical Knowledge Base (In a real system, this would be vast and dynamic)
MEDICAL_DOCUMENTS = [
    "Hypertension management in diabetic patients often involves ACE inhibitors or ARBs, aiming for a blood pressure target below 130/80 mmHg. Lifestyle modifications like diet and exercise are crucial.",
    "The latest guidelines for type 2 diabetes treatment emphasize early initiation of metformin, followed by GLP-1 receptor agonists or SGLT2 inhibitors for cardiovascular and renal benefits.",
    "Aspirin therapy for primary prevention of cardiovascular disease is not recommended for most adults without a high risk profile, due to the risk of bleeding. Consult patient-specific risk factors.",
    "Common side effects of statins include muscle pain and liver enzyme elevation. Regular monitoring is advised. Adherence to therapy is important for cholesterol reduction.",
    "Symptoms of acute appendicitis typically include sudden pain that begins around the navel and shifts to the lower right abdomen, often accompanied by nausea, vomiting, and fever.",
    "Diagnosis of deep vein thrombosis (DVT) usually involves ultrasound. Treatment involves anticoagulants such as heparin and warfarin, or direct oral anticoagulants (DOACs).",
    "Chronic kidney disease (CKD) is staged based on glomerular filtration rate (GFR). Management focuses on controlling blood pressure, diabetes, and reducing proteinuria with ACE inhibitors/ARBs.",
    "Diabetic retinopathy is a common complication of diabetes, affecting the eyes. Regular ophthalmic examinations are crucial for early detection and management to prevent vision loss.",
    "The COVID-19 pandemic guidelines suggest vaccination for all eligible individuals. Treatment protocols for severe cases may include antivirals like remdesivir and corticosteroids.",
    "Migraine headaches are often characterized by throbbing pain, usually on one side of the head, sensitivity to light and sound, and nausea. Triptans are a common class of abortive medications.",
]

@st.cache_resource
def setup_retriever():
    # Embedding Model
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    # Text Splitter for chunking documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.create_documents(MEDICAL_DOCUMENTS)

    # Vector Database (Chroma for local demo)
    vectorstore = Chroma.from_documents(docs, embeddings)

    return vectorstore.as_retriever()

# --- 3. RAG Orchestration Layer ---
@st.cache_resource
def setup_qa_chain(llm, retriever):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

# --- 4. User Interface (Streamlit) ---
st.title("Dynamic Clinical Decision Support Assistant")
st.markdown("This assistant combines a generative AI model with a medical knowledge base to provide evidence-based insights.")

# Load components
llm = load_llm()
retriever = setup_retriever()
qa_chain = setup_qa_chain(llm, retriever)

user_query = st.text_area("Enter your medical question or patient scenario:", 
                          "What are the recommended treatments for chronic kidney disease with proteinuria?")

if st.button("Get Clinical Insight"): 
    if user_query:
        with st.spinner("Fetching and synthesizing information..."):
            response = qa_chain.invoke({"query": user_query})
            
            st.subheader("Generated Insight:")
            st.write(response["result"])
            
            st.subheader("Sources Consulted:")
            if response["source_documents"]:
                for i, doc in enumerate(response["source_documents"]):
                    st.write(f"- Document {i+1}: {doc.page_content}")
            else:
                st.info("No specific source documents found for this query in the limited knowledge base.")
    else:
        st.warning("Please enter a query to get insights.")