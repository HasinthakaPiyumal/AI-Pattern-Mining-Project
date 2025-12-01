import streamlit as st
import os
import shutil
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

KNOWLEDGE_BASE_DIR = "medical_knowledge_base"
os.makedirs(KNOWLEDGE_BASE_DIR, exist_ok=True)

doc1_content = """
Acute appendicitis is a common surgical emergency characterized by inflammation of the vermiform appendix.
Symptoms typically include periumbilical pain migrating to the right lower quadrant, nausea, vomiting, and fever.
Diagnosis is primarily clinical, supported by imaging like ultrasound or CT scan. Treatment is surgical appendectomy.
"""
with open(os.path.join(KNOWLEDGE_BASE_DIR, "appendicitis.txt"), "w") as f:
    f.write(doc1_content)

doc2_content = """
Diabetes mellitus is a chronic metabolic disease characterized by elevated levels of blood glucose (hyperglycemia).
Type 1 diabetes results from the body's failure to produce insulin. Type 2 diabetes is characterized by insulin resistance.
Symptoms include frequent urination, increased thirst, and unexplained weight loss. Management involves diet, exercise, and medication.
"""
with open(os.path.join(KNOWLEDGE_BASE_DIR, "diabetes.txt"), "w") as f:
    f.write(doc2_content)

doc3_content = """
Hypertension, or high blood pressure, is a serious medical condition that significantly increases the risks of heart, brain, kidney, and other diseases.
It is defined as a systolic blood pressure equal to or above 140 mmHg and/or diastolic blood pressure equal to or above 90 mmHg.
Lifestyle modifications and medication are key to controlling hypertension. Regular monitoring is crucial.
"""
with open(os.path.join(KNOWLEDGE_BASE_DIR, "hypertension.txt"), "w") as f:
    f.write(doc3_content)

loaders = [TextLoader(os.path.join(KNOWLEDGE_BASE_DIR, fn)) for fn in os.listdir(KNOWLEDGE_BASE_DIR)]
documents = []
for loader in loaders:
    documents.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

embedding_model_name = "all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

persist_directory = './chroma_db'
vectordb = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=persist_directory)
vectordb.persist()

model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,
    temperature=0.7,
    torch_dtype=model.dtype,
    device=-1
)

llm = HuggingFacePipeline(pipeline=pipe)

retriever = vectordb.as_retriever(search_kwargs={"k": 3})
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

st.set_page_config(page_title="Medical Diagnosis Assistant")
st.title("👨‍⚕️ Medical Diagnosis Assistant")
st.markdown("---")

st.write("Ask a medical question and get evidence-backed suggestions from our knowledge base.")

user_query = st.text_area("Your medical question:", "What are the common symptoms and treatment for appendicitis?")

if st.button("Get Diagnosis"):
    if user_query:
        with st.spinner("Searching and generating response..."):
            response = qa_chain.run(user_query)
            st.success("Response Generated!")
            st.write("### Diagnostic Suggestion:")
            st.write(response)
    else:
        st.warning("Please enter a medical question.")

st.markdown("---")
st.info("This assistant combines a generative AI with a medical knowledge base for grounded insights. Always consult a qualified medical professional for actual diagnosis and treatment.")

shutil.rmtree(KNOWLEDGE_BASE_DIR)
shutil.rmtree(persist_directory)