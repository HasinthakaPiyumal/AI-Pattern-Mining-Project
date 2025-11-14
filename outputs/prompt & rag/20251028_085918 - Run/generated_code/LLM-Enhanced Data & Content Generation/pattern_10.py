import streamlit as st
import os
from typing import List, Dict

# Mocking LangChain and ChromaDB components for demonstration
# In a real application, you would install and import these libraries.

# --- Mock Embedding Model --- 
# In a real scenario, this would be: 
# from langchain_community.embeddings import HuggingFaceEmbeddings
# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
class MockEmbeddings:
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Simple mock: return a list of dummy vectors
        return [[float(ord(c))/100 for c in text[:16].ljust(16, 'a')] for text in texts]

    def embed_query(self, text: str) -> List[float]:
        # Simple mock: return a dummy vector
        return [float(ord(c))/100 for c in text[:16].ljust(16, 'a')]

embeddings = MockEmbeddings()

# --- Mock ChromaDB Client --- 
# In a real scenario, this would be: 
# import chromadb
# from langchain_community.vectorstores import Chroma
# db_client = chromadb.Client()
# vectorstore = Chroma(client=db_client, collection_name="medical_knowledge", embedding_function=embeddings)

class MockChromaCollection:
    def __init__(self, name: str):
        self.name = name
        self._documents = []
        self._embeddings = []
        self._metadatas = []
        self._ids = []
        self._next_id = 0

    def add(self, documents: List[str], metadatas: List[Dict], ids: List[str] = None):
        for i, doc in enumerate(documents):
            self.add_one(doc, metadatas[i], ids[i] if ids else None)

    def add_one(self, document: str, metadata: Dict, id: str = None):
        embedding = embeddings.embed_query(document)
        self._documents.append(document)
        self._embeddings.append(embedding)
        self._metadatas.append(metadata)
        new_id = id if id else str(self._next_id)
        self._ids.append(new_id)
        self._next_id += 1
        return new_id

    def query(self, query_embeddings: List[List[float]] = None, query_texts: List[str] = None, n_results: int = 4):
        if query_texts:
            query_embeddings = [embeddings.embed_query(text) for text in query_texts]
        
        # Simple mock: just return the first n_results documents for demonstration
        # In a real ChromaDB, this would be a similarity search.
        results = {
            "ids": [self._ids[i] for i in range(min(n_results, len(self._documents)))],
            "documents": [self._documents[i] for i in range(min(n_results, len(self._documents)))],
            "metadatas": [self._metadatas[i] for i in range(min(n_results, len(self._documents)))],
            "embeddings": [self._embeddings[i] for i in range(min(n_results, len(self._documents)))],
        }
        return results

class MockChroma:
    def __init__(self, collection_name: str, embedding_function):
        self._collection = MockChromaCollection(collection_name)
        self._embedding_function = embedding_function
    
    def add_documents(self, documents: List[str], metadatas: List[Dict]):
        # LangChain Document objects are typically used, but here we just pass text and metadata
        self._collection.add(documents, metadatas)

    def as_retriever(self, search_kwargs: Dict = None):
        return MockChromaRetriever(self._collection, search_kwargs or {"k": 4})

class MockChromaRetriever:
    def __init__(self, collection: MockChromaCollection, search_kwargs: Dict):
        self._collection = collection
        self._search_kwargs = search_kwargs

    def get_relevant_documents(self, query: str) -> List[Dict]:
        n_results = self._search_kwargs.get("k", 4)
        query_embedding = embeddings.embed_query(query)
        results = self._collection.query(query_embeddings=[query_embedding], n_results=n_results)
        
        # Convert to a format similar to LangChain Document objects
        documents = []
        for doc_content, metadata in zip(results["documents"], results["metadatas"]):
            documents.append({"page_content": doc_content, "metadata": metadata})
        return documents

# Initialize mock ChromaDB
vectorstore = MockChroma(collection_name="medical_knowledge", embedding_function=embeddings)

# --- Dummy Medical Knowledge Data --- 
# In a real app, this would be scraped/parsed from real medical sources
dummy_medical_data = [
    ("Hypertension (high blood pressure) is a common condition in which the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease. Symptoms include headaches, shortness of breath, or nosebleeds, but often there are no symptoms.", {"source": "NIH", "topic": "Hypertension"}),
    ("Diabetes mellitus is a chronic condition that affects the way your body processes blood sugar (glucose). Type 1 diabetes is an autoimmune disease, while Type 2 diabetes is often linked to lifestyle factors. Common symptoms include increased thirst, frequent urination, and unexplained weight loss.", {"source": "Mayo Clinic", "topic": "Diabetes"}),
    ("Pneumonia is an infection that inflames the air sacs in one or both lungs. The air sacs may fill with fluid or pus, causing cough with phlegm or pus, fever, chills, and difficulty breathing. Various organisms, including bacteria, viruses, and fungi, can cause pneumonia.", {"source": "CDC", "topic": "Pneumonia"}),
    ("Migraine is a severe headache that often comes with symptoms like throbbing in one area of the head, nausea, vomiting, and extreme sensitivity to light and sound. Migraine attacks can cause significant pain for hours to days.", {"source": "WHO", "topic": "Migraine"}),
    ("Chronic Kidney Disease (CKD) involves a gradual loss of kidney function. Symptoms often develop slowly over time and can include swelling in the ankles, feet, or hands, headaches, and a reduced amount of urine.", {"source": "NHS", "topic": "CKD"}),
    ("The drug Lisinopril is an ACE inhibitor used to treat high blood pressure (hypertension) and heart failure. It works by relaxing blood vessels so that blood flows more easily. Common side effects include dizziness and dry cough.", {"source": "DrugBank", "topic": "Lisinopril"}),
    ("Metformin is a medication used to treat type 2 diabetes. It helps to control high blood sugar by decreasing glucose production in the liver and improving insulin sensitivity. It's often the first-line treatment for type 2 diabetes.", {"source": "MedlinePlus", "topic": "Metformin"}),
]

# Add dummy data to the vector store
vectorstore.add_documents(
    documents=[item[0] for item in dummy_medical_data],
    metadatas=[item[1] for item in dummy_medical_data]
)

# --- Mock LLM Integration --- 
# In a real scenario, this would be: 
# from langchain_openai import ChatOpenAI (or other LLM providers)
# llm = ChatOpenAI(model_name="gpt-4", temperature=0.0)

class MockLLM:
    def invoke(self, prompt: str) -> str:
        # Simple mock LLM response based on keywords
        if "hypertension" in prompt.lower() or "blood pressure" in prompt.lower():
            return "Based on the information, the patient might have hypertension. Consider prescribing Lisinopril and recommending lifestyle changes."
        elif "diabetes" in prompt.lower() or "blood sugar" in prompt.lower():
            return "The symptoms are consistent with diabetes. Further tests for blood glucose levels are recommended. Metformin could be a suitable treatment."
        elif "pneumonia" in prompt.lower() or "cough" in prompt.lower() and "fever" in prompt.lower():
            return "The patient's symptoms strongly suggest pneumonia. A chest X-ray and antibiotic treatment may be necessary."
        elif "migraine" in prompt.lower() or "headache" in prompt.lower() and "nausea" in prompt.lower():
            return "The patient's severe headache and associated symptoms are indicative of migraine. Pain relief and preventative measures should be discussed."
        elif "kidney" in prompt.lower() or "ckd" in prompt.lower():
            return "Chronic Kidney Disease should be investigated given the symptoms. Renal function tests are advised."
        else:
            # Simulate LLM reasoning for a generic query combining facts
            retrieved_info_snippet = ""
            if "retrieved context:" in prompt.lower():
                start_idx = prompt.lower().find("retrieved context:") + len("retrieved context:")
                end_idx = prompt.lower().find("patient symptoms:", start_idx)
                if end_idx != -1:
                    retrieved_info_snippet = prompt[start_idx:end_idx].strip()
                else:
                    retrieved_info_snippet = prompt[start_idx:].strip()

            if retrieved_info_snippet:
                return f"Based on the patient's data and the retrieved medical knowledge (e.g., {retrieved_info_snippet.split('.')[0].strip()}...), a differential diagnosis could include several conditions. Further investigation into specific markers is needed."
            else:
                return "I need more specific medical context or patient details to provide a precise diagnostic suggestion."

llm = MockLLM()

# --- LangChain RAG Chain (Mock) --- 
# In a real scenario, this would be: 
# from langchain.chains import RetrievalQA
# qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())

class MockRetrievalQAChain:
    def __init__(self, llm_instance, retriever_instance):
        self.llm = llm_instance
        self.retriever = retriever_instance

    def invoke(self, inputs: Dict) -> Dict:
        question = inputs["query"]
        patient_symptoms = inputs.get("patient_symptoms", "No symptoms provided.")
        patient_history = inputs.get("patient_history", "No history provided.")
        patient_lab_results = inputs.get("patient_lab_results", "No lab results provided.")

        # Step 1: Retrieve relevant medical documents
        relevant_docs = self.retriever.get_relevant_documents(question + " " + patient_symptoms)
        context = "\n".join([doc["page_content"] for doc in relevant_docs])

        # Step 2: Formulate prompt for LLM
        full_prompt = f"""You are an intelligent medical diagnostic assistant. Your goal is to provide diagnostic suggestions and potential treatment considerations based on the patient's information and retrieved medical knowledge.

Retrieved Medical Context:
{context}

Patient Symptoms:
{patient_symptoms}

Patient Medical History:
{patient_history}

Patient Lab Results:
{patient_lab_results}

Diagnostic Question: {question}

Based on the above information, what are the most likely diagnoses and recommended next steps or treatments?"""

        # Step 3: Get LLM response
        llm_response = self.llm.invoke(full_prompt)
        
        return {"result": llm_response, "source_documents": relevant_docs}

qa_chain = MockRetrievalQAChain(llm_instance=llm, retriever_instance=vectorstore.as_retriever())


# --- Streamlit User Interface --- 
st.set_page_config(layout="wide", page_title="Intelligent Medical Diagnostic Assistant")
st.title("🩺 Intelligent Medical Diagnostic Assistant")
st.markdown("--- Developed using a Unified Retrieval and Reasoning (RAG) approach ---")

st.sidebar.header("Patient Information Input")

with st.sidebar.form("patient_form"):
    st.subheader("Patient Details")
    patient_name = st.text_input("Patient Name", "John Doe")
    patient_age = st.number_input("Age", min_value=0, max_value=120, value=55)
    patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    st.markdown("--- Patient Medical Data ---")
    symptoms = st.text_area("Symptoms (e.g., 'severe headache, nausea, sensitive to light')", "Severe headache, throbbing on one side, nausea, sensitive to light and sound for 2 days.")
    history = st.text_area("Medical History (e.g., 'history of migraines, no known allergies')", "History of occasional migraines, otherwise healthy.")
    lab_results = st.text_area("Relevant Lab Results (e.g., 'blood pressure 140/90, glucose 105')", "None available at this time.")
    diagnostic_question = st.text_input("Diagnostic Question", "What is the most likely diagnosis and what treatment options should be considered?")
    
    submitted = st.form_submit_button("Get Diagnostic Suggestion")


st.subheader("Diagnostic Output")

if submitted:
    with st.spinner("Processing diagnostic query..."):
        # Prepare inputs for the RAG chain
        inputs = {
            "query": diagnostic_question,
            "patient_symptoms": symptoms,
            "patient_history": history,
            "patient_lab_results": lab_results
        }
        
        # Invoke the RAG chain
        response = qa_chain.invoke(inputs)
        
        diagnosis = response["result"]
        source_docs = response["source_documents"]

        st.success("Diagnosis complete!")
        st.markdown(f"**Patient Name:** {patient_name}  |  **Age:** {patient_age}  |  **Gender:** {patient_gender}")
        st.markdown(f"**Symptoms:** {symptoms}")
        st.markdown(f"**Diagnostic Question:** {diagnostic_question}")
        st.markdown("### Diagnostic Suggestion:")
        st.info(diagnosis)

        st.markdown("### Supporting Medical Context (from knowledge base):")
        if source_docs:
            for i, doc in enumerate(source_docs):
                st.markdown(f"**Source Document {i+1} ({doc['metadata'].get('source', 'N/A')} - {doc['metadata'].get('topic', 'N/A')}):**")
                st.code(doc['page_content'], language="text")
        else:
            st.warning("No relevant supporting documents found.")
else:
    st.info("Enter patient details and a diagnostic question in the sidebar to get a suggestion.")
