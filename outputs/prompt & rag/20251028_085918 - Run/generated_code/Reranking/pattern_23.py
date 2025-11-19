import streamlit as st
import os
from dotenv import load_dotenv
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder

load_dotenv()

class MedicalAssistant:
    def __init__(self):
        self.embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        
        self.knowledge_base = self._initialize_knowledge_base()
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    def _initialize_knowledge_base(self):
        sample_data = [
            {"content": "Aspirin is a nonsteroidal anti-inflammatory drug (NSAID) used to reduce fever, pain, inflammation, and to prevent blood clots. Common side effects include stomach upset and increased bleeding risk.", "source": "Drug Reference Handbook"},
            {"content": "Diabetes mellitus is a chronic metabolic disease characterized by high blood sugar levels. Type 1 diabetes is an autoimmune condition, while Type 2 diabetes is often linked to lifestyle factors. Treatment involves insulin, oral medications, and lifestyle changes.", "source": "WHO Guidelines on Diabetes"},
            {"content": "Hypertension, or high blood pressure, is a common condition in which the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease. Lifestyle modifications like diet and exercise, and medications such as ACE inhibitors or diuretics, are common treatments.", "source": "Mayo Clinic"},
            {"content": "The COVID-19 vaccine helps protect against severe illness, hospitalization, and death from the virus. It works by teaching the immune system to recognize and fight the virus. Multiple doses may be required, and boosters are often recommended.", "source": "CDC Guidelines for Vaccination"},
            {"content": "Migraine is a severe type of headache characterized by throbbing pain on one side of the head, sensitivity to light and sound, and sometimes nausea or vomiting. Triggers can include stress, certain foods, and hormonal changes. Treatments range from over-the-counter pain relievers to specific migraine medications like triptans.", "source": "Neurology Journal, Vol 45, Issue 2"},
            {"content": "Common cold symptoms include a runny nose, sore throat, cough, and congestion. It is caused by viruses, mainly rhinoviruses, and usually resolves within 7-10 days. Treatment focuses on symptom relief, such as pain relievers, decongestants, and rest.", "source": "Family Health Guide"}
        ]
        
        docs = [Document(page_content=item["content"], metadata={"source": item["source"]}) for item in sample_data]
        
        return FAISS.from_documents(docs, self.embedding_model)

    def _conditional_retrieval(self, query: str) -> bool:
        # Simple heuristic: always retrieve for medical queries for this demo
        # In a real app, this could be a classifier or keyword-based check
        return True 

    def _retrieve_documents(self, query: str, k: int = 5):
        return self.knowledge_base.similarity_search(query, k=k)

    def _rerank_documents(self, query: str, documents: list[Document]) -> list[Document]:
        if not documents:
            return []
        
        sentences = [(query, doc.page_content) for doc in documents]
        scores = self.reranker.predict(sentences)
        
        ranked_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, score in ranked_docs]

    def _generate_response(self, query: str, context_docs: list[Document]) -> (str, list[str]):
        context_text = "\n\n".join([doc.page_content for doc in context_docs])
        
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful medical information assistant. Answer the user's question based on the provided medical context. If you cannot find the answer in the context, state that you don't have enough information. Always cite the sources of the information you use from the 'Source:' provided in the context."),
                ("human", "Context:\n{context}\n\nQuestion: {question}")
            ]
        )
        
        formatted_prompt = prompt_template.format_messages(context=context_text, question=query)
        response = self.llm.invoke(formatted_prompt)
        
        sources = sorted(list(set([doc.metadata["source"] for doc in context_docs if "source" in doc.metadata])))
        return response.content, sources

    def get_medical_information(self, query: str):
        if not self._conditional_retrieval(query):
            return "I can provide general information, but for specific medical queries, external knowledge retrieval is recommended. Please try rephrasing.", []
        
        retrieved_docs = self._retrieve_documents(query)
        if not retrieved_docs:
            return "I couldn't find relevant medical documents for your query in my knowledge base.", []
            
        reranked_docs = self._rerank_documents(query, retrieved_docs)
        
        response_content, sources = self._generate_response(query, reranked_docs[:3]) # Use top 3 reranked docs
        
        return response_content, sources

st.set_page_config(page_title="Medical Information Assistant", layout="centered")
st.title("🩺 Medical Information Assistant")
st.markdown("Get accurate and evidence-based answers to your medical queries. This assistant uses dynamic knowledge grounding to provide information from a simulated medical knowledge base.")

if "medical_assistant" not in st.session_state:
    st.session_state.medical_assistant = MedicalAssistant()

query = st.text_area("Enter your medical question here:", height=100)

if st.button("Get Information"):
    if query:
        with st.spinner("Searching and generating response..."):
            response, sources = st.session_state.medical_assistant.get_medical_information(query)
            st.subheader("Answer:")
            st.write(response)
            if sources:
                st.subheader("Sources:")
                for source in sources:
                    st.write(f"- {source}")
            else:
                st.info("No specific sources were cited for this response within the context provided.")
    else:
        st.warning("Please enter a medical question to get started.")