import streamlit as st
import networkx as nx
from sentence_transformers import SentenceTransformer
import chromadb
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM


def build_knowledge_graph_and_facts():
    G = nx.Graph()

    G.add_node("Phenylketonuria (PKU)", type="disease")
    G.add_node("Cystic Fibrosis (CF)", type="disease")
    G.add_node("Huntington's Disease", type="disease")
    G.add_node("Sickle Cell Anemia", type="disease")

    G.add_node("Intellectual Disability", type="symptom")
    G.add_node("Seizures", type="symptom")
    G.add_node("Fair Skin and Hair", type="symptom")
    G.add_node("Respiratory Problems", type="symptom")
    G.add_node("Digestive Issues", type="symptom")
    G.add_node("Involuntary Movements", type="symptom")
    G.add_node("Cognitive Decline", type="symptom")
    G.add_node("Chronic Pain", type="symptom")
    G.add_node("Anemia", type="symptom")

    G.add_node("PAH gene", type="gene")
    G.add_node("CFTR gene", type="gene")
    G.add_node("HTT gene", type="gene")
    G.add_node("HBB gene", type="gene")

    G.add_node("Special Diet (low phenylalanine)", type="treatment")
    G.add_node("Enzyme Replacement Therapy", type="treatment")
    G.add_node("Symptomatic Treatment", type="treatment")
    G.add_node("Bone Marrow Transplant", type="treatment")

    G.add_edge("Phenylketonuria (PKU)", "PAH gene", relation="caused_by_mutation_in")
    G.add_edge("Phenylketonuria (PKU)", "Intellectual Disability", relation="causes")
    G.add_edge("Phenylketonuria (PKU)", "Seizures", relation="causes")
    G.add_edge("Phenylketonuria (PKU)", "Fair Skin and Hair", relation="causes")
    G.add_edge("Phenylketonuria (PKU)", "Special Diet (low phenylalanine)", relation="treated_by")

    G.add_edge("Cystic Fibrosis (CF)", "CFTR gene", relation="caused_by_mutation_in")
    G.add_edge("Cystic Fibrosis (CF)", "Respiratory Problems", relation="causes")
    G.add_edge("Cystic Fibrosis (CF)", "Digestive Issues", relation="causes")
    G.add_edge("Cystic Fibrosis (CF)", "Enzyme Replacement Therapy", relation="treated_by")

    G.add_edge("Huntington's Disease", "HTT gene", relation="caused_by_mutation_in")
    G.add_edge("Huntington's Disease", "Involuntary Movements", relation="causes")
    G.add_edge("Huntington's Disease", "Cognitive Decline", relation="causes")
    G.add_edge("Huntington's Disease", "Symptomatic Treatment", relation="treated_by")

    G.add_edge("Sickle Cell Anemia", "HBB gene", relation="caused_by_mutation_in")
    G.add_edge("Sickle Cell Anemia", "Chronic Pain", relation="causes")
    G.add_edge("Sickle Cell Anemia", "Anemia", relation="causes")
    G.add_edge("Sickle Cell Anemia", "Bone Marrow Transplant", relation="treated_by")

    facts = []
    for u, v, data in G.edges(data=True):
        relation = data["relation"].replace("_", " ")
        fact_string = f"{u} {relation} {v}."
        facts.append(fact_string)
    
    for node, data in G.nodes(data=True):
        facts.append(f"{node} is a {data['type']}.")
    
    return facts

class CustomChromaRetriever:
    def __init__(self, collection, embedding_model):
        self.collection = collection
        self.embedding_model = embedding_model
    
    def get_relevant_documents(self, query: str):
        query_embedding = self.embedding_model.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=5,
            include=["documents"]
        )
        if results and results["documents"]:
            return [doc for sublist in results["documents"] for doc in sublist]
        return []

def initialize_vector_store(facts):
    model_name = "all-MiniLM-L6-v2"
    embedding_model = SentenceTransformer(model_name)

    client = chromadb.Client()
    collection_name = "rare_disease_facts"
    
    try:
        collection = client.get_or_create_collection(name=collection_name)
    except Exception as e:
        st.error(f"Error accessing ChromaDB collection: {e}")
        st.info("Attempting to delete and recreate collection (might happen if schema changes)")
        client.delete_collection(name=collection_name)
        collection = client.get_or_create_collection(name=collection_name)

    if collection.count() == 0:
        st.info("Populating vector store with knowledge graph facts...")
        embeddings = embedding_model.encode(facts).tolist()
        ids = [f"fact_{i}" for i in range(len(facts))]
        
        collection.add(
            documents=facts,
            embeddings=embeddings,
            ids=ids
        )
        st.success(f"Added {len(facts)} facts to the vector store.")
    else:
        st.info(f"Vector store already contains {collection.count()} facts.")

    return CustomChromaRetriever(collection, embedding_model)

def initialize_llm_chain(retriever):
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.95
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    template = """
    You are a clinical decision support system. Your goal is to suggest possible rare disease diagnoses based on patient symptoms and provided medical facts.
    If the facts do not contain enough information, state that you cannot provide a confident diagnosis.

    Medical Facts:
    {context}

    Patient Symptoms: {question}

    Based on the symptoms and the medical facts, what is the most probable rare disease diagnosis and why?
    Provide a concise answer, listing the diagnosis and the supporting facts.
    """
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
        verbose=True
    )
    return qa_chain

st.title("Rare Disease Diagnosis CDSS (RAG with KG)")

st.markdown("""
This system helps doctors suggest possible rare disease diagnoses by leveraging
a knowledge graph of medical facts and a Large Language Model.
Enter patient symptoms below to get a diagnostic suggestion.
""")

if "retriever" not in st.session_state:
    st.info("Initializing knowledge graph and vector store...")
    kg_facts = build_knowledge_graph_and_facts()
    st.session_state.retriever = initialize_vector_store(kg_facts)
    st.success("Knowledge graph and vector store initialized.")

if "qa_chain" not in st.session_state:
    st.info("Initializing LLM pipeline...")
    st.session_state.qa_chain = initialize_llm_chain(st.session_state.retriever)
    st.success("LLM pipeline initialized.")

patient_symptoms = st.text_area("Enter patient symptoms (e.g., 'intellectual disability, seizures, fair skin and hair'):", height=150)

if st.button("Get Diagnosis Suggestion"):
    if patient_symptoms:
        with st.spinner("Analyzing symptoms and retrieving facts..."):
            try:
                response = st.session_state.qa_chain.invoke({"query": patient_symptoms})
                
                st.subheader("Diagnostic Suggestion:")
                st.write(response["result"])

                st.subheader("Supporting Medical Facts:")
                for doc in response["source_documents"]:
                    st.markdown(f"- {doc}")

            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.warning("Please try again or check the console for more details.")
    else:
        st.warning("Please enter some symptoms to get a diagnosis.")
