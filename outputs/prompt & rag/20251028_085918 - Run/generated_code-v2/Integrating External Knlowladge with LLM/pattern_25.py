import streamlit as st
import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 1. Medical Knowledge Graph (Simplified with NetworkX)
G = nx.Graph()

G.add_node("Symptom: Fever", type="symptom", description="Elevated body temperature")
G.add_node("Symptom: Cough", type="symptom", description="Sudden expulsion of air from lungs")
G.add_node("Disease: Flu", type="disease", description="Influenza, a common viral infection")
G.add_node("Disease: Common Cold", type="disease", description="Viral infection of the nose and throat")
G.add_node("Treatment: Rest", type="treatment", description="Adequate sleep and reduced activity")
G.add_node("Treatment: Fluids", type="treatment", description="Drinking plenty of water and other liquids")
G.add_node("Drug: Paracetamol", type="drug", description="Pain reliever and fever reducer")

G.add_edge("Symptom: Fever", "Disease: Flu", relation="indicates")
G.add_edge("Symptom: Cough", "Disease: Flu", relation="indicates")
G.add_edge("Symptom: Fever", "Disease: Common Cold", relation="indicates")
G.add_edge("Symptom: Cough", "Disease: Common Cold", relation="indicates")
G.add_edge("Disease: Flu", "Treatment: Rest", relation="recommends")
G.add_edge("Disease: Flu", "Treatment: Fluids", relation="recommends")
G.add_edge("Disease: Flu", "Drug: Paracetamol", relation="prescribes")
G.add_edge("Disease: Common Cold", "Treatment: Rest", relation="recommends")
G.add_edge("Disease: Common Cold", "Treatment: Fluids", relation="recommends")

# Prepare text representations for embedding
kg_texts = []
kg_entities = []
for node, data in G.nodes(data=True):
    text = f"{data.get('type', 'entity')}: {node}. {data.get('description', '')}"
    kg_texts.append(text)
    kg_entities.append(node)

for u, v, data in G.edges(data=True):
    text = f"{u} {data.get('relation', 'has_relation')} {v}"
    kg_texts.append(text)
    kg_entities.append(text) # Storing the relation text as an entity for retrieval

# 2. KG Embedding Model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_embedding_model()
kg_embeddings = embedding_model.encode(kg_texts, convert_to_tensor=True)

# 3. Information Retrieval Module (Simplified with Cosine Similarity)
def retrieve_facts(query_embedding, top_k=3):
    similarities = cosine_similarity(query_embedding.cpu().numpy(), kg_embeddings.cpu().numpy())
    top_indices = np.argsort(similarities[0])[-top_k:][::-1]
    retrieved_facts = [kg_texts[i] for i in top_indices]
    return retrieved_facts

# 4. Large Language Model (LLM Placeholder)
def llm_generate_response(prompt):
    # This is a placeholder. In a real application, you'd use a model like GPT-4, Llama 2, etc.
    if "Flu" in prompt and "Paracetamol" in prompt:
        return "Based on the symptoms and knowledge graph, it is likely the patient has Flu. Treatment typically involves rest, fluids, and Paracetamol for symptom relief."
    elif "Common Cold" in prompt and "Rest" in prompt:
        return "Considering the symptoms and retrieved facts, a Common Cold is probable. Recommended treatments include rest and fluids."
    else:
        return "I am a simulated LLM. Based on the provided context, I can infer some medical information. Please note this is not real medical advice."

# 5. RAG Orchestration
def rag_pipeline(query):
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    retrieved_facts = retrieve_facts(query_embedding)

    context = "\n".join(retrieved_facts)
    prompt = f"Context from Medical Knowledge Graph:\n{context}\n\nPatient Query: {query}\n\nBased on the context, provide a medical opinion or recommendation."

    llm_response = llm_generate_response(prompt)
    return llm_response, retrieved_facts

# 6. User Interface (Streamlit)
st.set_page_config(layout="wide", page_title="Clinical Decision Support System")
st.title("🩺 Clinical Decision Support System (RAG for KGs)")
st.markdown("This system provides medical insights by combining a Knowledge Graph with a Large Language Model to reduce hallucinations.")

# Sidebar for KG visualization (optional, for demonstration)
st.sidebar.header("Medical Knowledge Graph Preview")
st.sidebar.write("Nodes (Entities):")
for node, data in G.nodes(data=True):
    st.sidebar.text(f"- {node} ({data.get('type', 'unknown')})")
st.sidebar.write("Edges (Relations):")
for u, v, data in G.edges(data=True):
    st.sidebar.text(f"- {u} --({data.get('relation', 'relates_to')})--> {v}")

user_query = st.text_area("Enter patient symptoms or a clinical question:", "Patient has fever and cough. What could be the diagnosis and treatment?")

if st.button("Get Medical Insight"):
    if user_query:
        with st.spinner("Processing query and retrieving insights..."):
            response, facts = rag_pipeline(user_query)
        
        st.subheader("Generated Medical Insight:")
        st.write(response)
        
        st.subheader("Retrieved Facts from Knowledge Graph:")
        for fact in facts:
            st.write(f"- {fact}")
    else:
        st.warning("Please enter a query.")
