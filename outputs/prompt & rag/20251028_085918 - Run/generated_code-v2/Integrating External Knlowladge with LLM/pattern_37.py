import streamlit as st
import networkx as nx
from pydantic import BaseModel, Field
import json


class MedicalEntities(BaseModel):
    diseases: list[str] = Field(default_factory=list, description="List of diseases mentioned in the text")
    drugs: list[str] = Field(default_factory=list, description="List of drugs mentioned in the text")
    genes: list[str] = Field(default_factory=list, description="List of genes mentioned in the text")
    symptoms: list[str] = Field(default_factory=list, description="List of symptoms mentioned in the text")
    treatments: list[str] = Field(default_factory=list, description="List of treatments mentioned in the text")


def mock_llm_entity_extraction(question: str) -> MedicalEntities:
    """Simulates an LLM extracting medical entities based on keywords."""
    # In a real system, this would involve a call to an actual LLM (e.g., via transformers/langchain)
    # and parsing its structured output. For this demo, we use keyword matching.
    
    question_lower = question.lower()
    entities = MedicalEntities()

    # Simple keyword matching for demonstration
    if "hypertension" in question_lower or "high blood pressure" in question_lower:
        entities.diseases.append("Hypertension")
    if "diabetes" in question_lower:
        entities.diseases.append("Diabetes")
    if "cancer" in question_lower:
        entities.diseases.append("Cancer")
    if "migraine" in question_lower:
        entities.diseases.append("Migraine")
    if "asthma" in question_lower:
        entities.diseases.append("Asthma")

    if "lisinopril" in question_lower:
        entities.drugs.append("Lisinopril")
    if "metformin" in question_lower:
        entities.drugs.append("Metformin")
    if "ibuprofen" in question_lower:
        entities.drugs.append("Ibuprofen")
    if "albuterol" in question_lower:
        entities.drugs.append("Albuterol")

    if "apoe gene" in question_lower or "apoe4" in question_lower:
        entities.genes.append("APOE gene")
    if "brca1" in question_lower:
        entities.genes.append("BRCA1 gene")

    if "headache" in question_lower:
        entities.symptoms.append("Headache")
    if "fatigue" in question_lower:
        entities.symptoms.append("Fatigue")
    if "chest pain" in question_lower:
        entities.symptoms.append("Chest Pain")
    if "cough" in question_lower:
        entities.symptoms.append("Cough")

    if "diet" in question_lower or "lifestyle changes" in question_lower:
        entities.treatments.append("Diet and Lifestyle Changes")
    if "chemotherapy" in question_lower:
        entities.treatments.append("Chemotherapy")
    if "inhaler" in question_lower:
        entities.treatments.append("Inhaler")

    return entities


def create_medical_knowledge_graph():
    """Creates a simplified medical knowledge graph using networkx."""
    G = nx.Graph()

    # Add nodes (medical concepts)
    G.add_nodes_from([
        "Hypertension", "Diabetes", "Cancer", "Migraine", "Asthma",
        "Lisinopril", "Metformin", "Ibuprofen", "Albuterol",
        "APOE gene", "BRCA1 gene",
        "Headache", "Fatigue", "Chest Pain", "Cough",
        "Diet and Lifestyle Changes", "Chemotherapy", "Inhaler",
        "Kidney Disease", "Heart Disease", "Obesity", "Alzheimer's Disease",
        "Breast Cancer", "Lung Cancer", "Insulin Resistance",
        "Stroke", "Anxiety", "Depression"
    ])

    # Add edges (relationships)
    G.add_edge("Hypertension", "Lisinopril", relation="treats")
    G.add_edge("Hypertension", "Heart Disease", relation="causes")
    G.add_edge("Hypertension", "Kidney Disease", relation="causes")
    G.add_edge("Hypertension", "Diet and Lifestyle Changes", relation="managed_by")

    G.add_edge("Diabetes", "Metformin", relation="treats")
    G.add_edge("Diabetes", "Obesity", relation="associated_with")
    G.add_edge("Diabetes", "Heart Disease", relation="causes")
    G.add_edge("Diabetes", "Diet and Lifestyle Changes", relation="managed_by")
    G.add_edge("Diabetes", "Insulin Resistance", relation="characterized_by")

    G.add_edge("Cancer", "Chemotherapy", relation="treats")
    G.add_edge("BRCA1 gene", "Breast Cancer", relation="increases_risk_of")
    G.add_edge("Cancer", "Fatigue", relation="symptom")

    G.add_edge("Migraine", "Headache", relation="symptom")
    G.add_edge("Migraine", "Ibuprofen", relation="treats")

    G.add_edge("Asthma", "Albuterol", relation="treats")
    G.add_edge("Asthma", "Inhaler", relation="device_for_treatment")
    G.add_edge("Asthma", "Cough", relation="symptom")

    G.add_edge("APOE gene", "Alzheimer's Disease", relation="increases_risk_of")

    G.add_edge("Headache", "Anxiety", relation="associated_with")
    G.add_edge("Fatigue", "Depression", relation="associated_with")

    return G

def retrieve_kg_info(kg: nx.Graph, entities: MedicalEntities, depth: int = 2):
    """Searches the knowledge graph for information related to extracted entities."""
    retrieved_info = []
    search_entities = []
    search_entities.extend(entities.diseases)
    search_entities.extend(entities.drugs)
    search_entities.extend(entities.genes)
    search_entities.extend(entities.symptoms)
    search_entities.extend(entities.treatments)
    
    found_nodes = set()

    for entity in search_entities:
        if entity in kg:
            found_nodes.add(entity)
            
            # Perform a limited-depth BFS from the entity
            for source, target in nx.bfs_edges(kg, source=entity, depth_limit=depth):
                if source == entity:
                    # Directly connected nodes
                    relation = kg[source][target].get("relation", "related to")
                    retrieved_info.append(f"{source} {relation} {target}")
                else:
                    # Indirect connections within depth limit
                    relation = kg[source][target].get("relation", "related to")
                    path_info = f"Path: {entity} -> ... -> {source} {relation} {target}"
                    if path_info not in retrieved_info: # Avoid duplicate path info for simplicity
                        retrieved_info.append(path_info)
    
    if not retrieved_info and found_nodes:
        retrieved_info.append(f"No direct or indirect relations found within depth {depth} for: {', '.join(list(found_nodes))}")
    elif not found_nodes:
        retrieved_info.append("No matching entities found in the knowledge graph.")

    return retrieved_info


# --- Streamlit Application ---
st.set_page_config(layout="wide", page_title="Medical QA System with LLM Entity Extraction")
st.title("Medical Research Question Answering System")
st.markdown("Ask a medical research question and get answers from a simulated knowledge graph.")

# Initialize KG (run once)
if 'kg' not in st.session_state:
    st.session_state.kg = create_medical_knowledge_graph()

question = st.text_area("Enter your medical research question here:", height=100)

if st.button("Get Answers") and question:
    st.subheader("1. LLM-based Entity Extraction")
    with st.spinner("Extracting entities..."):
        extracted_entities = mock_llm_entity_extraction(question)
    
    st.write("**Extracted Medical Entities:**")
    st.json(extracted_entities.model_dump_json(indent=2))

    if any(getattr(extracted_entities, field) for field in extracted_entities.model_fields):
        st.subheader("2. Knowledge Graph Search and Information Retrieval")
        with st.spinner("Searching knowledge graph..."):
            kg_info = retrieve_kg_info(st.session_state.kg, extracted_entities, depth=2)
        
        st.write("**Retrieved Information from Knowledge Graph:**")
        if kg_info:
            for item in kg_info:
                st.write(f"- {item}")
        else:
            st.write("No relevant information found based on extracted entities.")
    else:
        st.warning("No medical entities were extracted from your question. Please try a different query.")

