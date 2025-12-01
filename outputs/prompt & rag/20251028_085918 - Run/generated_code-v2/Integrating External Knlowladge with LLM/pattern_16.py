import streamlit as st
import openai
import networkx as nx
import json
import os

# Configure OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# 2a. LLM-based Entity Extractor
def extract_medical_entities(query: str) -> list[str]:
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert medical assistant. Extract key medical entities from the user's query and return them as a JSON list of strings."},
                {"role": "user", "content": f"Extract medical entities from this question: '{query}'. Examples: disease, symptom, drug, anatomical structure, procedure."}
            ],
            response_format={ "type": "json_object" }
        )
        content = response.choices[0].message.content
        entities = json.loads(content).get("entities", [])
        return entities if isinstance(entities, list) else []
    except Exception as e:
        st.error(f"Error extracting entities: {e}")
        return []

# 3. Medical Knowledge Graph (Conceptual/Demo)
def create_medical_knowledge_graph():
    kg = nx.Graph()

    # Nodes: Diseases, Symptoms, Treatments, Drugs, Organs
    kg.add_nodes_from([
        "Diabetes Mellitus Type 2", "Insulin Resistance", "High Blood Sugar", "Pancreas", "Metformin", "Lifestyle Changes",
        "Hypertension", "High Blood Pressure", "Headache", "Dizziness", "Diuretics", "ACE Inhibitors", "Kidneys",
        "Asthma", "Wheezing", "Shortness of Breath", "Bronchodilators", "Corticosteroids", "Lungs",
        "Fever", "Cough", "Sore Throat", "Influenza", "Antivirals", "Rest", "Hydration"
    ])

    # Edges: Relationships
    kg.add_edge("Diabetes Mellitus Type 2", "Insulin Resistance", relation="caused_by")
    kg.add_edge("Insulin Resistance", "High Blood Sugar", relation="leads_to")
    kg.add_edge("High Blood Sugar", "Diabetes Mellitus Type 2", relation="symptom_of")
    kg.add_edge("Pancreas", "Insulin Resistance", relation="affects")
    kg.add_edge("Metformin", "Diabetes Mellitus Type 2", relation="treats")
    kg.add_edge("Lifestyle Changes", "Diabetes Mellitus Type 2", relation="manages")

    kg.add_edge("Hypertension", "High Blood Pressure", relation="is_a")
    kg.add_edge("High Blood Pressure", "Headache", relation="symptom")
    kg.add_edge("High Blood Pressure", "Dizziness", relation="symptom")
    kg.add_edge("Diuretics", "Hypertension", relation="treats")
    kg.add_edge("ACE Inhibitors", "Hypertension", relation="treats")
    kg.add_edge("Kidneys", "Hypertension", relation="affected_by")

    kg.add_edge("Asthma", "Wheezing", relation="symptom_of")
    kg.add_edge("Asthma", "Shortness of Breath", relation="symptom_of")
    kg.add_edge("Bronchodilators", "Asthma", relation="treats")
    kg.add_edge("Corticosteroids", "Asthma", relation="treats")
    kg.add_edge("Lungs", "Asthma", relation="affected_organ")

    kg.add_edge("Influenza", "Fever", relation="symptom_of")
    kg.add_edge("Influenza", "Cough", relation="symptom_of")
    kg.add_edge("Influenza", "Sore Throat", relation="symptom_of")
    kg.add_edge("Antivirals", "Influenza", relation="treats")
    kg.add_edge("Rest", "Influenza", relation="manages")
    kg.add_edge("Hydration", "Influenza", relation="manages")

    return kg

# 4. Knowledge Graph Traversal & Reasoning Engine
def search_knowledge_graph(entities: list[str], kg: nx.Graph) -> dict:
    results = {}
    for entity in entities:
        if entity in kg.nodes:
            related_info = []
            for neighbor in kg.neighbors(entity):
                relation = kg.get_edge_data(entity, neighbor)['relation']
                related_info.append(f"{entity} {relation} {neighbor}")
            results[entity] = related_info
    return results

# 5. Response Synthesizer
def synthesize_response(kg_results: dict, original_query: str) -> str:
    if not kg_results:
        return "I couldn't find specific information related to your query in the knowledge graph. Please try rephrasing or asking about common medical terms."

    response_parts = [f"Based on your query '{original_query}', here's what I found in the medical knowledge graph:"]
    for entity, info_list in kg_results.items():
        response_parts.append(f"\nFor '{entity}':")
        if info_list:
            for info in info_list:
                response_parts.append(f"  - {info}")
        else:
            response_parts.append(f"  - No direct related information found for '{entity}'.")

    return "\n".join(response_parts)

# 1. User Interface (Streamlit)
st.set_page_config(page_title="Medical Knowledge Navigator")
st.title("Medical Knowledge Navigator")
st.write("Ask a natural language question about medical conditions, treatments, or symptoms.")

medical_question = st.text_area("Enter your medical question:", height=100)

if st.button("Search Knowledge Graph"):
    if medical_question:
        with st.spinner("Extracting entities and searching knowledge graph..."):
            # Extract entities
            extracted_entities = extract_medical_entities(medical_question)
            st.subheader("Extracted Entities:")
            if extracted_entities:
                st.write(extracted_entities)
            else:
                st.write("No key medical entities extracted.")

            # Create and search KG (for demo, recreate each time)
            medical_kg = create_medical_knowledge_graph()
            kg_search_results = search_knowledge_graph(extracted_entities, medical_kg)

            # Synthesize response
            final_response = synthesize_response(kg_search_results, medical_question)

            st.subheader("Knowledge Graph Insights:")
            st.info(final_response)
    else:
        st.warning("Please enter a medical question to search.")