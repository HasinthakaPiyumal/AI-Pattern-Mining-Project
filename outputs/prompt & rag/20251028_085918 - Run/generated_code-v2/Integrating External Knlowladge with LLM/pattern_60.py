import streamlit as st
import networkx as nx
import random

def build_medical_kg():
    kg = nx.DiGraph()

    # Entities
    diseases = ["Influenza", "Common Cold", "Pneumonia", "Bronchitis", "COVID-19", "Allergies"]
    symptoms = ["Fever", "Cough", "Sore Throat", "Runny Nose", "Headache", "Body Aches", "Fatigue", "Shortness of Breath", "Sneezing", "Itchy Eyes"]
    treatments = ["Paracetamol", "Ibuprofen", "Antihistamines", "Antibiotics", "Rest", "Fluids", "Oxygen Therapy"]
    drugs = ["Amoxicillin", "Azithromycin", "Cetirizine"]

    kg.add_nodes_from(diseases, type="disease")
    kg.add_nodes_from(symptoms, type="symptom")
    kg.add_nodes_from(treatments, type="treatment")
    kg.add_nodes_from(drugs, type="drug")

    # Relationships
    # Influenza
    kg.add_edge("Fever", "Influenza", relation="HAS_SYMPTOM")
    kg.add_edge("Cough", "Influenza", relation="HAS_SYMPTOM")
    kg.add_edge("Body Aches", "Influenza", relation="HAS_SYMPTOM")
    kg.add_edge("Fatigue", "Influenza", relation="HAS_SYMPTOM")
    kg.add_edge("Influenza", "Paracetamol", relation="TREATED_BY")
    kg.add_edge("Influenza", "Ibuprofen", relation="TREATED_BY")
    kg.add_edge("Influenza", "Rest", relation="RECOMMENDED_FOR")

    # Common Cold
    kg.add_edge("Runny Nose", "Common Cold", relation="HAS_SYMPTOM")
    kg.add_edge("Sore Throat", "Common Cold", relation="HAS_SYMPTOM")
    kg.add_edge("Sneezing", "Common Cold", relation="HAS_SYMPTOM")
    kg.add_edge("Common Cold", "Rest", relation="RECOMMENDED_FOR")
    kg.add_edge("Common Cold", "Fluids", relation="RECOMMENDED_FOR")

    # Pneumonia
    kg.add_edge("Cough", "Pneumonia", relation="HAS_SYMPTOM")
    kg.add_edge("Fever", "Pneumonia", relation="HAS_SYMPTOM")
    kg.add_edge("Shortness of Breath", "Pneumonia", relation="HAS_SYMPTOM")
    kg.add_edge("Pneumonia", "Antibiotics", relation="TREATED_BY")
    kg.add_edge("Antibiotics", "Amoxicillin", relation="IS_A")
    kg.add_edge("Pneumonia", "Oxygen Therapy", relation="MAY_REQUIRE")

    # Bronchitis
    kg.add_edge("Cough", "Bronchitis", relation="HAS_SYMPTOM")
    kg.add_edge("Sore Throat", "Bronchitis", relation="HAS_SYMPTOM")
    kg.add_edge("Bronchitis", "Rest", relation="RECOMMENDED_FOR")

    # COVID-19
    kg.add_edge("Fever", "COVID-19", relation="HAS_SYMPTOM")
    kg.add_edge("Cough", "COVID-19", relation="HAS_SYMPTOM")
    kg.add_edge("Shortness of Breath", "COVID-19", relation="HAS_SYMPTOM")
    kg.add_edge("Fatigue", "COVID-19", relation="HAS_SYMPTOM")
    kg.add_edge("COVID-19", "Rest", relation="RECOMMENDED_FOR")
    kg.add_edge("COVID-19", "Fluids", relation="RECOMMENDED_FOR")

    # Allergies
    kg.add_edge("Sneezing", "Allergies", relation="HAS_SYMPTOM")
    kg.add_edge("Runny Nose", "Allergies", relation="HAS_SYMPTOM")
    kg.add_edge("Itchy Eyes", "Allergies", relation="HAS_SYMPTOM")
    kg.add_edge("Allergies", "Antihistamines", relation="TREATED_BY")
    kg.add_edge("Antihistamines", "Cetirizine", relation="IS_A")

    return kg

def find_reasoning_paths(kg, symptoms, max_hops=3, num_paths_per_symptom=2):
    relevant_paths = []
    symptom_nodes = [s for s in symptoms if s in kg.nodes]

    for symptom_node in symptom_nodes:
        paths_found = 0
        for neighbor in list(kg.successors(symptom_node)) + list(kg.predecessors(symptom_node)):
            for path in nx.all_simple_paths(kg, source=symptom_node, target=neighbor, cutoff=max_hops):
                if len(path) > 1 and paths_found < num_paths_per_symptom:
                    formatted_path = []
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i+1]
                        if kg.has_edge(u, v):
                            relation = kg.get_edge_data(u, v)['relation']
                            formatted_path.append(f"({u}, {relation}, {v})")
                        elif kg.has_edge(v, u):
                            relation = kg.get_edge_data(v, u)['relation']
                            formatted_path.append(f"({v}, {relation}, {u})")
                    if formatted_path:
                        relevant_paths.append("; ".join(formatted_path))
                        paths_found += 1
    return relevant_paths

def construct_prompt(patient_history, symptoms, kg_triples):
    prompt_parts = []
    prompt_parts.append(f"Patient History: {patient_history}")
    prompt_parts.append(f"Current Symptoms: {', '.join(symptoms)}")
    prompt_parts.append("Relevant medical knowledge (entity-relation-entity triples):")
    for triple_path in kg_triples:
        prompt_parts.append(f"- {triple_path}")
    prompt_parts.append("Based on the patient's history, symptoms, and the provided medical knowledge, please generate a potential diagnosis and a concise, explainable reasoning path. Also, suggest potential treatments or further tests.")
    return "\n".join(prompt_parts)

def mock_llm_response(prompt):
    if "Pneumonia" in prompt and "Antibiotics" in prompt:
        return {
            "diagnosis": "Pneumonia",
            "explanation": "Based on symptoms like Fever, Cough, Shortness of Breath and the knowledge that (Cough, HAS_SYMPTOM, Pneumonia); (Fever, HAS_SYMPTOM, Pneumonia); (Shortness of Breath, HAS_SYMPTOM, Pneumonia), a diagnosis of Pneumonia is indicated. Treatment with Antibiotics like Amoxicillin is commonly prescribed.",
            "recommendations": "Prescribe Amoxicillin or Azithromycin. Monitor oxygen levels. Consider chest X-ray."
        }
    elif "Influenza" in prompt and "Paracetamol" in prompt:
        return {
            "diagnosis": "Influenza",
            "explanation": "The presence of Fever, Cough, and Body Aches aligns with Influenza. Knowledge (Fever, HAS_SYMPTOM, Influenza); (Cough, HAS_SYMPTOM, Influenza); (Body Aches, HAS_SYMPTOM, Influenza) supports this. Rest and symptom relief with Paracetamol are recommended.",
            "recommendations": "Recommend rest, increased fluid intake, and over-the-counter pain relievers like Paracetamol."
        }
    elif "Allergies" in prompt and "Antihistamines" in prompt:
        return {
            "diagnosis": "Allergies",
            "explanation": "Symptoms like Sneezing, Runny Nose, and Itchy Eyes are typical for Allergies. The knowledge (Sneezing, HAS_SYMPTOM, Allergies); (Runny Nose, HAS_SYMPTOM, Allergies); (Itchy Eyes, HAS_SYMPTOM, Allergies) points to this. Antihistamines are a common treatment.",
            "recommendations": "Suggest antihistamines like Cetirizine. Advise on avoiding allergens if known."
        }
    else:
        return {
            "diagnosis": "Undetermined",
            "explanation": "Insufficient information or complex case. More data needed.",
            "recommendations": "Consult a specialist for further evaluation. Perform more specific diagnostic tests."
        }

def main():
    st.title("Medical Diagnosis Assistant with Explainable AI")
    st.write("Enter patient symptoms and history to get a potential diagnosis and explanation.")

    kg = build_medical_kg()

    patient_history = st.text_area("Patient Medical History (e.g., 'No significant past medical history. Smoker for 10 years.')")
    symptoms_input = st.text_input("Current Symptoms (comma-separated, e.g., 'Fever, Cough, Shortness of Breath')")

    if st.button("Get Diagnosis"):
        if not symptoms_input:
            st.warning("Please enter at least one symptom.")
        else:
            symptoms = [s.strip() for s in symptoms_input.split(',') if s.strip()]

            st.subheader("1. Knowledge Retrieval from KG")
            st.write("Searching medical knowledge graph for relevant paths...")
            kg_triples = find_reasoning_paths(kg, symptoms)
            if kg_triples:
                st.write("Found relevant knowledge graph triples:")
                for triple in kg_triples:
                    st.text(f"- {triple}")
            else:
                st.write("No direct reasoning paths found for the given symptoms in the KG.")

            st.subheader("2. LLM Prompt Construction")
            prompt = construct_prompt(patient_history, symptoms, kg_triples)
            st.text_area("Prompt sent to LLM:", prompt, height=300)

            st.subheader("3. LLM Diagnosis and Explanation")
            st.write("Querying Large Language Model for diagnosis...")
            llm_output = mock_llm_response(prompt)

            st.success(f"Potential Diagnosis: {llm_output['diagnosis']}")
            st.write("### Explanation:")
            st.info(llm_output['explanation'])
            st.write("### Recommendations:")
            st.warning(llm_output['recommendations'])

if __name__ == "__main__":
    main()