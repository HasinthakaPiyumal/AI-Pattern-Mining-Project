import networkx as nx
import streamlit as st

# --- 1. Medical Knowledge Graph (KG) Database using networkx ---
class MedicalKnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self._build_kg()

    def _build_kg(self):
        # Add Symptoms
        self.graph.add_node("Headache", type="symptom")
        self.graph.add_node("Fever", type="symptom")
        self.graph.add_node("Muscle Pain", type="symptom")
        self.graph.add_node("Cough", type="symptom")
        self.graph.add_node("Sore Throat", type="symptom")
        self.graph.add_node("Nausea", type="symptom")
        self.graph.add_node("Vomiting", type="symptom")
        self.graph.add_node("Fatigue", type="symptom")

        # Add Diseases
        self.graph.add_node("Influenza", type="disease")
        self.graph.add_node("Common Cold", type="disease")
        self.graph.add_node("Migraine", type="disease")
        self.graph.add_node("Food Poisoning", type="disease")
        self.graph.add_node("Pneumonia", type="disease")

        # Add Treatments
        self.graph.add_node("Paracetamol", type="treatment")
        self.graph.add_node("Ibuprofen", type="treatment")
        self.graph.add_node("Fluids", type="treatment")
        self.graph.add_node("Rest", type="treatment")
        self.graph.add_node("Antibiotics", type="treatment")
        self.graph.add_node("Antiemetics", type="treatment")
        self.graph.add_node("Hydration Therapy", type="treatment")

        # Add Contraindications (simplified)
        self.graph.add_node("Aspirin_Child", type="contraindication")

        # Add Relationships (Symptom -> Disease)
        self.graph.add_edge("Headache", "Influenza", relation="has_symptom")
        self.graph.add_edge("Fever", "Influenza", relation="has_symptom")
        self.graph.add_edge("Muscle Pain", "Influenza", relation="has_symptom")
        self.graph.add_edge("Cough", "Influenza", relation="has_symptom")
        self.graph.add_edge("Fatigue", "Influenza", relation="has_symptom")

        self.graph.add_edge("Headache", "Common Cold", relation="has_symptom")
        self.graph.add_edge("Cough", "Common Cold", relation="has_symptom")
        self.graph.add_edge("Sore Throat", "Common Cold", relation="has_symptom")

        self.graph.add_edge("Headache", "Migraine", relation="has_symptom")
        self.graph.add_edge("Nausea", "Migraine", relation="has_symptom")

        self.graph.add_edge("Nausea", "Food Poisoning", relation="has_symptom")
        self.graph.add_edge("Vomiting", "Food Poisoning", relation="has_symptom")
        self.graph.add_edge("Fever", "Food Poisoning", relation="has_symptom")

        self.graph.add_edge("Cough", "Pneumonia", relation="has_symptom")
        self.graph.add_edge("Fever", "Pneumonia", relation="has_symptom")
        self.graph.add_edge("Fatigue", "Pneumonia", relation="has_symptom")

        # Add Relationships (Disease -> Treatment)
        self.graph.add_edge("Influenza", "Paracetamol", relation="treats")
        self.graph.add_edge("Influenza", "Fluids", relation="treats")
        self.graph.add_edge("Influenza", "Rest", relation="treats")

        self.graph.add_edge("Common Cold", "Paracetamol", relation="treats")
        self.graph.add_edge("Common Cold", "Fluids", relation="treats")
        self.graph.add_edge("Common Cold", "Rest", relation="treats")

        self.graph.add_edge("Migraine", "Ibuprofen", relation="treats")
        self.graph.add_edge("Migraine", "Rest", relation="treats")

        self.graph.add_edge("Food Poisoning", "Antiemetics", relation="treats")
        self.graph.add_edge("Food Poisoning", "Hydration Therapy", relation="treats")
        self.graph.add_edge("Food Poisoning", "Rest", relation="treats")

        self.graph.add_edge("Pneumonia", "Antibiotics", relation="treats")
        self.graph.add_edge("Pneumonia", "Rest", relation="treats")
        self.graph.add_edge("Pneumonia", "Fluids", relation="treats")

        # Add Relationships (Treatment -> Contraindication) - simplified for demo
        # This would typically link to patient attributes or other conditions
        # For now, let's assume Aspirin is contraindicated for children
        self.graph.add_edge("Aspirin_Child", "Aspirin", relation="contraindicated_with") # Conceptual link

    def get_neighbors(self, node, relation_type=None):
        neighbors = []
        for neighbor in self.graph.neighbors(node):
            edge_data = self.graph.get_edge_data(node, neighbor)
            if relation_type is None or (edge_data and edge_data.get("relation") == relation_type):
                neighbors.append((neighbor, edge_data.get("relation") if edge_data else None))
        return neighbors

    def get_node_type(self, node):
        return self.graph.nodes[node].get("type") if node in self.graph else None


# --- 2. LLM Agent Module (Simulated) ---
class LLMAgent:
    def __init__(self, kg):
        self.kg = kg
        self.known_symptoms = [s for s, data in kg.graph.nodes(data=True) if data.get("type") == "symptom"]

    def extract_entities(self, query: str) -> dict:
        """Simulates LLM entity extraction: identifies symptoms from a natural language query."""
        extracted_symptoms = []
        query_lower = query.lower()
        for symptom in self.known_symptoms:
            if symptom.lower() in query_lower:
                extracted_symptoms.append(symptom)
        return {"symptoms": extracted_symptoms}

    def explore_kg(self, entities: dict, max_depth=3, beam_width=2) -> list:
        """Simulates LLM-guided KG exploration using a beam search approach to find reasoning paths."""
        symptoms = entities.get("symptoms", [])
        if not symptoms:
            return []

        # Initial paths: (symptom) -> []
        # Path format: [(entity1, relation1, entity2), (entity2, relation2, entity3), ...]
        # We store paths as a list of nodes for now, and reconstruct triples later
        current_beam = []
        for sym in symptoms:
            current_beam.append([(sym, self.kg.get_node_type(sym))])

        all_found_paths = []

        for depth in range(max_depth):
            next_beam = []
            for path in current_beam:
                last_node_in_path = path[-1][0]
                last_node_type = path[-1][1]

                # Extend paths from symptoms to diseases
                if last_node_type == "symptom":
                    for neighbor, relation in self.kg.get_neighbors(last_node_in_path, relation_type="has_symptom"):
                        if self.kg.get_node_type(neighbor) == "disease" and (neighbor, "disease") not in path:
                            new_path = path + [(neighbor, "disease")]
                            next_beam.append(new_path)

                # Extend paths from diseases to treatments
                elif last_node_type == "disease":
                    for neighbor, relation in self.kg.get_neighbors(last_node_in_path, relation_type="treats"):
                        if self.kg.get_node_type(neighbor) == "treatment" and (neighbor, "treatment") not in path:
                            new_path = path + [(neighbor, "treatment")]
                            next_beam.append(new_path)

                # If no extension, or if it's a terminal path, add to all_found_paths
                else:
                    all_found_paths.append(path)

            # Pruning step: Keep only the top 'beam_width' paths based on some heuristic.
            # For simplicity, we'll just take the first 'beam_width' unique paths.
            # In a real system, this would involve LLM evaluation or more complex scoring.
            current_beam = list(next_beam[:beam_width])

            if not current_beam and depth > 0: # If no paths were extended in this iteration, stop early
                break

        # Add any remaining paths in the current_beam after max_depth is reached
        all_found_paths.extend(current_beam)

        # Deduplicate and format paths as (entity, relation, entity) triples
        formatted_paths = []
        unique_paths_set = set()

        for path_nodes in all_found_paths:
            triples_for_path = []
            for i in range(len(path_nodes) - 1):
                source_node_name = path_nodes[i][0]
                target_node_name = path_nodes[i+1][0]
                edge_data = self.kg.graph.get_edge_data(source_node_name, target_node_name)
                if edge_data:
                    relation = edge_data.get("relation")
                    triples_for_path.append((source_node_name, relation, target_node_name))
            if triples_for_path:
                path_tuple = tuple(triples_for_path) # Convert to tuple for set hashing
                if path_tuple not in unique_paths_set:
                    unique_paths_set.add(path_tuple)
                    formatted_paths.append(triples_for_path)
        return formatted_paths


# --- 3. Reasoning & Generation Module (Simulated) ---
class ReasoningEngine:
    def __init__(self):
        pass

    def _format_kg_path_for_llm(self, kg_paths: list) -> str:
        """Formats retrieved KG paths into a human-readable string for simulated LLM reasoning."""
        if not kg_paths:
            return "No relevant knowledge graph paths found."

        formatted_output = []
        for i, path in enumerate(kg_paths):
            path_str = f"Path {i+1}: "
            path_elements = []
            for triple in path:
                path_elements.append(f"{triple[0]} {triple[1]} {triple[2]}")
            formatted_output.append(path_str + " -> ".join(path_elements))
        return "\n".join(formatted_output)

    def reason_and_generate(self, patient_query: str, kg_paths: list) -> dict:
        """Simulates LLM reasoning and generation based on query and KG paths.
           Generates diagnosis and treatment recommendations with explanations."""

        formatted_kg_info = self._format_kg_path_for_llm(kg_paths)
        diagnosis_suggestions = []
        treatment_suggestions = []
        explanations = []

        if not kg_paths:
            diagnosis_suggestions.append("Unable to provide specific diagnosis due to lack of relevant KG information.")
            treatment_suggestions.append("Symptomatic relief recommended. Please consult a medical professional.")
            explanations.append("No direct links found in the medical knowledge graph for the given symptoms.")
        else:
            # Simple rule-based reasoning for demonstration based on the paths
            diseases_found = set()
            treatments_found = set()
            for path in kg_paths:
                for triple in path:
                    if self.kg.get_node_type(triple[0]) == "symptom" and self.kg.get_node_type(triple[2]) == "disease" and triple[1] == "has_symptom":
                        diseases_found.add(triple[2])
                    elif self.kg.get_node_type(triple[0]) == "disease" and self.kg.get_node_type(triple[2]) == "treatment" and triple[1] == "treats":
                        treatments_found.add(triple[2])

            if diseases_found:
                diagnosis_suggestions.append(f"Potential diagnoses include: {', '.join(diseases_found)}.")
                explanations.append(f"These diagnoses are suggested because the patient's symptoms are known to be associated with: {', '.join(diseases_found)}.")
            else:
                diagnosis_suggestions.append("No clear diagnosis could be directly identified from the provided symptoms and KG paths.")

            if treatments_found:
                treatment_suggestions.append(f"Recommended treatments: {', '.join(treatments_found)}. Always consult a medical professional before starting any treatment.")
                explanations.append(f"Treatments are recommended based on their association with the potential diagnoses identified.")
            else:
                treatment_suggestions.append("No specific treatments could be identified from the KG for the potential diagnoses.")

            explanations.append(f"Knowledge Graph Paths Explored:\n{formatted_kg_info}")

        return {
            "diagnosis": diagnosis_suggestions,
            "treatment_recommendations": treatment_suggestions,
            "explanation": explanations
        }


# --- Streamlit User Interface ---
st.set_page_config(layout="wide", page_title="Intelligent Medical Diagnosis")
st.title("🧠 Intelligent Medical Diagnosis & Treatment Recommendation System")
st.markdown("--- Developed by a Knowledge-Graph Guided Agentic LLM --- ")

# Initialize KG and Agents
@st.cache_resource
def initialize_system():
    kg = MedicalKnowledgeGraph()
    llm_agent = LLMAgent(kg)
    reasoning_engine = ReasoningEngine()
    # Attach kg to reasoning engine for node type lookups during explanation generation
    reasoning_engine.kg = kg 
    return kg, llm_agent, reasoning_engine

kg, llm_agent, reasoning_engine = initialize_system()

st.subheader("Patient Query")
patient_query = st.text_area(
    "Describe the patient's symptoms and history:",
    "Patient has a headache, fever, and muscle pain. Also has a cough and feels fatigued."
)

if st.button("Get Diagnosis and Treatment Recommendations"):
    if patient_query:
        with st.spinner("Extracting entities..."):
            extracted_entities = llm_agent.extract_entities(patient_query)
            st.write("**Extracted Symptoms:**", ", ".join(extracted_entities.get("symptoms", ["None"])))

        with st.spinner("Exploring Knowledge Graph..."):
            kg_paths = llm_agent.explore_kg(extracted_entities)
            # st.write("**Discovered KG Paths (Raw):**", kg_paths) # For debugging

        with st.spinner("Generating Diagnosis and Recommendations..."):
            results = reasoning_engine.reason_and_generate(patient_query, kg_paths)

        st.subheader("Diagnosis Suggestions")
        for diag in results["diagnosis"]:
            st.success(diag)

        st.subheader("Treatment Recommendations")
        for treat in results["treatment_recommendations"]:
            st.info(treat)

        st.subheader("Explanation")
        for exp in results["explanation"]:
            st.markdown(exp)

    else:
        st.warning("Please enter a patient query to get recommendations.")

st.sidebar.subheader("How it works (Simplified)")
st.sidebar.markdown(
    "This system simulates an AI agent that uses a medical Knowledge Graph to assist with diagnoses and treatment planning."
)
st.sidebar.markdown(
    "1. **Entity Extraction:** Identifies key symptoms from your text query."
)
st.sidebar.markdown(
    "2. **KG Exploration:** Performs a 'beam search' on the graph to find multi-hop connections (e.g., Symptom -> Disease -> Treatment)."
)
st.sidebar.markdown(
    "3. **Reasoning & Generation:** Provides potential diagnoses and treatments, along with explanations based on the discovered KG paths."
)

st.sidebar.markdown("--- **Disclaimer:** This is a demonstration system and should NOT be used for actual medical advice. Always consult with a qualified medical professional. ---")
