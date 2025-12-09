import json
import networkx as nx
import streamlit as st
import random
from typing import List, Tuple, Dict, Any, Optional

# prompts.py content
ENTITY_EXTRACTION_PROMPT = """
Extract key medical entities (symptoms, conditions, patient demographics, lab results) from the following patient case description. List them as a comma-separated string.
Patient Case: {patient_case_description}
Entities: """

EXPLORATION_SUGGESTION_PROMPT = """
Given the current medical reasoning path and patient context, suggest relevant medical relations and potential next entities to explore in a Knowledge Graph to extend this path. Focus on plausible connections that lead to a diagnosis.
Current Path: {current_path}
Patient Context: {patient_context}
Suggest potential (relation, entity) pairs, one per line, formatted as 'relation,entity'.
Suggestions: """

PATH_PRUNING_PROMPT = """
Evaluate the following candidate medical reasoning paths based on their relevance and probability given the patient's context. Rank them from most probable to least probable. List the top {beam_width} paths. Each path is a sequence of (entity, relation, entity) tuples.
Candidate Paths: {candidate_paths}
Patient Context: {patient_context}
Ranked Paths (comma-separated entity->relation->entity):
"""

DIAGNOSIS_GENERATION_PROMPT = """
Based on the following medical reasoning paths and patient context, provide a probable diagnosis and a detailed, explainable reasoning. Explain why each step in the path is relevant.
Reasoning Paths: {reasoning_paths}
Patient Context: {patient_context}
Diagnosis and Explanation: """

# medical_kg.json content (simplified and embedded)
MOCK_KG_DATA = {
    "nodes": [
        {"id": "Fever", "type": "Symptom"},
        {"id": "Cough", "type": "Symptom"},
        {"id": "Headache", "type": "Symptom"},
        {"id": "Fatigue", "type": "Symptom"},
        {"id": "Nausea", "type": "Symptom"},
        {"id": "Influenza", "type": "Disease"},
        {"id": "Common Cold", "type": "Disease"},
        {"id": "Migraine", "type": "Disease"},
        {"id": "Pneumonia", "type": "Disease"},
        {"id": "COVID-19", "type": "Disease"},
        {"id": "Antibiotics", "type": "Treatment"},
        {"id": "Antivirals", "type": "Treatment"},
        {"id": "Pain Relievers", "type": "Treatment"}
    ],
    "edges": [
        {"source": "Fever", "target": "Influenza", "relation": "presents_with"},
        {"source": "Cough", "target": "Influenza", "relation": "presents_with"},
        {"source": "Fatigue", "target": "Influenza", "relation": "presents_with"},
        {"source": "Fever", "target": "Common Cold", "relation": "presents_with"},
        {"source": "Cough", "target": "Common Cold", "relation": "presents_with"},
        {"source": "Headache", "target": "Migraine", "relation": "presents_with"},
        {"source": "Nausea", "target": "Migraine", "relation": "presents_with"},
        {"source": "Cough", "target": "Pneumonia", "relation": "presents_with"},
        {"source": "Fever", "target": "Pneumonia", "relation": "presents_with"},
        {"source": "Influenza", "target": "Antivirals", "relation": "treatable_by"},
        {"source": "Common Cold", "target": "Pain Relievers", "relation": "treatable_by"},
        {"source": "Migraine", "target": "Pain Relievers", "relation": "treatable_by"},
        {"source": "Pneumonia", "target": "Antibiotics", "relation": "treatable_by"},
        {"source": "Fever", "target": "COVID-19", "relation": "presents_with"},
        {"source": "Cough", "target": "COVID-19", "relation": "presents_with"},
        {"source": "Fatigue", "target": "COVID-19", "relation": "presents_with"}
    ]
}

# knowledge_graph.py content
class KnowledgeGraph:
    def __init__(self, kg_data: Dict[str, List[Dict[str, Any]]]):
        self.graph = nx.DiGraph()
        for node in kg_data["nodes"]:
            self.graph.add_node(node["id"], **{k: v for k, v in node.items() if k != "id"})
        for edge in kg_data["edges"]:
            self.graph.add_edge(edge["source"], edge["target"], relation=edge["relation"])

    def get_neighbors(self, entity: str, relation_type: Optional[str] = None) -> List[Tuple[str, str, str]]:
        neighbors = []
        if entity not in self.graph:
            return neighbors

        for neighbor in self.graph.neighbors(entity):
            edge_data = self.graph.get_edge_data(entity, neighbor)
            if edge_data and "relation" in edge_data:
                if relation_type is None or edge_data["relation"] == relation_type:
                    neighbors.append((entity, edge_data["relation"], neighbor))
        return neighbors

    def check_path_validity(self, path: List[Tuple[str, str, str]]) -> bool:
        if not path:
            return True
        for i in range(len(path) - 1):
            if path[i][2] != path[i+1][0]:
                return False
            if not self.graph.has_edge(path[i][0], path[i][2]):
                return False
            if self.graph.get_edge_data(path[i][0], path[i][2]).get("relation") != path[i][1]:
                return False
        if not self.graph.has_edge(path[-1][0], path[-1][2]):
            return False
        if self.graph.get_edge_data(path[-1][0], path[-1][2]).get("relation") != path[-1][1]:
            return False
        return True

    def get_entity_details(self, entity: str) -> Dict[str, Any]:
        if entity in self.graph:
            return self.graph.nodes[entity]
        return {}

# llm_agent.py content (mock implementation)
class LLMAgent:
    def __init__(self, llm_model: Any = None): # llm_model is a placeholder, not used in mock
        self.llm_model = llm_model

    def extract_initial_entities(self, patient_case_description: str) -> List[str]:
        # Mock LLM response for entity extraction
        lower_desc = patient_case_description.lower()
        entities = []
        if "fever" in lower_desc: entities.append("Fever")
        if "cough" in lower_desc: entities.append("Cough")
        if "headache" in lower_desc: entities.append("Headache")
        if "fatigue" in lower_desc: entities.append("Fatigue")
        if "nausea" in lower_desc: entities.append("Nausea")
        return entities if entities else ["General Ailment"]

    def suggest_exploration_steps(self, current_path: List[Tuple[str, str, str]], patient_context: str) -> List[Tuple[str, str]]:
        # Mock LLM response for exploration suggestions
        suggestions = []
        if not current_path:
            return [("presents_with", "Influenza"), ("presents_with", "Common Cold")]
        
        last_entity = current_path[-1][2]
        if last_entity == "Influenza":
            suggestions.append(("treatable_by", "Antivirals"))
        elif last_entity == "Common Cold":
            suggestions.append(("treatable_by", "Pain Relievers"))
        elif last_entity == "Migraine":
            suggestions.append(("treatable_by", "Pain Relievers"))
        elif last_entity == "Pneumonia":
            suggestions.append(("treatable_by", "Antibiotics"))
        elif last_entity == "COVID-19":
            suggestions.append(("treatable_by", "Antivirals")) # Mock treatment
        
        # Add some random plausible relations for broader exploration
        possible_relations = ["presents_with", "causes", "associated_with", "treatable_by"]
        possible_entities = ["Fever", "Cough", "Headache", "Influenza", "Pneumonia", "Antibiotics"]
        for _ in range(min(len(possible_relations), len(possible_entities))):
            rel = random.choice(possible_relations)
            ent = random.choice(possible_entities)
            if (rel, ent) not in suggestions: # Avoid duplicates
                suggestions.append((rel, ent))
        return suggestions

    def prune_paths(self, candidate_paths: List[List[Tuple[str, str, str]]], patient_context: str) -> List[List[Tuple[str, str, str]]]:
        # Mock LLM response for path pruning: simple heuristic based on path length and specific diseases
        scored_paths = []
        for path in candidate_paths:
            score = 0
            if "Influenza" in str(path): score += 3
            if "Pneumonia" in str(path): score += 4
            if "COVID-19" in str(path): score += 5
            if "Migraine" in str(path): score += 2
            score -= len(path) * 0.5 # Prefer shorter paths
            
            # Add context relevance
            if "fever" in patient_context.lower() and "Fever" in str(path): score += 1
            if "cough" in patient_context.lower() and "Cough" in str(path): score += 1
            scored_paths.append((score, path))
        
        scored_paths.sort(key=lambda x: x[0], reverse=True)
        return [path for score, path in scored_paths]

    def generate_diagnosis_and_explanation(self, reasoning_paths: List[List[Tuple[str, str, str]]], patient_context: str) -> Tuple[str, str]:
        # Mock LLM response for diagnosis generation
        diagnosis = "Undetermined Condition"
        explanation = "Could not definitively determine a diagnosis based on available information and exploration." 
        
        if not reasoning_paths:
            return diagnosis, explanation

        # Simple heuristic to pick a diagnosis based on the final entity in the best path
        best_path = reasoning_paths[0]
        if best_path:
            final_entity = best_path[-1][2]
            # Check if final_entity is a disease (from our mock KG)
            disease_entities = [n["id"] for n in MOCK_KG_DATA["nodes"] if n.get("type") == "Disease"]
            if final_entity in disease_entities:
                diagnosis = final_entity
                explanation = f"Based on the most probable reasoning path, the patient likely has {diagnosis}.\n\nReasoning Path:\n"
                for s, r, o in best_path:
                    explanation += f"- {s} {r} {o}\n"
                explanation += f"\nThis path was derived by starting from initial symptoms like '{best_path[0][0]}' and exploring related conditions in the knowledge graph. The symptoms '{patient_context}' align with this diagnosis."
            else: # If the final entity is not a disease, try to infer from the path
                for _, _, entity in best_path:
                    if entity in disease_entities:
                        diagnosis = entity
                        explanation = f"Based on exploration, '{entity}' is a strong candidate diagnosis.\n\nReasoning Path leading to {entity}:\n"
                        for s, r, o in best_path:
                            explanation += f"- {s} {r} {o}\n"
                        explanation += f"\nThis diagnosis considers the symptoms mentioned in '{patient_context}'."
                        break
        
        if diagnosis == "Undetermined Condition" and "fever" in patient_context.lower() and "cough" in patient_context.lower():
            diagnosis = "Influenza or Common Cold"
            explanation = "Patient presents with common symptoms of respiratory infections. Further investigation might be needed to differentiate."

        return diagnosis, explanation

# tog_framework.py content
class ToGFramework:
    def __init__(self, llm_agent: LLMAgent, kg: KnowledgeGraph, max_exploration_depth: int = 3, beam_width: int = 3):
        self.llm_agent = llm_agent
        self.kg = kg
        self.max_exploration_depth = max_exploration_depth
        self.beam_width = beam_width

    def _search_and_prune(self, current_paths: List[List[Tuple[str, str, str]]], patient_context: str) -> List[List[Tuple[str, str, str]]]:
        extended_paths = []
        for path in current_paths:
            last_entity = path[-1][2] if path else None
            if not last_entity: # Handle empty path case for initial exploration
                # For initial step, we expand from extracted entities directly
                initial_entities = self.llm_agent.extract_initial_entities(patient_context)
                for entity in initial_entities:
                    if entity in self.kg.graph.nodes:
                        neighbors = self.kg.get_neighbors(entity)
                        for s, r, o in neighbors:
                            extended_paths.append([(s, r, o)])
                continue

            llm_suggestions = self.llm_agent.suggest_exploration_steps(path, patient_context)
            
            for relation_type, target_entity_hint in llm_suggestions:
                neighbors = self.kg.get_neighbors(last_entity, relation_type=relation_type)
                for s, r, o in neighbors:
                    # Only extend if the suggested entity matches a neighbor, or if the LLM suggested a general relation
                    if not target_entity_hint or o == target_entity_hint:
                        new_path = path + [(s, r, o)]
                        if self.kg.check_path_validity(new_path):
                            extended_paths.append(new_path)
        
        # Prune the extended paths using the LLM agent
        pruned_paths = self.llm_agent.prune_paths(extended_paths, patient_context)
        return pruned_paths[:self.beam_width] # Return only top N paths after pruning

    def run_diagnostic_cycle(self, patient_case_description: str) -> Tuple[str, str, List[List[Tuple[str, str, str]]]]:
        patient_context = patient_case_description # Simplification for context passing

        # 1. Initialization Phase
        initial_entities = self.llm_agent.extract_initial_entities(patient_context)
        current_paths: List[List[Tuple[str, str, str]]] = []

        # Start paths from initial entities by finding their direct connections
        for entity in initial_entities:
            if entity in self.kg.graph.nodes:
                neighbors = self.kg.get_neighbors(entity)
                for s, r, o in neighbors:
                    current_paths.append([(s, r, o)])
        
        current_paths = self.llm_agent.prune_paths(current_paths, patient_context)[:self.beam_width]

        # 2. Exploration Phase (Iterative Beam Search)
        for depth in range(self.max_exploration_depth):
            if not current_paths:
                break
            st.write(f"Exploring depth {depth + 1}...")
            new_paths = []
            for path in current_paths:
                last_entity_in_path = path[-1][2]
                llm_suggestions = self.llm_agent.suggest_exploration_steps(path, patient_context)
                for suggested_relation, suggested_entity_hint in llm_suggestions:
                    # Try to extend with actual KG neighbors matching suggestion
                    possible_extensions = self.kg.get_neighbors(last_entity_in_path, relation_type=suggested_relation)
                    for s, r, o in possible_extensions:
                        if not suggested_entity_hint or o == suggested_entity_hint:
                            extended_path = path + [(s, r, o)]
                            if self.kg.check_path_validity(extended_path):
                                new_paths.append(extended_path)
            
            if not new_paths and current_paths: # If no new paths found but old paths exist, keep them
                st.write("No new paths found at this depth. Keeping current best paths.")
                break
            
            if new_paths:
                current_paths = self.llm_agent.prune_paths(new_paths, patient_context)[:self.beam_width]
            elif not new_paths and not current_paths and initial_entities: # Fallback if initial entities didn't lead to paths
                st.warning("No paths could be generated from initial entities. Trying a broader search.")
                # A more robust fallback would be to suggest more general relations/entities or use LLM's inherent knowledge more directly.
                break

        # 3. Reasoning Phase
        diagnosis, explanation = self.llm_agent.generate_diagnosis_and_explanation(current_paths, patient_context)
        return diagnosis, explanation, current_paths

# main.py content (Streamlit UI)
st.set_page_config(layout="wide")
st.title("🧠 Medical Diagnostic Assistant with Explainable Reasoning")
st.markdown("--- Developed using the ThinkonGraph (ToG) Algorithmic Framework --- ")

# Initialize KG and LLM Agent
medical_kg = KnowledgeGraph(MOCK_KG_DATA)
llm_agent = LLMAgent() # Mock LLM Agent

# Initialize ToG Framework
tog_framework = ToGFramework(llm_agent=llm_agent, kg=medical_kg, max_exploration_depth=3, beam_width=3)

st.sidebar.header("Configuration")
max_depth = st.sidebar.slider("Max Exploration Depth", 1, 5, 3)
beam_width = st.sidebar.slider("Beam Width", 1, 5, 3)
tog_framework.max_exploration_depth = max_depth
tog_framework.beam_width = beam_width

st.header("Patient Case Input")
patient_case = st.text_area(
    "Describe the patient's symptoms, medical history, and any relevant lab results:",
    "A 45-year-old male presents with a persistent cough, fever, and fatigue for the past 3 days. He also reports body aches.",
    height=150
)

if st.button("Diagnose"): # Use st.button directly for execution
    if patient_case:
        st.subheader("Running Diagnostic Cycle...")
        with st.spinner("Processing..."):
            diagnosis, explanation, reasoning_paths = tog_framework.run_diagnostic_cycle(patient_case)

        st.subheader("Diagnosis")
        st.success(f"**{diagnosis}**")

        st.subheader("Explanation")
        st.info(explanation)

        st.subheader("Top Reasoning Paths Found")
        if reasoning_paths:
            for i, path in enumerate(reasoning_paths):
                st.markdown(f"**Path {i+1}:**")
                path_str = []
                for s, r, o in path:
                    path_str.append(f"{s} -[{r}]-> {o}")
                st.write(" -> ".join(path_str))
        else:
            st.warning("No clear reasoning paths were constructed.")
    else:
        st.error("Please provide a patient case description to start the diagnosis.")

st.markdown("""
---
#### About this Demo:
This demonstration utilizes a mock LLM agent and a simplified in-memory medical knowledge graph (`medical_kg.json`) to illustrate the ThinkonGraph (ToG) framework. The LLM agent's responses are hardcoded or based on simple heuristics, not actual LLM API calls.
""")