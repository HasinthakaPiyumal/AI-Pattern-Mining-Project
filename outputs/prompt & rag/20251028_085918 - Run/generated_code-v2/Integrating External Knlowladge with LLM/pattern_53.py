import networkx as nx
import gradio as gr
from typing import List, Dict, Any, Tuple

PROMPTS = {
    "entity_extraction": "Extract medical entities (symptoms, conditions, etc.) from the following patient description: '{patient_description}'. Return as a comma-separated list.",
    "search_step": "Given the current reasoning path '{current_path}', and available next relations/entities: {available_options}. Identify the top 3 most relevant next steps to explore for diagnosing '{patient_query}'. Return as a comma-separated list of chosen options.",
    "prune_step": "Given the candidate reasoning paths: {candidate_paths}. Select the top {top_n} most promising paths to continue exploration for diagnosing '{patient_query}'. Return as a comma-separated list of chosen paths.",
    "reasoning_step": "Based on the following reasoning paths: {reasoning_paths}. Provide a differential diagnosis for '{patient_query}' and explain the reasoning. State 'Continue Exploration' if more steps are needed, otherwise provide diagnosis."
}

class MedicalKnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self._build_sample_graph()

    def _build_sample_graph(self):
        self.add_node("Fever", {"type": "Symptom"})
        self.add_node("Cough", {"type": "Symptom"})
        self.add_node("Headache", {"type": "Symptom"})
        self.add_node("Fatigue", {"type": "Symptom"})
        self.add_node("Shortness of Breath", {"type": "Symptom"})
        self.add_node("Muscle Aches", {"type": "Symptom"})
        self.add_node("Sore Throat", {"type": "Symptom"})
        self.add_node("Chest Pain", {"type": "Symptom"})

        self.add_node("Influenza", {"type": "Disease"})
        self.add_node("Common Cold", {"type": "Disease"})
        self.add_node("Pneumonia", {"type": "Disease"})
        self.add_node("Bronchitis", {"type": "Disease"})
        self.add_node("COVID-19", {"type": "Disease"})
        self.add_node("Migraine", {"type": "Disease"})
        self.add_node("Tuberculosis", {"type": "Disease"})
        self.add_node("Asthma", {"type": "Disease"})

        self.add_node("Antivirals", {"type": "Treatment"})
        self.add_node("Antibiotics", {"type": "Treatment"})
        self.add_node("Pain Relievers", {"type": "Treatment"})
        self.add_node("Rest", {"type": "Treatment"})
        self.add_node("Bronchodilators", {"type": "Treatment"})
        self.add_node("Oxygen Therapy", {"type": "Treatment"})

        self.add_edge("Influenza", "has_symptom", "Fever")
        self.add_edge("Influenza", "has_symptom", "Cough")
        self.add_edge("Influenza", "has_symptom", "Fatigue")
        self.add_edge("Influenza", "has_symptom", "Muscle Aches")
        self.add_edge("Influenza", "treats", "Antivirals")

        self.add_edge("Common Cold", "has_symptom", "Cough")
        self.add_edge("Common Cold", "has_symptom", "Sore Throat")
        self.add_edge("Common Cold", "treats", "Rest")
        self.add_edge("Common Cold", "treats", "Pain Relievers")

        self.add_edge("Pneumonia", "has_symptom", "Fever")
        self.add_edge("Pneumonia", "has_symptom", "Cough")
        self.add_edge("Pneumonia", "has_symptom", "Shortness of Breath")
        self.add_edge("Pneumonia", "has_symptom", "Chest Pain")
        self.add_edge("Pneumonia", "treats", "Antibiotics")
        self.add_edge("Pneumonia", "treats", "Oxygen Therapy")
        self.add_edge("Pneumonia", "associated_with", "Influenza")

        self.add_edge("Bronchitis", "has_symptom", "Cough")
        self.add_edge("Bronchitis", "has_symptom", "Shortness of Breath")
        self.add_edge("Bronchitis", "treats", "Bronchodilators")

        self.add_edge("COVID-19", "has_symptom", "Fever")
        self.add_edge("COVID-19", "has_symptom", "Cough")
        self.add_edge("COVID-19", "has_symptom", "Shortness of Breath")
        self.add_edge("COVID-19", "has_symptom", "Fatigue")
        self.add_edge("COVID-19", "treats", "Antivirals")

        self.add_edge("Migraine", "has_symptom", "Headache")
        self.add_edge("Migraine", "treats", "Pain Relievers")

        self.add_edge("Fever", "indicates", "Influenza")
        self.add_edge("Fever", "indicates", "Pneumonia")
        self.add_edge("Fever", "indicates", "COVID-19")
        self.add_edge("Cough", "indicates", "Influenza")
        self.add_edge("Cough", "indicates", "Common Cold")
        self.add_edge("Cough", "indicates", "Pneumonia")
        self.add_edge("Cough", "indicates", "Bronchitis")
        self.add_edge("Cough", "indicates", "COVID-19")
        self.add_edge("Shortness of Breath", "indicates", "Pneumonia")
        self.add_edge("Shortness of Breath", "indicates", "COVID-19")
        self.add_edge("Shortness of Breath", "indicates", "Bronchitis")
        self.add_edge("Headache", "indicates", "Migraine")
        self.add_edge("Headache", "indicates", "Fever")
        self.add_edge("Fatigue", "indicates", "Influenza")
        self.add_edge("Fatigue", "indicates", "COVID-19")

    def add_node(self, node_id: str, attributes: Dict[str, Any] = None):
        if not self.graph.has_node(node_id):
            self.graph.add_node(node_id, **(attributes if attributes else {}))

    def add_edge(self, u: str, relation: str, v: str, attributes: Dict[str, Any] = None):
        self.add_node(u)
        self.add_node(v)
        self.graph.add_edge(u, v, relation=relation, **(attributes if attributes else {}))

    def get_neighbors(self, entity: str) -> List[Tuple[str, str, str]]:
        neighbors = []
        if self.graph.has_node(entity):
            for neighbor in self.graph.neighbors(entity):
                relation = self.graph.get_edge_data(entity, neighbor)["relation"]
                neighbors.append((entity, relation, neighbor))
            for predecessor in self.graph.predecessors(entity):
                relation = self.graph.get_edge_data(predecessor, entity)["relation"]
                neighbors.append((predecessor, relation + "_rev", entity))
        return neighbors

    def get_node_type(self, node_id: str) -> str:
        return self.graph.nodes[node_id].get("type", "Unknown") if self.graph.has_node(node_id) else "Unknown"

class ToGAgent:
    def __init__(self, kg: MedicalKnowledgeGraph, max_depth: int = 3, top_n_paths: int = 5):
        self.kg = kg
        self.max_depth = max_depth
        self.top_n_paths = top_n_paths

    def _simulate_llm_extract_entities(self, patient_description: str) -> List[str]:
        entities = []
        patient_description_lower = patient_description.lower()
        for node in self.kg.graph.nodes():
            if node.lower() in patient_description_lower:
                entities.append(node)
        return list(set(entities))

    def _simulate_llm_search(self, current_path: List[str], available_options: List[Tuple[str, str, str]], patient_query: str) -> List[Tuple[str, str, str]]:
        selected_options = []
        disease_options = [opt for opt in available_options if self.kg.get_node_type(opt[2]) == "Disease" or opt[1] == "indicates"]
        symptom_options = [opt for opt in available_options if self.kg.get_node_type(opt[2]) == "Symptom" and opt[1] != "indicates"]
        other_options = [opt for opt in available_options if opt not in disease_options and opt not in symptom_options]

        chosen_count = 0
        for opt in disease_options:
            if chosen_count < self.top_n_paths:
                selected_options.append(opt)
                chosen_count += 1
        for opt in symptom_options:
            if chosen_count < self.top_n_paths:
                selected_options.append(opt)
                chosen_count += 1
        for opt in other_options:
            if chosen_count < self.top_n_paths:
                selected_options.append(opt)
                chosen_count += 1

        return selected_options[:self.top_n_paths]

    def _simulate_llm_prune(self, candidate_paths: List[List[str]], patient_query: str) -> List[List[str]]:
        pruned_paths = sorted(candidate_paths, key=lambda p: (self.kg.get_node_type(p[-1]) == "Disease", len(p)), reverse=True)
        return pruned_paths[:self.top_n_paths]

    def _simulate_llm_reason(self, reasoning_paths: List[List[str]], patient_query: str, current_depth: int) -> Tuple[str, bool]:
        disease_paths = [p for p in reasoning_paths if self.kg.get_node_type(p[-1]) == "Disease"]

        if disease_paths and current_depth >= 1:
            diagnosis_info = []
            for path in disease_paths:
                diagnosis_info.append(f"Path: {' -> '.join(path)}. Leading to a potential diagnosis of {path[-1]}.")
            
            explanation = "\n".join(diagnosis_info)
            explanation += "\n\nBased on the explored paths, here is a differential diagnosis. Further investigation may be required."
            
            potential_diseases = list(set([p[-1] for p in disease_paths]))
            if len(potential_diseases) == 1:
                diagnosis = f"Probable diagnosis: {potential_diseases[0]}.";
            elif potential_diseases:
                diagnosis = f"Differential diagnosis: {', '.join(potential_diseases)}."
            else:
                diagnosis = "No clear diagnosis from current paths."

            return f"{diagnosis}\n\nReasoning:\n{explanation}", True
        elif current_depth >= self.max_depth:
            return "Maximum exploration depth reached. Could not find a definitive diagnosis based on the knowledge graph. Falling back to inherent knowledge (no further KG exploration).", True
        else:
            return "Continue Exploration", False

    def initialize_paths(self, patient_description: str) -> List[List[str]]:
        initial_entities = self._simulate_llm_extract_entities(patient_description)
        return [[entity] for entity in initial_entities if self.kg.graph.has_node(entity)]

    def _get_path_str(self, path: List[str]) -> str:
        return " -> ".join(path)

    def _get_path_from_search_option(self, current_path: List[str], option: Tuple[str, str, str]) -> List[str]:
        if option[1].endswith("_rev"):
            return [option[0]] + current_path
        else:
            return current_path + [option[2]]

    def explore_paths(self, current_paths: List[List[str]], patient_query: str) -> List[List[str]]:
        new_candidate_paths = []
        for path in current_paths:
            last_entity = path[-1]
            neighbors = self.kg.get_neighbors(last_entity)
            
            if neighbors:
                selected_options_for_path = self._simulate_llm_search(
                    current_path=path,
                    available_options=neighbors,
                    patient_query=patient_query
                )

                for option in selected_options_for_path:
                    new_candidate_paths.append(self._get_path_from_search_option(path, option))
            else:
                new_candidate_paths.append(path)

        pruned_paths = self._simulate_llm_prune(new_candidate_paths, patient_query)
        return pruned_paths

    def run_diagnosis(self, patient_query: str) -> Tuple[str, List[str]]:
        reasoning_paths = self.initialize_paths(patient_query)
        all_reasoning_paths_history = []

        if not reasoning_paths:
            return "Could not extract initial medical entities from your query. Please provide more specific symptoms or conditions.", []

        for depth in range(self.max_depth):
            current_diagnosis, is_final = self._simulate_llm_reason(reasoning_paths, patient_query, depth)
            if is_final:
                final_paths_str = [self._get_path_str(p) for p in reasoning_paths]
                return current_diagnosis, final_paths_str

            reasoning_paths = self.explore_paths(reasoning_paths, patient_query)
            all_reasoning_paths_history.extend([self._get_path_str(p) for p in reasoning_paths])

            if not reasoning_paths:
                return "Exploration stopped: No new relevant paths could be found.", all_reasoning_paths_history

        final_diagnosis, _ = self._simulate_llm_reason(reasoning_paths, patient_query, self.max_depth)
        final_paths_str = [self._get_path_str(p) for p in reasoning_paths]
        return final_diagnosis, final_paths_str

def medical_diagnostic_app(patient_input: str) -> Tuple[str, str]:
    kg = MedicalKnowledgeGraph()
    agent = ToGAgent(kg, max_depth=3, top_n_paths=5)
    
    diagnosis, reasoning_paths = agent.run_diagnosis(patient_input)
    
    formatted_reasoning = ""
    if reasoning_paths:
        formatted_reasoning = "Explored Reasoning Paths:\n" + "\n".join([f"- {path}" for path in reasoning_paths])
    else:
        formatted_reasoning = "No specific reasoning paths were found or generated."
        
    return diagnosis, formatted_reasoning

with gr.Blocks() as demo:
    gr.Markdown("# Medical Diagnostic Assistant (ToG Framework)")
    gr.Markdown("This assistant leverages the ThinkonGraph (ToG) algorithmic framework to provide differential diagnoses based on a medical knowledge graph.")
    
    with gr.Row():
        patient_query_input = gr.Textbox(
            label="Enter Patient Symptoms/Query",
            placeholder="e.g., 'Patient has fever, cough, and fatigue for 3 days.'"
        )
    
    diagnose_btn = gr.Button("Get Diagnosis")
    
    with gr.Column():
        diagnosis_output = gr.Textbox(label="Differential Diagnosis", interactive=False)
        reasoning_output = gr.Textbox(label="Reasoning Paths", interactive=False, lines=10)
        
    diagnose_btn.click(
        fn=medical_diagnostic_app,
        inputs=patient_query_input,
        outputs=[diagnosis_output, reasoning_output]
    )

if __name__ == "__main__":
    demo.launch()