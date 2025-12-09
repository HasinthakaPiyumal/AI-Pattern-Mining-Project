import networkx as nx
import gradio as gr
import os
from loguru import logger
from dotenv import dotenv_values

# Load environment variables (e.g., for API keys if using actual LLMs)
config = dotenv_values(".env")

# Configure logger
logger.remove()
logger.add(os.sys.stderr, level="INFO")

class MedicalKnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self._build_sample_kg()

    def _build_sample_kg(self):
        # Nodes: Medical Entities
        self.graph.add_nodes_from([
            ("Symptom:Headache", {"type": "symptom"}),
            ("Symptom:Fever", {"type": "symptom"}),
            ("Symptom:Cough", {"type": "symptom"}),
            ("Symptom:Fatigue", {"type": "symptom"}),
            ("Disease:Migraine", {"type": "disease"}),
            ("Disease:CommonCold", {"type": "disease"}),
            ("Disease:Influenza", {"type": "disease"}),
            ("Disease:Hypertension", {"type": "disease"}),
            ("Drug:Paracetamol", {"type": "drug"}),
            ("Drug:Ibuprofen", {"type": "drug"}),
            ("Drug:Antihistamine", {"type": "drug"}),
            ("Drug:Lisinopril", {"type": "drug"}),
            ("Treatment:Rest", {"type": "treatment"}),
            ("Treatment:Hydration", {"type": "treatment"})
        ])

        # Edges: Relations
        self.graph.add_edges_from([
            ("Symptom:Headache", "Disease:Migraine", {"relation": "is_symptom_of"}),
            ("Symptom:Fever", "Disease:CommonCold", {"relation": "is_symptom_of"}),
            ("Symptom:Cough", "Disease:CommonCold", {"relation": "is_symptom_of"}),
            ("Symptom:Fever", "Disease:Influenza", {"relation": "is_symptom_of"}),
            ("Symptom:Cough", "Disease:Influenza", {"relation": "is_symptom_of"}),
            ("Symptom:Fatigue", "Disease:Influenza", {"relation": "is_symptom_of"}),
            ("Disease:Migraine", "Drug:Ibuprofen", {"relation": "treated_by"}),
            ("Disease:Migraine", "Drug:Paracetamol", {"relation": "treated_by"}),
            ("Disease:CommonCold", "Drug:Paracetamol", {"relation": "treated_by"}),
            ("Disease:CommonCold", "Treatment:Rest", {"relation": "treated_by"}),
            ("Disease:CommonCold", "Treatment:Hydration", {"relation": "treated_by"}),
            ("Disease:Influenza", "Drug:Paracetamol", {"relation": "treated_by"}),
            ("Disease:Influenza", "Treatment:Rest", {"relation": "treated_by"}),
            ("Disease:Influenza", "Treatment:Hydration", {"relation": "treated_by"}),
            ("Disease:Hypertension", "Drug:Lisinopril", {"relation": "treated_by"}),
            ("Drug:Ibuprofen", "Disease:Hypertension", {"relation": "contraindicated_with"}),
            ("Drug:Paracetamol", "Drug:Ibuprofen", {"relation": "interacts_with_mildly"})
        ])
        logger.info("Medical Knowledge Graph initialized with sample data.")

    def get_relations(self):
        return list(set(d["relation"] for u, v, d in self.graph.edges(data=True) if "relation" in d))

    def get_entity_types(self):
        return list(set(d["type"] for n, d in self.graph.nodes(data=True) if "type" in d))

    def get_neighbors(self, node, relation_type=None):
        if node not in self.graph:
            return []
        neighbors = []
        for u, v, data in self.graph.edges(node, data=True):
            if relation_type is None or data.get("relation") == relation_type:
                neighbors.append((v, data.get("relation")))
        return neighbors

    def get_paths(self, start_node, end_node=None, relation_path=None, max_length=3):
        if start_node not in self.graph:
            logger.warning(f"Start node '{start_node}' not found in KG.")
            return []

        found_paths = []

        if relation_path:
            # Try to follow the specific relation path
            paths = list(nx.all_simple_paths(self.graph, source=start_node, target=end_node, cutoff=max_length))
            for path in paths:
                # Check if this path matches the relation_path pattern
                current_relation_sequence = []
                for i in range(len(path) - 1):
                    u, v = path[i], path[i+1]
                    edge_data = self.graph.get_edge_data(u, v)
                    if edge_data and "relation" in edge_data:
                        current_relation_sequence.append(edge_data["relation"])
                    else:
                        break # Path has no relation data
                if current_relation_sequence == relation_path[:len(current_relation_sequence)]:
                    found_paths.append(path)
        else:
            # Simple pathfinding without specific relation constraint (up to max_length)
            for path in nx.all_simple_paths(self.graph, source=start_node, target=end_node, cutoff=max_length):
                found_paths.append(path)
        
        return found_paths

class PlanningModule:
    def __init__(self, kg: MedicalKnowledgeGraph, llm_model_name: str = "dummy-llm"):
        self.kg = kg
        self.llm_model_name = llm_model_name
        logger.info(f"PlanningModule initialized with {llm_model_name}.")

    def generate_plan(self, patient_query: str) -> list:
        logger.info(f"Planning for query: '{patient_query}'")
        # --- LLM Simulation for Plan Generation ---
        # In a real scenario, an LLM (e.g., via transformers or langchain) would analyze the query
        # and propose a sequence of relations based on its understanding and the KG schema.
        
        # Dummy LLM logic: Based on keywords, suggest a simple plan
        proposed_relation_paths = []
        if "symptom" in patient_query.lower() and "disease" in patient_query.lower():
            proposed_relation_paths = [["is_symptom_of", "treated_by"]]
        elif "treatment" in patient_query.lower() and "disease" in patient_query.lower():
            proposed_relation_paths = [["treated_by"]]
        elif "interaction" in patient_query.lower() and "drug" in patient_query.lower():
             proposed_relation_paths = [["interacts_with_mildly"], ["contraindicated_with"]]
        else:
            proposed_relation_paths = [["is_symptom_of"]]

        # --- KG Grounding Validation ---
        # Verify if the proposed relations actually exist in the KG
        valid_plans = []
        existing_relations = self.kg.get_relations()
        for plan in proposed_relation_paths:
            is_valid_plan = all(rel in existing_relations for rel in plan)
            if is_valid_plan:
                valid_plans.append(plan)
            else:
                logger.warning(f"Proposed plan {plan} contains invalid relations against KG schema.")
        
        if not valid_plans:
            logger.warning("No valid plans generated. Falling back to a generic path.")
            return [[]] # Return an empty plan if no specific valid plan, allowing retrieval to do a broader search

        logger.info(f"Generated and validated plans: {valid_plans}")
        return valid_plans

class RetrievalModule:
    def __init__(self, kg: MedicalKnowledgeGraph):
        self.kg = kg
        logger.info("RetrievalModule initialized.")

    def retrieve_paths(self, patient_query: str, planned_relation_paths: list) -> list:
        logger.info(f"Retrieving paths for query: '{patient_query}' with plans: {planned_relation_paths}")
        retrieved_reasoning_paths = []

        # Extract potential starting entities from the query
        query_entities = []
        for node in self.kg.graph.nodes:
            if node.lower() in patient_query.lower():
                query_entities.append(node)
        
        # If no specific entity found, try to infer based on common entity types
        if not query_entities:
            if "headache" in patient_query.lower():
                query_entities.append("Symptom:Headache")
            if "fever" in patient_query.lower():
                query_entities.append("Symptom:Fever")
            if "cough" in patient_query.lower():
                query_entities.append("Symptom:Cough")
            if "migraine" in patient_query.lower():
                query_entities.append("Disease:Migraine")
            if "paracetamol" in patient_query.lower():
                query_entities.append("Drug:Paracetamol")
        
        logger.info(f"Identified query entities for retrieval: {query_entities}")

        for start_node in query_entities:
            for plan in planned_relation_paths:
                # This is a simplification. A more robust retrieval would handle multi-hop with specific relations.
                # For now, we find paths that *can* follow the general idea of the plan.
                # The get_paths method in KG is enhanced to handle relation_path suggestion.
                if plan:
                    # Try to find paths specifically following the relations in the plan
                    paths = self.kg.get_paths(start_node=start_node, relation_path=plan, max_length=len(plan)+1)
                    for p in paths:
                        path_description = [p[0]]
                        for i in range(len(p) - 1):
                            u, v = p[i], p[i+1]
                            edge_data = self.kg.graph.get_edge_data(u, v)
                            if edge_data and "relation" in edge_data:
                                path_description.append(f"--{edge_data['relation']}-->")
                                path_description.append(v)
                            else:
                                path_description.append(f"--UNKNOWN_RELATION-->")
                                path_description.append(v)
                        retrieved_reasoning_paths.append(" ".join(path_description))
                else:
                    # If plan is empty, do a broader search around the start node
                    for path in nx.all_simple_paths(self.kg.graph, source=start_node, target=None, cutoff=2):
                        path_description = [path[0]]
                        for i in range(len(path) - 1):
                            u, v = path[i], path[i+1]
                            edge_data = self.kg.graph.get_edge_data(u, v)
                            if edge_data and "relation" in edge_data:
                                path_description.append(f"--{edge_data['relation']}-->")
                                path_description.append(v)
                            else:
                                path_description.append(f"--UNKNOWN_RELATION-->")
                                path_description.append(v)
                        retrieved_reasoning_paths.append(" ".join(path_description))

        # Deduplicate and limit for brevity
        retrieved_reasoning_paths = list(set(retrieved_reasoning_paths))
        logger.info(f"Retrieved paths ({len(retrieved_reasoning_paths)}): {retrieved_reasoning_paths[:5]}...")
        return retrieved_reasoning_paths

class ReasoningModule:
    def __init__(self, llm_model_name: str = "dummy-llm"):
        self.llm_model_name = llm_model_name
        logger.info(f"ReasoningModule initialized with {llm_model_name}.")

    def reason_and_explain(self, patient_query: str, retrieved_paths: list) -> dict:
        logger.info(f"Reasoning for query: '{patient_query}' with {len(retrieved_paths)} retrieved paths.")
        
        # --- LLM Simulation for Reasoning and Explanation ---
        # In a real scenario, an LLM would consume the query and retrieved paths
        # to synthesize a diagnosis, treatment, and explanation.
        
        diagnosis = "Unclear, further information needed."
        treatment = "Symptomatic relief recommended based on available information."
        explanation = "The system attempted to reason based on the provided query and retrieved knowledge graph paths. "

        if not retrieved_paths:
            explanation += "However, no relevant paths were retrieved from the knowledge graph, leading to a generic response."
        else:
            explanation += "Based on the following retrieved facts from the medical knowledge graph:\n"
            for i, path in enumerate(retrieved_paths):
                explanation += f"- {path}\n"
            explanation += "\nUsing these facts, the system infers: "

            # Dummy reasoning based on retrieved paths
            has_headache_path = any("Symptom:Headache --is_symptom_of--> Disease:Migraine" in path for path in retrieved_paths)
            has_fever_cough_path = any("Symptom:Fever --is_symptom_of--> Disease:CommonCold" in path and "Symptom:Cough" in path for path in retrieved_paths)
            has_hypertension_drug_contraindication = any("Disease:Hypertension --treated_by--> Drug:Lisinopril" in path and "Drug:Ibuprofen --contraindicated_with--> Disease:Hypertension" in path for path in retrieved_paths)

            if has_headache_path:
                diagnosis = "Possible Migraine"
                treatment = "Consider Ibuprofen or Paracetamol. Avoid triggers."
                explanation += "The presence of 'Headache' linked to 'Migraine' suggests this diagnosis. 'Ibuprofen' and 'Paracetamol' are common treatments for Migraine."
            elif has_fever_cough_path:
                diagnosis = "Possible Common Cold or Influenza"
                treatment = "Rest, hydration, and Paracetamol for symptomatic relief."
                explanation += "Symptoms like 'Fever' and 'Cough' are indicative of 'Common Cold' or 'Influenza'. Recommended treatments include 'Rest', 'Hydration', and 'Paracetamol'."
            elif has_hypertension_drug_contraindication:
                diagnosis = "Hypertension detected with potential drug interaction risk."
                treatment = "Lisinopril for hypertension. Avoid Ibuprofen due to contraindication."
                explanation += "Hypertension is identified, for which Lisinopril is a treatment. It's crucial to note that Ibuprofen is contraindicated with hypertension."
            else:
                explanation += "No specific condition could be definitively identified from the retrieved paths. The diagnosis and treatment are general."

        result = {
            "diagnosis": diagnosis,
            "treatment_plan": treatment,
            "explanation": explanation.strip()
        }
        logger.info(f"Reasoning result: {result}")
        return result

class ClinicalDecisionSupportSystem:
    def __init__(self):
        self.kg = MedicalKnowledgeGraph()
        self.planning_module = PlanningModule(self.kg)
        self.retrieval_module = RetrievalModule(self.kg)
        self.reasoning_module = ReasoningModule()
        logger.info("CDSS initialized.")

    def process_query(self, patient_query: str) -> dict:
        logger.info(f"Processing patient query: '{patient_query}'")
        
        # 1. Planning Module
        planned_paths = self.planning_module.generate_plan(patient_query)
        
        # 2. Retrieval Module
        retrieved_facts = self.retrieval_module.retrieve_paths(patient_query, planned_paths)
        
        # 3. Reasoning Module
        reasoning_output = self.reasoning_module.reason_and_explain(patient_query, retrieved_facts)
        
        logger.info("CDSS query processing complete.")
        return reasoning_output

# Initialize the CDSS
cdss_system = ClinicalDecisionSupportSystem()

# Gradio Interface
def cdss_interface(patient_symptoms: str, patient_history: str, specific_question: str) -> tuple:
    query_parts = []
    if patient_symptoms: query_parts.append(f"Symptoms: {patient_symptoms}")
    if patient_history: query_parts.append(f"History: {patient_history}")
    if specific_question: query_parts.append(f"Question: {specific_question}")
    
    full_query = ". ".join(query_parts)
    if not full_query: 
        return "Please provide some input.", "", ""

    result = cdss_system.process_query(full_query)
    
    diagnosis = result.get("diagnosis", "N/A")
    treatment_plan = result.get("treatment_plan", "N/A")
    explanation = result.get("explanation", "N/A")
    
    return diagnosis, treatment_plan, explanation

# UI definition using Gradio
iface = gr.Interface(
    fn=cdss_interface,
    inputs=[
        gr.Textbox(label="Patient Symptoms (e.g., headache, fever, cough)", placeholder="e.g., severe headache"),
        gr.Textbox(label="Patient Medical History (optional)", placeholder="e.g., history of hypertension"),
        gr.Textbox(label="Specific Clinical Question (optional)", placeholder="e.g., what is the best treatment for this?")
    ],
    outputs=[
        gr.Textbox(label="Diagnosis"),
        gr.Textbox(label="Treatment Plan"),
        gr.Textbox(label="Explanation")
    ],
    title="Clinical Decision Support System (RoG Framework)",
    description="Enter patient information to get a potential diagnosis, treatment plan, and an interpretable explanation powered by a Knowledge Graph and LLM (simulated)."
)

# Launch the Gradio app
if __name__ == "__main__":
    # Create a dummy .env file if it doesn't exist for dotenv_values to work without error
    if not os.path.exists(".env"):
        with open(".env", "w") as f:
            f.write("# Add your API keys here if using external LLMs (e.g., OPENAI_API_KEY=your_key)")
    
    logger.info("Starting CDSS Gradio interface...")
    iface.launch()
