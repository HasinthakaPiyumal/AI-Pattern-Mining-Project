import networkx as nx
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self._load_sample_data()

    def _load_sample_data(self):
        # Sample medical knowledge graph data
        self.graph.add_nodes_from(["fever", "cough", "sore throat", "fatigue", "headache", 
                                   "influenza", "common cold", "strep throat", 
                                   "oseltamivir", "rest", "fluids", "antibiotics"])

        self.graph.add_edge("fever", "has_symptom", relation="has_symptom")
        self.graph.add_edge("cough", "has_symptom", relation="has_symptom")
        self.graph.add_edge("sore throat", "has_symptom", relation="has_symptom")
        self.graph.add_edge("fatigue", "has_symptom", relation="has_symptom")
        self.graph.add_edge("headache", "has_symptom", relation="has_symptom")

        self.graph.add_edge("influenza", "fever", relation="causes")
        self.graph.add_edge("influenza", "cough", relation="causes")
        self.graph.add_edge("influenza", "fatigue", relation="causes")

        self.graph.add_edge("common cold", "cough", relation="causes")
        self.graph.add_edge("common cold", "sore throat", relation="causes")

        self.graph.add_edge("strep throat", "sore throat", relation="causes")
        self.graph.add_edge("strep throat", "fever", relation="causes")

        self.graph.add_edge("oseltamivir", "treats", relation="treats")
        self.graph.add_edge("treats", "influenza", relation="treats")
        self.graph.add_edge("rest", "treats", relation="treats")
        self.graph.add_edge("treats", "common cold", relation="treats")
        self.graph.add_edge("fluids", "treats", relation="treats")
        self.graph.add_edge("antibiotics", "treats", relation="treats")
        self.graph.add_edge("treats", "strep throat", relation="treats")

    def get_candidate_paths(self, start_nodes, depth=2):
        all_paths = []
        for start_node in start_nodes:
            if start_node not in self.graph:
                continue
            for path in nx.all_simple_paths(self.graph, source=start_node, target=None, cutoff=depth):
                path_str = []
                for i in range(len(path) - 1):
                    path_str.append(path[i])
                    edge_data = self.graph.get_edge_data(path[i], path[i+1])
                    if edge_data and "relation" in edge_data:
                        path_str.append(edge_data["relation"])
                path_str.append(path[-1])
                all_paths.append(path_str)
        return all_paths

class LightweightPruner:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def prune_paths(self, candidate_paths, patient_symptoms_text, threshold=0.5):
        symptom_embedding = self.model.encode(patient_symptoms_text)
        pruned_paths = []
        for path in candidate_paths:
            path_description = " ".join(path)
            path_embedding = self.model.encode(path_description)
            similarity = 1 - cosine(symptom_embedding, path_embedding)
            if similarity >= threshold:
                pruned_paths.append((path, similarity))
        pruned_paths.sort(key=lambda x: x[1], reverse=True)
        return [path for path, _ in pruned_paths]

class LLMReasoning:
    def reason(self, pruned_promising_paths, patient_symptoms_text):
        final_diagnoses = []
        if not pruned_promising_paths:
            return ["No definitive diagnosis could be made based on the provided symptoms and knowledge."]
        
        # Mock LLM reasoning
        # In a real scenario, an LLM would analyze the paths and symptoms for a nuanced diagnosis.
        for path in pruned_promising_paths:
            path_str = " -> ".join(path)
            diagnosis_candidates = [node for node in path if node in ["influenza", "common cold", "strep throat"]]
            if diagnosis_candidates:
                for diagnosis in diagnosis_candidates:
                    final_diagnoses.append(f"Potential Diagnosis: {diagnosis}. Reasoning based on path: {path_str}. Symptoms: {patient_symptoms_text}.")
            else:
                final_diagnoses.append(f"Further investigation needed for path: {path_str}. Symptoms: {patient_symptoms_text}.")

        return list(set(final_diagnoses)) # Remove duplicates

class DiagnosticAssistant:
    def __init__(self):
        self.kg = KnowledgeGraph()
        self.pruner = LightweightPruner()
        self.llm_reasoner = LLMReasoning()

    def diagnose(self, patient_symptoms_text, kg_search_depth=3, pruning_threshold=0.5):
        print(f"\nPatient Symptoms: {patient_symptoms_text}")

        # Step 1: KG Exploration to get initial candidate paths
        symptoms_list = [s.strip() for s in patient_symptoms_text.lower().split(',') if s.strip() in self.kg.graph.nodes]
        if not symptoms_list:
            return ["No recognized symptoms found in the knowledge graph."]

        initial_candidate_paths = self.kg.get_candidate_paths(symptoms_list, depth=kg_search_depth)
        print(f"Initial candidate paths found ({len(initial_candidate_paths)}):\n {initial_candidate_paths[:5]}...")

        # Step 2: Lightweight Pruning
        pruned_promising_paths = self.pruner.prune_paths(initial_candidate_paths, patient_symptoms_text, pruning_threshold)
        print(f"Pruned promising paths ({len(pruned_promising_paths)}):\n {pruned_promising_paths[:5]}...")

        # Step 3: LLM Reasoning
        final_diagnoses = self.llm_reasoner.reason(pruned_promising_paths, patient_symptoms_text)
        
        return final_diagnoses

if __name__ == "__main__":
    assistant = DiagnosticAssistant()

    symptoms1 = "I have a fever and a cough"
    diagnoses1 = assistant.diagnose(symptoms1)
    print("\n--- Final Diagnoses for Symptoms 1 ---")
    for diag in diagnoses1:
        print(diag)

    symptoms2 = "sore throat, fatigue, headache"
    diagnoses2 = assistant.diagnose(symptoms2)
    print("\n--- Final Diagnoses for Symptoms 2 ---")
    for diag in diagnoses2:
        print(diag)

    symptoms3 = "runny nose"
    diagnoses3 = assistant.diagnose(symptoms3)
    print("\n--- Final Diagnoses for Symptoms 3 ---")
    for diag in diagnoses3:
        print(diag)

    symptoms4 = "severe stomach pain, blurry vision"
    diagnoses4 = assistant.diagnose(symptoms4)
    print("\n--- Final Diagnoses for Symptoms 4 ---")
    for diag in diagnoses4:
        print(diag)
