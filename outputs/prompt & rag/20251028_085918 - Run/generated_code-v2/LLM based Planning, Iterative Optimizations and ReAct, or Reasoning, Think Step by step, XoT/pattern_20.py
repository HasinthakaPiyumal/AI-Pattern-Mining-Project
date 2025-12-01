class PatientDataModule:
    def __init__(self):
        self.patient_data = {}

    def load_patient_data(self, patient_id, data):
        self.patient_data[patient_id] = data
        return self.patient_data[patient_id]

    def get_patient_data(self, patient_id):
        return self.patient_data.get(patient_id, {})

class InternalKBModule:
    def __init__(self):
        self.knowledge_base = {
            "Fever": {"symptoms": ["high temperature", "chills", "headache"], "possible_diagnoses": ["Flu", "Common Cold", "Bacterial Infection"]},
            "Cough": {"symptoms": ["sore throat", "fatigue"], "possible_diagnoses": ["Bronchitis", "Allergies"]},
            "Flu": {"causes": "Virus", "treatment": "Rest, fluids"},
            "Common Cold": {"causes": "Virus", "treatment": "Symptomatic relief"}
        }

    def get_knowledge(self, topic):
        return self.knowledge_base.get(topic, {})

class ExternalToolIntegrationModule:
    def __init__(self):
        self.external_data = {
            "Flu": {"recent_research": {"diagnosis": "Influenza (A or B)", "prevalence": "Seasonal", "new_treatment": "Antivirals for severe cases"}},
            "Common Cold": {"recent_research": {"diagnosis": "Rhinovirus infection", "prevalence": "Year-round"}}
        }

    def query_external_database(self, query_topic):
        return self.external_data.get(query_topic, {})

class KnowledgeGraph:
    def __init__(self):
        self.graph = {}

    def add_data(self, source, data):
        self.graph[source] = data

    def get_unified_knowledge(self):
        unified = {}
        for source, data in self.graph.items():
            for key, value in data.items():
                if key not in unified:
                    unified[key] = []
                unified[key].append({"source": source, "value": value})
        return unified

class ConflictResolutionEngine:
    def __init__(self):
        self.trust_scores = {
            "internal_kb": 0.8,
            "patient_data": 0.9,
            "external_tool": 0.7
        }

    def detect_conflicts(self, unified_knowledge):
        conflicts = []
        diagnoses = {}

        for key, sources_data in unified_knowledge.items():
            if key == "possible_diagnoses" or key == "diagnosis":
                for item in sources_data:
                    if isinstance(item["value"], list):
                        for diag in item["value"]:
                            diagnoses.setdefault(diag, []).append(item["source"])
                    else:
                        diagnoses.setdefault(item["value"], []).append(item["source"])

        for diag, sources in diagnoses.items():
            if len(sources) > 1 and len(set(sources)) > 1: # Conflict if same diagnosis from multiple *different* sources
                # More sophisticated conflict detection would compare details, but for simplicity, we focus on differing sources for the same diagnosis.
                pass # The conflict resolution will handle weighing

        return conflicts # Currently just an empty list, resolution happens in next step

    def evaluate_trustworthiness(self, source):
        return self.trust_scores.get(source, 0.5) # Default trustworthiness

    def resolve_conflicts(self, unified_knowledge):
        potential_diagnoses = {}
        explanations = []

        for key, sources_data in unified_knowledge.items():
            if key == "possible_diagnoses" or key == "diagnosis":
                for item in sources_data:
                    source = item["source"]
                    trust_score = self.evaluate_trustworthiness(source)
                    
                    current_diagnoses = []
                    if isinstance(item["value"], list):
                        current_diagnoses.extend(item["value"])
                    else:
                        current_diagnoses.append(item["value"])

                    for diag in current_diagnoses:
                        if diag not in potential_diagnoses:
                            potential_diagnoses[diag] = {"score": 0, "sources": {}}
                        
                        # Add trustworthiness score for this diagnosis from this source
                        potential_diagnoses[diag]["score"] += trust_score
                        potential_diagnoses[diag]["sources"].setdefault(source, []).append(item["value"])

        ranked_diagnoses = sorted(potential_diagnoses.items(), key=lambda item: item[1]["score"], reverse=True)

        final_diagnoses_with_explanation = []
        for diag, data in ranked_diagnoses:
            explanation = f"Diagnosis: {diag} (Score: {data['score']:.2f})\nSupported by: "
            source_explanations = []
            for source, values in data["sources"].items():
                source_explanations.append(f"{source} (Trust: {self.evaluate_trustworthiness(source):.2f})")
            explanation += ", ".join(source_explanations)
            final_diagnoses_with_explanation.append((diag, explanation))
        
        return final_diagnoses_with_explanation


# --- Main Application Flow --- 
if __name__ == "__main__":
    # 1. Initialize Modules
    patient_data_module = PatientDataModule()
    internal_kb_module = InternalKBModule()
    external_tool_module = ExternalToolIntegrationModule()
    knowledge_graph = KnowledgeGraph()
    conflict_resolution_engine = ConflictResolutionEngine()

    # 2. Ingest Data
    patient_id = "P001"
    patient_info = {
        "symptoms": ["high temperature", "headache", "sore throat"],
        "vitals": {"temperature": 101.5, "heart_rate": 85},
        "lab_results": {"WBC": "normal"}
    }
    patient_data_module.load_patient_data(patient_id, patient_info)
    knowledge_graph.add_data("patient_data", patient_data_module.get_patient_data(patient_id))

    # Get internal KB knowledge related to symptoms
    kb_fever = internal_kb_module.get_knowledge("Fever")
    kb_cough = internal_kb_module.get_knowledge("Cough") # Example of potentially conflicting/additional info
    knowledge_graph.add_data("internal_kb", {"Fever_KB": kb_fever, "Cough_KB": kb_cough})

    # Query external tool for "Flu" and "Common Cold"
    ext_flu = external_tool_module.query_external_database("Flu")
    ext_cold = external_tool_module.query_external_database("Common Cold")
    knowledge_graph.add_data("external_tool", {"Flu_External": ext_flu, "CommonCold_External": ext_cold})

    # 3. Get Unified Knowledge
    unified_knowledge = knowledge_graph.get_unified_knowledge()

    # 4. Detect and Resolve Conflicts
    # conflicts = conflict_resolution_engine.detect_conflicts(unified_knowledge) # Currently not used for direct output
    final_diagnoses_with_explanation = conflict_resolution_engine.resolve_conflicts(unified_knowledge)

    # 5. Present Results
    print("\n--- Medical Diagnostic Assistant Results ---")
    for diag, explanation in final_diagnoses_with_explanation:
        print(explanation)
        print("------------------------------------------")

    # Example with specific conflict
    print("\n--- Scenario with more explicit conflict --- ")
    patient_data_module_2 = PatientDataModule()
    internal_kb_module_2 = InternalKBModule()
    external_tool_module_2 = ExternalToolIntegrationModule()
    knowledge_graph_2 = KnowledgeGraph()
    conflict_resolution_engine_2 = ConflictResolutionEngine()

    patient_info_2 = {
        "symptoms": ["joint pain", "fatigue"],
        "vitals": {"temperature": 99.0},
        "possible_diagnosis_from_sensor": "Lyme Disease" # A sensor or initial AI guess
    }
    patient_data_module_2.load_patient_data("P002", patient_info_2)
    knowledge_graph_2.add_data("patient_data", patient_data_module_2.get_patient_data("P002"))

    internal_kb_module_2.knowledge_base["joint pain"] = {"symptoms": ["swelling", "stiffness"], "possible_diagnoses": ["Arthritis", "Gout"]}
    knowledge_graph_2.add_data("internal_kb", internal_kb_module_2.get_knowledge("joint pain"))

    external_tool_module_2.external_data["Lyme Disease"] = {"recent_research": {"diagnosis": "Borrelia infection", "key_symptom": "bullseye rash", "prevalence": "Tick-borne"}}
    knowledge_graph_2.add_data("external_tool", external_tool_module_2.query_external_database("Lyme Disease"))

    unified_knowledge_2 = knowledge_graph_2.get_unified_knowledge()
    final_diagnoses_with_explanation_2 = conflict_resolution_engine_2.resolve_conflicts(unified_knowledge_2)

    for diag, explanation in final_diagnoses_with_explanation_2:
        print(explanation)
        print("------------------------------------------")