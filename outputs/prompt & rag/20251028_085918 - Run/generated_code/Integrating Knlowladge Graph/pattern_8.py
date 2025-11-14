import gradio as gr
import networkx as nx
from collections import defaultdict

# 1. Medical Knowledge Graph (MKG) - Integrated
class MedicalKnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self._populate_sample_data()

    def _populate_sample_data(self):
        # Diseases
        self.add_entity("Flu", "Disease", {"description": "Influenza, a common viral infection."})
        self.add_entity("Common Cold", "Disease", {"description": "A viral infectious disease of the upper respiratory tract."})
        self.add_entity("Pneumonia", "Disease", {"description": "An inflammatory condition of the lung affecting primarily the small air sacs called alveoli."})
        self.add_entity("Bronchitis", "Disease", {"description": "Inflammation of the lining of your bronchial tubes."})
        self.add_entity("Asthma", "Disease", {"description": "A condition in which your airways narrow and swell."})

        # Symptoms
        self.add_entity("Fever", "Symptom", {})
        self.add_entity("Cough", "Symptom", {})
        self.add_entity("Sore Throat", "Symptom", {})
        self.add_entity("Headache", "Symptom", {})
        self.add_entity("Fatigue", "Symptom", {})
        self.add_entity("Runny Nose", "Symptom", {})
        self.add_entity("Body Aches", "Symptom", {})
        self.add_entity("Shortness of Breath", "Symptom", {})
        self.add_entity("Chest Pain", "Symptom", {})
        self.add_entity("Wheezing", "Symptom", {})
        self.add_entity("Chills", "Symptom", {})

        # Treatments
        self.add_entity("Rest", "Treatment", {})
        self.add_entity("Fluids", "Treatment", {})
        self.add_entity("Pain Relievers", "Treatment", {})
        self.add_entity("Antibiotics", "Treatment", {})
        self.add_entity("Bronchodilators", "Treatment", {})
        self.add_entity("Steroids", "Treatment", {})

        # Relations
        self.add_relation("Flu", "Fever", "has_symptom", {})
        self.add_relation("Flu", "Cough", "has_symptom", {})
        self.add_relation("Flu", "Body Aches", "has_symptom", {})
        self.add_relation("Flu", "Fatigue", "has_symptom", {})
        self.add_relation("Flu", "Headache", "has_symptom", {})
        self.add_relation("Flu", "Chills", "has_symptom", {})
        self.add_relation("Flu", "Rest", "treats", {})
        self.add_relation("Flu", "Fluids", "treats", {})
        self.add_relation("Flu", "Pain Relievers", "treats", {})

        self.add_relation("Common Cold", "Runny Nose", "has_symptom", {})
        self.add_relation("Common Cold", "Sore Throat", "has_symptom", {})
        self.add_relation("Common Cold", "Cough", "has_symptom", {})
        self.add_relation("Common Cold", "Fatigue", "has_symptom", {})
        self.add_relation("Common Cold", "Rest", "treats", {})
        self.add_relation("Common Cold", "Fluids", "treats", {})

        self.add_relation("Pneumonia", "Cough", "has_symptom", {})
        self.add_relation("Pneumonia", "Fever", "has_symptom", {})
        self.add_relation("Pneumonia", "Shortness of Breath", "has_symptom", {})
        self.add_relation("Pneumonia", "Chest Pain", "has_symptom", {})
        self.add_relation("Pneumonia", "Fatigue", "has_symptom", {})
        self.add_relation("Pneumonia", "Antibiotics", "treats", {})
        self.add_relation("Pneumonia", "Rest", "treats", {})

        self.add_relation("Bronchitis", "Cough", "has_symptom", {})
        self.add_relation("Bronchitis", "Fatigue", "has_symptom", {})
        self.add_relation("Bronchitis", "Shortness of Breath", "has_symptom", {})
        self.add_relation("Bronchitis", "Chest Pain", "has_symptom", {})
        self.add_relation("Bronchitis", "Antibiotics", "treats", {"note": "if bacterial"})
        self.add_relation("Bronchitis", "Bronchodilators", "treats", {})

        self.add_relation("Asthma", "Wheezing", "has_symptom", {})
        self.add_relation("Asthma", "Shortness of Breath", "has_symptom", {})
        self.add_relation("Asthma", "Cough", "has_symptom", {})
        self.add_relation("Asthma", "Chest Pain", "has_symptom", {})
        self.add_relation("Asthma", "Bronchodilators", "treats", {})
        self.add_relation("Asthma", "Steroids", "treats", {})


    def add_entity(self, entity_id, entity_type, attributes=None):
        if attributes is None:
            attributes = {}
        self.graph.add_node(entity_id, type=entity_type, **attributes)

    def add_relation(self, entity1_id, entity2_id, relation_type, attributes=None):
        if attributes is None:
            attributes = {}
        if not self.graph.has_node(entity1_id):
            print(f"Warning: Entity {entity1_id} not found in graph.")
            return
        if not self.graph.has_node(entity2_id):
            print(f"Warning: Entity {entity2_id} not found in graph.")
            return
        self.graph.add_edge(entity1_id, entity2_id, type=relation_type, **attributes)

    def query_by_symptom(self, symptom_name, depth=2):
        results = set()
        if symptom_name not in self.graph or self.graph.nodes[symptom_name].get("type") != "Symptom":
            return results

        # Find entities that have this symptom
        for u, v, data in self.graph.in_edges(symptom_name, data=True):
            if data.get("type") == "has_symptom" and self.graph.nodes[u].get("type") == "Disease":
                results.add(u)
        return list(results)

    def get_all_symptoms_for_disease(self, disease_name):
        symptoms = set()
        if disease_name not in self.graph or self.graph.nodes[disease_name].get("type") != "Disease":
            return symptoms

        for u, v, data in self.graph.out_edges(disease_name, data=True):
            if data.get("type") == "has_symptom" and self.graph.nodes[v].get("type") == "Symptom":
                symptoms.add(v)
        return list(symptoms)

    def get_treatments_for_disease(self, disease_name):
        treatments = set()
        if disease_name not in self.graph or self.graph.nodes[disease_name].get("type") != "Disease":
            return treatments

        for u, v, data in self.graph.out_edges(disease_name, data=True):
            if data.get("type") == "treats" and self.graph.nodes[v].get("type") == "Treatment":
                treatments.add(v)
        return list(treatments)

    def prune_graph_by_relevance(self, subgraph_nodes, query_terms, threshold=0.1): # Simplified for simulation
        pruned_nodes = []
        for node_id in subgraph_nodes:
            # Simple relevance check: if node_id or its type/description contains any query term
            is_relevant = False
            node_data = self.graph.nodes[node_id]
            text_to_check = node_id.lower()
            if "description" in node_data:
                text_to_check += " " + node_data["description"].lower()
            text_to_check += " " + node_data.get("type", "").lower()

            for term in query_terms:
                if term.lower() in text_to_check:
                    is_relevant = True
                    break
            if is_relevant:
                pruned_nodes.append(node_id)
        return pruned_nodes


# 2. LLM Agent - Integrated (with MockLLM)
class MockLLM:
    def _generate_response(self, prompt):
        # Simulate LLM responses based on keywords in the prompt
        prompt_lower = prompt.lower()
        if "initial query" in prompt_lower:
            if "symptoms: fever, cough" in prompt_lower:
                return "QUERY: Find diseases with symptoms like Fever, Cough, and their related entities."
            return f"QUERY: Based on the input symptoms, find related medical entities in the knowledge graph. Input: {prompt_lower}"
        elif "refined query" in prompt_lower:
            if "flu" in prompt_lower and "pneumonia" in prompt_lower and "shortness of breath" in prompt_lower:
                return "QUERY: Further investigate 'Pneumonia' vs 'Flu' given 'Shortness of Breath'. Focus on distinguishing symptoms."
            return f"QUERY: Refine search for diseases based on collected facts and new insights. Input: {prompt_lower}"
        elif "reasoning and explanation" in prompt_lower:
            if "flu" in prompt_lower and "fever" in prompt_lower and "cough" in prompt_lower:
                diagnosis = "Flu"
                explanation = "Based on the symptoms of fever, cough, and body aches, and lack of severe shortness of breath, the most likely diagnosis is Influenza (Flu). The knowledge graph indicates Flu commonly presents with these symptoms and is treated with rest and fluids. Pneumonia, while sharing some symptoms, typically involves more severe respiratory distress."
                confidence = 0.9
                return f"DIAGNOSIS: {diagnosis}\nEXPLANATION: {explanation}\nCONFIDENCE: {confidence:.2f}"
            else:
                diagnosis = "Undetermined Condition"
                explanation = "The LLM agent requires more information or further context to provide a definitive diagnosis. Based on the provided facts, a general set of conditions might be considered."
                confidence = 0.5
                return f"DIAGNOSIS: {diagnosis}\nEXPLANATION: {explanation}\nCONFIDENCE: {confidence:.2f}"
        return "LLM_RESPONSE: I am a mock LLM and cannot fully process this request."

class LLMAgent:
    def __init__(self):
        self.llm = MockLLM()

    def generate_initial_kg_query(self, symptoms):
        prompt = f"Generate an initial query for a medical knowledge graph to find diseases and related information based on the following patient symptoms: {', '.join(symptoms)}. Provide this as a natural language query."
        return self.llm._generate_response(prompt)

    def generate_refined_kg_query(self, current_kg_data, symptoms):
        kg_facts_str = "\n".join(current_kg_data)
        prompt = (
            f"Given the current knowledge graph facts:\n{kg_facts_str}\n\nAnd the patient's symptoms: {', '.join(symptoms)}\n\nGenerate a refined query to further explore or confirm potential diagnoses. Focus on distinguishing similar conditions or identifying missing information."
        )
        return self.llm._generate_response(prompt)

    def reason_and_explain(self, patient_history, kg_facts):
        kg_facts_str = "\n".join(kg_facts)
        prompt = (
            f"Based on the patient's history: '{patient_history}'\n\nAnd the following structured knowledge graph facts:\n{kg_facts_str}\n\nProvide a diagnosis, a confidence score (0-1), and a comprehensive, traceable explanation. Ensure the explanation explicitly links the diagnosis to the symptoms and KG facts, mitigating hallucinations."
        )
        return self.llm._generate_response(prompt)


# 3. KGAR Orchestrator & UI - `main_diagnosis_app.py`
def diagnose_patient(patient_symptoms_input, patient_history_input):
    mk_graph = MedicalKnowledgeGraph()
    llm_agent = LLMAgent()

    symptoms_list = [s.strip() for s in patient_symptoms_input.split(',') if s.strip()]

    # --- Stage 1: Initial KG Query Generation and Exploration ---
    initial_llm_query = llm_agent.generate_initial_kg_query(symptoms_list)
    # In a real scenario, this would be parsed to form a structured KG query
    # For this simulation, we'll directly use symptoms to query the MKG

    potential_diseases = set()
    for symptom in symptoms_list:
        diseases_for_symptom = mk_graph.query_by_symptom(symptom.strip())
        potential_diseases.update(diseases_for_symptom)

    kg_facts = []
    for disease in potential_diseases:
        kg_facts.append(f"Disease: {disease}")
        for symptom in mk_graph.get_all_symptoms_for_disease(disease):
            kg_facts.append(f"  - Has symptom: {symptom}")
        for treatment in mk_graph.get_treatments_for_disease(disease):
            kg_facts.append(f"  - Treated by: {treatment}")

    # --- Stage 2: Iterative KG Exploration & Pruning (Simulated) ---
    # Simulate refinement and pruning
    if len(potential_diseases) > 1:
        refined_llm_query = llm_agent.generate_refined_kg_query(kg_facts, symptoms_list)
        # Simulate parsing refined_llm_query and using it for pruning
        query_terms = [term for term in symptoms_list + [d.lower() for d in potential_diseases] if len(term) > 2]
        # Pruning based on relevance to initial symptoms and potential diseases
        pruned_kg_facts = []
        for fact in kg_facts:
            is_relevant = False
            for term in query_terms:
                if term.lower() in fact.lower():
                    is_relevant = True
                    break
            if is_relevant:
                pruned_kg_facts.append(fact)
        kg_facts = pruned_kg_facts

    # --- Stage 3: Reasoning & Explanation ---
    llm_reasoning_output = llm_agent.reason_and_explain(patient_history_input, kg_facts)

    # Parse LLM output (simple parsing for mock responses)
    diagnosis = "N/A"
    explanation = "N/A"
    confidence = "N/A"

    for line in llm_reasoning_output.split('\n'):
        if line.startswith("DIAGNOSIS:"):
            diagnosis = line.replace("DIAGNOSIS:", "").strip()
        elif line.startswith("EXPLANATION:"):
            explanation = line.replace("EXPLANATION:", "").strip()
        elif line.startswith("CONFIDENCE:"):
            confidence = line.replace("CONFIDENCE:", "").strip()

    full_explanation = f"**Diagnosis:** {diagnosis}\n"
    full_explanation += f"**Confidence:** {float(confidence)*100:.0f}%\n\n"
    full_explanation += f"**Detailed Explanation:** {explanation}\n\n"
    full_explanation += "---\n\n**Knowledge Graph Facts Used for Reasoning:**\n"
    if not kg_facts:
        full_explanation += "No specific KG facts were found relevant for this reasoning after pruning."
    else:
        for fact in kg_facts:
            full_explanation += f"- {fact}\n"

    return diagnosis, full_explanation

# Gradio Interface
if __name__ == "__main__":
    # Example usage for direct testing
    # mk_graph = MedicalKnowledgeGraph()
    # agent = LLMAgent()
    # print(agent.generate_initial_kg_query(["Fever", "Cough"]))
    # print(mk_graph.query_by_symptom("Fever"))
    # print(mk_graph.get_all_symptoms_for_disease("Flu"))

    interface = gr.Interface(
        fn=diagnose_patient,
        inputs=[
            gr.Textbox(label="Patient Symptoms (comma-separated)", placeholder="e.g., Fever, Cough, Headache, Body Aches"),
            gr.Textbox(label="Patient History (briefly)", placeholder="e.g., 35-year-old male, symptoms started 2 days ago, no prior medical conditions.")
        ],
        outputs=[
            gr.Textbox(label="Likely Diagnosis"),
            gr.Markdown(label="Reasoning and Explanation")
        ],
        title="Medical Diagnosis Assistant (KGAR Demo)",
        description="Leveraging Knowledge Graph Agentic Reasoning (KGAR) to provide traceable and reliable medical diagnoses based on symptoms and patient history.",
        examples=[
            ["Fever, Cough, Body Aches, Fatigue", "30-year-old female, symptoms started yesterday, feels run down."],
            ["Runny Nose, Sore Throat, Mild Cough", "5-year-old child, feeling unwell for 2 days."],
            ["Cough, Shortness of Breath, Chest Pain, Fever", "60-year-old male with history of smoking, symptoms worsening over a week."]
        ]
    )

    interface.launch(debug=True)
