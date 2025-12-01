import networkx as nx
import gradio as gr
import random
from collections import defaultdict


class MedicalKnowledgeBase:
    def __init__(self):
        self.symptoms_to_diseases = {
            "fever": ["influenza", "malaria", "pneumonia"],
            "cough": ["influenza", "bronchitis", "pneumonia"],
            "headache": ["influenza", "migraine", "meningitis"],
            "fatigue": ["influenza", "anemia", "chronic_fatigue_syndrome"],
            "sore_throat": ["influenza", "strep_throat"],
            "muscle_pain": ["influenza", "fibromyalgia"],
            "shortness_of_breath": ["pneumonia", "asthma"],
            "rash": ["measles", "chickenpox"],
            "nausea": ["food_poisoning", "influenza"],
            "vomiting": ["food_poisoning", "influenza"],
            "abdominal_pain": ["appendicitis", "food_poisoning"],
            "diarrhea": ["food_poisoning", "gastroenteritis"]
        }

        self.diseases_to_tests = {
            "influenza": ["rapid_flu_test", "throat_swab"],
            "malaria": ["blood_smear"],
            "pneumonia": ["chest_xray", "sputum_culture"],
            "bronchitis": ["chest_xray"],
            "migraine": ["neurological_exam"],
            "meningitis": ["lumbar_puncture"],
            "anemia": ["cbc_test"],
            "strep_throat": ["rapid_strep_test"],
            "appendicitis": ["ct_scan_abdomen"],
            "food_poisoning": ["stool_culture"]
        }

        self.diseases_to_treatments = {
            "influenza": ["antivirals", "rest", "fluids"],
            "malaria": ["antimalarials"],
            "pneumonia": ["antibiotics", "oxygen_therapy"],
            "bronchitis": ["cough_suppressants", "bronchodilators"],
            "migraine": ["pain_relievers", "triptans"],
            "meningitis": ["antibiotics"], # Bacterial meningitis
            "anemia": ["iron_supplements", "dietary_changes"],
            "strep_throat": ["antibiotics"],
            "appendicitis": ["surgery"],
            "food_poisoning": ["rest", "hydration"],
            "measles": ["rest", "vitamin_a_supplements"],
            "chickenpox": ["antihistamines", "calamine_lotion"],
            "gastroenteritis": ["hydration", "electrolytes"]
        }

        self.test_results_impact = {
            "positive_rapid_flu_test": {"confirms": "influenza", "negates": []},
            "negative_rapid_flu_test": {"confirms": [], "negates": ["influenza"]},
            "positive_blood_smear": {"confirms": "malaria", "negates": []},
            "negative_blood_smear": {"confirms": [], "negates": ["malaria"]},
            "abnormal_chest_xray": {"confirms": ["pneumonia", "bronchitis"], "negates": []},
            "normal_chest_xray": {"confirms": [], "negates": ["pneumonia", "bronchitis"]}
        }

    def get_potential_diseases(self, symptoms):
        potential_diseases = set()
        for symptom in symptoms:
            if symptom in self.symptoms_to_diseases:
                potential_diseases.update(self.symptoms_to_diseases[symptom])
        return list(potential_diseases)

    def get_related_tests(self, diseases):
        related_tests = set()
        for disease in diseases:
            if disease in self.diseases_to_tests:
                related_tests.update(self.diseases_to_tests[disease])
        return list(related_tests)

    def get_treatments(self, disease):
        return self.diseases_to_treatments.get(disease, [])

    def interpret_test_result(self, test_result):
        return self.test_results_impact.get(test_result, {"confirms": [], "negates": []})


class MockLLM:
    def __init__(self, knowledge_base: MedicalKnowledgeBase):
        self.kb = knowledge_base

    def generate_thoughts(self, prompt: str, current_graph_state: nx.DiGraph):
        # This is a highly simplified simulation of an LLM's reasoning process.
        # In a real GoT, the LLM would dynamically generate these thoughts.
        thoughts = []
        edges = []

        current_symptoms = [node_data["content"] for node, node_data in current_graph_state.nodes(data=True) if node_data["type"] == "symptom"]
        current_diseases = [node_data["content"] for node, node_data in current_graph_state.nodes(data=True) if node_data["type"] == "disease"]
        current_tests = [node_data["content"] for node, node_data in current_graph_state.nodes(data=True) if node_data["type"] == "test"]
        current_test_results = [node_data["content"] for node, node_data in current_graph_state.nodes(data=True) if node_data["type"] == "test_result"]

        if "initial symptoms" in prompt.lower():
            potential_diseases = self.kb.get_potential_diseases(current_symptoms)
            for disease in potential_diseases:
                if disease not in current_diseases:
                    thoughts.append({"type": "disease", "content": disease, "confidence": 0.6 + random.random() * 0.2}) # Simulated confidence
                    for symptom in current_symptoms:
                        if symptom in self.kb.symptoms_to_diseases and disease in self.kb.symptoms_to_diseases[symptom]:
                             edges.append((symptom, disease, {"relation": "suggests"}))

        elif "suggest diagnostic tests" in prompt.lower() and current_diseases:
            for disease in current_diseases:
                related_tests = self.kb.get_related_tests([disease])
                for test in related_tests:
                    if test not in current_tests:
                        thoughts.append({"type": "test", "content": test, "confidence": 0.7})
                        edges.append((disease, test, {"relation": "test_for"}))

        elif "interpret test results" in prompt.lower() and current_test_results:
            for result in current_test_results:
                impact = self.kb.interpret_test_result(result)
                for confirmed_disease in impact["confirms"]:
                    if confirmed_disease not in current_diseases:
                        thoughts.append({"type": "disease", "content": confirmed_disease, "confidence": 0.95})
                        edges.append((result, confirmed_disease, {"relation": "confirms"}))
                for negated_disease in impact["negates"]:
                    # In a real system, we'd adjust confidence or prune paths
                    pass # For simplicity, just adding confirmations

        elif "propose treatments" in prompt.lower() and current_diseases:
            for disease in current_diseases:
                existing_treatments = [node_data["content"] for node, node_data in current_graph_state.nodes(data=True) if node_data["type"] == "treatment" and node_data.get("for_disease") == disease]
                related_treatments = self.kb.get_treatments(disease)
                for treatment in related_treatments:
                    if treatment not in existing_treatments:
                        thoughts.append({"type": "treatment", "content": treatment, "confidence": 0.8, "for_disease": disease})
                        edges.append((disease, treatment, {"relation": "treats"}))

        return thoughts, edges


class GoTReasoningEngine:
    def __init__(self, knowledge_base: MedicalKnowledgeBase, llm):
        self.graph = nx.DiGraph()
        self.kb = knowledge_base
        self.llm = llm # MockLLM instance
        self.thought_counter = 0

    def _generate_thought_id(self, thought_type, content):
        self.thought_counter += 1
        return f"{thought_type}_{content.replace(' ', '_')}_{self.thought_counter}"

    def add_thought(self, thought_type: str, content: str, confidence: float = 0.5):
        thought_id = self._generate_thought_id(thought_type, content)
        self.graph.add_node(thought_id, type=thought_type, content=content, confidence=confidence)
        return thought_id

    def add_dependency(self, source_thought_id: str, target_thought_id: str, relation: str = "related_to"):
        self.graph.add_edge(source_thought_id, target_thought_id, relation=relation)

    def _get_current_graph_summary(self) -> str:
        summary_parts = []
        symptoms = []
        diseases = []
        tests = []
        test_results = []

        for node, data in self.graph.nodes(data=True):
            if data["type"] == "symptom":
                symptoms.append(data["content"])
            elif data["type"] == "disease":
                diseases.append(f"{data['content']} (confidence: {data['confidence']:.2f})")
            elif data["type"] == "test":
                tests.append(data["content"])
            elif data["type"] == "test_result":
                test_results.append(data["content"])

        if symptoms: summary_parts.append(f"Symptoms: {', '.join(symptoms)}")
        if diseases: summary_parts.append(f"Potential Diseases: {', '.join(diseases)}")
        if tests: summary_parts.append(f"Suggested Tests: {', '.join(tests)}")
        if test_results: summary_parts.append(f"Test Results: {', '.join(test_results)}")

        return ". ".join(summary_parts) if summary_parts else "No information in graph yet."

    def _generate_prompt_for_llm(self, step: str) -> str:
        current_summary = self._get_current_graph_summary()
        if step == "symptoms_to_diseases":
            return f"Given the current symptoms: {current_summary}, what are the most likely diseases to consider?"
        elif step == "diseases_to_tests":
            return f"Based on the potential diseases: {current_summary}, what diagnostic tests should be suggested?"
        elif step == "interpret_tests":
            return f"Considering the existing test results and potential diseases: {current_summary}, how should these results influence the diagnosis?"
        elif step == "diagnose_and_treat":
            return f"Based on all gathered information and confirmed/negated diseases: {current_summary}, provide a final diagnosis and propose treatments."
        return f"Continue reasoning based on: {current_summary}"

    def reason(self, initial_symptoms: list, max_iterations: int = 5):
        # Initialize graph with initial symptoms
        symptom_ids = {}
        for symptom in initial_symptoms:
            symptom_id = self.add_thought("symptom", symptom, 1.0)
            symptom_ids[symptom] = symptom_id

        current_diseases_in_graph = set()
        current_tests_in_graph = set()
        current_test_results_in_graph = set()

        # Iterative GoT reasoning process
        for i in range(max_iterations):
            # Step 1: Symptoms to Potential Diseases
            prompt_symptoms_to_diseases = self._generate_prompt_for_llm("symptoms_to_diseases")
            new_thoughts, new_edges = self.llm.generate_thoughts(prompt_symptoms_to_diseases, self.graph)
            for thought in new_thoughts:
                if thought["type"] == "disease" and thought["content"] not in current_diseases_in_graph:
                    thought_id = self.add_thought(thought["type"], thought["content"], thought["confidence"])
                    current_diseases_in_graph.add(thought["content"])
                    for src, dest, data in new_edges:
                        if dest == thought["content"] and src in symptom_ids:
                            self.add_dependency(symptom_ids[src], thought_id, data["relation"])

            # Step 2: Diseases to Diagnostic Tests
            prompt_diseases_to_tests = self._generate_prompt_for_llm("diseases_to_tests")
            new_thoughts, new_edges = self.llm.generate_thoughts(prompt_diseases_to_tests, self.graph)
            for thought in new_thoughts:
                if thought["type"] == "test" and thought["content"] not in current_tests_in_graph:
                    thought_id = self.add_thought(thought["type"], thought["content"], thought["confidence"])
                    current_tests_in_graph.add(thought["content"])
                    for src, dest, data in new_edges:
                        src_node_id = next((node for node, d in self.graph.nodes(data=True) if d['content'] == src and d['type'] == 'disease'), None)
                        if src_node_id:
                            self.add_dependency(src_node_id, thought_id, data["relation"])
                    # Simulate a test result immediately after suggesting a test
                    test_result_content = f"{'positive' if random.random() > 0.3 else 'negative'}_{thought['content']}"
                    test_result_id = self.add_thought("test_result", test_result_content, 0.9)
                    current_test_results_in_graph.add(test_result_content)
                    self.add_dependency(thought_id, test_result_id, "has_result")

            # Step 3: Interpret Test Results to refine Diseases
            prompt_interpret_tests = self._generate_prompt_for_llm("interpret_tests")
            new_thoughts, new_edges = self.llm.generate_thoughts(prompt_interpret_tests, self.graph)
            for thought in new_thoughts:
                if thought["type"] == "disease" and thought["content"] not in current_diseases_in_graph:
                    thought_id = self.add_thought(thought["type"], thought["content"], thought["confidence"])
                    current_diseases_in_graph.add(thought["content"])
                    for src, dest, data in new_edges:
                        src_node_id = next((node for node, d in self.graph.nodes(data=True) if d['content'] == src and d['type'] == 'test_result'), None)
                        if src_node_id:
                            self.add_dependency(src_node_id, thought_id, data["relation"])


        # Aggregation Logic: Determine the most probable diagnosis and treatment
        final_diagnosis = self.aggregate_results()

        return final_diagnosis

    def aggregate_results(self):
        disease_scores = defaultdict(float)
        disease_paths = defaultdict(list)

        # Propagate confidence through the graph to score diseases
        for node in nx.topological_sort(self.graph):
            node_data = self.graph.nodes[node]
            node_type = node_data["type"]
            node_content = node_data["content"]
            node_confidence = node_data["confidence"]

            if node_type == "disease":
                disease_scores[node_content] += node_confidence
                disease_paths[node_content].append(node_content)

            # Simple propagation: parents contribute to child's score
            for pred in self.graph.predecessors(node):
                pred_data = self.graph.nodes[pred]
                pred_content = pred_data["content"]
                pred_confidence = pred_data["confidence"]
                edge_relation = self.graph[pred][node]["relation"]

                if node_type == "disease":
                    disease_scores[node_content] += pred_confidence * 0.5 # Example weight
                    disease_paths[node_content].append(f"{pred_content} -> {node_content}")
                elif node_type == "test_result":
                    # If a test result confirms a disease, boost that disease's score
                    impact = self.kb.interpret_test_result(node_content)
                    for confirmed_disease in impact["confirms"]:
                        disease_scores[confirmed_disease] += 1.5 # Strong boost

        # Filter out diseases with low scores or no strong evidence
        primary_diagnosis = None
        max_score = 0
        for disease, score in disease_scores.items():
            if score > max_score and score > 1.0: # Threshold for a valid diagnosis
                max_score = score
                primary_diagnosis = disease

        if primary_diagnosis:
            treatments = self.kb.get_treatments(primary_diagnosis)
            reasoning_path = list(set(disease_paths[primary_diagnosis])) # Unique elements
            return {
                "diagnosis": primary_diagnosis,
                "confidence": max_score,
                "treatments": treatments,
                "reasoning": reasoning_path
            }
        else:
            return {
                "diagnosis": "Undetermined",
                "confidence": 0,
                "treatments": [],
                "reasoning": ["Could not converge on a clear diagnosis with current information."]
            }


def diagnose_patient(symptoms_input):
    kb = MedicalKnowledgeBase()
    llm = MockLLM(kb)
    engine = GoTReasoningEngine(kb, llm)

    initial_symptoms = [s.strip().lower() for s in symptoms_input.split(',') if s.strip()]
    if not initial_symptoms:
        return "Please enter at least one symptom."

    result = engine.reason(initial_symptoms)

    output_str = f"## Diagnosis Result\n\n"
    output_str += f"**Diagnosis:** {result['diagnosis'].replace('_', ' ').title()}\n"
    output_str += f"**Confidence Score:** {result['confidence']:.2f}\n"
    
    if result['treatments']:
        output_str += f"**Recommended Treatments:** {', '.join([t.replace('_', ' ').title() for t in result['treatments']])}\n"
    else:
        output_str += "**Recommended Treatments:** No specific treatments found.\n"
    
    output_str += f"\n**Reasoning Path (Simplified):**\n"
    for step in result['reasoning']:
        output_str += f"- {step}\n"
    
    output_str += f"\n### Full Graph (Nodes and Edges):\n"
    output_str += f"**Nodes ({len(engine.graph.nodes)}):**\n"
    for node_id, data in engine.graph.nodes(data=True):
        output_str += f"- ID: {node_id}, Type: {data['type']}, Content: {data['content']}, Confidence: {data.get('confidence', 0):.2f}\n"
    
    output_str += f"\n**Edges ({len(engine.graph.edges)}):**\n"
    for u, v, data in engine.graph.edges(data=True):
        output_str += f"- {engine.graph.nodes[u]['content']} --({data.get('relation', 'related_to')})--> {engine.graph.nodes[v]['content']}\n"


    return output_str


if __name__ == "__main__":
    # Example Usage without Gradio (for testing):
    # kb = MedicalKnowledgeBase()
    # llm = MockLLM(kb)
    # engine = GoTReasoningEngine(kb, llm)
    # initial_symptoms = ["fever", "cough"]
    # diagnosis = engine.reason(initial_symptoms)
    # print(diagnosis)

    iface = gr.Interface(
        fn=diagnose_patient,
        inputs=gr.Textbox(lines=2, placeholder="Enter symptoms separated by commas (e.g., fever, cough, headache)"),
        outputs="markdown",
        title="Intelligent Medical Diagnosis System (GoT Demo)",
        description="Enter a list of symptoms to get a potential diagnosis and treatment recommendations based on a Graph-of-Thoughts reasoning process."
    )

    iface.launch()