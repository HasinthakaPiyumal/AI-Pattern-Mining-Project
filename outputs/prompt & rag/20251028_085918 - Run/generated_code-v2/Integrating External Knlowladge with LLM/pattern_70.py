
class KnowledgeGraph:
    """
    A highly simplified Knowledge Graph implementation using dictionaries.
    Simulates storing medical entities and their relationships.
    """
    def __init__(self):
        self.graph = {} # Adjacency list: {entity: {relation: [target_entities]}}
        self.entities = set()
        self.relations = set()

    def add_entity(self, entity_name):
        if entity_name not in self.graph:
            self.graph[entity_name] = {}
            self.entities.add(entity_name)

    def add_relation(self, source, relation, target):
        self.add_entity(source)
        self.add_entity(target)
        if relation not in self.graph[source]:
            self.graph[source][relation] = []
        self.graph[source][relation].append(target)
        self.relations.add(relation)

    def get_neighbors(self, entity, relation=None):
        """Returns entities connected to 'entity' via 'relation'."""
        if entity not in self.graph:
            return []
        if relation:
            return self.graph[entity].get(relation, [])
        else:
            neighbors = []
            for rel, targets in self.graph[entity].items():
                neighbors.extend(targets)
            return list(set(neighbors)) # Remove duplicates

    def find_paths(self, start_entity, end_entity, max_depth=3):
        """
        A very simplified breadth-first search (BFS-like) for demonstration.
        Returns a list of paths, where each path is a list of (entity, relation, entity) triples.
        """
        if start_entity not in self.graph or end_entity not in self.graph:
            return []

        paths = []
        # Queue stores tuples: (current_path_entities, current_path_triples)
        queue = [([start_entity], [(start_entity, None, start_entity)])]

        while queue:
            current_entities_path, current_triples_path = queue.pop(0)
            current_node = current_entities_path[-1]

            if current_node == end_entity:
                paths.append(current_triples_path)
                continue

            if len(current_entities_path) -1 >= max_depth: # -1 because the first element is the start
                continue

            for relation, targets in self.graph.get(current_node, {}).items():
                for target_node in targets:
                    if target_node not in current_entities_path: # Avoid simple cycles
                        new_entities_path = current_entities_path + [target_node]
                        new_triples_path = current_triples_path + [(current_node, relation, target_node)]
                        queue.append((new_entities_path, new_triples_path))
        return paths

    def populate_example_data(self):
        """Populates the KG with a small set of rare disease-related information."""
        self.add_entity("Patient A")
        self.add_entity("Symptom: Fatigue")
        self.add_entity("Symptom: Muscle Weakness")
        self.add_entity("Symptom: Dyspnea")
        self.add_entity("Disease: Myasthenia Gravis")
        self.add_entity("Disease: ALS")
        self.add_entity("Disease: Pompe Disease")
        self.add_entity("Genetic Marker: CHRNE mutation")
        self.add_entity("Genetic Marker: SOD1 mutation")
        self.add_entity("Genetic Marker: GAA mutation")
        self.add_entity("Test: EMG")
        self.add_entity("Test: Genetic Test CHRNE")
        self.add_entity("Test: Genetic Test SOD1") # Added for examples
        self.add_entity("Test: Genetic Test GAA")   # Added for examples
        self.add_entity("Test: Muscle Biopsy")
        self.add_entity("Treatment: Pyridostigmine")
        self.add_entity("Treatment: Enzyme Replacement Therapy")
        self.add_entity("Treatment: Riluzole")
        self.add_entity("Research: Autoimmune Neuromuscular Disorders")
        self.add_entity("Research: Motor Neuron Disease Genetics")
        self.add_entity("Research: Lysosomal Storage Disorders")

        self.add_relation("Patient A", "has_symptom", "Symptom: Fatigue")
        self.add_relation("Patient A", "has_symptom", "Symptom: Muscle Weakness")

        self.add_relation("Disease: Myasthenia Gravis", "has_symptom", "Symptom: Fatigue")
        self.add_relation("Disease: Myasthenia Gravis", "has_symptom", "Symptom: Muscle Weakness")
        self.add_relation("Disease: Myasthenia Gravis", "associated_with", "Genetic Marker: CHRNE mutation")
        self.add_relation("Disease: Myasthenia Gravis", "diagnosed_by", "Test: EMG")
        self.add_relation("Disease: Myasthenia Gravis", "treated_by", "Treatment: Pyridostigmine")
        self.add_relation("Disease: Myasthenia Gravis", "falls_under", "Research: Autoimmune Neuromuscular Disorders")

        self.add_relation("Disease: ALS", "has_symptom", "Symptom: Muscle Weakness")
        self.add_relation("Disease: ALS", "has_symptom", "Symptom: Dyspnea")
        self.add_relation("Disease: ALS", "associated_with", "Genetic Marker: SOD1 mutation")
        self.add_relation("Disease: ALS", "diagnosed_by", "Test: EMG")
        self.add_relation("Disease: ALS", "treated_by", "Treatment: Riluzole")
        self.add_relation("Disease: ALS", "falls_under", "Research: Motor Neuron Disease Genetics")

        self.add_relation("Disease: Pompe Disease", "has_symptom", "Symptom: Muscle Weakness")
        self.add_relation("Disease: Pompe Disease", "has_symptom", "Symptom: Fatigue")
        self.add_relation("Disease: Pompe Disease", "associated_with", "Genetic Marker: GAA mutation")
        self.add_relation("Disease: Pompe Disease", "diagnosed_by", "Test: Muscle Biopsy")
        self.add_relation("Disease: Pompe Disease", "treated_by", "Treatment: Enzyme Replacement Therapy")
        self.add_relation("Disease: Pompe Disease", "falls_under", "Research: Lysosomal Storage Disorders")

        self.add_relation("Genetic Marker: CHRNE mutation", "detected_by", "Test: Genetic Test CHRNE")
        self.add_relation("Genetic Marker: SOD1 mutation", "detected_by", "Test: Genetic Test SOD1")
        self.add_relation("Genetic Marker: GAA mutation", "detected_by", "Test: Genetic Test GAA")

def generate_planning_examples(kg_data):
    """
    Generates synthetic instruction-following examples for Planning Optimization.
    These examples train the LLM to generate KG-grounded diagnostic paths.
    Args:
        kg_data: The KnowledgeGraph instance to draw data from (conceptually).
    Returns:
        A list of dictionaries, each with 'instruction', 'input', 'output'.
    """
    examples = []

    # Example 1: Simple symptom to disease path
    instruction_1 = "Generate a diagnostic plan (relation path) to connect symptoms to a potential rare disease."
    patient_symptoms_1 = "Symptom: Fatigue, Symptom: Muscle Weakness"
    target_disease_1 = "Disease: Myasthenia Gravis"
    expected_path_output_1 = "Path: Symptom: Fatigue has_symptom Disease: Myasthenia Gravis; Symptom: Muscle Weakness has_symptom Disease: Myasthenia Gravis; Disease: Myasthenia Gravis diagnosed_by Test: EMG; Disease: Myasthenia Gravis associated_with Genetic Marker: CHRNE mutation"
    examples.append({
        "instruction": instruction_1,
        "input": f"Patient symptoms: {patient_symptoms_1}. Consider potential disease: {target_disease_1}.",
        "output": expected_path_output_1
    })

    # Example 2: More complex path involving genetic markers
    instruction_2 = "Generate a diagnostic plan (relation path) to connect symptoms to a potential rare disease."
    patient_symptoms_2 = "Symptom: Muscle Weakness, Symptom: Dyspnea"
    target_disease_2 = "Disease: ALS"
    expected_path_output_2 = "Path: Symptom: Muscle Weakness has_symptom Disease: ALS; Symptom: Dyspnea has_symptom Disease: ALS; Disease: ALS associated_with Genetic Marker: SOD1 mutation; Genetic Marker: SOD1 mutation detected_by Test: Genetic Test SOD1"
    examples.append({
        "instruction": instruction_2,
        "input": f"Patient symptoms: {patient_symptoms_2}. Consider potential disease: {target_disease_2}.",
        "output": expected_path_output_2
    })

    # Example 3: Suggesting tests for a given disease based on symptoms
    instruction_3 = "Given patient symptoms and a suspected disease, suggest relevant diagnostic tests based on the knowledge graph."
    patient_symptoms_3 = "Symptom: Muscle Weakness, Symptom: Fatigue"
    suspected_disease_3 = "Disease: Pompe Disease"
    expected_tests_3 = "Suggested Tests: Test: Muscle Biopsy (for Disease: Pompe Disease); Test: Genetic Test GAA (for Genetic Marker: GAA mutation associated with Disease: Pompe Disease)"
    examples.append({
        "instruction": instruction_3,
        "input": f"Patient symptoms: {patient_symptoms_3}. Suspected disease: {suspected_disease_3}.",
        "output": expected_tests_3
    })
    return examples

def generate_reasoning_examples(kg_data):
    """
    Generates synthetic instruction-following examples for Retrieval-Reasoning Optimization.
    These examples train the LLM to reason over retrieved KG reasoning paths.
    Args:
        kg_data: The KnowledgeGraph instance to draw data from (conceptually).
    Returns:
        A list of dictionaries, each with 'instruction', 'input', 'output'.
    """
    examples = []

    # Example 1: Reasoning about disease likelihood based on a retrieved path
    instruction_1 = "Evaluate the likelihood of a disease and explain your reasoning based on the provided KG path."
    retrieved_path_1 = "Symptom: Fatigue has_symptom Disease: Myasthenia Gravis; Symptom: Muscle Weakness has_symptom Disease: Myasthenia Gravis; Disease: Myasthenia Gravis associated_with Genetic Marker: CHRNE mutation"
    patient_context_1 = "Patient presents with chronic fatigue and progressive muscle weakness."
    expected_reasoning_1 = (
        "Reasoning: The patient's symptoms (Fatigue, Muscle Weakness) directly match symptoms "
        "of Myasthenia Gravis as per the KG path. The path further strengthens this by associating "
        "Myasthenia Gravis with the Genetic Marker: CHRNE mutation. "
        "Likelihood: High. Missing Steps: Confirm presence of CHRNE mutation via a specific test, perform EMG test. "
        "Suggested Treatment: Pyridostigmine (as per KG)."
    )
    examples.append({
        "instruction": instruction_1,
        "input": f"Patient context: {patient_context_1}. Retrieved KG path: {retrieved_path_1}",
        "output": expected_reasoning_1
    })

    # Example 2: Reasoning about an alternative disease with its path
    instruction_2 = "Evaluate the likelihood of a disease and explain your reasoning based on the provided KG path."
    retrieved_path_2 = "Symptom: Muscle Weakness has_symptom Disease: ALS; Symptom: Dyspnea has_symptom Disease: ALS; Disease: ALS associated_with Genetic Marker: SOD1 mutation"
    patient_context_2 = "Patient presents with muscle weakness and shortness of breath (dyspnea)."
    expected_reasoning_2 = (
        "Reasoning: The patient's symptoms (Muscle Weakness, Dyspnea) align with ALS as per the KG path. "
        "The path also indicates an association with Genetic Marker: SOD1 mutation. "
        "Likelihood: Medium-High. Missing Steps: Confirm SOD1 mutation via genetic test, assess severity of dyspnea, further neurological examination. "
        "Suggested Treatment: Riluzole (as per KG)."
    )
    examples.append({
        "instruction": instruction_2,
        "input": f"Patient context: {patient_context_2}. Retrieved KG path: {retrieved_path_2}",
        "output": expected_reasoning_2
    })

    # Example 3: Reasoning with general symptoms and broader context for Pompe Disease
    instruction_3 = "Evaluate the likelihood of a disease and explain your reasoning based on the provided KG path."
    retrieved_path_3 = "Symptom: Muscle Weakness has_symptom Disease: Pompe Disease; Symptom: Fatigue has_symptom Disease: Pompe Disease; Disease: Pompe Disease associated_with Genetic Marker: GAA mutation"
    patient_context_3 = "Patient reports generalized muscle weakness and chronic fatigue, with a family history of an undiagnosed neuromuscular disorder."
    expected_reasoning_3 = (
        "Reasoning: The patient's general symptoms (Muscle Weakness, Fatigue) are consistent with Pompe Disease "
        "as shown in the KG path. The family history adds to the suspicion. "
        "Likelihood: Medium. Missing Steps: Conduct Muscle Biopsy, perform Genetic Test GAA. "
        "Suggested Treatment: Enzyme Replacement Therapy (as per KG)."
    )
    examples.append({
        "instruction": instruction_3,
        "input": f"Patient context: {patient_context_3}. Retrieved KG path: {retrieved_path_3}",
        "output": expected_reasoning_3
    })

    return examples

class SimulatedLLM:
    """
    A highly simplified, simulated LLM for demonstration purposes.
    It 'learns' by storing input-output mappings from generated instruction data.
    In a real scenario, this would be a fine-tuned LLM (e.g., from Hugging Face Transformers).
    """
    def __init__(self):
        self.knowledge_base = {} # Maps (instruction, input) -> output
        self.fallback_response = "I need more information or specific training to answer that." # Default response

    def simulate_finetuning(self, instruction_data):
        """
        Simulates the fine-tuning process by populating the internal knowledge base.
        In a real LLM, this would involve gradient descent on a large dataset.
        """
        print("Simulating LLM fine-tuning with provided instruction data...")
        for example in instruction_data:
            key = (example["instruction"], example["input"])
            self.knowledge_base[key] = example["output"]
        print(f"Simulated LLM trained on {len(instruction_data)} examples.")

    def predict(self, instruction, input_text):
        """
        Simulates LLM inference. It tries to find an exact match in its
        "knowledge_base". In a real LLM, it would generalize from training.
        """
        key = (instruction, input_text)
        if key in self.knowledge_base:
            print(f"SimulatedLLM: Found exact match for '{instruction}' with input '{input_text[:50]}...'\n")
            return self.knowledge_base[key]
        else:
            print(f"SimulatedLLM: No exact match found for '{instruction}' with input '{input_text[:50]}...'. Using fallback.\n")
            return self.fallback_response


class RareDiseaseDiagnosticAssistant:
    """
    The main application class for the Rare Disease Diagnostic Assistant.
    It orchestrates the interaction between the Knowledge Graph and the Simulated LLM.
    """
    def __init__(self):
        self.kg = KnowledgeGraph()
        self.kg.populate_example_data()
        self.llm = SimulatedLLM()
        self._initialize_llm_training()

    def _initialize_llm_training(self):
        """Gathers all synthetic training data and 'fine-tunes' the simulated LLM."""
        print("\n--- Initializing LLM Training for RDDA ---")
        planning_data = generate_planning_examples(self.kg)
        reasoning_data = generate_reasoning_examples(self.kg)
        
        # Combine all training data for the simulated LLM
        all_training_data = planning_data + reasoning_data
        self.llm.simulate_finetuning(all_training_data)
        print("--- LLM Training Initialization Complete ---")

    def find_relevant_kg_paths(self, symptoms, diseases_of_interest=None):
        """
        Simulates retrieving relevant KG paths based on symptoms and potential diseases.
        In a real system, this would involve complex KG queries, embeddings, or advanced graph algorithms.
        For this simulation, we'll try to find direct relations and short paths.
        """
        relevant_paths = []
        symptom_entities = [s for s in self.kg.entities if s.startswith("Symptom:") and any(sym.strip().lower() in s.lower() for sym in symptoms.split(","))]
        
        # First, find diseases that have these symptoms
        potential_diseases = set()
        for symptom in symptom_entities:
            for disease_candidate in self.kg.entities:
                if disease_candidate.startswith("Disease:"):
                    # Check if disease has this symptom
                    if symptom in self.kg.get_neighbors(disease_candidate, "has_symptom"):
                        potential_diseases.add(disease_candidate)
                        path_str = f"{symptom} has_symptom {disease_candidate}"
                        relevant_paths.append(path_str)
        
        # If specific diseases are of interest, prioritize them
        if diseases_of_interest:
            target_diseases = [d for d in self.kg.entities if d.startswith("Disease:") and any(do.strip().lower() in d.lower() for do in diseases_of_interest.split(","))]
            potential_diseases.update(target_diseases)

        # Now, for each potential disease, find associated tests, genetic markers, treatments
        for disease in potential_diseases:
            # Find paths from disease to tests, genetic markers, etc.
            for rel, targets in self.kg.graph.get(disease, {}).items():
                for target_node in targets:
                    path_str = f"{disease} {rel} {target_node}"
                    relevant_paths.append(path_str)
            
            # Also, find paths from associated genetic markers to their tests
            genetic_markers = self.kg.get_neighbors(disease, "associated_with")
            for marker in genetic_markers:
                for rel, targets in self.kg.graph.get(marker, {}).items():
                    for target_node in targets:
                        path_str = f"{marker} {rel} {target_node}"
                        relevant_paths.append(path_str)

        return list(set(relevant_paths)) # Remove duplicates

    def diagnose_patient(self, symptoms, patient_history=None):
        """
        Orchestrates the diagnostic process using the KG and fine-tuned LLM.
        """
        print(f"\n--- Diagnosing Patient with Symptoms: {symptoms} ---")
        
        patient_context = f"Patient presents with symptoms: {symptoms}."
        if patient_history:
            patient_context += f" Additional history: {patient_history}."

        # Step 1: LLM for Planning Optimization (initial diagnostic plan/suspected diseases)
        # We try to hit one of the pre-trained planning examples based on symptoms
        llm_planning_output = "No specific pre-trained diagnostic plan found for these symptoms." 
        suspected_diseases_for_reasoning = []

        if "fatigue" in symptoms.lower() and "muscle weakness" in symptoms.lower():
            # Try to match example 1 for planning
            input_for_planning = "Patient symptoms: Symptom: Fatigue, Symptom: Muscle Weakness. Consider potential disease: Disease: Myasthenia Gravis."
            planning_instruction = "Generate a diagnostic plan (relation path) to connect symptoms to a potential rare disease."
            llm_planning_output = self.llm.predict(planning_instruction, input_for_planning)
            if llm_planning_output != self.llm.fallback_response:
                suspected_diseases_for_reasoning = ["Disease: Myasthenia Gravis"]
            # Also try matching example 3 for suggesting tests for Pompe Disease (if relevant)
            input_for_tests = "Patient symptoms: Symptom: Muscle Weakness, Symptom: Fatigue. Suspected disease: Disease: Pompe Disease."
            instruction_for_tests = "Given patient symptoms and a suspected disease, suggest relevant diagnostic tests based on the knowledge graph."
            test_suggestions = self.llm.predict(instruction_for_tests, input_for_tests)
            if test_suggestions != self.llm.fallback_response:
                llm_planning_output += f"\nAdditional suggestions: {test_suggestions}"
                suspected_diseases_for_reasoning.append("Disease: Pompe Disease")

        elif "muscle weakness" in symptoms.lower() and "dyspnea" in symptoms.lower():
            # Try to match example 2 for planning
            input_for_planning = "Patient symptoms: Symptom: Muscle Weakness, Symptom: Dyspnea. Consider potential disease: Disease: ALS."
            planning_instruction = "Generate a diagnostic plan (relation path) to connect symptoms to a potential rare disease."
            llm_planning_output = self.llm.predict(planning_instruction, input_for_planning)
            if llm_planning_output != self.llm.fallback_response:
                suspected_diseases_for_reasoning = ["Disease: ALS"]

        print(f"LLM Planning Output (Initial Diagnostic Plan):\n{llm_planning_output}")
        
        # Step 2: Retrieve relevant KG paths based on symptoms and suspected diseases
        # We use the diseases identified by the planning LLM or general symptom matching
        retrieved_paths_for_reasoning = []
        if suspected_diseases_for_reasoning:
            for disease in suspected_diseases_for_reasoning:
                retrieved_paths_for_reasoning.extend(self.find_relevant_kg_paths(symptoms, disease))
        else:
            retrieved_paths_for_reasoning.extend(self.find_relevant_kg_paths(symptoms))
        
        retrieved_paths_for_reasoning = list(set(retrieved_paths_for_reasoning)) # Remove duplicates

        print(f"\nRetrieved KG Paths (for reasoning):")
        if retrieved_paths_for_reasoning:
            for i, path in enumerate(retrieved_paths_for_reasoning):
                print(f"  Path {i+1}: {path}")
        else:
            print("  No specific KG paths retrieved for detailed reasoning based on current input.")

        # Step 3: LLM for Retrieval-Reasoning Optimization
        # We try to hit one of the pre-trained reasoning examples
        llm_reasoning_output = self.llm.fallback_response
        
        if "fatigue" in symptoms.lower() and "muscle weakness" in symptoms.lower() and "myasthenia gravis" in str(suspected_diseases_for_reasoning).lower():
            # Match reasoning example 1
            reasoning_instruction = "Evaluate the likelihood of a disease and explain your reasoning based on the provided KG path."
            input_for_reasoning = f"Patient context: {patient_context}. Retrieved KG path: Symptom: Fatigue has_symptom Disease: Myasthenia Gravis; Symptom: Muscle Weakness has_symptom Disease: Myasthenia Gravis; Disease: Myasthenia Gravis associated_with Genetic Marker: CHRNE mutation"
            llm_reasoning_output = self.llm.predict(reasoning_instruction, input_for_reasoning)
        
        elif "muscle weakness" in symptoms.lower() and "dyspnea" in symptoms.lower() and "als" in str(suspected_diseases_for_reasoning).lower():
            # Match reasoning example 2
            reasoning_instruction = "Evaluate the likelihood of a disease and explain your reasoning based on the provided KG path."
            input_for_reasoning = f"Patient context: {patient_context}. Retrieved KG path: Symptom: Muscle Weakness has_symptom Disease: ALS; Symptom: Dyspnea has_symptom Disease: ALS; Disease: ALS associated_with Genetic Marker: SOD1 mutation"
            llm_reasoning_output = self.llm.predict(reasoning_instruction, input_for_reasoning)
        
        elif "muscle weakness" in symptoms.lower() and "fatigue" in symptoms.lower() and "pompe disease" in str(suspected_diseases_for_reasoning).lower():
             # Match reasoning example 3
            reasoning_instruction = "Evaluate the likelihood of a disease and explain your reasoning based on the provided KG path."
            input_for_reasoning = f"Patient context: Patient reports generalized muscle weakness and chronic fatigue, with a family history of an undiagnosed neuromuscular disorder.. Retrieved KG path: Symptom: Muscle Weakness has_symptom Disease: Pompe Disease; Symptom: Fatigue has_symptom Disease: Pompe Disease; Disease: Pompe Disease associated_with Genetic Marker: GAA mutation"
            llm_reasoning_output = self.llm.predict(reasoning_instruction, input_for_reasoning)


        print(f"\nLLM Reasoning Output:\n{llm_reasoning_output}")

        print("\n--- Diagnostic Process Complete ---")


# --- Demonstration --- 
if __name__ == "__main__":
    rdda = RareDiseaseDiagnosticAssistant()

    print("\n>>> Scenario 1: Patient with Fatigue and Muscle Weakness <<<")
    rdda.diagnose_patient("Fatigue, Muscle Weakness")

    print("\n>>> Scenario 2: Patient with Muscle Weakness and Dyspnea <<<")
    rdda.diagnose_patient("Muscle Weakness, Dyspnea")

    print("\n>>> Scenario 3: Patient with general symptoms and family history (for Pompe Disease check) <<<")
    rdda.diagnose_patient("Muscle Weakness, Fatigue", patient_history="family history of an undiagnosed neuromuscular disorder")

    print("\n>>> Scenario 4: Patient with symptoms not explicitly in training data (will use fallback) <<<")
    rdda.diagnose_patient("Severe Headache, Blurred Vision")


