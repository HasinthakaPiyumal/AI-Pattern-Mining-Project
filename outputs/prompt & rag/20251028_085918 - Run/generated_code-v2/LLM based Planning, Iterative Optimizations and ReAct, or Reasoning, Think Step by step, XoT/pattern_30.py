import collections

class Symptom:
    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return f"Symptom('{self.name}')"

    def __eq__(self, other):
        return isinstance(other, Symptom) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

class TestResult:
    def __init__(self, test_name: str, result: str):
        self.test_name = test_name
        self.result = result

    def __repr__(self):
        return f"TestResult(test_name='{self.test_name}', result='{self.result}')"

class MedicalCondition:
    def __init__(self, name: str, symptoms: list[Symptom], required_tests: list[str]):
        self.name = name
        self.symptoms = set(symptoms)
        self.required_tests = set(required_tests)

    def __repr__(self):
        return f"MedicalCondition('{self.name}')"

    def __eq__(self, other):
        return isinstance(other, MedicalCondition) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

class DiagnosisState:
    def __init__(self, patient_id: str, symptoms_observed: list[Symptom], medical_history: str, test_results: list[TestResult], current_hypotheses: list[MedicalCondition] = None):
        self.patient_id = patient_id
        self.symptoms_observed = set(symptoms_observed)
        self.medical_history = medical_history
        self.test_results = {tr.test_name: tr.result for tr in test_results}
        self.current_hypotheses = current_hypotheses if current_hypotheses is not None else []

    def __repr__(self):
        return f"DiagnosisState(patient_id='{self.patient_id}', symptoms={len(self.symptoms_observed)}, tests={len(self.test_results)}, hypotheses={len(self.current_hypotheses)})"

    def clone(self):
        return DiagnosisState(
            self.patient_id,
            list(self.symptoms_observed),
            self.medical_history,
            [TestResult(k, v) for k, v in self.test_results.items()],
            list(self.current_hypotheses)
        )

class Thought:
    def __init__(self, thought_type: str, content: any):
        self.thought_type = thought_type  # e.g., "hypothesis", "test_recommendation", "question"
        self.content = content

    def __repr__(self):
        return f"Thought(type='{self.thought_type}', content='{self.content}')"

class Node:
    def __init__(self, diagnosis_state: DiagnosisState, thought: Thought = None, evaluation_score: float = 0.0, parent=None):
        self.diagnosis_state = diagnosis_state
        self.thought = thought
        self.evaluation_score = evaluation_score
        self.parent = parent
        self.children = []

    def add_child(self, child_node):
        self.children.append(child_node)

    def __repr__(self):
        return f"Node(score={self.evaluation_score:.2f}, thought={self.thought.thought_type if self.thought else 'Root'})"

# Simplified Medical Knowledge Base
MEDICAL_KNOWLEDGE = {
    "Influenza": MedicalCondition(
        "Influenza",
        [Symptom("fever"), Symptom("cough"), Symptom("sore throat"), Symptom("body aches")],
        ["Influenza Test"]
    ),
    "Common Cold": MedicalCondition(
        "Common Cold",
        [Symptom("runny nose"), Symptom("sneezing"), Symptom("sore throat")],
        []
    ),
    "Streptococcal Pharyngitis": MedicalCondition(
        "Streptococcal Pharyngitis",
        [Symptom("sore throat"), Symptom("fever"), Symptom("swollen lymph nodes")],
        ["Rapid Strep Test"]
    ),
    "Pneumonia": MedicalCondition(
        "Pneumonia",
        [Symptom("cough"), Symptom("fever"), Symptom("shortness of breath"), Symptom("chest pain")],
        ["Chest X-ray", "Blood Culture"]
    ),
}

class DiagnosisAssistant:
    def __init__(self, medical_knowledge: dict, max_iterations: int = 5, max_branches: int = 3, diagnosis_threshold: float = 0.8):
        self.medical_knowledge = medical_knowledge
        self.max_iterations = max_iterations
        self.max_branches = max_branches
        self.diagnosis_threshold = diagnosis_threshold

    def _simulate_llm_generate_hypotheses(self, state: DiagnosisState) -> list[Thought]:
        possible_conditions = []
        for condition_name, condition_obj in self.medical_knowledge.items():
            # Simple heuristic: if a patient has at least one symptom of the condition
            if state.symptoms_observed.intersection(condition_obj.symptoms):
                possible_conditions.append(Thought("hypothesis", condition_obj))
        return possible_conditions

    def _simulate_llm_generate_test_recommendations(self, state: DiagnosisState) -> list[Thought]:
        recommended_tests = []
        for hypothesis in state.current_hypotheses:
            for test in hypothesis.required_tests:
                if test not in state.test_results:
                    recommended_tests.append(Thought("test_recommendation", test))
        return list(collections.Counter(recommended_tests).keys()) # Deduplicate recommendations

    def _simulate_llm_evaluate_thought(self, thought: Thought, state: DiagnosisState) -> float:
        if thought.thought_type == "hypothesis":
            condition = thought.content
            matched_symptoms = state.symptoms_observed.intersection(condition.symptoms)
            symptom_match_score = len(matched_symptoms) / max(1, len(condition.symptoms)) if condition.symptoms else 0.0

            # Check if all required tests for this hypothesis are done and consistent
            test_confirmation_score = 0.0
            if condition.required_tests:
                completed_tests = {t for t in condition.required_tests if t in state.test_results}
                if completed_tests == condition.required_tests: # All required tests completed
                    test_confirmation_score = 1.0 # Assume positive confirmation for simplicity
                elif completed_tests: # Some tests done, but not all required
                    test_confirmation_score = len(completed_tests) / len(condition.required_tests) * 0.5 # Partial confirmation
            else: # No required tests, rely on symptoms more
                test_confirmation_score = 0.5

            # Combine scores
            score = (symptom_match_score * 0.6) + (test_confirmation_score * 0.4)
            return score

        elif thought.thought_type == "test_recommendation":
            # Higher score for tests that are not yet performed and relevant to current hypotheses
            test_name = thought.content
            if test_name not in state.test_results:
                relevance_score = 0.0
                for hypothesis in state.current_hypotheses:
                    if test_name in hypothesis.required_tests:
                        relevance_score = 0.7 # A test that could confirm a current hypothesis
                        break
                return 0.5 + relevance_score # Base score + relevance boost
            return 0.1 # Already performed, low value
        return 0.0

    def _evaluate_and_add_nodes(self, parent_node: Node, thoughts: list[Thought]) -> list[Node]:
        new_nodes = []
        for thought in thoughts:
            score = self._simulate_llm_evaluate_thought(thought, parent_node.diagnosis_state)
            new_state = parent_node.diagnosis_state.clone()
            if thought.thought_type == "hypothesis":
                # If it's a new hypothesis, add it to the state for subsequent evaluations
                if thought.content not in new_state.current_hypotheses:
                    new_state.current_hypotheses.append(thought.content)
            new_node = Node(new_state, thought, score, parent_node)
            parent_node.add_child(new_node)
            new_nodes.append(new_node)
        return new_nodes

    def _prune_nodes(self, nodes: list[Node]) -> list[Node]:
        # Sort nodes by evaluation score in descending order and keep top N
        nodes.sort(key=lambda node: node.evaluation_score, reverse=True)
        return nodes[:self.max_branches]

    def _simulate_perform_test(self, test_name: str) -> TestResult:
        # Simulate a test result. In a real app, this would query a lab system.
        # For simplicity, let's assume some common results.
        if "Strep" in test_name or "Influenza" in test_name:
            result = "Positive" if collections.random.random() < 0.6 else "Negative" # 60% chance of positive
        elif "X-ray" in test_name:
            result = "Abnormal" if collections.random.random() < 0.4 else "Normal"
        elif "Blood Culture" in test_name:
            result = "Bacteria Detected" if collections.random.random() < 0.3 else "No Bacteria Detected"
        else:
            result = "Normal"
        return TestResult(test_name, result)

    def diagnose(self, patient_id: str, initial_symptoms: list[Symptom], initial_history: str) -> dict:
        initial_state = DiagnosisState(patient_id, initial_symptoms, initial_history, [])
        root_node = Node(initial_state, evaluation_score=0.5) # Initial baseline score
        active_nodes = [root_node]
        best_diagnosis_node = None
        highest_score = 0.0

        for i in range(self.max_iterations):
            print(f"\n--- Iteration {i+1}/{self.max_iterations} --- Current Active Nodes: {len(active_nodes)}")
            newly_generated_nodes = []

            # Sort active nodes to prioritize expansion of promising paths
            active_nodes.sort(key=lambda node: node.evaluation_score, reverse=True)
            current_round_nodes_to_expand = active_nodes[:self.max_branches]

            for node in current_round_nodes_to_expand:
                print(f"Expanding node (score: {node.evaluation_score:.2f}, thought: {node.thought.content if node.thought else 'Initial State'}) from patient {node.diagnosis_state.patient_id}")

                # Step 1: Generate Hypotheses
                hypotheses_thoughts = self._simulate_llm_generate_hypotheses(node.diagnosis_state)
                generated_hypothesis_nodes = self._evaluate_and_add_nodes(node, hypotheses_thoughts)
                newly_generated_nodes.extend(generated_hypothesis_nodes)

                # Update best diagnosis if any hypothesis is strong
                for h_node in generated_hypothesis_nodes:
                    if h_node.thought.thought_type == "hypothesis" and h_node.evaluation_score > highest_score:
                        highest_score = h_node.evaluation_score
                        best_diagnosis_node = h_node
                        print(f"  New best hypothesis: {h_node.thought.content.name} with score {highest_score:.2f}")

                # Step 2: Generate Test Recommendations based on current hypotheses in the node's state
                test_recommendation_thoughts = self._simulate_llm_generate_test_recommendations(node.diagnosis_state)
                generated_test_nodes = self._evaluate_and_add_nodes(node, test_recommendation_thoughts)
                newly_generated_nodes.extend(generated_test_nodes)

                # Step 3: Simulate performing a recommended test if applicable and create a new state
                if generated_test_nodes:
                    # Pick the highest scoring test recommendation to simulate
                    best_test_node = max(generated_test_nodes, key=lambda n: n.evaluation_score)
                    if best_test_node.thought.thought_type == "test_recommendation":
                        test_to_perform = best_test_node.thought.content
                        print(f"  Simulating test: {test_to_perform}")
                        test_result_obj = self._simulate_perform_test(test_to_perform)

                        # Create a new DiagnosisState with the test result for a new node
                        state_after_test = node.diagnosis_state.clone()
                        state_after_test.test_results[test_result_obj.test_name] = test_result_obj.result

                        # Update current hypotheses in the new state, re-evaluating their plausibility
                        re_evaluated_hypotheses = []
                        for existing_hypothesis_obj in state_after_test.current_hypotheses:
                            re_evaluated_score = self._simulate_llm_evaluate_thought(Thought("hypothesis", existing_hypothesis_obj), state_after_test)
                            if re_evaluated_score > 0.1: # Keep plausible hypotheses
                                re_evaluated_hypotheses.append(existing_hypothesis_obj)
                        state_after_test.current_hypotheses = re_evaluated_hypotheses

                        # Create a new node representing the state after the test
                        test_result_thought = Thought("test_result", test_result_obj)
                        # Re-evaluate the overall state after test for the new node's score
                        state_score = self._simulate_llm_evaluate_thought(Thought("state_evaluation", state_after_test), state_after_test) # Use a placeholder thought type
                        node_after_test = Node(state_after_test, test_result_thought, state_score, node)
                        node.add_child(node_after_test)
                        newly_generated_nodes.extend([node_after_test]) # Add to consider for next round

            # Combine current active nodes with newly generated ones
            active_nodes.extend(newly_generated_nodes)

            # Prune less promising branches for the next iteration
            active_nodes = self._prune_nodes(active_nodes)

            # Check for conclusive diagnosis in the current active nodes
            for node in active_nodes:
                if node.thought and node.thought.thought_type == "hypothesis" and node.evaluation_score >= self.diagnosis_threshold:
                    if node.evaluation_score > highest_score:
                        highest_score = node.evaluation_score
                        best_diagnosis_node = node
                    print(f"  Potential conclusive diagnosis found: {node.thought.content.name} (Score: {node.evaluation_score:.2f})")

        final_diagnosis = "No conclusive diagnosis reached." if best_diagnosis_node is None else f"Diagnosed with {best_diagnosis_node.thought.content.name} (Confidence: {best_diagnosis_node.evaluation_score:.2f})"
        return {"final_diagnosis": final_diagnosis, "best_node_path": self._get_path_to_node(best_diagnosis_node) if best_diagnosis_node else []}

    def _get_path_to_node(self, node: Node) -> list[dict]:
        path = []
        current = node
        while current:
            path.append({
                "thought_type": current.thought.thought_type if current.thought else "Initial State",
                "content": str(current.thought.content) if current.thought else "N/A",
                "evaluation_score": current.evaluation_score,
                "diagnosis_state_summary": {
                    "symptoms": [s.name for s in current.diagnosis_state.symptoms_observed],
                    "test_results": current.diagnosis_state.test_results,
                    "hypotheses": [h.name for h in current.diagnosis_state.current_hypotheses]
                }
            })
            current = current.parent
        return path[::-1] # Reverse to get path from root to target

# Example Usage:
if __name__ == "__main__":
    # Initialize the assistant
    assistant = DiagnosisAssistant(medical_knowledge=MEDICAL_KNOWLEDGE, max_iterations=4, max_branches=2, diagnosis_threshold=0.75)

    # Define initial patient symptoms
    patient_symptoms_1 = [Symptom("fever"), Symptom("cough"), Symptom("sore throat")]
    patient_history_1 = "Patient has a history of seasonal allergies."

    print("\n--- Starting Diagnosis for Patient 001 ---")
    result_1 = assistant.diagnose("Patient_001", patient_symptoms_1, patient_history_1)
    print(f"\nFinal Diagnosis for Patient 001: {result_1['final_diagnosis']}")
    print("Diagnosis Path:")
    for step in result_1['best_node_path']:
        print(f"  -> Type: {step['thought_type']}, Content: {step['content']}, Score: {step['evaluation_score']:.2f}, State: {step['diagnosis_state_summary']}")

    print("\n--- Starting Diagnosis for Patient 002 (more severe symptoms) ---")
    patient_symptoms_2 = [Symptom("fever"), Symptom("cough"), Symptom("shortness of breath"), Symptom("chest pain")]
    patient_history_2 = "Patient is elderly with pre-existing heart condition."
    result_2 = assistant.diagnose("Patient_002", patient_symptoms_2, patient_history_2)
    print(f"\nFinal Diagnosis for Patient 002: {result_2['final_diagnosis']}")
    print("Diagnosis Path:")
    for step in result_2['best_node_path']:
        print(f"  -> Type: {step['thought_type']}, Content: {step['content']}, Score: {step['evaluation_score']:.2f}, State: {step['diagnosis_state_summary']}")
