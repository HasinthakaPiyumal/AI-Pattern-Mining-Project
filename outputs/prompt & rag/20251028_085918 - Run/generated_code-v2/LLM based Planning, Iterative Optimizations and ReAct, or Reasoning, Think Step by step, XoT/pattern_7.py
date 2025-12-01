import uuid

class ToTNode:
    def __init__(self, thought_text, parent=None, evidence=None, plausibility_score=0.0):
        self.id = str(uuid.uuid4())
        self.thought_text = thought_text
        self.evidence = evidence if evidence is not None else []
        self.plausibility_score = plausibility_score
        self.parent = parent
        self.children = []

class TreeOfThoughtsManager:
    def __init__(self, patient_data, depth_limit=3, branch_factor=2, pruning_threshold=0.3):
        self.patient_data = patient_data
        self.root_nodes = []
        self.depth_limit = depth_limit
        self.branch_factor = branch_factor
        self.pruning_threshold = pruning_threshold

    def _simulate_llm_generate_thoughts(self, current_thought, current_evidence, depth):
        if depth == 0:
            # Initial broad hypotheses
            return [
                {"thought": "Possible respiratory infection", "evidence": [f"Patient has {self.patient_data.get('symptoms', ['cough'])[0]}"]},
                {"thought": "Possible cardiovascular issue", "evidence": [f"Patient has {self.patient_data.get('symptoms', ['chest pain'])[0]}"]},
                {"thought": "Possible neurological disorder", "evidence": [f"Patient has {self.patient_data.get('symptoms', ['headache'])[0]}"]},
            ]
        
        # Simulate generating more specific thoughts based on current thought and evidence
        if "respiratory infection" in current_thought.lower():
            return [
                {"thought": f"Consider bacterial pneumonia given {current_evidence[0] if current_evidence else ''}", "evidence": ["High fever", "Productive cough"]},
                {"thought": f"Consider viral bronchitis given {current_evidence[0] if current_evidence else ''}", "evidence": ["Mild fever", "Dry cough"]},
            ]
        elif "cardiovascular issue" in current_thought.lower():
            return [
                {"thought": f"Investigate angina given {current_evidence[0] if current_evidence else ''}", "evidence": ["Chest pain with exertion"]},
                {"thought": f"Rule out myocardial infarction given {current_evidence[0] if current_evidence else ''}", "evidence": ["Severe chest pain", "ECG changes"]},
            ]
        elif "neurological disorder" in current_thought.lower():
            return [
                {"thought": f"Assess for migraine given {current_evidence[0] if current_evidence else ''}", "evidence": ["Unilateral headache", "Photophobia"]},
                {"thought": f"Consider tension headache given {current_evidence[0] if current_evidence else ''}", "evidence": ["Bilateral headache", "Stress-related"]},
            ]
        return []

    def add_thought(self, parent_node, thought_text, evidence=None, plausibility_score=0.0):
        node = ToTNode(thought_text, parent_node, evidence, plausibility_score)
        if parent_node:
            parent_node.children.append(node)
        else:
            self.root_nodes.append(node)
        return node

    def evaluate_and_prune(self, node):
        # Simple heuristic for simulation: higher evidence count, higher score
        if node.evidence:
            node.plausibility_score = len(node.evidence) * 0.1 + 0.5 # Base score + bonus for evidence
        
        # In a real scenario, this would involve LLM evaluation or a knowledge base lookup
        # Pruning based on a threshold
        return node.plausibility_score >= self.pruning_threshold

    def explore_branch(self, current_node, depth):
        if depth >= self.depth_limit:
            return

        llm_thoughts = self._simulate_llm_generate_thoughts(current_node.thought_text, current_node.evidence, depth)
        
        generated_children = []
        for i, thought_data in enumerate(llm_thoughts):
            if i >= self.branch_factor: # Limit branching
                break
            child_node = self.add_thought(current_node, thought_data["thought"], thought_data["evidence"])
            generated_children.append(child_node)
        
        # Evaluate and prune before further exploration
        for child in generated_children:
            if self.evaluate_and_prune(child):
                self.explore_branch(child, depth + 1)
            else:
                # Optionally, mark for pruning or remove immediately
                pass # For this simulation, we just don't explore further pruned branches

    def backtrack(self, node):
        # In a real system, this would involve choosing a different path from the parent
        # For this simulation, we just return the parent to signal backtracking
        return node.parent

    def run_diagnosis_tot(self):
        # Initial broad hypotheses
        initial_thoughts = self._simulate_llm_generate_thoughts("", [], 0)
        for thought_data in initial_thoughts:
            root = self.add_thought(None, thought_data["thought"], thought_data["evidence"], 0.7) # Give initial thoughts a base plausibility
            if self.evaluate_and_prune(root):
                self.explore_branch(root, depth=1)

    def get_prioritized_diagnoses(self, node=None, diagnoses=None):
        if diagnoses is None:
            diagnoses = []
            for root in self.root_nodes:
                self.get_prioritized_diagnoses(root, diagnoses)
            diagnoses.sort(key=lambda x: x['score'], reverse=True)
            return diagnoses

        if node.children:
            for child in node.children:
                self.get_prioritized_diagnoses(child, diagnoses)
        else:
            # If it's a leaf node or a branch end, consider it a potential diagnosis
            diagnoses.append({"diagnosis": node.thought_text, "score": node.plausibility_score, "path_id": node.id})
        return diagnoses # This return is for recursive calls, final result is in the first call

# Main Execution Logic
if __name__ == "__main__":
    patient_data = {
        "symptoms": ["fever", "cough", "shortness of breath", "fatigue"],
        "medical_history": ["asthma"],
        "lab_results": {"WBC": "elevated", "CRP": "high"}
    }

    print("Starting AI-powered Differential Diagnosis Assistant...")
    tot_manager = TreeOfThoughtsManager(patient_data, depth_limit=3, branch_factor=2, pruning_threshold=0.4)
    tot_manager.run_diagnosis_tot()

    prioritized_diagnoses = tot_manager.get_prioritized_diagnoses()

    print("\nPrioritized Differential Diagnoses:")
    if prioritized_diagnoses:
        for diagnosis in prioritized_diagnoses:
            print(f"- {diagnosis['diagnosis']} (Plausibility: {diagnosis['score']:.2f})")
    else:
        print("No diagnoses could be determined.")

    print("\n--- Full Tree Exploration (for visualization) ---")
    all_nodes = []
    queue = list(tot_manager.root_nodes)
    while queue:
        current = queue.pop(0)
        all_nodes.append(current)
        for child in current.children:
            queue.append(child)

    for node in all_nodes:
        parent_id = node.parent.id if node.parent else "None"
        print(f"Node ID: {node.id[:4]}..., Parent ID: {parent_id[:4]}..., Thought: {node.thought_text[:50]}..., Score: {node.plausibility_score:.2f}")
