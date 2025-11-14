import networkx as nx
from PIL import Image # Used for simulating image data handling
import io # Used to simulate image data

class PatientData:
    def __init__(self, patient_id: str, clinical_notes: str, lab_results: dict, medical_images_data: dict):
        self.patient_id = patient_id
        self.clinical_notes = clinical_notes
        self.lab_results = lab_results  # Dictionary: {"test_name": value}
        self.medical_images_data = medical_images_data  # Dictionary: {"image_type": bytes_data}

    def __repr__(self):
        return f"PatientData(ID: {self.patient_id})"

class ClinicalDecisionSupportSystem:
    def __init__(self):
        print("Clinical Decision Support System initialized.")
        self.thought_graph = nx.DiGraph() # Represents the structured reasoning process
        self.diagnostic_path = [] # Stores the sequence of reasoning steps taken
        self.current_patient = None # To hold the patient data being processed

    def _extract_text_features(self, notes: str) -> dict:
        """Simulates extracting features from clinical notes using a placeholder method."""
        print(f"  Extracting text features from notes...")
        # In a real system, this would involve NLP models (e.g., Transformers for embeddings, entity recognition).
        features = {}
        if "fever" in notes.lower():
            features["has_fever"] = True
        if "cough" in notes.lower():
            features["has_cough"] = True
        if "chest pain" in notes.lower() or "dyspnea" in notes.lower():
            features["respiratory_symptoms"] = True
        return features

    def _extract_lab_features(self, lab_results: dict) -> dict:
        """Simulates extracting features from lab results using a placeholder method."""
        print(f"  Extracting lab features from results...")
        features = {}
        if lab_results.get("CRP") is not None and lab_results["CRP"] > 5:  # C-reactive protein for inflammation
            features["inflammation_marker_elevated"] = True
        if lab_results.get("WBC") is not None and lab_results["WBC"] > 10.0:  # White blood cell count
            features["leukocytosis"] = True
        return features

    def _extract_image_features(self, image_data: bytes) -> dict:
        """Simulates extracting features from medical images using a placeholder method."""
        print(f"  Extracting image features...")
        # In a real system, this would use a CNN or specialized vision model for anomaly detection.
        image_features = {}
        # Example: Simulating a lung X-ray analysis based on a placeholder flag
        if self.current_patient and 'xray_lung' in self.current_patient.medical_images_data:
            # Simulate detection of opacities for a respiratory context
            image_features["has_opacities"] = True
            image_features["opacity_location"] = "lower right lobe"
        return image_features

    def _generate_sub_questions(self, initial_findings: dict) -> list:
        """Generates sub-questions based on initial extracted findings to guide further reasoning."""
        print(f"  Generating sub-questions based on findings: {list(initial_findings.keys())}")
        sub_questions = []
        if initial_findings.get("respiratory_symptoms") or initial_findings.get("has_cough"):
            sub_questions.append("Are there structural abnormalities in the imaging (e.g., X-ray for lungs)?")
        if initial_findings.get("inflammation_marker_elevated") or initial_findings.get("has_fever"):
            sub_questions.append("Are there specific signs of infection in lab results or notes?")
        return sub_questions

    def _generate_visual_explanation(self, image_data: bytes, textual_findings: dict) -> str:
        """Simulates generating an annotated image for visual interpretation and sequential reasoning."""
        print(f"  Generating visual explanation for image...")
        # In a real system, this would involve image processing (e.g., OpenCV, Matplotlib)
        # to overlay annotations (bounding boxes, heatmaps) based on the textual findings.
        explanation_text = "Generated a mock annotated image. "
        if "has_opacities" in textual_findings:
            explanation_text += f"Highlighted opacities in the {textual_findings.get('opacity_location', 'image')}."
        return explanation_text + " (Image bytes with annotations would be returned here)"

    def _build_and_traverse_thought_graph(self, patient_data: PatientData, initial_features: dict) -> dict:
        """
        Builds and traverses a dynamic thought graph using networkx to guide the diagnostic process.
        Each node represents a reasoning step or information integration point.
        """
        print("\n--- Building and Traversing Thought Graph ---")
        self.thought_graph.clear()
        self.diagnostic_path = []

        # Define nodes representing various reasoning steps and data modalities
        self.thought_graph.add_node("Start", type="initial")
        self.thought_graph.add_node("Text_Analysis", type="modal_analysis")
        self.thought_graph.add_node("Lab_Analysis", type="modal_analysis")
        self.thought_graph.add_node("Image_Analysis", type="modal_analysis")
        self.thought_graph.add_node("Synthesize_Modalities", type="integration")
        self.thought_graph.add_node("Sub_Question_Gen", type="reasoning_step")
        self.thought_graph.add_node("Visual_Explanation_Gen", type="reasoning_step")
        self.thought_graph.add_node("Hypothesis_Formulation", type="decision_point")
        self.thought_graph.add_node("Diagnosis", type="final_output")

        # Add edges representing the flow of reasoning
        self.thought_graph.add_edge("Start", "Text_Analysis")
        self.thought_graph.add_edge("Start", "Lab_Analysis")
        if patient_data.medical_images_data: # Only add image analysis if images are present
            self.thought_graph.add_edge("Start", "Image_Analysis")

        # All modal analyses feed into synthesis
        self.thought_graph.add_edge("Text_Analysis", "Synthesize_Modalities")
        self.thought_graph.add_edge("Lab_Analysis", "Synthesize_Modalities")
        if patient_data.medical_images_data:
            self.thought_graph.add_edge("Image_Analysis", "Synthesize_Modalities")

        # Synthesis leads to sub-question generation and direct hypothesis formulation
        self.thought_graph.add_edge("Synthesize_Modalities", "Sub_Question_Gen")
        self.thought_graph.add_edge("Synthesize_Modalities", "Hypothesis_Formulation")
        self.thought_graph.add_edge("Sub_Question_Gen", "Hypothesis_Formulation") # Sub-questions inform hypothesis

        # Conditional path for visual explanation, which can refine the hypothesis
        if patient_data.medical_images_data and (initial_features.get("respiratory_symptoms") or "xray_lung" in patient_data.medical_images_data):
            self.thought_graph.add_edge("Hypothesis_Formulation", "Visual_Explanation_Gen")
            # This edge shows that visual explanation can feedback to refine hypothesis
            self.thought_graph.add_edge("Visual_Explanation_Gen", "Hypothesis_Formulation", weight=0.5, label="Refine hypothesis")

        self.thought_graph.add_edge("Hypothesis_Formulation", "Diagnosis")

        # --- Traverse and execute the graph ---
        current_findings = initial_features.copy()
        current_diagnosis_confidence = {"unknown": 1.0} # Initial placeholder for diagnosis
        processed_nodes = set()
        queue = ["Start"]  # Simple BFS-like traversal for demonstration

        while queue:
            node = queue.pop(0)
            if node in processed_nodes:
                continue
            processed_nodes.add(node)
            self.diagnostic_path.append(node) # Record the path
            print(f"  Processing node: {node} (Type: {self.thought_graph.nodes[node]['type']})")

            # Execute logic based on the node type
            if node == "Start":
                pass # Initial setup
            elif node == "Text_Analysis":
                current_findings.update(self._extract_text_features(patient_data.clinical_notes))
            elif node == "Lab_Analysis":
                current_findings.update(self._extract_lab_features(patient_data.lab_results))
            elif node == "Image_Analysis" and patient_data.medical_images_data:
                if 'xray_lung' in patient_data.medical_images_data:
                    current_findings.update(self._extract_image_features(patient_data.medical_images_data['xray_lung']))
            elif node == "Synthesize_Modalities":
                print(f"    Synthesizing findings: {current_findings.keys()}")
                # In a real system, this would involve a complex multimodal fusion model.
                # Simple aggregation for now to drive the diagnosis.
                if current_findings.get("respiratory_symptoms") and current_findings.get("has_opacities") and current_findings.get("inflammation_marker_elevated"):
                    current_findings["strong_respiratory_disease_indication"] = True
            elif node == "Sub_Question_Gen":
                sub_questions = self._generate_sub_questions(current_findings)
                current_findings["generated_sub_questions"] = sub_questions
                print(f"    Generated sub-questions: {sub_questions}")
            elif node == "Visual_Explanation_Gen":
                if 'xray_lung' in patient_data.medical_images_data:
                    visual_output_desc = self._generate_visual_explanation(
                        patient_data.medical_images_data['xray_lung'], current_findings
                    )
                    current_findings["visual_explanation"] = visual_output_desc
                    print(f"    Visual explanation generated: {visual_output_desc}")
            elif node == "Hypothesis_Formulation":
                print(f"    Formulating hypothesis with current findings: {current_findings.keys()}")
                # Simple rule-based hypothesis for demonstration
                if current_findings.get("strong_respiratory_disease_indication") and current_findings.get("leukocytosis"):
                    current_diagnosis_confidence = {"Pneumonia": 0.85, "Bronchitis": 0.10, "Other Respiratory Infection": 0.05}
                elif current_findings.get("has_fever") and current_findings.get("inflammation_marker_elevated"):
                    current_diagnosis_confidence = {"Influenza": 0.6, "Common Cold": 0.3, "Bacterial Infection (unspecified)": 0.1}
                else:
                    current_diagnosis_confidence = {"Unclear": 0.7, "Requires More Data": 0.3}
                print(f"    Current hypothesis: {current_diagnosis_confidence}")

            # Add neighbors to queue for further processing
            for neighbor in self.thought_graph.successors(node):
                if neighbor not in processed_nodes:
                    queue.append(neighbor)

        print("--- Thought Graph Traversal Complete ---")
        return {"final_diagnosis": current_diagnosis_confidence, "supporting_evidence": current_findings, "diagnostic_path": self.diagnostic_path}

    def diagnose_patient(self, patient_data: PatientData):
        """Initiates the multimodal structured reasoning process for a given patient."""
        self.current_patient = patient_data # Set current patient for context in helper methods
        print(f"\nProcessing patient: {patient_data.patient_id}")

        # Initial findings before graph traversal (can be integrated into graph 'Start' node logic)
        initial_context_features = {"patient_id": patient_data.patient_id}

        diagnosis_output = self._build_and_traverse_thought_graph(patient_data, initial_context_features)

        print("\n--- Final Diagnostic Output ---")
        print(f"Patient ID: {patient_data.patient_id}")
        print(f"Probabilistic Diagnosis: {diagnosis_output['final_diagnosis']}")
        print(f"Key Supporting Evidence: { {k: v for k, v in diagnosis_output['supporting_evidence'].items() if k not in ['patient_id', 'generated_sub_questions'] } }")
        print(f"Reasoning Path (Nodes Visited): {' -> '.join(diagnosis_output['diagnostic_path'])}")

        return diagnosis_output

# Helper function to create dummy image data for simulation
def create_dummy_image_bytes(width=100, height=100, color=(255, 0, 0)) -> bytes:
    image = Image.new("RGB", (width, height), color)
    byte_arr = io.BytesIO()
    image.save(byte_arr, format='PNG')
    return byte_arr.getvalue()

# --- Main Execution Block ---
if __name__ == "__main__":
    system = ClinicalDecisionSupportSystem()

    # Sample Patient 1: With respiratory symptoms, inflammation, and imaging
    patient1_data = PatientData(
        patient_id="P001",
        clinical_notes="Patient presents with persistent cough, shortness of breath, and mild fever for 3 days. Reports chest tightness. On examination, decreased breath sounds in lower right lung. No travel history.",
        lab_results={"CRP": 12.5, "WBC": 14.2, "Hemoglobin": 13.0, "ESR": 35},
        medical_images_data={
            "xray_lung": create_dummy_image_bytes(color=(200, 200, 200)) # Simulate a grey X-ray image
        }
    )
    print("\n" + "="*50 + "\nProcessing Patient 1\n" + "="*50)
    diagnosis1 = system.diagnose_patient(patient1_data)

    # Sample Patient 2: General symptoms, less specific, no imaging
    patient2_data = PatientData(
        patient_id="P002",
        clinical_notes="Patient complains of general malaise, headache, and body aches for 2 days. No cough or fever reported. Denies respiratory or gastrointestinal issues. Just feels unwell.",
        lab_results={"CRP": 3.0, "WBC": 7.5, "Hemoglobin": 14.5, "Glucose": 95},
        medical_images_data={} # No imaging for this patient
    )
    print("\n" + "="*50 + "\nProcessing Patient 2\n" + "="*50)
    diagnosis2 = system.diagnose_patient(patient2_data)
