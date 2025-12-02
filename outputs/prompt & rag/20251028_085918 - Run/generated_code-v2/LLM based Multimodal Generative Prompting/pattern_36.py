import os
import re
import networkx as nx
from typing import List, Dict, Any

class ImageCaptioner:
    def __init__(self):
        pass

    def generate_caption(self, image_path: str) -> str:
        if not os.path.exists(image_path):
            return f"A simulated caption for an image. The image shows signs of potential {os.path.basename(image_path).replace('.jpg', '').replace('_', ' ')}."
        return "A simulated X-ray image showing some opacities in the lower left lung." if "xray" in image_path.lower() else "A simulated medical image with general findings."

class TextProcessor:
    def clean_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def extract_keywords(self, text: str) -> list[str]:
        cleaned_text = self.clean_text(text)
        words = cleaned_text.split()
        medical_stopwords = {"the", "a", "an", "is", "are", "was", "were", "patient", "report", "note", "and", "or", "of", "in", "for", "with", "on", "at", "from", "by"}
        keywords = [word for word in words if len(word) > 2 and word not in medical_stopwords]
        return list(set(keywords))

class ThoughtGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.text_processor = TextProcessor()
        self.node_id_counter = 0
        self.node_map = {}

    def _get_or_create_node(self, concept: str, node_type: str = "concept") -> str:
        if concept not in self.node_map:
            node_id = f"node_{self.node_id_counter}"
            self.graph.add_node(node_id, label=concept, type=node_type)
            self.node_map[concept] = node_id
            self.node_id_counter += 1
        return self.node_map[concept]

    def add_relationship(self, source_concept: str, target_concept: str, relation_type: str, attributes: Dict[str, Any] = None):
        source_node_id = self._get_or_create_node(source_concept)
        target_node_id = self._get_or_create_node(target_concept)
        
        edge_attributes = {'relation': relation_type}
        if attributes:
            edge_attributes.update(attributes)

        self.graph.add_edge(source_node_id, target_node_id, **edge_attributes)

    def build_graph_from_data(self, patient_history: str, image_caption: str, doctor_notes: str):
        all_text = patient_history + " " + image_caption + " " + doctor_notes
        keywords = self.text_processor.extract_keywords(all_text)

        for keyword in keywords:
            self._get_or_create_node(keyword)
            
        if "fever" in keywords and "cough" in keywords:
            self.add_relationship("fever", "cough", "associated_symptom")
        
        if "opacities" in keywords and "lung" in keywords:
            self.add_relationship("opacities", "lung", "located_in")
            
        if "history" in patient_history.lower() and "diabetes" in patient_history.lower():
             self.add_relationship("patient", "diabetes", "has_medical_history", {'source': 'patient_history'})
             
        if "enlarged" in image_caption.lower() and "lymph" in image_caption.lower():
            self.add_relationship("enlarged lymph nodes", "infection", "suggests_potential", {'source': 'image_caption'})

        print(f"Thought graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")

    def get_graph(self) -> nx.DiGraph:
        return self.graph

    def find_paths_to_concept(self, target_concept: str, max_depth: int = 3) -> List[List[str]]:
        target_node_id = self.node_map.get(target_concept)
        if not target_node_id:
            return []

        paths = []
        for source_node_id in self.graph.nodes:
            if source_node_id == target_node_id:
                continue
            for path in nx.all_simple_paths(self.graph, source=source_node_id, target=target_node_id, cutoff=max_depth):
                paths.append([self.graph.nodes[node_id]['label'] for node_id in path])
        return paths

class DiagnosticAidSystem:
    def __init__(self):
        self.image_captioner = ImageCaptioner()
        self.text_processor = TextProcessor()
        self.thought_graph_builder = ThoughtGraph()

    def diagnose(self, image_path: str, patient_history: str, doctor_notes: str) -> Dict[str, Any]:
        print(f"\n--- Starting Diagnosis for image: {image_path} ---")

        image_caption = self.image_captioner.generate_caption(image_path)
        print(f"Generated Image Caption: {image_caption}")

        self.thought_graph_builder.build_graph_from_data(patient_history, image_caption, doctor_notes)
        graph = self.thought_graph_builder.get_graph()
        
        potential_diagnoses = []
        reasoning_steps = []
        
        common_medical_concepts = ["pneumonia", "infection", "tumor", "inflammation", "diabetes", "fracture"]
        
        for concept in common_medical_concepts:
            paths = self.thought_graph_builder.find_paths_to_concept(concept)
            if paths:
                potential_diagnoses.append(concept)
                for path in paths:
                    reasoning_steps.append(f"Path to '{concept}': {' -> '.join(path)}")

        if not potential_diagnoses:
            potential_diagnoses.append("Undetermined: Further investigation needed.")

        rationale = (
            f"Based on the multimodal patient data (image, history, notes) and graph analysis:\n"
            f"Image Description: {image_caption}\n"
            f"Key patient history insights: {self.text_processor.extract_keywords(patient_history)}\n"
            f"Key doctor's notes insights: {self.text_processor.extract_keywords(doctor_notes)}\n"
            f"Graph analysis revealed the following potential reasoning paths: {'; '.join(reasoning_steps) if reasoning_steps else 'No specific reasoning paths found.'}\n"
            f"Therefore, potential diagnoses include: {', '.join(potential_diagnoses)}."
            "Further clinical correlation and tests are recommended."
        )

        print(f"Rationale Generated: {rationale}")
        print(f"Potential Diagnoses: {', '.join(potential_diagnoses)}")
        print("---"" Diagnosis Complete --- ")

        return {
            "rationale": rationale,
            "potential_diagnoses": potential_diagnoses,
            "graph_nodes": list(graph.nodes(data=True)),
            "graph_edges": list(graph.edges(data=True))
        }

def main():
    print("Initializing Multimodal GraphofThought Diagnostic Aid System...")
    system = DiagnosticAidSystem()

    image_path_1 = "data/xray_lung_opacities.jpg"
    patient_history_1 = (
        "Patient presented with chronic cough, fever for 3 days, and shortness of breath. "
        "History of smoking for 10 years. No known allergies."
    )
    doctor_notes_1 = (
        "Initial examination reveals crackles in the lower left lung. "
        "Patient appears fatigued. Ordered chest X-ray and blood tests."
    )

    os.makedirs(os.path.dirname(image_path_1), exist_ok=True)
    
    print("\n--- Running Diagnosis Example 1 (Lung Opacities) ---")
    result_1 = system.diagnose(image_path_1, patient_history_1, doctor_notes_1)
    print("\n--- Diagnosis Result 1 ---")
    print(f"Final Rationale: {result_1['rationale']}")
    print(f"Suggested Diagnoses: {', '.join(result_1['potential_diagnoses'])}")

    print("\n----------------------------------------------------\n")

    image_path_2 = "data/arm_fracture.jpg"
    patient_history_2 = (
        "Patient fell off a bicycle, complaining of severe pain in the right forearm. "
        "Unable to move the arm. No previous bone injuries."
    )
    doctor_notes_2 = (
        "Swelling and deformity noted in the right forearm. Palpation elicits tenderness. "
        "Suspect forearm fracture. Ordered X-ray."
    )
    
    os.makedirs(os.path.dirname(image_path_2), exist_ok=True)

    print("\n--- Running Diagnosis Example 2 (Arm Fracture) ---")
    result_2 = system.diagnose(image_path_2, patient_history_2, doctor_notes_2)
    print("\n--- Diagnosis Result 2 ---")
    print(f"Final Rationale: {result_2['rationale']}")
    print(f"Suggested Diagnoses: {', '.join(result_2['potential_diagnoses'])}")

if __name__ == "__main__":
    main()