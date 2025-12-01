import pydicom
import cv2
import numpy as np
from PIL import Image
import io
import spacy
import networkx as nx
from transformers import pipeline, AutoProcessor, BlipForConditionalGeneration # Example VLM

# Placeholder for a medical knowledge base (could be a database, ontologies, etc.)
MEDICAL_KNOWLEDGE_BASE = {
    "fever": {"causes": ["infection", "inflammation"], "symptoms_of": ["flu", "pneumonia"]},
    "cough": {"causes": ["infection", "allergy"], "symptoms_of": ["flu", "bronchitis"]},
    "pneumonia": {"symptoms": ["fever", "cough", "shortness of breath"], "tests": ["chest X-ray"]},
    "infection": {"treatments": ["antibiotics"]},
    "chest X-ray": {"detects": ["pneumonia", "fracture"]},
    "shortness of breath": {"causes": ["pneumonia", "asthma"]}
}

class InputModule:
    def __init__(self):
        pass

    def process_image_input(self, image_data: bytes, filename: str):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            return image
        elif filename.lower().endswith(('.dcm')):
            dicom_data = pydicom.dcmread(io.BytesIO(image_data))
            # Convert DICOM pixel data to PIL Image
            if 'PixelData' in dicom_data:
                if dicom_data.PhotometricInterpretation == "MONOCHROME1":
                    # Invert if MONOCHROME1 to get typical grayscale display
                    image_array = dicom_data.pixel_array.astype(np.float32) * (-1)
                    image_array = cv2.normalize(image_array, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                    image = Image.fromarray(image_array).convert("L")
                else:
                    image_array = dicom_data.pixel_array.astype(np.float32)
                    image_array = cv2.normalize(image_array, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                    image = Image.fromarray(image_array).convert("L") # Assuming grayscale for most medical DICOMs
                return image
            else:
                raise ValueError("DICOM file does not contain pixel data.")
        else:
            raise ValueError("Unsupported image format.")

    def process_text_input(self, text_data: str):
        return text_data

class PreprocessingModule:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm") # Basic English model; for medical, fine-tuned models are better
        # Initialize a pre-trained BLIP model for image captioning
        self.vlm_processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.vlm_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    def preprocess_image(self, image: Image.Image):
        # Basic resizing and normalization (can be expanded)
        image = image.resize((224, 224))
        return image

    def caption_image(self, image: Image.Image) -> str:
        inputs = self.vlm_processor(images=image, return_tensors="pt")
        outputs = self.vlm_model.generate(**inputs, max_new_tokens=50)
        caption = self.vlm_processor.decode(outputs[0], skip_special_tokens=True)
        return caption

    def extract_nlu_entities(self, text: str) -> dict:
        doc = self.nlp(text)
        entities = {"symptoms": [], "diseases": [], "tests": [], "anatomy": []}
        # This is a very basic example; for real medical NLU, use specialized models
        for ent in doc.ents:
            if "symptom" in ent.label_.lower(): # Example: if custom NER model labels symptoms
                entities["symptoms"].append(ent.text)
            elif "disease" in ent.label_.lower():
                entities["diseases"].append(ent.text)
            elif "test" in ent.label_.lower():
                entities["tests"].append(ent.text)
            elif "organ" in ent.label_.lower() or "anatomy" in ent.label_.lower():
                entities["anatomy"].append(ent.text)
            else:
                # Fallback or general entities might be categorized here
                pass

        # Simple keyword-based extraction for demonstration if no custom NER is available
        medical_keywords = {
            "fever": "symptom", "cough": "symptom", "pneumonia": "disease",
            "chest X-ray": "test", "shortness of breath": "symptom", "infection": "disease"
        }
        for word in text.lower().split():
            if word in medical_keywords and medical_keywords[word] in entities:
                if word not in entities[medical_keywords[word]]:
                    entities[medical_keywords[word]].append(word)

        return entities

class ThoughtGraphConstructionModule:
    def __init__(self, knowledge_base: dict = MEDICAL_KNOWLEDGE_BASE):
        self.graph = nx.DiGraph()
        self.knowledge_base = knowledge_base
        self._define_initial_schema()

    def _define_initial_schema(self):
        # Define potential node types and edge types for the graph
        # Nodes: Symptoms, Diseases, Tests, Treatments, Anatomy, Findings
        # Edges: causes, indicates, located_in, suggests, treatments
        pass # Schema is implicitly defined by how nodes/edges are added

    def integrate_multimodal_info(self, nlu_entities: dict, image_caption: str):
        # Add entities from NLU as nodes
        for entity_type, entities_list in nlu_entities.items():
            for entity in entities_list:
                if not self.graph.has_node(entity):
                    self.graph.add_node(entity, type=entity_type)

        # Integrate information from image caption
        # This is a simplified approach; more advanced would involve NLU on caption
        caption_doc = spacy.load("en_core_web_sm")(image_caption) # Re-using spacy for caption NLU
        for ent in caption_doc.ents:
            # Example: If a disease or finding is mentioned in caption
            if "disease" in ent.label_.lower() or "finding" in ent.label_.lower():
                if not self.graph.has_node(ent.text):
                    self.graph.add_node(ent.text, type="finding")

        # Simple keyword matching from caption to add potential links
        for keyword in self.knowledge_base.keys():
            if keyword in image_caption.lower():
                if not self.graph.has_node(keyword):
                    self.graph.add_node(keyword, type="finding_from_caption")

    def enrich_from_knowledge_base(self):
        # Expand the graph using the medical knowledge base
        nodes_to_process = list(self.graph.nodes())
        for node in nodes_to_process:
            if node in self.knowledge_base:
                kb_info = self.knowledge_base[node]
                for relation, targets in kb_info.items():
                    for target in targets:
                        if not self.graph.has_node(target):
                            self.graph.add_node(target, type="kb_entity") # Type can be more specific
                        self.graph.add_edge(node, target, relation=relation)

    def expand_and_reason(self) -> dict:
        # Example reasoning: find paths from symptoms to diseases
        diagnostic_hypotheses = {}
        symptoms = [n for n, data in self.graph.nodes(data=True) if data.get('type') == 'symptom']
        diseases = [n for n, data in self.graph.nodes(data=True) if data.get('type') == 'disease' or data.get('type') == 'kb_entity']

        for symptom in symptoms:
            for disease in diseases:
                if nx.has_path(self.graph, symptom, disease):
                    # Find all simple paths
                    paths = list(nx.all_simple_paths(self.graph, symptom, disease))
                    if paths:
                        if disease not in diagnostic_hypotheses:
                            diagnostic_hypotheses[disease] = []
                        for path in paths:
                            path_description = " -> ".join(path)
                            diagnostic_hypotheses[disease].append(f"Path from {symptom} to {disease}: {path_description}")

        # Add a simple rationale based on graph structure
        rationale = "Based on the integrated information and knowledge base, the following diagnostic hypotheses and supporting evidence were generated:\n"
        if not diagnostic_hypotheses:
            rationale += "No direct diagnostic hypotheses found based on the current graph and knowledge."
        else:
            for disease, evidence_list in diagnostic_hypotheses.items():
                rationale += f"\nDisease: {disease}\n  Evidence:\n"
                for evidence in evidence_list:
                    rationale += f"    - {evidence}\n"
        
        # Also list all nodes and edges for full context
        graph_summary = {
            "nodes": list(self.graph.nodes(data=True)),
            "edges": list(self.graph.edges(data=True))
        }

        return {"rationale": rationale, "graph_summary": graph_summary}

class MedicalDiagnosticAssistant:
    def __init__(self):
        self.input_module = InputModule()
        self.preprocessing_module = PreprocessingModule()
        self.graph_module = ThoughtGraphConstructionModule()

    def diagnose(self, text_input: str = None, image_data: bytes = None, image_filename: str = None):
        patient_text_context = ""
        image_caption = ""
        nlu_entities = {}

        if text_input:
            patient_text_context = self.input_module.process_text_input(text_input)
            nlu_entities = self.preprocessing_module.extract_nlu_entities(patient_text_context)

        if image_data and image_filename:
            image = self.input_module.process_image_input(image_data, image_filename)
            preprocessed_image = self.preprocessing_module.preprocess_image(image)
            image_caption = self.preprocessing_module.caption_image(preprocessed_image)

            # Integrate caption findings into NLU entities for graph construction
            caption_nlu = self.preprocessing_module.extract_nlu_entities(image_caption)
            for ent_type, ents in caption_nlu.items():
                nlu_entities[ent_type] = list(set(nlu_entities.get(ent_type, []) + ents))

        self.graph_module.integrate_multimodal_info(nlu_entities, image_caption)
        self.graph_module.enrich_from_knowledge_base()
        reasoning_result = self.graph_module.expand_and_reason()
        return reasoning_result

if __name__ == "__main__":
    assistant = MedicalDiagnosticAssistant()

    # Example 1: Text-only input
    print("\n--- TEXT ONLY DIAGNOSIS ---")
    text_case_1 = "Patient reports severe cough and fever for 3 days."
    result_text = assistant.diagnose(text_input=text_case_1)
    print(result_text["rationale"])
    # print("Graph Nodes:", result_text["graph_summary"]["nodes"])
    # print("Graph Edges:", result_text["graph_summary"]["edges"])

    # Example 2: Multimodal input (text + dummy image)
    print("\n--- MULTIMODAL DIAGNOSIS (Dummy Image) ---")
    # Create a dummy image for demonstration
    dummy_image = Image.new('RGB', (60, 30), color = 'red')
    buf = io.BytesIO()
    dummy_image.save(buf, format='PNG')
    dummy_image_data = buf.getvalue()

    text_case_2 = "Patient has shortness of breath and a persistent cough. Doctor ordered a chest X-ray."
    result_multimodal = assistant.diagnose(text_input=text_case_2, image_data=dummy_image_data, image_filename="dummy_chest_xray.png")
    print(result_multimodal["rationale"])
    # print("Graph Nodes:", result_multimodal["graph_summary"]["nodes"])
    # print("Graph Edges:", result_multimodal["graph_summary"]["edges"])

    # Example 3: Text input with findings mentioned in text directly related to KB
    print("\n--- TEXT ONLY DIAGNOSIS (KB-rich text) ---")
    text_case_3 = "The patient's X-ray showed signs of pneumonia. They have a fever and cough."
    result_text_kb = assistant.diagnose(text_input=text_case_3)
    print(result_text_kb["rationale"])

    # Example 4: DICOM image example (requires a real DICOM file for full functionality)
    # For this to work, you'd need a valid DICOM file to load and convert.
    # For now, this is a placeholder demonstrating the expected usage.
    # try:
    #     with open("path/to/your/dicom_image.dcm", "rb") as f:
    #         dicom_data_example = f.read()
    #     text_case_4 = "Patient presents with severe chest pain."
    #     result_dicom = assistant.diagnose(text_input=text_case_4, image_data=dicom_data_example, image_filename="example_dicom.dcm")
    #     print("\n--- DICOM MULTIMODAL DIAGNOSIS ---")
    #     print(result_dicom["rationale"])
    # except FileNotFoundError:
    #     print("\n--- DICOM EXAMPLE SKIPPED: dicom_image.dcm not found. ---")
    # except Exception as e:
    #     print(f"\n--- ERROR PROCESSING DICOM: {e} ---")


