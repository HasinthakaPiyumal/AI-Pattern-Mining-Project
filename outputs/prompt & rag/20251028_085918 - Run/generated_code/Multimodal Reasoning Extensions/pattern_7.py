import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import cv2 # For potential future image processing, currently Pillow suffices for basic operations
import numpy as np
import networkx as nx
import os

# --- 1. Input Layer ---

class InputLayer:
    def __init__(self):
        pass

    def ingest_textual_data(self, symptoms_text: str, lab_results_df: pd.DataFrame, medical_history_text: str):
        """
        Ingests textual patient data.
        """
        print("\n--- Input Layer: Textual Data Ingestion ---")
        print(f"Symptoms: {symptoms_text}")
        print(f"Lab Results:\n{lab_results_df.to_string()}")
        print(f"Medical History: {medical_history_text}")
        return {"symptoms": symptoms_text, "lab_results": lab_results_df, "medical_history": medical_history_text}

    def ingest_image_data(self, image_path: str):
        """
        Ingests image data (e.g., X-ray, MRI) and performs basic preprocessing.
        """
        print(f"\n--- Input Layer: Image Data Ingestion ---")
        if not os.path.exists(image_path):
            print(f"Dummy image not found at {image_path}. Creating one for demonstration.")
            # Create a dummy image
            img = Image.new('RGB', (256, 256), color = 'white')
            d = ImageDraw.Draw(img)
            # Try to load a default font, if not available, text might not render perfectly but won't crash
            try:
                fnt = ImageFont.load_default()
            except IOError:
                fnt = None # Fallback if default font isn't found
            
            if fnt: d.text((10,10), "Simulated X-Ray (Anomaly here)", fill=(0,0,0), font=fnt)
            else: d.text((10,10), "Simulated X-Ray (Anomaly here)", fill=(0,0,0))
            img.save(image_path)
            print(f"Dummy image saved to {image_path}")

        image = Image.open(image_path).convert("RGB")
        print(f"Loaded image from {image_path} with size {image.size}")
        # Basic preprocessing: resize
        resized_image = image.resize((224, 224)) # Common input size for vision models
        print(f"Resized image to {resized_image.size}")
        return {"image": resized_image, "image_path": image_path}

# --- 2. Multimodal Fusion & Embedding Layer ---

class MultimodalEmbeddingLayer:
    def __init__(self):
        # Simulate model loading for demonstration
        print("\n--- Multimodal Embedding Layer: Initializing (Simulated) ---")
        # In a real application, you'd load models like:
        # self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
        # self.vision_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        # self.vision_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        pass

    def get_text_embedding(self, text_data: dict):
        """
        Generates embeddings for textual data. (Simulated)
        """
        print("\n--- Multimodal Embedding Layer: Text Embedding (Simulated) ---")
        combined_text = f"{text_data['symptoms']} {text_data['medical_history']} {text_data['lab_results'].to_string()}"
        # Simulate embedding generation
        embedding = np.random.rand(768) # Example: a 768-dim embedding
        print(f"Generated dummy text embedding of shape {embedding.shape}")
        return embedding

    def get_image_embedding(self, image_data: dict):
        """
        Generates embeddings for image data. (Simulated)
        """
        print("\n--- Multimodal Embedding Layer: Image Embedding (Simulated) ---")
        # Simulate image preprocessing for a vision model (e.g., CLIP)
        # In a real scenario: inputs = self.vision_processor(images=image_data['image'], return_tensors="pt")
        # with torch.no_grad():
        #    image_features = self.vision_model.get_image_features(**inputs)
        embedding = np.random.rand(512) # Example: a 512-dim CLIP embedding
        print(f"Generated dummy image embedding of shape {embedding.shape}")
        return embedding

    def fuse_embeddings(self, text_embedding: np.ndarray, image_embedding: np.ndarray):
        """
        Combines text and image embeddings. (Simple Concatenation)
        """
        print("\n--- Multimodal Embedding Layer: Fusion Mechanism ---")
        fused_embedding = np.concatenate((text_embedding, image_embedding))
        print(f"Fused embeddings into a vector of shape {fused_embedding.shape}")
        return fused_embedding

# --- 3. Reasoning Engine (Multimodal Structured Reasoning) ---

class ReasoningNode:
    """Represents a node in the thought graph."""
    def __init__(self, node_id: str, question: str, modality: str = "multimodal"):
        self.node_id = node_id
        self.question = question
        self.modality = modality # "text", "image", "multimodal"
        self.status = "pending"
        self.input_data = {} # Data received from previous nodes or initial input
        self.output = None
        self.explanation = ""

    def __repr__(self):
        return f"Node(ID:{self.node_id}, Q:'{self.question[:30]}...', Status:{self.status})"

class ReasoningEngine:
    def __init__(self, llm_api_key=None):
        self.graph = nx.DiGraph()
        self.llm_api_key = llm_api_key # Placeholder for LLM integration
        # self.llm = OpenAI(api_key=llm_api_key) # Example if using langchain

        print("\n--- Reasoning Engine: Initializing ---")
        print("Thought Graph will be constructed using NetworkX.")

    def _simulate_llm_call(self, prompt: str, input_data: dict):
        """Simulates an LLM call with a placeholder response."""
        print(f"  [Simulating LLM call for: '{prompt[:70]}...']")
        if "symptoms" in input_data and "fever" in input_data["symptoms"].lower():
            return "Based on symptoms, 'fever' is a key indicator for potential infections."
        return f"Simulated LLM response for: {prompt}. Relevant data: {list(input_data.keys())}"

    def _simulate_vlm_call(self, prompt: str, image_data: Image.Image):
        """Simulates a VLM (Vision Language Model) call."""
        print(f"  [Simulating VLM call for: '{prompt[:70]}...' on image of size {image_data.size}]")
        # In a real scenario, you'd send the image and prompt to a model like Gemini Pro Vision or GPT-4V
        return "Simulated VLM response: Image shows some cloudiness in the upper lung region."


    def decompose_problem(self, initial_query: str):
        """
        Decomposes a complex diagnostic query into sub-questions. (Simulated)
        """
        print(f"\n--- Reasoning Engine: Problem Decomposition for '{initial_query}' ---")
        sub_questions = [
            ReasoningNode("Q1", "What are the primary symptoms and their severity?", "text"),
            ReasoningNode("Q2", "Are there any abnormal findings in the lab results?", "text"),
            ReasoningNode("Q3", "What does the medical history suggest about pre-existing conditions or risk factors?", "text"),
            ReasoningNode("Q4", "Are there any visual anomalies in the provided medical images (e.g., X-rays)?", "image"),
            ReasoningNode("Q5", "Correlate textual symptoms with visual findings to form initial hypotheses.", "multimodal"),
            ReasoningNode("Q6", "Evaluate hypotheses against lab results and medical history.", "multimodal"),
            ReasoningNode("Q7", "Formulate a preliminary diagnosis and suggest further investigations.", "multimodal")
        ]
        print(f"Decomposed into {len(sub_questions)} sub-questions.")
        return sub_questions

    def build_thought_graph(self, sub_questions: list[ReasoningNode]):
        """
        Constructs a thought graph from sub-questions.
        Defines a simple sequential flow for demonstration.
        """
        print("\n--- Reasoning Engine: Building Thought Graph ---")
        for i, node in enumerate(sub_questions):
            self.graph.add_node(node.node_id, obj=node)
            if i > 0:
                self.graph.add_edge(sub_questions[i-1].node_id, node.node_id)
        print(f"Graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")
        return self.graph

    def execute_reasoning_step(self, node_id: str, patient_data: dict, fused_embedding: np.ndarray = None):
        """
        Executes a reasoning step for a given node.
        """
        node = self.graph.nodes[node_id]['obj']
        print(f"\n--- Reasoning Engine: Executing Step for Node '{node.node_id}' ({node.question[:50]}...) ---")
        node.status = "executing"
        node.input_data = patient_data # Pass all patient data for now

        output_text = ""
        intermediate_image = None

        if node.modality == "text":
            # Simulate textual reasoning with an LLM
            prompt = f"Analyze the following patient data to answer: {node.question}\nSymptoms: {patient_data['symptoms']}\nLab Results:\n{patient_data['lab_results'].to_string()}\nMedical History: {patient_data['medical_history']}"
            output_text = self._simulate_llm_call(prompt, patient_data)
        elif node.modality == "image":
            # Simulate visual reasoning with a VLM
            if "image" in patient_data:
                prompt = f"Analyze the medical image to answer: {node.question}"
                output_text = self._simulate_vlm_call(prompt, patient_data['image'])
                # Simulate intermediate output: annotated image
                intermediate_image = self._annotate_image(patient_data['image'], output_text)
            else:
                output_text = "No image data provided for visual reasoning."
        elif node.modality == "multimodal":
            # Simulate cross-modal inference using fused embedding and previous reasoning
            prompt = f"Correlate all available information (text, image insights) to answer: {node.question}"
            # In a real scenario, this would involve a complex VLM or a specialized model
            output_text = self._simulate_llm_call(prompt, patient_data)
            output_text += f"\n (Fusing insights from text and image embeddings, which has shape {fused_embedding.shape if fused_embedding is not None else 'N/A'})."

        node.output = output_text
        node.explanation = f"Completed {node.modality} reasoning for '{node.question}'."
        if intermediate_image:
            node.output_image = intermediate_image # Store generated image
            node.explanation += f" Generated intermediate image: {node.node_id}_output.png"
            intermediate_image.save(f"{node.node_id}_output.png") # Save for inspection
            print(f"  Intermediate annotated image saved as {node.node_id}_output.png")

        node.status = "completed"
        print(f"  Output: {node.output}")
        print(f"  Status: {node.status}")
        return node.output

    def _annotate_image(self, image: Image.Image, annotation_text: str):
        """
        Simulates annotating an image with text.
        """
        print(f"  Annotating image with text: '{annotation_text[:50]}...'\n")
        img_copy = image.copy()
        draw = ImageDraw.Draw(img_copy)
        try:
            fnt = ImageFont.load_default()
        except IOError:
            fnt = None
        
        text_position = (10, img_copy.height - 30)
        text_color = (255, 0, 0) # Red color for annotation

        if fnt: draw.text(text_position, annotation_text, fill=text_color, font=fnt)
        else: draw.text(text_position, annotation_text, fill=text_color)
        return img_copy

# --- 4. Output Layer ---

class OutputLayer:
    def __init__(self):
        pass

    def generate_diagnostic_summary(self, thought_graph: nx.DiGraph):
        """
        Compiles the results from the thought graph into a diagnostic summary.
        """
        print("\n--- Output Layer: Generating Diagnostic Summary ---")
        summary_sections = []
        for node_id in nx.topological_sort(thought_graph):
            node = thought_graph.nodes[node_id]['obj']
            if node.output:
                summary_sections.append(f"  - {node.question.strip()}: {node.output.strip()}")
        final_summary = "\n".join(summary_sections)
        print("\n--- Final Diagnostic Summary ---")
        print(final_summary)
        return final_summary

    def generate_explanation(self, thought_graph: nx.DiGraph):
        """
        Provides a step-by-step explanation of the reasoning process.
        """
        print("\n--- Output Layer: Generating Reasoning Explanation ---")
        explanation_steps = []
        for node_id in nx.topological_sort(thought_graph):
            node = thought_graph.nodes[node_id]['obj']
            explanation_steps.append(f"Step {node.node_id}: {node.question}")
            explanation_steps.append(f"  Status: {node.status}")
            if node.explanation:
                explanation_steps.append(f"  Details: {node.explanation}")
            if node.output:
                explanation_steps.append(f"  Output: {node.output[:100]}...") # Truncate for brevity
            explanation_steps.append("-" * 20)

        full_explanation = "\n".join(explanation_steps)
        print("\n--- Step-by-Step Reasoning Explanation ---")
        print(full_explanation)
        return full_explanation

# --- Main Application Logic ---

def run_diagnosis_assistant(symptoms: str, lab_results_data: dict, medical_history: str, image_filepath: str):
    """
    Main function to run the Intelligent Medical Diagnosis Assistant.
    """
    print("--- Starting Intelligent Medical Diagnosis Assistant ---")

    # 1. Input Layer
    input_layer = InputLayer()
    text_data = input_layer.ingest_textual_data(symptoms, pd.DataFrame(lab_results_data), medical_history)
    image_data = input_layer.ingest_image_data(image_filepath)

    # 2. Multimodal Fusion & Embedding Layer
    embedding_layer = MultimodalEmbeddingLayer()
    text_embedding = embedding_layer.get_text_embedding(text_data)
    image_embedding = embedding_layer.get_image_embedding(image_data)
    fused_embedding = embedding_layer.fuse_embeddings(text_embedding, image_embedding)

    # Combine all raw and embedded data for the reasoning engine
    all_patient_data = {**text_data, **image_data, "fused_embedding": fused_embedding}

    # 3. Reasoning Engine
    reasoning_engine = ReasoningEngine()
    initial_query = "Diagnose the patient's condition based on all available data."
    sub_questions = reasoning_engine.decompose_problem(initial_query)
    thought_graph = reasoning_engine.build_thought_graph(sub_questions)

    # Execute nodes in topological order
    for node_id in nx.topological_sort(thought_graph):
        reasoning_engine.execute_reasoning_step(node_id, all_patient_data, fused_embedding)

    # 4. Output Layer
    output_layer = OutputLayer()
    diagnostic_summary = output_layer.generate_diagnostic_summary(thought_graph)
    reasoning_explanation = output_layer.generate_explanation(thought_graph)

    print("\n--- Intelligent Medical Diagnosis Assistant Finished ---")
    return diagnostic_summary, reasoning_explanation


# --- Example Usage ---
if __name__ == "__main__":
    # Dummy Patient Data
    patient_symptoms = "Patient presents with persistent cough, mild fever, and shortness of breath for 3 days."
    patient_lab_results = {
        'Test': ['WBC Count', 'CRP', 'Oxygen Saturation'],
        'Value': [12.5, 8.2, 92],
        'Unit': ['x10^9/L', 'mg/L', '%'],
        'Normal Range': ['4.0-10.0', '0.0-5.0', '95-100']
    }
    patient_medical_history = "Smoker for 10 years. No known allergies. History of seasonal asthma."
    patient_image_path = "patient_chest_xray.png" # This image will be created if it doesn't exist

    summary, explanation = run_diagnosis_assistant(
        symptoms=patient_symptoms,
        lab_results_data=patient_lab_results,
        medical_history=patient_medical_history,
        image_filepath=patient_image_path
    )
