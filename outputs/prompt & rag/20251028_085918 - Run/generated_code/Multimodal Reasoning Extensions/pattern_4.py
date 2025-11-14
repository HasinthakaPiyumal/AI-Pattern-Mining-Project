import networkx as nx
from PIL import Image
import io
import base64
import cv2
import numpy as np

class MultimodalDiagnosisAssistant:
    """
    Orchestrates multimodal structured reasoning for medical diagnosis.
    Integrates text processing (LLM simulation) and image analysis (vision model simulation)
    to break down problems, generate intermediate outputs, and formulate a diagnosis.
    """
    def __init__(self):
        self.reasoning_graph = nx.DiGraph() # Conceptually represents the thought process as a graph
        self.llm = self._init_llm_model() # Placeholder for Language Model
        self.vision_model = self._init_vision_model() # Placeholder for Vision Model

    def _init_llm_model(self):
        # In a real application, initialize a powerful LLM (e.g., from transformers library or an API client)
        # For this example, we simulate its behavior.
        return "Simulated LLM"

    def _init_vision_model(self):
        # In a real application, initialize a specialized vision model for medical images
        # (e.g., from Hugging Face, torchvision, or custom trained models)
        # For this example, we simulate its behavior.
        return "Simulated Vision Model"

    def _simulate_llm_inference(self, prompt):
        # A dummy function to simulate LLM responses based on keywords in the prompt.
        if "symptoms" in prompt and "history" in prompt:
            return "Initial assessment suggests inflammatory or infectious process based on combined symptoms and history."
        if "lab results" in prompt:
            return "Lab results indicate elevated markers consistent with acute inflammation or tissue damage."
        if "image findings" in prompt:
            return "LLM interprets image findings: \'Pleural Effusion\' strongly suggests fluid accumulation around the lungs. \'Lung Nodule\' requires further investigation."
        if "Correlate" in prompt:
            return "Correlation reveals a consistent picture of respiratory distress, possibly due to inflammation/infection exacerbated by fluid accumulation."
        if "Interpret image findings in context" in prompt:
            return "Contextual interpretation: The pleural effusion observed in the image aligns with reported shortness of breath and chest pain, and could be a sequela of an inflammatory condition indicated by lab results."
        if "Synthesize" in prompt:
            return "**Most likely Diagnosis: Acute Pleurisy with Effusion (Confidence: 80%)**\nAlternative: Pneumonia (Confidence: 65%). Further tests for differential diagnosis recommended."
        return f"LLM thought for: {prompt}"

    def _simulate_vision_inference(self, pil_image):
        # A dummy function to simulate vision model findings and bounding box coordinates.
        # In a real application, this would involve actual object detection or segmentation models.
        # For demonstration, we use fixed dummy findings.
        dummy_findings = [
            {"label": "Pleural Effusion", "confidence": 0.85, "bbox": [50, 50, 400, 350]}, # x1, y1, x2, y2
            {"label": "Lung Nodule (possible)", "confidence": 0.60, "bbox": [280, 150, 350, 220]}
        ]
        return dummy_findings

    def _annotate_image(self, pil_image, annotations):
        # Converts a PIL image to OpenCV format, draws annotations, and converts back to PIL.
        opencv_image = np.array(pil_image.convert("RGB"))
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)

        for ann in annotations:
            x1, y1, x2, y2 = ann["bbox"]
            label = ann["label"]
            confidence = ann["confidence"]
            color = (0, 255, 0) # Green BGR

            cv2.rectangle(opencv_image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(opencv_image, f"{label} ({confidence:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        pil_annotated_image = Image.fromarray(cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB))
        return pil_annotated_image

    def _add_reasoning_step(self, step_id, description, inputs, outputs, multimodal_output=None):
        # Adds a node to the reasoning graph representing a step in the diagnosis process.
        self.reasoning_graph.add_node(step_id, description=description, inputs=inputs, outputs=outputs, multimodal_output=multimodal_output)
        # For sequential flow, connect to the previous step if available for graph visualization
        if len(self.reasoning_graph.nodes) > 1:
            prev_node = list(self.reasoning_graph.nodes)[-2] # Get the second-to-last added node
            self.reasoning_graph.add_edge(prev_node, step_id)

    def diagnose(self, text_data, image_file):
        """
        Performs a multimodal structured diagnosis.

        Args:
            text_data (str): Patient's textual medical information (symptoms, history, lab results).
            image_file (str or io.BytesIO): Path to the medical image or an in-memory image file.

        Returns:
            tuple: (list of reasoning steps, final diagnostic hypothesis, base64 encoded initial annotated image)
        """
        self.reasoning_graph = nx.DiGraph() # Reset graph for a new diagnosis
        reasoning_steps_list = []
        final_hypothesis = "No diagnosis could be formulated."
        initial_annotated_image_b64 = None

        # --- 1. Multimodal Input Processing ---
        # Text analysis
        text_analysis_prompt = f"Analyze patient symptoms, medical history, and lab results: {text_data}"
        text_summary = self._simulate_llm_inference(text_analysis_prompt)
        self._add_reasoning_step("step_1_text_analysis", "Initial Textual Data Analysis", {"raw_text": text_data}, {"summary": text_summary})
        reasoning_steps_list.append({"description": "Initial text summary and analysis", "output": text_summary})

        # Image analysis
        pil_image = Image.open(image_file) if isinstance(image_file, str) else Image.open(io.BytesIO(image_file.read()))
        vision_findings = self._simulate_vision_inference(pil_image)
        self._add_reasoning_step("step_2_image_analysis", "Initial Medical Image Analysis", {"raw_image_input": "binary_image_data"}, {"findings": vision_findings})
        reasoning_steps_list.append({"description": "Initial image analysis findings", "output": str(vision_findings)})

        # Generate and store an initial annotated image as an intermediate multimodal output
        annotated_pil_image = self._annotate_image(pil_image.copy(), vision_findings)
        with io.BytesIO() as buffer:
            annotated_pil_image.save(buffer, format="PNG")
            initial_annotated_image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        reasoning_steps_list.append({"description": "Visual interpretation of initial image findings (intermediate output)", "multimodal_output": initial_annotated_image_b64})


        # --- 2. Problem Decomposition and Structured Reasoning ---
        # Sub-question 1: Correlate textual findings
        sub_q1_prompt = f"Correlate the patient's summarized text data: '{text_summary}' with potential conditions based on general medical knowledge."
        sub_q1_answer = self._simulate_llm_inference(sub_q1_prompt)
        self._add_reasoning_step("step_3_textual_correlation", "Correlate Textual Findings",
                                  {"text_summary": text_summary}, {"correlation_output": sub_q1_answer})
        reasoning_steps_list.append({"description": "Correlation of textual findings", "output": sub_q1_answer})

        # Sub-question 2: Contextualize image findings with textual data
        sub_q2_prompt = f"Interpret image findings: {vision_findings} in the context of textual analysis: '{text_summary}'."
        sub_q2_answer = self._simulate_llm_inference(sub_q2_prompt)
        self._add_reasoning_step("step_4_image_contextualization", "Contextualize Image Findings",
                                  {"vision_findings": vision_findings, "text_summary": text_summary}, {"contextual_interpretation": sub_q2_answer})
        reasoning_steps_list.append({"description": "Contextualized image interpretation", "output": sub_q2_answer})


        # --- 3. Diagnostic Hypothesis Formulation ---
        # Synthesize all prior reasoning steps and multimodal outputs
        synthesis_prompt = (
            f"Synthesize all gathered information for a diagnostic hypothesis:\n"
            f"- Text Analysis: '{text_summary}'\n"
            f"- Image Findings: {vision_findings}\n"
            f"- Textual Correlation: '{sub_q1_answer}'\n"
            f"- Contextualized Image Interpretation: '{sub_q2_answer}'\n"
            f"Propose a final diagnosis with confidence levels and brief justification."
        )
        final_hypothesis = self._simulate_llm_inference(synthesis_prompt)
        self._add_reasoning_step("step_5_final_synthesis", "Formulate Diagnostic Hypothesis",
                                  {"all_intermediate_outputs": "..." }, {"diagnosis": final_hypothesis})
        reasoning_steps_list.append({"description": "Final diagnostic hypothesis formulation", "output": final_hypothesis})

        return reasoning_steps_list, final_hypothesis, initial_annotated_image_b64
