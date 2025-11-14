import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModel, AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import networkx as nx
import matplotlib.pyplot as plt
import io
import base64
from abc import ABC, abstractmethod

# --- Configuration --- #
# For demonstration, using general models. In a real application, fine-tuned medical models would be used.
TEXT_MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
VISION_LANGUAGE_MODEL_NAME = "Salesforce/blip-vqa-base"

# --- Abstract Base Classes for Modularity --- #

class MultimodalFeatureExtractor(ABC):
    @abstractmethod
    def extract_features(self, text_data: str, image_data: Image.Image):
        pass

class ReasoningEngine(ABC):
    @abstractmethod
    def diagnose(self, multimodal_features: dict, patient_query: str):
        pass


# --- Implementations --- #

class ClinicalBERTTextFeatureExtractor(MultimodalFeatureExtractor):
    def __init__(self, model_name=TEXT_MODEL_NAME):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def extract_features(self, text_data: str, image_data: Image.Image = None):
        inputs = self.tokenizer(text_data, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Using the [CLS] token embedding as the text feature
        return outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()


class BLIPMultimodalFeatureExtractor(MultimodalFeatureExtractor):
    def __init__(self, model_name=VISION_LANGUAGE_MODEL_NAME):
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForVision2Seq.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def extract_features(self, text_data: str, image_data: Image.Image):
        # For BLIP, we can get a multimodal embedding by asking a VQA-like question
        # or use it to generate a description for the image to integrate with text.
        # For feature extraction, we'll simulate a combined representation.

        # Simple approach: generate an image caption and use it with the text data
        # More advanced: use BLIP's internal representations directly
        pixel_values = self.processor(images=image_data, return_tensors="pt").pixel_values.to(self.device)

        # Generate a caption for the image
        image_caption_ids = self.model.generate(pixel_values=pixel_values, max_new_tokens=50)
        image_caption = self.processor.decode(image_caption_ids[0], skip_special_tokens=True)

        combined_text = f"Patient medical history: {text_data}. Image description: {image_caption}."

        inputs = self.processor(text=combined_text, images=image_data, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # BLIP's forward pass gives different outputs depending on the task.
        # For a general feature, we might take the pooled output from the language model
        # after seeing image features. This is a simplification.
        with torch.no_grad():
            # This is a simplified way to get a 'multimodal' feature by passing text+image to VQA model
            # In a true feature extraction, one would access specific intermediate hidden states.
            outputs = self.model(input_ids=inputs['input_ids'], pixel_values=inputs['pixel_values'], attention_mask=inputs['attention_mask'])
            # We're taking the last hidden state of the text output as a proxy for multimodal feature
            # This is a simplification for a demonstration.
            multimodal_feature = outputs.language_model_outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

        return {"text_features": combined_text, "multimodal_feature": multimodal_feature, "image_caption": image_caption}


class MultimodalDiagnosticReasoningEngine(ReasoningEngine):
    def __init__(self, llm_model_for_reasoning=None):
        # In a real app, this would integrate with a powerful LLM like GPT-4, Llama, etc.
        # For this example, we'll simulate LLM behavior with simple string manipulation and a basic Graph-of-Thought.
        self.llm_model = llm_model_for_reasoning # Placeholder for an actual LLM client
        self.graph = nx.DiGraph()
        self.diagnosis_history = []

    def _decompose_problem(self, patient_query: str, patient_data: str, image_caption: str) -> list[str]:
        # Simulate LLM problem decomposition
        prompts = [
            f"Given the patient query '{patient_query}' and medical history: '{patient_data}', and image description: '{image_caption}', what are the key symptoms and concerns?",
            f"Are there any abnormalities or significant findings in the image as described by '{image_caption}' that correlate with the symptoms?",
            f"Based on all available information, what are potential differential diagnoses and supporting evidence?"
        ]
        return prompts

    def _build_or_update_graph_of_thought(self, step_name: str, observation: str, relation_to_nodes: list = None):
        # Simulate building a Graph-of-Thought
        node_id = f"{step_name}_{len(self.graph.nodes)}"
        self.graph.add_node(node_id, label=observation, type=step_name)
        if relation_to_nodes:
            for target_node_id, relation_type in relation_to_nodes:
                if target_node_id in self.graph:
                    self.graph.add_edge(node_id, target_node_id, type=relation_type)
                else:
                    print(f"Warning: Target node {target_node_id} not found in graph. Skipping edge.")

        # Example of adding initial nodes if not present
        if not self.graph.has_node("Patient_History_Start"):
            self.graph.add_node("Patient_History_Start", label="Patient Initial Data", type="start")
            self.graph.add_edge("Patient_History_Start", node_id, type="provides_context")

    def _generate_intermediate_visuals(self, image_data: Image.Image, multimodal_features: dict):
        # This is highly simplified. In a real scenario, attention maps from BLIP or other VLM would be used.
        # Here, we'll just draw a placeholder bounding box.
        from PIL import ImageDraw
        if image_data:
            draw = ImageDraw.Draw(image_data.copy())
            width, height = image_data.size
            # Simulate highlighting a region
            draw.rectangle([(width*0.2, height*0.2), (width*0.8, height*0.8)], outline="red", width=5)
            buf = io.BytesIO()
            image_data.save(buf, format="PNG")
            encoded_image = base64.b64encode(buf.getvalue()).decode('utf-8')
            return f"data:image/png;base64,{encoded_image}"
        return None

    def _get_graph_visualization(self):
        pos = nx.spring_layout(self.graph, seed=42)
        plt.figure(figsize=(10, 8))
        node_labels = nx.get_node_attributes(self.graph, 'label')
        edge_labels = nx.get_edge_attributes(self.graph, 'type')
        nx.draw(self.graph, pos, with_labels=True, labels=node_labels, node_color='lightblue', node_size=2000, font_size=8, font_weight='bold')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_color='red')
        buf = io.BytesIO()
        plt.savefig(buf, format="PNG")
        plt.close()
        encoded_graph = base64.b64encode(buf.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{encoded_graph}"

    def diagnose(self, multimodal_features: dict, patient_query: str, image_data: Image.Image):
        patient_data = multimodal_features.get("text_features", "")
        image_caption = multimodal_features.get("image_caption", "No image description.")

        # 1. Problem Decomposition
        sub_questions = self._decompose_problem(patient_query, patient_data, image_caption)
        reasoning_steps = []
        self.graph.clear() # Clear graph for new diagnosis

        # 2. Iterate through sub-questions (simulated reasoning)
        for i, question in enumerate(sub_questions):
            # Simulate LLM answering sub-question
            answer = f"Simulated answer for '{question}': This is a complex case. Based on the data, potential issues related to..."
            reasoning_steps.append(f"**Sub-question {i+1}:** {question}\n**Answer:** {answer}")
            
            # Update Graph-of-Thought
            self._build_or_update_graph_of_thought(f"Question_{i+1}", question)
            self._build_or_update_graph_of_thought(f"Answer_{i+1}", answer, [(f"Question_{i+1}", "answers")])

        # 3. Generate Intermediate Visuals
        highlighted_image_b64 = self._generate_intermediate_visuals(image_data, multimodal_features)
        graph_vis_b64 = self._get_graph_visualization()

        # 4. Diagnostic Hypothesis Generation (Simulated)
        final_diagnosis_hypothesis = (
            f"**Final Diagnostic Hypothesis (Simulated):**\n" +
            f"Based on the multimodal input and structured reasoning, the primary hypothesis is related to [Simulated Diagnosis, e.g., 'Chronic Obstructive Pulmonary Disease (COPD)'] with a confidence score of 85%.\n\n" +
            f"**Supporting Evidence:**\n" +
            f"- Patient history: {patient_data[:200]}... (Extracted key symptoms like dyspnea, chronic cough)\n" +
            f"- Imaging findings: {image_caption}. (Visual highlights indicate [Simulated Finding, e.g., 'hyperinflation, flattened diaphragm'])\n" +
            f"- Reasoning path: The system sequentially analyzed symptoms, imaging, and correlated findings through the knowledge graph to reach this conclusion.\n\n" +
            f"**Recommended Next Steps:**\n" +
            f"- Further pulmonary function tests.\n" +
            f"- Consult with a pulmonologist.\n" 
        )
        self.diagnosis_history.append(final_diagnosis_hypothesis)

        return {
            "reasoning_steps": "\n\n".join(reasoning_steps),
            "highlighted_image": highlighted_image_b64,
            "reasoning_graph": graph_vis_b64,
            "final_diagnosis": final_diagnosis_hypothesis
        }


# --- Gradio Interface --- #

text_extractor = ClinicalBERTTextFeatureExtractor()
multimodal_extractor = BLIPMultimodalFeatureExtractor()
diagnostic_engine = MultimodalDiagnosticReasoningEngine()

def process_patient_case(medical_history: str, medical_image: Image.Image):
    if not medical_history and not medical_image:
        return "Please provide at least medical history text or an image.", None, None, "", ""

    patient_query = "Diagnose the patient's condition based on provided data."

    # Extract multimodal features
    if medical_image and medical_history:
        multimodal_features = multimodal_extractor.extract_features(medical_history, medical_image)
    elif medical_history:
        # Fallback if only text is provided, use only text features
        text_features = text_extractor.extract_features(medical_history)
        multimodal_features = {"text_features": medical_history, "multimodal_feature": text_features, "image_caption": "No image provided."} # Simplified
    elif medical_image:
        # Fallback if only image is provided, generate caption and use as text feature
        pixel_values = multimodal_extractor.processor(images=medical_image, return_tensors="pt").pixel_values.to(multimodal_extractor.device)
        image_caption_ids = multimodal_extractor.model.generate(pixel_values=pixel_values, max_new_tokens=50)
        image_caption = multimodal_extractor.processor.decode(image_caption_ids[0], skip_special_tokens=True)
        multimodal_features = {"text_features": f"Image description: {image_caption}.", "multimodal_feature": None, "image_caption": image_caption} # Simplified
    else:
        return "Error: No input data provided.", None, None, "", ""

    # Perform diagnosis using the reasoning engine
    diagnosis_output = diagnostic_engine.diagnose(multimodal_features, patient_query, medical_image)

    return (
        diagnosis_output["final_diagnosis"],
        diagnosis_output["highlighted_image"],
        diagnosis_output["reasoning_graph"],
        diagnosis_output["reasoning_steps"]
    )


iface = gr.Interface(
    fn=process_patient_case,
    inputs=[
        gr.Textbox(label="Patient Medical History (Text)", placeholder="e.g., Patient presents with chronic cough, shortness of breath for 6 months..."),
        gr.Image(type="pil", label="Medical Image (X-ray, MRI, etc.)", optional=True)
    ],
    outputs=[
        gr.Markdown(label="Diagnostic Hypothesis & Explanation"),
        gr.Image(type="image", label="Image with Visual Highlights (Simulated)", optional=True),
        gr.Image(type="image", label="Reasoning Graph (Graph-of-Thought)", optional=True),
        gr.Markdown(label="Detailed Reasoning Steps")
    ],
    title="AI-powered Multimodal Diagnostic Assistant",
    description="Upload patient medical history (text) and an optional medical image to get a diagnostic hypothesis and the structured reasoning process.",
    allow_flagging="manual",
    flagging_dir="flagged_diagnoses",
    theme="soft"
)

if __name__ == "__main__":
    # To run this, you'll need to install:
    # pip install gradio torch transformers pillow networkx matplotlib
    # pip install "accelerate>=0.21.0" "bitsandbytes>=0.40.1" "transformers>=4.33.0"
    print("Starting Gradio interface...")
    iface.launch()

