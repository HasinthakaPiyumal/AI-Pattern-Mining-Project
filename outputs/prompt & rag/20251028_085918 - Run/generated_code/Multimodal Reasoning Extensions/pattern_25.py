import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from transformers import AutoTokenizer, AutoModel
import networkx as nx
import numpy as np
import cv2
import matplotlib.pyplot as plt
import gradio as gr
import io


class InputPreprocessor:
    def __init__(self):
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") # Placeholder for medical specific tokenizer

    def preprocess_image(self, image_input):
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        else: # Assuming it's a PIL Image or similar from Gradio
            image = image_input.convert("RGB")
        return self.image_transform(image).unsqueeze(0) # Add batch dimension

    def preprocess_text(self, text_data):
        return self.tokenizer(text_data, return_tensors="pt", truncation=True, padding=True)


class ImageFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # Placeholder for a pre-trained Vision Transformer or ResNet
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 512) # Outputting a 512-dim embedding
        )

    def forward(self, x):
        return self.model(x)


class TextFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # Placeholder for a pre-trained BERT-like model
        self.model = AutoModel.from_pretrained("bert-base-uncased") # Placeholder for medical specific model
        self.linear = nn.Linear(self.model.config.hidden_size, 512) # Map to 512-dim embedding

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return self.linear(pooled_output)


class MultimodalFusionLayer(nn.Module):
    def __init__(self, image_embedding_dim, text_embedding_dim, fused_embedding_dim):
        super().__init__()
        self.fusion_layer = nn.Linear(image_embedding_dim + text_embedding_dim, fused_embedding_dim)
        self.relu = nn.ReLU()

    def forward(self, image_features, text_features):
        fused_features = torch.cat((image_features, text_features), dim=-1)
        return self.relu(self.fusion_layer(fused_features))


class DutyDistinctChainOfThought:
    def __init__(self):
        self.sub_questions = [
            "What are the primary visual anomalies in the image?",
            "Are these anomalies consistent with the patient's reported symptoms and medical history?",
            "What potential diagnoses can be inferred from the combined visual and textual evidence?",
            "Are there any confounding factors or differential diagnoses to consider?"
        ]

    def generate_subquestions(self, fused_features):
        # In a real scenario, this would dynamically generate sub-questions
        # based on input features and context. Here, we use predefined ones.
        print(f"DEBUG: Fused features shape for DDCoT: {fused_features.shape}")
        return self.sub_questions

    def analyze_focused(self, sub_question, image_features, text_features):
        # Simulate focused analysis based on the sub-question
        if "visual anomalies" in sub_question.lower():
            return "Simulated visual analysis: Detected opacity in upper right lung field."
        elif "consistent with symptoms" in sub_question.lower():
            return "Simulated consistency check: Opacity is consistent with patient's cough and fever."
        elif "potential diagnoses" in sub_question.lower():
            return "Simulated diagnosis inference: Possible pneumonia or tuberculosis."
        else:
            return f"Simulated analysis for: {sub_question}"


class MultimodalGraphOfThought:
    def __init__(self):
        self.graph = nx.DiGraph()

    def build_graph(self, multimodal_observations, dd_cot_answers):
        self.graph.clear()
        self.graph.add_node("Patient", type="patient")

        # Add observations as nodes
        for i, obs in enumerate(multimodal_observations):
            node_name = f"Observation_{i}"
            self.graph.add_node(node_name, type="observation", description=obs)
            self.graph.add_edge("Patient", node_name, relation="has_observation")

        # Add DDCoT answers as reasoning nodes
        for i, answer in enumerate(dd_cot_answers):
            node_name = f"Reasoning_{i}"
            self.graph.add_node(node_name, type="reasoning_step", content=answer)
            # Link reasoning steps
            if i > 0:
                self.graph.add_edge(f"Reasoning_{i-1}", node_name, relation="leads_to")
            else:
                self.graph.add_edge("Patient", node_name, relation="starts_reasoning")

        # Simulate extracting entities and relationships (very simplified)
        for obs in multimodal_observations:
            if "opacity" in obs.lower():
                self.graph.add_node("Opacity", type="visual_feature")
                self.graph.add_edge(obs.replace("Simulated visual analysis: ", ""), "Opacity", relation="contains")
            if "cough" in obs.lower() or "fever" in obs.lower():
                self.graph.add_node("Symptoms", type="textual_feature")
                self.graph.add_edge(obs.replace("Simulated consistency check: ", ""), "Symptoms", relation="mentions")
            if "pneumonia" in obs.lower() or "tuberculosis" in obs.lower():
                self.graph.add_node("Pneumonia", type="diagnosis_candidate")
                self.graph.add_node("Tuberculosis", type="diagnosis_candidate")
                self.graph.add_edge(obs.replace("Simulated diagnosis inference: ", ""), "Pneumonia", relation="suggests")
                self.graph.add_edge(obs.replace("Simulated diagnosis inference: ", ""), "Tuberculosis", relation="suggests")

        # Example of inferred relationships
        if "Opacity" in self.graph.nodes and "Symptoms" in self.graph.nodes:
            self.graph.add_edge("Opacity", "Symptoms", relation="associated_with")

        return self.graph

    def get_graph_summary(self):
        if not self.graph.nodes:
            return "Graph is empty."
        summary = "\nGraph Nodes:\n"
        for node, data in self.graph.nodes(data=True):
            summary += f"  - {node} (Type: {data.get('type', 'N/A')}, Desc: {data.get('description', data.get('content', 'N/A'))})\n"
        summary += "\nGraph Edges:\n"
        for u, v, data in self.graph.edges(data=True):
            summary += f"  - {u} --({data.get('relation', 'N/A')})--> {v}\n"
        return summary


class ChainOfImages:
    def __init__(self):
        pass

    def generate_attention_heatmap(self, image_tensor, feature_map):
        # Dummy Grad-CAM like visualization
        # image_tensor shape: (1, 3, H, W)
        # feature_map shape: (1, C, H', W')

        if image_tensor.dim() == 4 and image_tensor.shape[0] == 1:
            original_image = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        else:
            original_image = np.zeros((224, 224, 3)) # Placeholder for invalid input

        # Normalize image for display
        original_image = (original_image - original_image.min()) / (original_image.max() - original_image.min() + 1e-8)
        original_image = (original_image * 255).astype(np.uint8)

        if feature_map is None or feature_map.dim() < 2:
            heatmap = np.zeros(original_image.shape[:2], dtype=np.uint8)
        else:
            # Simple average of feature map channels as a heatmap proxy
            heatmap = torch.mean(feature_map, dim=1).squeeze(0).cpu().detach().numpy()
            heatmap = np.maximum(heatmap, 0)
            heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        superimposed_img = heatmap * 0.4 + original_image * 0.6
        superimposed_img = np.uint8(np.clip(superimposed_img, 0, 255))

        return Image.fromarray(superimposed_img)

    def generate_roi_overlay(self, image_tensor, rois):
        # Dummy ROI overlay
        if image_tensor.dim() == 4 and image_tensor.shape[0] == 1:
            original_image = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        else:
            original_image = np.zeros((224, 224, 3)) # Placeholder for invalid input

        original_image = (original_image - original_image.min()) / (original_image.max() - original_image.min() + 1e-8)
        original_image = (original_image * 255).astype(np.uint8).copy()

        for roi in rois: # roi is [x, y, w, h] format
            x, y, w, h = roi
            cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(original_image, "ROI", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return Image.fromarray(original_image)


class DiagnosisClassifier(nn.Module):
    def __init__(self, fused_embedding_dim, num_classes):
        super().__init__()
        self.classifier = nn.Linear(fused_embedding_dim, num_classes)

    def forward(self, fused_features):
        return self.classifier(fused_features)


class ExplanationGenerator:
    def __init__(self):
        pass

    def generate_report(self, dd_cot_steps, m_got_summary, co_images, diagnosis_prediction):
        report = "## Medical Diagnostic Report\n\n"
        report += "### Structured Reasoning Steps (Chain-of-Thought):\n"
        for i, step in enumerate(dd_cot_steps):
            report += f"- **Step {i+1}:** {step}\n"
        
        report += "\n### Multimodal Graph-of-Thought Summary:\n"
        report += f"```\n{m_got_summary}\n```\n"

        report += "\n### Predicted Diagnosis:\n"
        report += f"**{diagnosis_prediction}**\n"

        report += "\n### Visual Explanations (Chain-of-Images):\n"
        # co_images will be a list of (image_title, PIL_Image) tuples
        image_data_list = []
        for title, img in co_images:
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            img_str = f"<img src='data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}' alt='{title}' style='width:300px;'>"
            image_data_list.append(f"**{title}:** {img_str}")
        
        report += "\n".join(image_data_list)

        return report


import base64

class MedicalDiagnosticAssistant:
    def __init__(self):
        self.preprocessor = InputPreprocessor()
        self.image_feature_extractor = ImageFeatureExtractor()
        self.text_feature_extractor = TextFeatureExtractor()
        self.multimodal_fusion_layer = MultimodalFusionLayer(512, 512, 256)
        self.dd_cot_module = DutyDistinctChainOfThought()
        self.m_got_module = MultimodalGraphOfThought()
        self.co_images_module = ChainOfImages()
        self.diagnosis_classifier = DiagnosisClassifier(256, num_classes=5) # Example: 5 possible diagnoses
        self.explanation_generator = ExplanationGenerator()
        self.diagnosis_labels = ["Normal", "Pneumonia", "Tuberculosis", "Fracture", "Tumor"]

    def diagnose(self, image_input, patient_text):
        # 1. Input and Preprocessing
        processed_image = self.preprocessor.preprocess_image(image_input)
        processed_text = self.preprocessor.preprocess_text(patient_text)

        # 2. Multimodal Feature Extraction
        image_features = self.image_feature_extractor(processed_image)
        text_features = self.text_feature_extractor(processed_text["input_ids"], processed_text["attention_mask"])
        fused_features = self.multimodal_fusion_layer(image_features, text_features)

        # 3. Structured Reasoning Engine
        # DDCoT
        sub_questions = self.dd_cot_module.generate_subquestions(fused_features)
        dd_cot_answers = []
        visual_explanations = []
        for i, sq in enumerate(sub_questions):
            answer = self.dd_cot_module.analyze_focused(sq, image_features, text_features)
            dd_cot_answers.append(f"Question: {sq}\nAnswer: {answer}")

            # Simulate Chain-of-Images generation based on sub-question
            if i == 0: # First question often involves visual anomalies
                heatmap_img = self.co_images_module.generate_attention_heatmap(processed_image, self.image_feature_extractor.model[0].weight.grad) # Placeholder for real Grad-CAM
                visual_explanations.append(("Attention Heatmap (Anomalies)", heatmap_img))
            elif i == 1: # Second question might involve ROI
                # Dummy ROI: Example [x, y, w, h]
                dummy_rois = [[50, 50, 100, 100], [150, 150, 70, 70]] 
                roi_img = self.co_images_module.generate_roi_overlay(processed_image, dummy_rois)
                visual_explanations.append(("ROI Overlay (Consistency Check)", roi_img))
            

        # MGoT
        graph = self.m_got_module.build_graph(dd_cot_answers, dd_cot_answers) # Using DDCoT answers as observations and reasoning steps
        m_got_summary = self.m_got_module.get_graph_summary()

        # 4. Diagnostic Prediction and Explanation
        logits = self.diagnosis_classifier(fused_features)
        predicted_class_idx = torch.argmax(logits, dim=1).item()
        predicted_diagnosis = self.diagnosis_labels[predicted_class_idx]

        # Generate comprehensive report
        report = self.explanation_generator.generate_report(
            dd_cot_answers, m_got_summary, visual_explanations, predicted_diagnosis
        )

        return report, visual_explanations[0][1] if visual_explanations else None # Return report and first visual explanation for display


assistant = MedicalDiagnosticAssistant()

def predict_diagnosis(image, text):
    report, first_viz = assistant.diagnose(image, text)
    return report, first_viz


if __name__ == "__main__":
    # Gradio Interface
    iface = gr.Interface(
        fn=predict_diagnosis,
        inputs=[
            gr.Image(type="pil", label="Medical Image (X-ray, MRI, CT)"),
            gr.Textbox(lines=5, label="Patient Medical History & Symptoms")
        ],
        outputs=[
            gr.Markdown(label="Diagnostic Report"),
            gr.Image(type="pil", label="Key Visual Explanation")
        ],
        title="Multimodal Medical Diagnostic Assistant",
        description="Upload a medical image and provide patient's medical history/symptoms for a structured diagnostic analysis.",
        allow_flagging="never"
    )
    iface.launch()
