import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import cv2
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import networkx as nx
import gradio as gr
import matplotlib.pyplot as plt
import faiss

# Suppress UserWarning from Hugging Face Transformers
import logging
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

# 1. Multimodal Input Layer
class ImageProcessor:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.model = models.resnet18(pretrained=True)
        self.model = nn.Sequential(*(list(self.model.children())[:-1]))  # Remove final classification layer
        self.model.eval()

    def process_image(self, image_input):
        if isinstance(image_input, np.ndarray):
            image = Image.fromarray(image_input)
        elif isinstance(image_input, Image.Image):
            image = image_input
        else:
            raise ValueError("Image input must be a NumPy array or PIL Image.")

        image_tensor = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            features = self.model(image_tensor)
        return features.squeeze().numpy() # Return numpy array for easier handling outside torch

class TextProcessor:
    def __init__(self, model_name="emilyalsentzer/Bio_ClinicalBERT"): # Using ClinicalBERT
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1) # Using for embeddings
        # We'll use the hidden states for embeddings, not the classification head directly

    def process_text(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        # Use the mean of the last hidden state as an embedding
        last_hidden_state = outputs.hidden_states[-1]
        embeddings = torch.mean(last_hidden_state, dim=1).squeeze().numpy()
        return embeddings

# 2. Multimodal Feature Fusion
class FusionNetwork(nn.Module):
    def __init__(self, image_feature_dim, text_feature_dim, fused_feature_dim=512):
        super().__init__()
        self.fusion_layer = nn.Linear(image_feature_dim + text_feature_dim, fused_feature_dim)
        self.relu = nn.ReLU()

    def forward(self, image_features, text_features):
        # Ensure image and text features are torch tensors
        if isinstance(image_features, np.ndarray):
            image_features = torch.from_numpy(image_features).float()
        if isinstance(text_features, np.ndarray):
            text_features = torch.from_numpy(text_features).float()

        if image_features.ndim == 1: # Add batch dimension if missing
            image_features = image_features.unsqueeze(0)
        if text_features.ndim == 1: # Add batch dimension if missing
            text_features = text_features.unsqueeze(0)

        fused_features = torch.cat((image_features, text_features), dim=1)
        fused_features = self.fusion_layer(fused_features)
        return self.relu(fused_features)

# 4. Knowledge Base Integration (simplified with FAISS)
class KnowledgeBase:
    def __init__(self, embedding_dim=768): # Assuming ClinicalBERT embedding dim
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.documents = [] # Store original text
        self.doc_to_idx = {}
        self.next_idx = 0

    def add_document(self, text, embedding):
        if text not in self.doc_to_idx:
            self.index.add(np.array([embedding], dtype=np.float32))
            self.documents.append(text)
            self.doc_to_idx[text] = self.next_idx
            self.next_idx += 1

    def retrieve(self, query_embedding, k=3):
        if self.index.ntotal == 0:
            return []
        query_embedding = np.array([query_embedding], dtype=np.float32)
        distances, indices = self.index.search(query_embedding, k)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                results.append((self.documents[idx], distances[0][i]))
        return results

# 3. Structured Reasoning Engine
class ReasoningEngine:
    def __init__(self, text_processor, knowledge_base):
        self.graph = nx.DiGraph()
        self.text_processor = text_processor
        self.knowledge_base = knowledge_base
        # Add some initial medical knowledge to the KB
        self.knowledge_base.add_document("pneumonia is a lung infection", self.text_processor.process_text("pneumonia is a lung infection"))
        self.knowledge_base.add_document("fever and cough are symptoms of pneumonia", self.text_processor.process_text("fever and cough are symptoms of pneumonia"))
        self.knowledge_base.add_document("x-ray shows consolidation in pneumonia", self.text_processor.process_text("x-ray shows consolidation in pneumonia"))

    def decompose_problem(self, patient_data):
        steps = [
            "1. Initial symptom assessment and patient history review.",
            "2. Visual analysis of medical images for anomalies.",
            "3. Correlation of symptoms and imaging findings with known conditions.",
            "4. Formulate differential diagnoses and seek supporting evidence."
        ]
        return steps

    def generate_visual_insights(self, image_np, reasoning_step):
        img_display = image_np.copy()
        if img_display.ndim == 2: # Grayscale to RGB for display
            img_display = cv2.cvtColor(img_display, cv2.COLOR_GRAY2RGB)
        
        height, width = img_display.shape[:2]

        # Placeholder for highlighting based on reasoning step
        if "image for anomalies" in reasoning_step:
            # Simulate highlighting a region (e.g., lung area)
            cv2.rectangle(img_display, (width // 4, height // 4), (width * 3 // 4, height * 3 // 4), (0, 255, 0), 5)
            cv2.putText(img_display, "Potential Anomaly", (width // 4 + 10, height // 4 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif "consolidation" in reasoning_step or "opacity" in reasoning_step:
             cv2.circle(img_display, (width // 2, height // 2), 50, (255, 0, 0), -1)
             cv2.putText(img_display, "Localized Opacity", (width // 2 - 80, height // 2 - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        plt.figure(figsize=(6, 6))
        plt.imshow(img_display)
        plt.title(f"Visual Insight: {reasoning_step}")
        plt.axis("off")
        plt.tight_layout()
        
        # Save plot to a numpy array for Gradio
        buf = plt.gcf()
        buf.canvas.draw()
        img_array = np.frombuffer(buf.canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(buf.canvas.get_width_height()[::-1] + (3,))
        plt.close(buf) # Close the plot to free memory
        return img_array


    def build_thought_graph(self, patient_text, image_findings, reasoning_step):
        self.graph.add_node("Patient", type="context", description=patient_text[:50] + "...")
        self.graph.add_node("ImageFindings", type="context", description=image_findings[:50] + "...")
        self.graph.add_edge("Patient", "ImageFindings", relation="associated_with")

        # Example: Extract entities and relationships from text (simplified)
        if "fever" in patient_text.lower():
            self.graph.add_node("Fever", type="symptom")
            self.graph.add_edge("Patient", "Fever", relation="has_symptom")
        if "cough" in patient_text.lower():
            self.graph.add_node("Cough", type="symptom")
            self.graph.add_edge("Patient", "Cough", relation="has_symptom")
        
        if "anomaly" in image_findings.lower() or "consolidation" in image_findings.lower():
            self.graph.add_node("LungAnomaly", type="finding")
            self.graph.add_edge("ImageFindings", "LungAnomaly", relation="shows")
            
        if "pneumonia" in reasoning_step.lower(): # If diagnosis mentioned in reasoning
            self.graph.add_node("Pneumonia", type="diagnosis")
            self.graph.add_edge("LungAnomaly", "Pneumonia", relation="suggests")
            self.graph.add_edge("Fever", "Pneumonia", relation="suggests")
            self.graph.add_edge("Cough", "Pneumonia", relation="suggests")

        return nx.to_dict_of_dicts(self.graph)

    def infer_diagnosis_from_graph(self, graph_dict, patient_text, image_findings):
        # Simple inference: check for common patterns
        if "Fever" in graph_dict and "Cough" in graph_dict and "LungAnomaly" in graph_dict:
            # Use KB to confirm
            query_emb = self.text_processor.process_text("fever cough lung anomaly pneumonia")
            kb_results = self.knowledge_base.retrieve(query_emb, k=1)
            if kb_results and "pneumonia" in kb_results[0][0].lower():
                return "Probable Diagnosis: Pneumonia (supported by symptoms, imaging, and knowledge base)"
        
        # More complex graph traversal could be implemented here
        if "fever" in patient_text.lower() and "rash" in patient_text.lower():
            return "Possible Diagnosis: Viral infection (further investigation needed)"

        return "Diagnosis: Undetermined (more data or reasoning steps required)"

# 5. Diagnostic Output & Explanation Module
class DiagnosisPredictor(nn.Module):
    def __init__(self, fused_feature_dim=512, num_diagnoses=2): # Example: 2 diagnoses (e.g., Pneumonia, Other)
        super().__init__()
        self.classifier = nn.Linear(fused_feature_dim, num_diagnoses)

    def forward(self, fused_features):
        return self.classifier(fused_features)

class ExplanationGenerator:
    def __init__(self, model_name="t5-small"): # Using a small T5 model for explanation generation
        self.generator = pipeline("text2text-generation", model=model_name)

    def generate_explanation(self, diagnosis, reasoning_steps, graph_info, patient_text, image_findings, kb_results):
        explanation_prompt = f"Diagnosis: {diagnosis}. \n\nReasoning Steps:\n" + "\n".join(reasoning_steps)
        
        if patient_text:
            explanation_prompt += f"\n\nPatient Symptoms/History: {patient_text}"
        if image_findings:
            explanation_prompt += f"\n\nImage Findings: {image_findings}"

        if kb_results:
            kb_text = ", ".join([res[0] for res in kb_results])
            explanation_prompt += f"\n\nRelevant Knowledge Base Information: {kb_text}"

        # Simplified graph info conversion for explanation
        graph_summary = "\nGraph Nodes: " + ", ".join([node for node in graph_info.keys()])
        # Add a simple edge summary
        edges_summary = []
        for u, v_data in graph_info.items():
            for v, data in v_data.items():
                edges_summary.append(f"{u} {data.get('relation', '->')} {v}")
        graph_summary += "\nGraph Edges: " + ", ".join(edges_summary)
        explanation_prompt += f"\n\nReasoning Graph Summary: {graph_summary}"

        generated_text = self.generator(explanation_prompt, max_new_tokens=200, num_return_sequences=1)[0]['generated_text']
        return generated_text


# Main application workflow
image_processor = ImageProcessor()
text_processor = TextProcessor()
fusion_network = FusionNetwork(image_feature_dim=512, text_feature_dim=768) # ResNet output, ClinicalBERT output
knowledge_base = KnowledgeBase()
reasoning_engine = ReasoningEngine(text_processor, knowledge_base)
diagnosis_predictor = DiagnosisPredictor(fused_feature_dim=512)
explanation_generator = ExplanationGenerator()

def diagnose_patient(image_input, patient_text):
    if image_input is None or patient_text is None or patient_text.strip() == "":
        return "Please provide both an image and patient text.", None, None, None, "", ""

    # 1. Multimodal Input Layer
    image_features = image_processor.process_image(image_input)
    text_features = text_processor.process_text(patient_text)
    
    image_findings = "No specific findings identified from initial image analysis." # Placeholder
    if "anomaly" in patient_text.lower() or "consolidation" in patient_text.lower() or "opacity" in patient_text.lower():
        image_findings = "Initial image scan suggests potential anomalies like consolidation/opacity."

    # 2. Multimodal Feature Fusion
    fused_features = fusion_network(image_features, text_features)

    # 3. Structured Reasoning Engine
    reasoning_steps = reasoning_engine.decompose_problem(patient_text)
    visual_insights = []
    for step in reasoning_steps:
        img_insight = reasoning_engine.generate_visual_insights(image_input, step)
        visual_insights.append((step, img_insight))
    
    # Build thought graph (simplified)
    graph_info = reasoning_engine.build_thought_graph(patient_text, image_findings, "")
    
    # Example: Query knowledge base based on initial findings
    query_for_kb = patient_text + " " + image_findings
    kb_query_embedding = text_processor.process_text(query_for_kb)
    kb_results = knowledge_base.retrieve(kb_query_embedding)
    
    # Infer diagnosis from graph and KB
    # Temporarily feed a reasoning step containing diagnosis keyword to build_thought_graph for demonstration
    if "fever" in patient_text.lower() and "cough" in patient_text.lower() and ("anomaly" in image_findings.lower() or "consolidation" in image_findings.lower()):
        reasoning_engine.build_thought_graph(patient_text, image_findings, "suggests pneumonia") # Update graph with potential diagnosis

    final_diagnosis = reasoning_engine.infer_diagnosis_from_graph(reasoning_engine.graph, patient_text, image_findings)

    # 5. Diagnostic Output & Explanation Module
    # Placeholder for actual prediction from fusion network
    # We'll use the graph inferred diagnosis for explanation generation here.
    
    # 6. Generate Explanation
    explanation = explanation_generator.generate_explanation(
        final_diagnosis, reasoning_steps, graph_info, patient_text, image_findings, kb_results
    )

    # Prepare outputs for Gradio
    reasoning_output_str = "\n".join([f"- {step}" for step in reasoning_steps])

    # Create a single figure for all visual insights
    num_insights = len(visual_insights)
    fig, axes = plt.subplots(1, num_insights, figsize=(6 * num_insights, 6))
    if num_insights == 1:
        axes = [axes]
    for i, (step_title, img_arr) in enumerate(visual_insights):
        axes[i].imshow(img_arr)
        axes[i].set_title(step_title, fontsize=10)
        axes[i].axis("off")
    plt.tight_layout()
    
    # Convert this single figure to a numpy array for Gradio
    buf = plt.gcf()
    buf.canvas.draw()
    combined_visual_insights_np = np.frombuffer(buf.canvas.tostring_rgb(), dtype=np.uint8)
    combined_visual_insights_np = combined_visual_insights_np.reshape(buf.canvas.get_width_height()[::-1] + (3,))
    plt.close(buf)


    # Create a string representation of the graph for display
    graph_display = "Nodes:\n"
    for node, data in reasoning_engine.graph.nodes(data=True):
        graph_display += f"  - {node} (Type: {data.get('type', 'unknown')}, Desc: {data.get('description', 'N/A')})\n"
    graph_display += "\nEdges:\n"
    for u, v, data in reasoning_engine.graph.edges(data=True):
        graph_display += f"  - {u} --({data.get('relation', 'related_to')})--> {v}\n"

    return final_diagnosis, reasoning_output_str, combined_visual_insights_np, graph_display, explanation, kb_results

# Gradio Interface
if __name__ == "__main__":
    demo = gr.Interface(
        fn=diagnose_patient,
        inputs=[
            gr.Image(type="numpy", label="Upload Medical Image (X-ray, MRI, CT)", value=None),
            gr.Textbox(lines=5, label="Patient Text Data (Symptoms, Medical History, Doctor's Notes)", placeholder="e.g., Patient presents with fever, cough, and shortness of breath for 3 days. Past medical history includes asthma.")
        ],
        outputs=[
            gr.Textbox(label="Probable Diagnosis"),
            gr.Textbox(label="Structured Reasoning Steps"),
            gr.Image(label="Intermediate Visual Insights"),
            gr.Textbox(label="Thought Graph (Nodes and Edges)"),
            gr.Textbox(label="Detailed Explanation"),
            gr.JSON(label="Relevant Knowledge Base Results")
        ],
        title="Medical Diagnostic Assistant with Multimodal Structured Reasoning",
        description="This AI assistant integrates medical images and patient text to provide structured reasoning and a probable diagnosis."
    )
    demo.launch()
