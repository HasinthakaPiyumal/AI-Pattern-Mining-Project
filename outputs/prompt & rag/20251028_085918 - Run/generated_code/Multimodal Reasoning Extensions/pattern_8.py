"""
AI-Powered Multimodal Diagnostic Assistant for Dermatology

This script demonstrates a simplified architecture for a multimodal diagnostic assistant 
for dermatology, integrating text and image analysis with structured reasoning. 
It uses dummy implementations for complex AI models and databases to be self-contained 
and runnable as a single Python file.

Architecture Components Simulated:
- Gradio (Frontend): User interface for input and output.
- FastAPI (Backend): Simulated by direct function calls within Gradio interface.
- Textual Analysis Module: Dummy for extracting info from patient history.
- Visual Analysis Module: Dummy for extracting features and generating attention maps from images.
- Chroma (Vector Database): Simulated with an in-memory dictionary for similar case retrieval.
- LLM for Reasoning: Dummy for guiding problem decomposition and explanation generation.
- Differential Diagnosis Classifier: Dummy for predicting diagnoses.
- Explanation & Output Generation: Dummy for synthesizing results.
"""

import gradio as gr
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import random
import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

# --- 1. Pydantic Models for Data Structure ---

class PatientInput(BaseModel):
    medical_history: str
    image_path: Optional[str] = None # In a real app, this would be image data or URL

class DiagnosisOutput(BaseModel):
    diagnosis_rank: List[Dict[str, Any]] # e.g., [{'condition': 'Melanoma', 'confidence': 0.95}]
    explanation: str
    visual_report: str
    attention_map_base64: Optional[str] = None # Base64 encoded image for Gradio
    comparative_cases: List[Dict[str, Any]] # e.g., [{'id': 'case_123', 'description': 'Similar lesions...', 'image_base64': '...'}]

# --- 2. Dummy Models/Services for AI Reasoning Core ---

class DummyTextualAnalyzer:
    """Simulates a ClinicalBERT-like model for textual analysis."""
    def analyze_history(self, text: str) -> Dict[str, Any]:
        print(f"[Textual Analyzer] Analyzing text: {text[:50]}...")
        # Simulate extracting key symptoms and generating initial hypotheses
        symptoms = []
        hypotheses = []
        if "mole" in text.lower() or "lesion" in text.lower():
            symptoms.append("skin lesion")
        if "itchy" in text.lower():
            symptoms.append("itchiness")
        if "grew" in text.lower() or "changed" in text.lower():
            symptoms.append("lesion change/growth")

        if "melanoma history" in text.lower():
            hypotheses.append("Melanoma")
        else:
            hypotheses.append("Benign Nevus")
            hypotheses.append("Seborrheic Keratosis")
            
        return {
            "extracted_symptoms": symptoms,
            "initial_hypotheses": hypotheses,
            "text_embedding": [random.random() for _ in range(128)] # Dummy embedding
        }

class DummyVisualAnalyzer(nn.Module):
    """Simulates a PyTorch vision model for feature extraction and Grad-CAM."""
    def __init__(self):
        super().__init__()
        # Simulate a simplified feature extractor (e.g., a small CNN)
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(32, 2) # Dummy output classes

    def forward(self, x):
        features = self.features(x)
        out = self.avgpool(features)
        out = torch.flatten(out, 1)
        return out, features # Return features for attention map

    def extract_features_and_attention(self, image: Image.Image) -> Dict[str, Any]:
        print(f"[Visual Analyzer] Processing image...")
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        img_tensor = transform(image).unsqueeze(0) # Add batch dimension

        # Simulate feature extraction
        feature_embedding, conv_features = self.forward(img_tensor)
        feature_embedding = feature_embedding.detach().numpy().flatten().tolist()

        # Simulate Grad-CAM: create a dummy heatmap
        # In a real Grad-CAM, you'd hook into the gradients of the target layer
        heatmap = np.random.rand(conv_features.shape[2], conv_features.shape[3]) # Dummy heatmap
        heatmap = cv2.resize(heatmap, (image.width, image.height))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Overlay heatmap on original image (for demonstration)
        img_np = np.array(image.convert("RGB"))
        overlaid_img = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
        
        # Convert overlaid_img back to PIL Image and then to base64 for Gradio
        _, buffer = cv2.imencode('.png', cv2.cvtColor(overlaid_img, cv2.COLOR_RGB2BGR))
        import base64
        base64_image = base64.b64encode(buffer).decode('utf-8')

        return {
            "visual_features": feature_embedding,
            "attention_map_base64": base64_image,
            "key_visual_cues": random.choice([
                "asymmetric shape", "irregular border", "varied color", 
                "uniform pigmentation", "clear borders", "small diameter"
            ]) # Dummy visual cue
        }

class DummyChromaDB:
    """Simulates an in-memory Chroma DB for vector search."""
    def __init__(self):
        self.cases = [] # Stores {'id': str, 'embedding': List[float], 'metadata': Dict}
        self._populate_dummy_cases()

    def _populate_dummy_cases(self):
        # Add some dummy cases with random embeddings
        for i in range(5):
            self.cases.append({
                "id": f"case_{i+1}",
                "embedding": [random.random() for _ in range(256)],
                "metadata": {
                    "diagnosis": random.choice(["Melanoma", "Basal Cell Carcinoma", "Squamous Cell Carcinoma", "Benign Nevus"]),
                    "description": f"A typical presentation of {{diagnosis}} with {random.choice(['irregular borders', 'uniform color'])} and {random.choice(['rapid growth', 'slow progression'])}.",
                    "image_base64": self._generate_dummy_image_base64()
                }
            })

    def _generate_dummy_image_base64(self):
        # Generates a tiny grey image as a base64 string
        dummy_img = Image.new('RGB', (50, 50), color = (random.randint(0,255), random.randint(0,255), random.randint(0,255)))
        import io, base64
        buffered = io.BytesIO()
        dummy_img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def search_similar_cases(self, query_embedding: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
        print(f"[Chroma DB] Searching for similar cases...")
        # Simulate cosine similarity search (dot product for normalized vectors)
        similarities = []
        for case in self.cases:
            # Dummy similarity calculation
            sim = np.dot(query_embedding, case["embedding"]) / (np.linalg.norm(query_embedding) * np.linalg.norm(case["embedding"])) if np.linalg.norm(query_embedding) and np.linalg.norm(case["embedding"]) else 0
            similarities.append((sim, case))
        
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        results = []
        for sim, case in similarities[:top_k]:
            case["similarity"] = sim
            results.append(case)
        return results

class DummyLLM:
    """Simulates an LLM for reasoning and explanation generation."""
    def reason_and_explain(self, context: Dict[str, Any]) -> Dict[str, Any]:
        print(f"[LLM] Performing multimodal reasoning...")
        history_analysis = context.get("history_analysis", {})
        visual_analysis = context.get("visual_analysis", {})
        initial_hypotheses = history_analysis.get("initial_hypotheses", [])
        key_visual_cues = visual_analysis.get("key_visual_cues", "no specific visual cues")

        # Simulate a structured reasoning process
        reasoning_steps = []
        reasoning_steps.append(f"1. Initial assessment based on patient history suggests potential conditions: {', '.join(initial_hypotheses)}.")
        reasoning_steps.append(f"2. Visual analysis identified key features such as: {key_visual_cues}.")
        
        # Problem decomposition and cross-referencing simulation
        if "lesion change/growth" in history_analysis.get("extracted_symptoms", []) and "asymmetric shape" == key_visual_cues:
            reasoning_steps.append("3. The observed changes in growth coupled with asymmetric shape raise concern for malignancy.")
            question = "Is this lesion likely malignant or benign?"
            answer = "Given the progressive changes and visual asymmetry, malignancy is a significant consideration."
        else:
            question = "What are the most probable diagnoses?"
            answer = "Further investigation is required, but benign conditions are also possible."
        
        full_explanation = "\n".join(reasoning_steps) + f"\n\nProblem Question: {question}\nLLM Answer: {answer}"
        
        visual_report_text = f"Visual Report: Lesion exhibits {key_visual_cues}. Further detailed analysis is provided by the attention map."
        
        return {
            "reasoning_explanation": full_explanation,
            "structured_visual_report": visual_report_text,
            "llm_problem_decomposition_answer": answer # For problem decomposition output
        }

class DummyDifferentialDiagnosisClassifier:
    """Simulates a simple classifier for differential diagnosis."""
    def classify(self, fused_embedding: List[float], initial_hypotheses: List[str]) -> List[Dict[str, Any]]:
        print(f"[Classifier] Classifying based on fused features...")
        # In a real model, this would be a trained classifier outputting probabilities
        possible_diagnoses = ["Melanoma", "Basal Cell Carcinoma", "Squamous Cell Carcinoma", "Benign Nevus", "Seborrheic Keratosis"]
        
        # Prioritize based on initial hypotheses if available
        if "Melanoma" in initial_hypotheses: # Dummy logic
            diagnosis_candidates = [
                {"condition": "Melanoma", "confidence": round(random.uniform(0.7, 0.95), 2)},
                {"condition": "Basal Cell Carcinoma", "confidence": round(random.uniform(0.3, 0.6), 2)},
                {"condition": "Benign Nevus", "confidence": round(random.uniform(0.1, 0.3), 2)}
            ]
        else:
             diagnosis_candidates = [
                {"condition": "Benign Nevus", "confidence": round(random.uniform(0.7, 0.95), 2)},
                {"condition": "Seborrheic Keratosis", "confidence": round(random.uniform(0.5, 0.8), 2)},
                {"condition": "Melanoma", "confidence": round(random.uniform(0.1, 0.4), 2)}
            ]
        
        # Sort by confidence
        diagnosis_candidates.sort(key=lambda x: x['confidence'], reverse=True)
        return diagnosis_candidates

# --- 3. Multimodal Integration & Reasoning Orchestrator (Core Engine) ---

class MultimodalReasoningEngine:
    """Orchestrates the entire multimodal reasoning process."""
    def __init__(self):
        self.text_analyzer = DummyTextualAnalyzer()
        self.visual_analyzer = DummyVisualAnalyzer()
        self.chroma_db = DummyChromaDB()
        self.llm = DummyLLM()
        self.classifier = DummyDifferentialDiagnosisClassifier()

    def process_case(self, medical_history: str, image: Optional[Image.Image]) -> DiagnosisOutput:
        print("\n--- Starting Case Processing ---")
        # 1. Initial Symptom and History Analysis
        history_analysis = self.text_analyzer.analyze_history(medical_history)
        text_embedding = history_analysis["text_embedding"]

        # 2. Visual Feature Extraction and Interpretation
        visual_analysis = {"visual_features": [0.0]*128, "attention_map_base64": None, "key_visual_cues": "no image provided"}
        visual_embedding = [0.0]*128
        if image:
            visual_analysis = self.visual_analyzer.extract_features_and_attention(image)
            visual_embedding = visual_analysis["visual_features"]

        # 3. Multimodal Integration and Cross-Referencing (Embedding Fusion)
        # A simple concatenation for demonstration
        fused_embedding = text_embedding + visual_embedding 
        print(f"[Engine] Fused embedding length: {len(fused_embedding)}")

        # 4. Problem Decomposition and LLM Reasoning
        llm_context = {
            "history_analysis": history_analysis,
            "visual_analysis": visual_analysis,
            "fused_embedding_stub": fused_embedding[:10] # Pass a stub to LLM for context
        }
        llm_output = self.llm.reason_and_explain(llm_context)
        
        # 5. Differential Diagnosis Generation
        differential_diagnoses = self.classifier.classify(
            fused_embedding,
            history_analysis.get("initial_hypotheses", [])
        )

        # 6. Comparative Image Generation (via Vector Search)
        comparative_cases = self.chroma_db.search_similar_cases(fused_embedding, top_k=3)
        
        print("--- Case Processing Complete ---")
        return DiagnosisOutput(
            diagnosis_rank=differential_diagnoses,
            explanation=llm_output["reasoning_explanation"],
            visual_report=llm_output["structured_visual_report"],
            attention_map_base64=visual_analysis["attention_map_base64"],
            comparative_cases=comparative_cases
        )

# --- 4. Gradio Interface (Frontend) ---

# Instantiate the reasoning engine
engine = MultimodalReasoningEngine()

def diagnose(medical_history_text: str, skin_lesion_image: Image.Image) -> List[Any]:
    if not medical_history_text and not skin_lesion_image:
        return ["Please provide either medical history or an image.", None, None, [], []]

    try:
        output = engine.process_case(medical_history_text, skin_lesion_image)
        
        diagnosis_str = "Top Differential Diagnoses:\n"
        for diag in output.diagnosis_rank:
            diagnosis_str += f"- {diag['condition']} (Confidence: {diag['confidence']:.2f})\n"

        comparative_case_outputs = []
        for case in output.comparative_cases:
            comparative_case_outputs.append(gr.Image.update(value=Image.open(io.BytesIO(base64.b64decode(case['metadata']['image_base64']))), label=f"{case['metadata']['diagnosis']} (Sim: {case['similarity']:.2f})"))
            comparative_case_outputs.append(gr.Markdown.update(value=f"**Description:** {case['metadata']['description']}"))

        # Fill up remaining comparative case outputs with empty values if less than 3
        while len(comparative_case_outputs) < 6: # 3 images + 3 markdown descriptions
            comparative_case_outputs.append(gr.Image.update(value=None))
            comparative_case_outputs.append(gr.Markdown.update(value=""))

        return [
            diagnosis_str,
            output.explanation,
            output.visual_report,
            Image.open(io.BytesIO(base64.b64decode(output.attention_map_base64))) if output.attention_map_base64 else None,
            *comparative_case_outputs
        ]
    except Exception as e:
        return [f"An error occurred: {str(e)}", None, None, None, [], [], [], [], [], []]


# Gradio Interface layout
with gr.Blocks() as demo:
    gr.Markdown("# AI-Powered Multimodal Diagnostic Assistant for Dermatology")
    gr.Markdown(
        "This assistant integrates patient medical history (text) and skin lesion images "
        "to provide differential diagnoses, explanations, visual reports, and comparative cases." 
        "_Note: This is a simplified demonstration with dummy AI models and in-memory data._"
    )

    with gr.Row():
        with gr.Column():
            medical_history_input = gr.Textbox(
                label="Patient Medical History",
                placeholder="E.g., 60-year-old male with a new, rapidly growing mole on his back. Reports occasional itchiness. No family history of melanoma.",
                lines=5
            )
            image_input = gr.Image(type="pil", label="Upload Skin Lesion Image")
            diagnose_btn = gr.Button("Get Diagnosis")
        
        with gr.Column():
            diagnosis_output = gr.Textbox(label="Differential Diagnoses", lines=3)
            explanation_output = gr.Textbox(label="Reasoning Explanation", lines=10)
            visual_report_output = gr.Textbox(label="Structured Visual Report", lines=3)
            attention_map_output = gr.Image(label="Visual Attention Map", show_share_button=False)
            
    gr.Markdown("### Similar Comparative Cases")
    with gr.Row():
        comp_case_img_1 = gr.Image(label="Case 1", show_share_button=False, height=150)
        comp_case_desc_1 = gr.Markdown(label="Case 1 Description")
        comp_case_img_2 = gr.Image(label="Case 2", show_share_button=False, height=150)
        comp_case_desc_2 = gr.Markdown(label="Case 2 Description")
        comp_case_img_3 = gr.Image(label="Case 3", show_share_button=False, height=150)
        comp_case_desc_3 = gr.Markdown(label="Case 3 Description")

    diagnose_btn.click(
        diagnose,
        inputs=[medical_history_input, image_input],
        outputs=[
            diagnosis_output,
            explanation_output,
            visual_report_output,
            attention_map_output,
            comp_case_img_1, comp_case_desc_1,
            comp_case_img_2, comp_case_desc_2,
            comp_case_img_3, comp_case_desc_3
        ]
    )

# To run the Gradio app
if __name__ == "__main__":
    demo.launch()
