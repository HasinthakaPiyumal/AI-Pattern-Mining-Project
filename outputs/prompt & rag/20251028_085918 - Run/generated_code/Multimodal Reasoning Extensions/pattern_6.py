"""main.py: Orchestrates the Medical Diagnostic Assistant workflow."""

import os
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import spacy
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import networkx as nx

# --- 1. Data Models/Structures (multimodal_data.py) ---

class PatientData(BaseModel):
    patient_id: str
    medical_history: str
    doctor_notes: str
    lab_results: str
    xray_paths: List[str] = []
    mri_paths: List[str] = []
    ct_paths: List[str] = []
    pathology_slide_paths: List[str] = []

# --- 2. Processing Modules ---

class TextAnalyzer:
    """Analyzes textual medical data."""
    def __init__(self):
        try:
            self.nlp_ner = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading en_core_web_sm model...")
            spacy.cli.download("en_core_web_sm")
            self.nlp_ner = spacy.load("en_core_web_sm")
        
        self.summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        self.sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        doc = self.nlp_ner(text)
        entities = {}
        for ent in doc.ents:
            entities.setdefault(ent.label_, []).append(ent.text)
        return entities

    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        result = self.sentiment_analyzer(text)[0]
        return {result["label"]: result["score"]}

    def summarize_text(self, text: str, max_length: int = 100, min_length: int = 30) -> str:
        if len(text.split()) < min_length: # Handle short texts that summarizer might struggle with
            return text
        summary = self.summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]["summary_text"]
        return summary

    def process_text(self, patient_data: PatientData) -> Dict[str, Any]:
        all_text = f"{patient_data.medical_history} {patient_data.doctor_notes} {patient_data.lab_results}"
        return {
            "extracted_entities": self.extract_entities(all_text),
            "overall_sentiment": self.analyze_sentiment(all_text),
            "medical_history_summary": self.summarize_text(patient_data.medical_history),
            "doctor_notes_summary": self.summarize_text(patient_data.doctor_notes),
        }

class ImageAnalyzer:
    """Analyzes medical images (X-rays, MRI, CT, pathology slides)."""
    def __init__(self):
        # Load a pre-trained ResNet model for feature extraction (example)
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model.eval() # Set to evaluation mode
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _load_image(self, image_path: str) -> Optional[Image.Image]:
        if not os.path.exists(image_path):
            print(f"Warning: Image file not found at {image_path}")
            return None
        try:
            return Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

    def extract_features(self, image: Image.Image) -> torch.Tensor:
        input_tensor = self.preprocess(image)
        input_batch = input_tensor.unsqueeze(0) # Create a mini-batch as expected by the model
        with torch.no_grad():
            features = self.model(input_batch)
        return features.squeeze(0) # Remove batch dimension

    def detect_anomalies_mock(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Mock anomaly detection: In a real system, this would use specialized CV models."""
        width, height = image.size
        anomalies = []
        # Simulate detecting an anomaly in a quarter of the image
        if np.random.rand() > 0.6: # 40% chance of no anomaly
            x1 = int(width * np.random.rand() * 0.5)
            y1 = int(height * np.random.rand() * 0.5)
            x2 = x1 + int(width * (0.1 + np.random.rand() * 0.2))
            y2 = y1 + int(height * (0.1 + np.random.rand() * 0.2))
            anomalies.append({
                "bbox": (x1, y1, x2, y2),
                "label": np.random.choice(["mass", "lesion", "fracture", "inflammation"]),
                "confidence": round(0.7 + np.random.rand() * 0.2, 2)
            })
        return anomalies

    def process_images(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        results = []
        for path in image_paths:
            img = self._load_image(path)
            if img:
                features = self.extract_features(img)
                anomalies = self.detect_anomalies_mock(img)
                results.append({
                    "image_path": path,
                    "features": features.tolist(), # Convert tensor to list for JSON compatibility
                    "anomalies": anomalies,
                    "image_dimensions": img.size
                })
        return results

# --- 3. Reasoning Engine (reasoning_engine.py) ---

class MultimodalReasoningEngine:
    """Performs structured reasoning by integrating multimodal data."""
    def __init__(self, llm_pipeline=None):
        # Using a simplified text generation pipeline for mock LLM interactions
        self.llm_pipeline = llm_pipeline or pipeline("text-generation", model="distilgpt2")
        self.knowledge_graph = nx.DiGraph() # Conceptual graph for reasoning steps

    def _get_llm_response(self, prompt: str) -> str:
        # Simulate LLM interaction, can be replaced with actual LLM API calls
        print(f"\nLLM Prompt: {prompt[:200]}...")
        # For demonstration, we'll hardcode some simple responses based on keywords
        if "primary symptoms" in prompt.lower():
            return "Based on the notes, the primary symptoms appear to be persistent cough and fatigue."
        elif "abnormalities in the X-ray" in prompt.lower():
            return "The X-ray shows a suspicious opacity in the upper left lung lobe."
        elif "integrate findings" in prompt.lower():
            return "Integrating the textual symptoms with the X-ray findings suggests a possible respiratory infection or early-stage lung disease. Further investigation is recommended."
        elif "diagnosis options" in prompt.lower():
            return "Possible diagnoses include pneumonia, bronchitis, or early-stage lung cancer."
        return self.llm_pipeline(prompt, max_new_tokens=50, num_return_sequences=1)[0]["generated_text"]

    def problem_decomposition(self, initial_query: str, text_features: Dict, image_features: List) -> List[str]:
        """Decomposes a complex query into sub-questions."""
        sub_questions = [
            "What are the patient's primary symptoms and relevant medical history?",
            "Are there any abnormalities visible in the provided medical images (X-rays, MRI, etc.)?",
            "How do the lab results correlate with the reported symptoms and image findings?",
            "What are the potential differential diagnoses based on all available evidence?"
        ]
        # Simulate LLM refining sub-questions based on input
        llm_prompt = f"Given the patient's data, refine these general diagnostic questions: {'; '.join(sub_questions)}. Focus on: {initial_query}"
        refined_questions_raw = self._get_llm_response(llm_prompt)
        # A simple parsing for demonstration
        refined_questions = [q.strip() for q in refined_questions_raw.split('.') if q.strip()]
        if not refined_questions: # Fallback
            refined_questions = sub_questions
        return refined_questions[:4] # Limit for example

    def chain_of_thought_reasoning(self, patient_data: PatientData, text_features: Dict, image_features: List) -> List[str]:
        """Sequentially processes information to build a chain of thought."""
        reasoning_steps = []

        # Step 1: Analyze primary symptoms and history from text
        prompt_symptoms = f"Analyze the patient's medical history: '{patient_data.medical_history}' and doctor's notes: '{patient_data.doctor_notes}'. What are the primary symptoms and relevant historical factors?"
        symptoms_response = self._get_llm_response(prompt_symptoms)
        reasoning_steps.append(f"Step 1 (Textual Analysis - Symptoms): {symptoms_response}")
        self.knowledge_graph.add_node("Primary Symptoms", type="textual", content=symptoms_response)

        # Step 2: Analyze image findings
        image_summaries = []
        for img_data in image_features:
            anomalies_str = ", ".join([f"{a['label']} (confidence: {a['confidence']:.2f}) at {a['bbox']}" for a in img_data['anomalies']])
            if anomalies_str:
                image_summaries.append(f"Image {img_data['image_path']}: Detected {anomalies_str}.")
            else:
                image_summaries.append(f"Image {img_data['image_path']}: No significant anomalies detected.")
        image_analysis_summary = "\n".join(image_summaries)
        reasoning_steps.append(f"Step 2 (Image Analysis - Findings): {image_analysis_summary}")
        self.knowledge_graph.add_node("Image Findings", type="visual", content=image_analysis_summary)
        self.knowledge_graph.add_edge("Primary Symptoms", "Image Findings", relation="informs")

        # Step 3: Integrate findings
        prompt_integration = (
            f"Integrate the following: \n" +
            f"Symptoms/History: {symptoms_response}\n" +
            f"Image Findings: {image_analysis_summary}\n" +
            f"What initial inferences can be made by combining this multimodal evidence?"
        )
        integration_response = self._get_llm_response(prompt_integration)
        reasoning_steps.append(f"Step 3 (Multimodal Integration): {integration_response}")
        self.knowledge_graph.add_node("Integrated Inferences", type="multimodal", content=integration_response)
        self.knowledge_graph.add_edge("Image Findings", "Integrated Inferences", relation="leads_to")

        # Step 4: Generate potential diagnoses (Least-to-Most style progression)
        prompt_diagnoses = (
            f"Based on the integrated inferences: '{integration_response}', and patient's lab results: '{patient_data.lab_results}', " +
            f"what are the most probable differential diagnoses? Provide a concise list."
        )
        diagnoses_response = self._get_llm_response(prompt_diagnoses)
        reasoning_steps.append(f"Step 4 (Hypothesis Generation): {diagnoses_response}")
        self.knowledge_graph.add_node("Differential Diagnoses", type="reasoning", content=diagnoses_response)
        self.knowledge_graph.add_edge("Integrated Inferences", "Differential Diagnoses", relation="suggests")

        return reasoning_steps

# --- 4. Output Generator (output_generator.py) ---

class OutputGenerator:
    """Generates human-readable diagnostic outputs."""
    def __init__(self):
        # Try to load a default font, fall back to a generic one if not found
        self.font = None
        try:
            self.font = ImageFont.truetype("arial.ttf", 20) # Common font on Windows/Linux
        except IOError:
            try:
                self.font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 20) # macOS path
            except IOError:
                print("Warning: Could not load Arial.ttf. Using default PIL font, which may not support all characters.")
                self.font = ImageFont.load_default()

    def annotate_image(self, image_path: str, anomalies: List[Dict[str, Any]], output_dir: str = "annotated_images") -> Optional[str]:
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        for anomaly in anomalies:
            bbox = anomaly["bbox"]
            label = anomaly["label"]
            confidence = anomaly["confidence"]

            draw.rectangle(bbox, outline="red", width=3)
            text = f"{label} ({confidence:.2f})"
            # Position text above the bounding box
            text_x, text_y = bbox[0], bbox[1] - 25
            if text_y < 0: text_y = bbox[3] + 5 # If too high, place below

            # Draw a semi-transparent background for text
            text_bbox = draw.textbbox((text_x, text_y), text, font=self.font)
            draw.rectangle(text_bbox, fill=(255, 0, 0, 128)) # Red with 50% transparency
            draw.text((text_x, text_y), text, fill="white", font=self.font)
        
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"annotated_{os.path.basename(image_path)}")
        img.save(output_path)
        return output_path

    def generate_reasoning_summary(self, reasoning_steps: List[str]) -> str:
        summary = "### Multimodal Reasoning Process Summary\n\n"
        for i, step in enumerate(reasoning_steps):
            summary += f"{i+1}. {step}\n"
        return summary

    def generate_diagnostic_report(self, patient_data: PatientData, text_analysis: Dict, image_analysis_results: List[Dict], reasoning_summary: str) -> Dict[str, Any]:
        report = {
            "patient_id": patient_data.patient_id,
            "timestamp": "2023-10-27 10:30:00", # Placeholder
            "overview": "Comprehensive diagnostic report based on multimodal structured reasoning.",
            "patient_summary": {
                "medical_history_summary": text_analysis["medical_history_summary"],
                "doctor_notes_summary": text_analysis["doctor_notes_summary"],
                "extracted_entities": text_analysis["extracted_entities"],
                "overall_sentiment": text_analysis["overall_sentiment"]
            },
            "image_findings": [],
            "reasoning_process": reasoning_summary,
            "final_diagnosis_hypothesis": "Refer to reasoning process for specific diagnoses.", # Will be updated by reasoning engine
            "recommendations": "Further clinical evaluation and specific tests recommended based on findings."
        }

        annotated_image_paths = []
        for img_result in image_analysis_results:
            if img_result["anomalies"]:
                annotated_path = self.annotate_image(img_result["image_path"], img_result["anomalies"])
                if annotated_path: annotated_image_paths.append(annotated_path)
            report["image_findings"].append({
                "original_path": img_result["image_path"],
                "annotated_path": annotated_path if img_result["anomalies"] else "N/A",
                "anomalies_detected": img_result["anomalies"]
            })
        report["annotated_image_outputs"] = annotated_image_paths

        # Extract final diagnosis hypothesis from reasoning_summary if possible
        if "Step 4 (Hypothesis Generation)" in reasoning_summary:
            start_idx = reasoning_summary.find("Step 4 (Hypothesis Generation):") + len("Step 4 (Hypothesis Generation):")
            end_idx = reasoning_summary.find("\n", start_idx)
            if end_idx == -1: end_idx = len(reasoning_summary)
            report["final_diagnosis_hypothesis"] = reasoning_summary[start_idx:end_idx].strip()

        return report

# --- 5. Main Application (main.py) ---

def create_dummy_image(filename: str, width: int = 500, height: int = 400, add_noise: bool = True):
    """Creates a dummy image file for testing."""
    img_array = np.zeros((height, width, 3), dtype=np.uint8) + 50 # Dark background
    if add_noise:
        img_array = img_array + np.random.randint(0, 30, (height, width, 3), dtype=np.uint8)

    # Draw a simple shape to simulate an 'anomaly'
    if 'xray' in filename.lower():
        cv2.circle(img_array, (width // 3, height // 2), 40, (0, 0, 200), -1) # Red circle
        cv2.putText(img_array, "Suspect Opacity", (width // 3 - 60, height // 2 - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    elif 'mri' in filename.lower():
        cv2.rectangle(img_array, (width // 2, height // 3), (width * 3 // 4, height * 2 // 3), (0, 200, 0), -1) # Green rectangle
        cv2.putText(img_array, "Tissue Anomaly", (width // 2, height // 3 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        cv2.ellipse(img_array, (width // 2, height // 2), (60, 30), 0, 0, 360, (200, 200, 0), -1) # Yellow ellipse

    Image.fromarray(img_array).save(filename)
    print(f"Created dummy image: {filename}")

def main():
    print("Initializing Medical Diagnostic Assistant...")

    # Create dummy image files for demonstration
    dummy_image_dir = "./dummy_medical_images"
    os.makedirs(dummy_image_dir, exist_ok=True)
    xray_path = os.path.join(dummy_image_dir, "patient_001_xray_chest.png")
    mri_path = os.path.join(dummy_image_dir, "patient_001_mri_brain.png")
    create_dummy_image(xray_path)
    create_dummy_image(mri_path)

    # Sample Patient Data
    patient_data = PatientData(
        patient_id="P001",
        medical_history=(
            "Patient is a 65-year-old male with a history of hypertension and occasional shortness of breath. "
            "Reports persistent dry cough for the last 3 weeks, accompanied by mild fatigue. "
            "No fever or chills. Smoked for 30 years, quit 10 years ago. Family history of lung cancer." 
        ),
        doctor_notes=(
            "Patient presented with chronic cough. Chest auscultation revealed diminished breath sounds in the left upper lobe. "
            "Ordered chest X-ray and blood tests. Concerned about potential respiratory infection or malignancy given smoking history." 
        ),
        lab_results="CBC normal, CRP slightly elevated (12 mg/L), D-dimer negative. Sputum culture pending.",
        xray_paths=[xray_path],
        mri_paths=[mri_path],
        ct_paths=[],
        pathology_slide_paths=[]
    )

    # 1. Initialize Processing Modules
    text_analyzer = TextAnalyzer()
    image_analyzer = ImageAnalyzer()

    # 2. Process Data
    print("\n--- Analyzing Text Data ---")
    text_analysis_results = text_analyzer.process_text(patient_data)
    print("Text Analysis Complete.")
    # print(json.dumps(text_analysis_results, indent=2))

    print("\n--- Analyzing Image Data ---")
    all_image_paths = patient_data.xray_paths + patient_data.mri_paths + patient_data.ct_paths + patient_data.pathology_slide_paths
    image_analysis_results = image_analyzer.process_images(all_image_paths)
    print("Image Analysis Complete.")
    # print(json.dumps(image_analysis_results, indent=2))

    # 3. Initialize and Run Reasoning Engine
    reasoning_engine = MultimodalReasoningEngine()
    print("\n--- Running Multimodal Reasoning Engine ---")
    
    # Problem Decomposition
    initial_query = "Diagnose the patient's current respiratory condition."
    sub_questions = reasoning_engine.problem_decomposition(initial_query, text_analysis_results, image_analysis_results)
    print("Problem decomposed into sub-questions:")
    for sq in sub_questions: print(f"- {sq}")

    # Chain-of-Thought Reasoning
    reasoning_steps = reasoning_engine.chain_of_thought_reasoning(patient_data, text_analysis_results, image_analysis_results)
    print("Multimodal Reasoning Steps Generated.")

    # 4. Generate Outputs
    output_generator = OutputGenerator()
    print("\n--- Generating Outputs ---")

    reasoning_summary = output_generator.generate_reasoning_summary(reasoning_steps)
    print("Reasoning Summary:\n", reasoning_summary)

    diagnostic_report = output_generator.generate_diagnostic_report(
        patient_data,
        text_analysis_results,
        image_analysis_results,
        reasoning_summary
    )

    print("\n--- Final Diagnostic Report ---")
    # Print report in a more readable format
    print(f"Patient ID: {diagnostic_report['patient_id']}")
    print(f"Overview: {diagnostic_report['overview']}")
    print("\nPatient Summary:")
    print(f"  Medical History Summary: {diagnostic_report['patient_summary']['medical_history_summary']}")
    print(f"  Doctor Notes Summary: {diagnostic_report['patient_summary']['doctor_notes_summary']}")
    print(f"  Extracted Entities: {diagnostic_report['patient_summary']['extracted_entities']}")
    print(f"  Overall Sentiment: {diagnostic_report['patient_summary']['overall_sentiment']}")

    print("\nImage Findings:")
    if diagnostic_report['image_findings']:
        for img_f in diagnostic_report['image_findings']:
            print(f"  Original Image: {img_f['original_path']}")
            print(f"  Annotated Image: {img_f['annotated_path']}")
            print(f"  Anomalies: {img_f['anomalies_detected']}")
    else:
        print("  No images processed or no findings.")
    
    print("\nReasoning Process Summary:")
    print(diagnostic_report['reasoning_process'])

    print(f"\nFinal Diagnosis Hypothesis: {diagnostic_report['final_diagnosis_hypothesis']}")
    print(f"Recommendations: {diagnostic_report['recommendations']}")

    print("\nAnnotated images saved to ./annotated_images/")
    print("Medical Diagnostic Assistant workflow complete.")

if __name__ == "__main__":
    main()
