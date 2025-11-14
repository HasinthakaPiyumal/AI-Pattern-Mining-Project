import pandas as pd
import spacy
from PIL import Image
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from transformers import AutoTokenizer, AutoModel # Placeholder for BERT-like model
# import open_clip # Uncomment and install if using OpenCLIP
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Placeholder for an actual LLM client, e.g., from langchain_openai import ChatOpenAI
# For this example, we'll use a dummy LLM to simulate responses
class DummyLLM:
    def invoke(self, prompt_text):
        if "decompose" in prompt_text.lower():
            return "1. Is there evidence of inflammation? 2. What are the key symptoms? 3. Is there any abnormality in the X-ray? 4. How does the patient's history relate to current findings?"
        elif "diagnosis" in prompt_text.lower():
            # Simulate a diagnosis based on the dummy input
            if "infiltrate in lower left lung" in prompt_text.lower() and "cough" in prompt_text.lower():
                return "Based on the patient's persistent cough, mild fever, and the reported infiltrate in the lower left lung from the X-ray, a preliminary diagnosis of pneumonia or a severe respiratory infection is strongly indicated. Further tests such as sputum culture and CBC with differential are recommended to confirm the specific pathogen and assess the severity. Continue monitoring vital signs and respiratory status."
            return "Based on the input, preliminary diagnosis points towards a general respiratory issue. Further tests recommended."
        return "LLM response placeholder."

# Initialize dummy LLM
llm_model = DummyLLM()

# --- 1. Input and Preprocessing Layer ---

# Placeholder for Spacy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spacy model 'en_core_web_sm'. This might take a moment...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def preprocess_text_data(patient_history_path, symptoms_path, lab_results_path):
    """
    Loads and preprocesses patient textual data.
    Args:
        patient_history_path (str): Path to patient history text file.
        symptoms_path (str): Path to symptoms CSV file.
        lab_results_path (str): Path to lab results JSON file.
    Returns:
        dict: Processed text data.
    """
    # Load patient history
    with open(patient_history_path, 'r') as f:
        patient_history = f.read()

    # Process with Spacy for NER (dummy example)
    doc = nlp(patient_history)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    print(f"Spacy NER entities from history: {entities}")

    # Load symptoms (dummy using pandas)
    symptoms_df = pd.read_csv(symptoms_path)

    # Load lab results (dummy using pandas for simplicity, even if JSON suggested)
    lab_results_df = pd.read_json(lab_results_path)

    return {
        "patient_history_raw": patient_history,
        "patient_history_entities": entities,
        "symptoms": symptoms_df.to_dict(orient='records'),
        "lab_results": lab_results_df.to_dict(orient='records')
    }

def preprocess_image_data(image_path):
    """
    Loads and preprocesses a medical image.
    Args:
        image_path (str): Path to the medical image.
    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    # Using Pillow for robust image loading
    image = Image.open(image_path).convert("RGB")

    # Define transforms (example: resize, convert to tensor, normalize)
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # Common input size for many models
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0) # Add batch dimension

# --- 2. Multimodal Feature Extraction Layer ---

class TextEncoder:
    def __init__(self, model_name="bert-base-uncased"): # Placeholder for BioBERT/ClinicalBERT
        # In a real scenario, you'd load BioBERT/ClinicalBERT specific models
        # self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        # self.model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        print(f"Initializing dummy TextEncoder for {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        # Set to eval mode if it's a real model
        self.model.eval()

    def encode(self, text):
        """Generates embeddings for text."""
        # Dummy encoding for demonstration
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Use the [CLS] token embedding as sentence embedding
        return outputs.last_hidden_state[:, 0, :]

class ImageEncoder:
    def __init__(self, model_name="resnet18"): # Placeholder for ResNet/ViT
        # In a real scenario, you'd load a fine-tuned medical image model
        # from torchvision.models import resnet50, ResNet50_Weights
        # self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        print(f"Initializing dummy ImageEncoder for {model_name}")
        self.model = torch.hub.load('pytorch/vision:v0.10.0', model_name, pretrained=True)
        # If using CLIP/OpenCLIP:
        # self.model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
        self.model.eval()
        # Remove the classification head for feature extraction
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))

    def encode(self, image_tensor):
        """Generates embeddings for image."""
        # Dummy encoding for demonstration
        with torch.no_grad():
            features = self.model(image_tensor)
        return features.squeeze() # Remove spatial dimensions if present


# --- 3. Multimodal Fusion and Reasoning Engine ---

class MultimodalDiagnosisAssistant:
    def __init__(self, text_encoder_model="bert-base-uncased", image_encoder_model="resnet18"):
        self.text_encoder = TextEncoder(text_encoder_model)
        self.image_encoder = ImageEncoder(image_encoder_model)
        self.llm = llm_model # Using the global dummy LLM
        self.langchain_orchestrator = self._setup_langchain_orchestrator()

    def _setup_langchain_orchestrator(self):
        """Sets up the LangChain components for structured reasoning."""
        # Problem Decomposition Agent
        decompose_prompt = PromptTemplate.from_template(
            "Given the patient data:\n{patient_data}\nDecompose the diagnostic problem into a series of smaller, interdependent sub-questions. Focus on extracting key medical insights. Provide only the sub-questions." # The dummy LLM will handle the decomposition logic based on this prompt
        )
        decompose_chain = {"patient_data": RunnablePassthrough()} | decompose_prompt | self.llm

        # Structured Reasoning Agent
        reasoning_prompt = PromptTemplate.from_template(
            "Patient Data:\n{patient_data}\n\nMedical Image Context (embedding-derived, sample):\n{image_context}\n\nSub-questions to address:\n{sub_questions}\n\nBased on all available information, provide a comprehensive medical diagnosis and justify your reasoning by answering each sub-question. Also suggest next steps if needed."
        )
        reasoning_chain = reasoning_prompt | self.llm

        return {
            "decompose_chain": decompose_chain,
            "reasoning_chain": reasoning_chain
        }

    def _fuse_multimodal_embeddings(self, text_embedding, image_embedding):
        """
        Fuses text and image embeddings.
        Simple concatenation for demonstration. In a real system, this could be
        a more sophisticated cross-attention mechanism or a multimodal transformer.
        """
        # Ensure embeddings are 1D for concatenation
        text_embedding = text_embedding.flatten()
        image_embedding = image_embedding.flatten()
        fused_embedding = torch.cat((text_embedding, image_embedding), dim=0)
        return fused_embedding.numpy() # Convert to numpy for easier handling if not passed to torch model

    def diagnose(self, patient_text_data, image_tensor):
        """
        Performs multimodal structured reasoning to arrive at a diagnosis.
        Args:
            patient_text_data (dict): Preprocessed textual data.
            image_tensor (torch.Tensor): Preprocessed image tensor.
        Returns:
            dict: Diagnostic results, reasoning steps, and evidence.
        """
        # 1. Feature Extraction
        print("\n--- Feature Extraction ---")
        combined_text = (
            f"Patient History: {patient_text_data['patient_history_raw']}\n"
            f"Symptoms: {patient_text_data['symptoms']}\n"
            f"Lab Results: {patient_text_data['lab_results']}"
        )
        text_embedding = self.text_encoder.encode(combined_text)
        print(f"Text embedding shape: {text_embedding.shape}")

        image_embedding = self.image_encoder.encode(image_tensor)
        print(f"Image embedding shape: {image_embedding.shape}")

        # 2. Multimodal Fusion
        fused_context_embedding = self._fuse_multimodal_embeddings(text_embedding, image_embedding)
        print(f"Fused context embedding shape: {fused_context_embedding.shape}")
        # For passing to LLM, we'll represent a snippet of the fused embedding as a string
        image_context_str = f"Image features (sample): {fused_context_embedding[:5]}..."

        # 3. Problem Decomposition
        print("\n--- Problem Decomposition ---")
        # Prepare data for decomposition LLM. We give it the raw text data.
        llm_input_text_data = (
            f"Patient History: {patient_text_data['patient_history_raw']}\n"
            f"Symptoms: {patient_text_data['symptoms']}\n"
            f"Lab Results: {patient_text_data['lab_results']}"
        )
        sub_questions = self.langchain_orchestrator["decompose_chain"].invoke(llm_input_text_data)
        print(f"Decomposed sub-questions: {sub_questions}")

        # 4. Structured Reasoning
        print("\n--- Structured Reasoning ---")
        reasoning_input = {
            "patient_data": llm_input_text_data,
            "image_context": image_context_str, # Using string representation of image features
            "sub_questions": sub_questions
        }
        final_diagnosis = self.langchain_orchestrator["reasoning_chain"].invoke(reasoning_input)
        print(f"Final Diagnosis and Reasoning: {final_diagnosis}")

        return {
            "sub_questions": sub_questions,
            "final_diagnosis": final_diagnosis,
            "text_embedding_sample": text_embedding.flatten()[:5].tolist(), # Sample for output
            "image_embedding_sample": image_embedding.flatten()[:5].tolist(), # Sample for output
            "fused_embedding_sample": fused_context_embedding[:5].tolist()
        }

# --- Main execution block ---
if __name__ == "__main__":
    print("Setting up dummy data for demonstration...")

    # Create dummy text files
    with open("patient_history.txt", "w") as f:
        f.write("Patient is a 45-year-old male presenting with persistent cough for 2 weeks, mild fever, and fatigue. No known allergies. Smokes occasionally. History of seasonal allergies.")
    pd.DataFrame([{"symptom": "cough", "severity": "mild"}, {"symptom": "fever", "severity": "mild"}, {"symptom": "fatigue", "severity": "moderate"}]).to_csv("symptoms.csv", index=False)
    pd.DataFrame([{"test": "CBC", "result": "Normal"}, {"test": "X-ray", "finding": "Infiltrate in lower left lung"}]).to_json("lab_results.json", orient="records")

    # Create a dummy image (a black square with a white circle to simulate an abnormality)
    dummy_image = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.circle(dummy_image, (150, 150), 50, (255, 255, 255), -1) # White circle in center
    cv2.imwrite("dummy_xray.png", dummy_image)

    # Initialize the assistant
    print("\nInitializing MultimodalDiagnosisAssistant...")
    assistant = MultimodalDiagnosisAssistant()

    # Preprocess inputs
    print("\n--- Preprocessing Inputs ---")
    processed_text = preprocess_text_data("patient_history.txt", "symptoms.csv", "lab_results.json")
    processed_image = preprocess_image_data("dummy_xray.png")

    # Perform diagnosis
    print("\n--- Performing Diagnosis ---")
    diagnosis_results = assistant.diagnose(processed_text, processed_image)

    print("\n--- Diagnosis Complete ---")
    print(f"Sub-questions asked: {diagnosis_results['sub_questions']}")
    print(f"Final Diagnosis: {diagnosis_results['final_diagnosis']}")
    print("\nNote: This is a simulated output using dummy encoders and an LLM placeholder.")
    print("In a real application, actual BioBERT/ClinicalBERT, fine-tuned image models,")
    print("and a powerful LLM (e.g., GPT-4 or a specialized medical LLM) would be used.")