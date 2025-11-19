
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from transformers import AutoModel, AutoTokenizer, ViTModel, ViTImageProcessor
from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
import io

# Configuration
VIT_MODEL_NAME = "google/vit-base-patch16-224-224"
TEXT_MODEL_NAME = "microsoft/BiomedVLP-CXR-BERT-specialized"
IMAGE_FEATURE_DIM = 768  # ViT base model output dimension
TEXT_FEATURE_DIM = 768   # BERT base model output dimension
FUSION_EMBED_DIM = 512
NUM_CLASSES = 3  # Example: "Pneumonia", "Fracture", "Normal"

# Dummy class labels for demonstration
CLASS_LABELS = ["Pneumonia", "Fracture", "Normal"]

class MultimodalDiagnosticModel(nn.Module):
    def __init__(self, image_feature_dim, text_feature_dim, fusion_embed_dim, num_classes):
        super().__init__()
        self.image_encoder = ViTModel.from_pretrained(VIT_MODEL_NAME)
        self.text_encoder = AutoModel.from_pretrained(TEXT_MODEL_NAME)

        self.image_projection = nn.Linear(image_feature_dim, fusion_embed_dim)
        self.text_projection = nn.Linear(text_feature_dim, fusion_embed_dim)

        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_embed_dim * 2, fusion_embed_dim),
            nn.ReLU(),
            nn.Linear(fusion_embed_dim, fusion_embed_dim // 2),
            nn.ReLU()
        )

        self.classification_head = nn.Linear(fusion_embed_dim // 2, num_classes)

    def forward(self, pixel_values, input_ids, attention_mask):
        # Image features
        image_outputs = self.image_encoder(pixel_values=pixel_values)
        image_features = image_outputs.pooler_output  # [batch_size, image_feature_dim]
        projected_image_features = self.image_projection(image_features)

        # Text features
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.pooler_output  # [batch_size, text_feature_dim]
        projected_text_features = self.text_projection(text_features)

        # Multimodal Fusion
        fused_features = torch.cat((projected_image_features, projected_text_features), dim=1)
        fused_output = self.fusion_mlp(fused_features)

        # Classification
        logits = self.classification_head(fused_output)
        return logits

    def generate_reasoning_path(self, predicted_class_idx: int, patient_text_input: str) -> str:
        diagnosis = CLASS_LABELS[predicted_class_idx]
        
        # This is a simplified reasoning path generation as per the architecture description.
        # In a real-world application, this would involve a more complex LLM interaction
        # or a sophisticated rule-based system trained on medical knowledge graphs.
        
        reasoning = f"Based on the multimodal input, the AI assistant identified patterns consistent with {diagnosis}."
        
        if "shortness of breath" in patient_text_input.lower() or "cough" in patient_text_input.lower():
            if diagnosis == "Pneumonia":
                reasoning += " The presence of respiratory symptoms like cough and shortness of breath, combined with visual findings (e.g., infiltrates on X-ray), supports the diagnosis of Pneumonia."
            else:
                reasoning += " While respiratory symptoms were noted, the primary visual findings did not strongly align with a respiratory condition."
        
        if "pain" in patient_text_input.lower() or "injury" in patient_text_input.lower():
            if diagnosis == "Fracture":
                reasoning += " Patient's reported pain/injury, along with visual evidence of discontinuity in bone structure, indicates a Fracture."
            else:
                reasoning += " Although pain was mentioned, the visual evidence did not suggest a fracture."

        if diagnosis == "Normal":
            reasoning += " No significant abnormalities were detected in the provided medical image or strongly indicated by the patient's textual history, suggesting a Normal finding."

        return reasoning.strip()

# Initialize Models and Processors (Global for efficiency)
image_processor = ViTImageProcessor.from_pretrained(VIT_MODEL_NAME)
text_tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
model = MultimodalDiagnosticModel(
    image_feature_dim=IMAGE_FEATURE_DIM,
    text_feature_dim=TEXT_FEATURE_DIM,
    fusion_embed_dim=FUSION_EMBED_DIM,
    num_classes=NUM_CLASSES
)
model.eval()  # Set model to evaluation mode

# FastAPI App
app = FastAPI()

# Pydantic models for API
class DiagnosisResponse(BaseModel):
    diagnosis: str
    reasoning_path: str

@app.post("/diagnose", response_model=DiagnosisResponse)
async def diagnose(image_file: UploadFile = File(...), patient_history: str = Form(...)):
    # 1. Image Preprocessing
    image_bytes = await image_file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Note: For DICOM, 'pydicom' would be used here. For simplicity, we assume common image formats.
    # Example placeholder for DICOM (not implemented):
    # if image_file.filename.endswith('.dcm'):
    #     import pydicom
    #     ds = pydicom.dcmread(io.BytesIO(image_bytes))
    #     image_array = ds.pixel_array
    #     image = Image.fromarray(image_array)

    pixel_values = image_processor(images=image, return_tensors="pt").pixel_values

    # 2. Text Preprocessing
    text_inputs = text_tokenizer(
        patient_history,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512
    )

    input_ids = text_inputs.input_ids
    attention_mask = text_inputs.attention_mask

    # 3. Multimodal Inference
    with torch.no_grad():
        logits = model(pixel_values, input_ids, attention_mask)
        probabilities = torch.softmax(logits, dim=-1)
        predicted_class_idx = torch.argmax(probabilities, dim=-1).item()

    # 4. Generate Reasoning Path
    reasoning = model.generate_reasoning_path(predicted_class_idx, patient_history)

    return DiagnosisResponse(
        diagnosis=CLASS_LABELS[predicted_class_idx],
        reasoning_path=reasoning
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
