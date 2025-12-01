from pydantic import BaseModel, Field, conlist, confloat
from typing import Literal, Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException
import uvicorn
import base64

class ModelMetadata(BaseModel):
    name: str
    task: Literal["classification", "segmentation", "anomaly_detection"]
    accuracy: confloat(ge=0, le=1) = Field(...)
    f1_score: confloat(ge=0, le=1) = Field(default=0.0)
    gpu_memory_mb: int = Field(..., ge=0)
    inference_time_ms: int = Field(..., ge=0)
    computational_cost_units: int = Field(..., ge=0)
    regulatory_compliance: conlist(str, unique_items=True) = Field(default_factory=list)
    false_positive_rate: confloat(ge=0, le=1) = Field(default=0.0)

class Constraints(BaseModel):
    max_gpu_memory_mb: Optional[int] = Field(None, ge=0)
    max_inference_time_ms: Optional[int] = Field(None, ge=0)
    min_accuracy: Optional[confloat(ge=0, le=1)] = Field(None, ge=0, le=1)
    min_f1_score: Optional[confloat(ge=0, le=1)] = Field(None, ge=0, le=1)
    max_computational_cost_units: Optional[int] = Field(None, ge=0)
    required_regulatory_compliance: Optional[conlist(str, unique_items=True)] = Field(None)
    max_false_positive_rate: Optional[confloat(ge=0, le=1)] = Field(None, ge=0, le=1)

class MedicalImageQuery(BaseModel):
    image_data: str = Field(...)
    functional_requirement: Literal["classification", "segmentation", "anomaly_detection"]
    constraints: Optional[Constraints] = None

class ModelSelectionResult(BaseModel):
    selected_model_name: Optional[str] = None
    reason: str
    model_metadata: Optional[ModelMetadata] = None
    analysis_result: Optional[Dict[str, Any]] = None
    alternative_models: Optional[list[ModelMetadata]] = None

MODEL_LIBRARY: Dict[str, ModelMetadata] = {
    "efficientnet_b0_cls": ModelMetadata(
        name="efficientnet_b0_cls",
        task="classification",
        accuracy=0.77,
        gpu_memory_mb=100,
        inference_time_ms=50,
        computational_cost_units=10,
        regulatory_compliance=["HIPAA"],
    ),
    "resnet50_segmentation": ModelMetadata(
        name="resnet50_segmentation",
        task="segmentation",
        accuracy=0.82,
        f1_score=0.79,
        gpu_memory_mb=500,
        inference_time_ms=200,
        computational_cost_units=50,
        regulatory_compliance=["HIPAA", "GDPR"],
    ),
    "unet_lung_segmentation": ModelMetadata(
        name="unet_lung_segmentation",
        task="segmentation",
        accuracy=0.88,
        f1_score=0.85,
        gpu_memory_mb=800,
        inference_time_ms=300,
        computational_cost_units=70,
        regulatory_compliance=["HIPAA", "FDA_ClassII"],
    ),
    "autoencoder_anomaly_detection_v1": ModelMetadata(
        name="autoencoder_anomaly_detection_v1",
        task="anomaly_detection",
        accuracy=0.91,
        f1_score=0.89,
        gpu_memory_mb=300,
        inference_time_ms=150,
        computational_cost_units=30,
        regulatory_compliance=["GDPR"],
        false_positive_rate=0.05
    ),
    "fast_cls_model": ModelMetadata(
        name="fast_cls_model",
        task="classification",
        accuracy=0.70,
        gpu_memory_mb=50,
        inference_time_ms=20,
        computational_cost_units=5,
        regulatory_compliance=[],
    ),
    "high_acc_cls_model": ModelMetadata(
        name="high_acc_cls_model",
        task="classification",
        accuracy=0.90,
        gpu_memory_mb=1000,
        inference_time_ms=500,
        computational_cost_units=120,
        regulatory_compliance=["HIPAA", "FDA_ClassI"],
    ),
}

def select_optimal_model(
    query: MedicalImageQuery
) -> ModelSelectionResult:
    eligible_models: List[ModelMetadata] = []

    for model_name, model_meta in MODEL_LIBRARY.items():
        if model_meta.task == query.functional_requirement:
            eligible_models.append(model_meta)

    if not eligible_models:
        return ModelSelectionResult(
            reason=f"No models found for task: {query.functional_requirement}"
        )

    if query.constraints:
        constrained_models: List[ModelMetadata] = []
        constraints: Constraints = query.constraints

        for model in eligible_models:
            passes_constraints = True

            if constraints.max_gpu_memory_mb is not None and \
               model.gpu_memory_mb > constraints.max_gpu_memory_mb:
                passes_constraints = False
            if passes_constraints and constraints.max_inference_time_ms is not None and \
               model.inference_time_ms > constraints.max_inference_time_ms:
                passes_constraints = False
            if passes_constraints and constraints.min_accuracy is not None and \
               model.accuracy < constraints.min_accuracy:
                passes_constraints = False
            if passes_constraints and constraints.min_f1_score is not None and \
               model.f1_score < constraints.min_f1_score:
                passes_constraints = False
            if passes_constraints and constraints.max_computational_cost_units is not None and \
               model.computational_cost_units > constraints.max_computational_cost_units:
                passes_constraints = False
            if passes_constraints and constraints.required_regulatory_compliance:
                for req_comp in constraints.required_regulatory_compliance:
                    if req_comp not in model.regulatory_compliance:
                        passes_constraints = False
                        break
            if passes_constraints and constraints.max_false_positive_rate is not None and \
               model.false_positive_rate > constraints.max_false_positive_rate:
                passes_constraints = False

            if passes_constraints:
                constrained_models.append(model)
        
        eligible_models = constrained_models

    if not eligible_models:
        return ModelSelectionResult(
            reason=f"No models satisfy all specified constraints for task: {query.functional_requirement}"
        )

    if query.functional_requirement == "classification":
        eligible_models.sort(key=lambda m: (-m.accuracy, m.computational_cost_units, m.inference_time_ms))
    elif query.functional_requirement == "segmentation":
        eligible_models.sort(key=lambda m: (-m.f1_score, m.computational_cost_units, m.inference_time_ms))
    elif query.functional_requirement == "anomaly_detection":
        eligible_models.sort(key=lambda m: (-m.f1_score, m.false_positive_rate, m.computational_cost_units, m.inference_time_ms))

    selected_model = eligible_models[0]
    alternative_models = eligible_models[1:] if len(eligible_models) > 1 else None

    return ModelSelectionResult(
        selected_model_name=selected_model.name,
        reason="Optimal model selected based on functional requirement and constraints.",
        model_metadata=selected_model,
        alternative_models=alternative_models
    )

def perform_analysis(image_data: str, model_metadata: ModelMetadata) -> Dict[str, Any]:
    print(f"Performing analysis using model: {model_metadata.name} for task: {model_metadata.task}")
    
    if model_metadata.task == "classification":
        return {
            "diagnosis": f"Mock {model_metadata.name} classification: Likely benign lesion",
            "confidence": f"{model_metadata.accuracy * 0.95:.2f}",
            "model_used": model_metadata.name,
            "inference_time_ms": model_metadata.inference_time_ms
        }
    elif model_metadata.task == "segmentation":
        return {
            "segmentation_mask_url": f"mock_segmentation_url/{model_metadata.name}_mask.png",
            "segmented_regions": ["lung_left", "lung_right", "tumor_area"],
            "f1_score_achieved": f"{model_metadata.f1_score * 0.9:.2f}",
            "model_used": model_metadata.name,
            "inference_time_ms": model_metadata.inference_time_ms
        }
    elif model_metadata.task == "anomaly_detection":
        return {
            "anomaly_detected": True if "anomalous_data" in image_data else False,
            "anomaly_score": f"{model_metadata.accuracy * 0.1 + 0.8:.2f}",
            "false_positive_rate_expected": model_metadata.false_positive_rate,
            "model_used": model_metadata.name,
            "inference_time_ms": model_metadata.inference_time_ms
        }
    else:
        return {"status": "Analysis type not recognized", "model_used": model_metadata.name}

app = FastAPI(
    title="Medical Image Analysis Platform with Constraint-Aware API Selection",
    description="API for selecting optimal AI models for medical image analysis based on functional requirements and user-defined constraints."
)

@app.post("/analyze_image", response_model=ModelSelectionResult)
async def analyze_image(query: MedicalImageQuery):
    print(f"Received query for functional requirement: {query.functional_requirement}")
    if query.constraints:
        print(f"Constraints received: {query.constraints.dict()}")

    selection_result = select_optimal_model(query)

    if selection_result.selected_model_name:
        print(f"Selected model: {selection_result.selected_model_name}")
        analysis_output = perform_analysis(query.image_data, selection_result.model_metadata)
        selection_result.analysis_result = analysis_output
    else:
        print(f"Model selection failed: {selection_result.reason}")
        raise HTTPException(status_code=404, detail=selection_result.reason)

    return selection_result

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Medical Image Analysis Platform! Access /docs for API documentation."}

if __name__ == "__main__":
    print("To run the API, use: uvicorn main:app --reload")
    print("Access the API documentation at http://127.0.0.1:8000/docs")

    print("\n--- Example Usage (via FastAPI interactive docs or a client) ---")
    print("\n1. Request for classification with specific constraints:")
    example_query_classification = MedicalImageQuery(
        image_data=base64.b64encode(b"mock_xray_image_data").decode("utf-8"),
        functional_requirement="classification",
        constraints=Constraints(
            max_gpu_memory_mb=200,
            min_accuracy=0.75,
            max_inference_time_ms=100,
            required_regulatory_compliance=["HIPAA"]
        )
    )
    print("\n2. Request for segmentation with less strict constraints:")
    example_query_segmentation = MedicalImageQuery(
        image_data=base64.b64encode(b"mock_mri_image_data").decode("utf-8"),
        functional_requirement="segmentation",
        constraints=Constraints(
            min_f1_score=0.8,
            max_computational_cost_units=60
        )
    )

    print("\n3. Request for anomaly detection with no specific constraints (should pick the best available):")
    example_query_anomaly = MedicalImageQuery(
        image_data=base64.b64encode(b"mock_anomalous_data").decode("utf-8"),
        functional_requirement="anomaly_detection",
        constraints=Constraints(
            max_false_positive_rate=0.06
        )
    )

    print("\n4. Request that will likely fail due to strict constraints:")
    example_query_failure = MedicalImageQuery(
        image_data=base64.b64encode(b"mock_image").decode("utf-8"),
        functional_requirement="classification",
        constraints=Constraints(
            min_accuracy=0.95,
            max_gpu_memory_mb=50
        )
    )
