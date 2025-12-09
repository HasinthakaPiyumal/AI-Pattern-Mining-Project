import logging
from PIL import Image
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SubQuestion(BaseModel):
    id: str
    type: str
    query: str
    modality: str
    status: str = "pending"

class InputData(BaseModel):
    image_path: Optional[str] = None
    ehr_text: Optional[str] = None
    lab_results: Optional[str] = None

class InputPreprocessingModule:
    def load_image(self, image_path: str) -> Optional[Image.Image]:
        try:
            image = Image.open(image_path).convert("RGB")
            logger.info(f"Successfully loaded image from {image_path}")
            return image
        except FileNotFoundError:
            logger.error(f"Image file not found: {image_path}")
            return None
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None

    def preprocess_text(self, text: Optional[str]) -> Optional[str]:
        if text is None:
            return None
        processed_text = text.strip()
        logger.info("Text data preprocessed.")
        return processed_text

class ProblemDecompositionModule:
    def decompose_query(self, main_query: str, input_data: InputData) -> List[SubQuestion]:
        sub_questions = []
        logger.info(f"Decomposing main query: '{main_query}'")

        if ("visual" in main_query.lower() or "x-ray" in main_query.lower() or "mri" in main_query.lower()) and input_data.image_path:
            sub_questions.append(SubQuestion(
                id="sq_img_001",
                type="image_analysis",
                query="Analyze the provided medical image for abnormalities and key visual findings.",
                modality="image"
            ))
        if ("patient history" in main_query.lower() or "symptoms" in main_query.lower() or "ehr" in main_query.lower()) and input_data.ehr_text:
            sub_questions.append(SubQuestion(
                id="sq_txt_001",
                type="text_analysis",
                query="Extract key symptoms, medical history, and relevant clinical notes from the EHR.",
                modality="text"
            ))
        if ("lab results" in main_query.lower() or "bloodwork" in main_query.lower()) and input_data.lab_results:
            sub_questions.append(SubQuestion(
                id="sq_txt_002",
                type="text_analysis",
                query="Interpret the provided lab results for significant markers and abnormal values.",
                modality="text"
            ))

        if not sub_questions:
            logger.warning("No specific sub-questions generated based on query and input data. Generating a general reasoning sub-question.")
            sub_questions.append(SubQuestion(
                id="sq_gen_001",
                type="general_reasoning",
                query=f"Synthesize all available information to provide a comprehensive response to: {main_query}",
                modality="multimodal"
            ))

        return sub_questions

class MockImageAnalysisModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        logger.info("MockImageAnalysisModel initialized.")

    def forward(self, x: torch.Tensor):
        x = self.pool(self.relu(self.conv1(x)))
        return x

    def analyze(self, image: Image.Image) -> str:
        if image.width > 500 and image.height > 300:
            mock_finding = "Radiological review indicates a consolidation in the lower left lobe, suggestive of inflammation."
        else:
            mock_finding = "Image analysis suggests general chest structures are visible, but resolution limits detailed assessment."
        logger.info("Image analysis completed (mock).")
        return f"Image Analysis Result: {mock_finding}"

class MockTextAnalysisModel:
    def __init__(self):
        logger.info("MockTextAnalysisModel initialized.")

    def analyze(self, text_data: Optional[str]) -> str:
        if not text_data:
            return "No text data provided for analysis."

        mock_findings = []
        if "cough" in text_data.lower() and "fever" in text_data.lower():
            mock_findings.append("EHR reports acute respiratory symptoms including cough and fever.")
        if "fatigue" in text_data.lower():
            mock_findings.append("Patient describes general fatigue.")
        if "wbc: 12.5" in text_data.lower() or "wbc high" in text_data.lower():
            mock_findings.append("Lab results show elevated white blood cell count (WBC).")
        if "crp: 45" in text_data.lower() or "crp high" in text_data.lower():
            mock_findings.append("Lab results show elevated C-reactive protein (CRP).")
        if "d-dimer: 850" in text_data.lower() or "d-dimer elevated" in text_data.lower():
            mock_findings.append("Lab results indicate elevated D-dimer levels, suggesting thrombotic activity.")

        if not mock_findings:
            return "Text analysis found no specific medical findings."
        return f"Text Analysis Result: {' '.join(mock_findings)}"

class IntegrationSynthesisModule:
    def synthesize_findings(self, sub_question_answers: Dict[str, str], main_query: str) -> Dict[str, Any]:
        logger.info("Synthesizing findings from sub-question answers.")
        overall_findings = []
        for q_id, answer in sub_question_answers.items():
            overall_findings.append(f"[{q_id}] {answer}")

        combined_findings_str = "\n".join(overall_findings)

        diagnostic_hypothesis = "Initial assessment suggests a complex presentation requiring further clinical correlation."
        confidence = "Low"

        if "consolidation in the lower left lobe" in combined_findings_str and ("cough" in combined_findings_str or "fever" in combined_findings_str) and ("elevated white blood cell count" in combined_findings_str or "elevated C-reactive protein" in combined_findings_str):
            diagnostic_hypothesis = "High suspicion for pneumonia, supported by radiological findings, respiratory symptoms, and inflammatory markers."
            confidence = "High"
        elif "elevated D-dimer" in combined_findings_str:
            diagnostic_hypothesis = "Elevated D-dimer suggests a need to investigate for thrombotic events such as pulmonary embolism, especially if other symptoms are present."
            confidence = "Medium"
        elif "EHR reports acute respiratory symptoms" in combined_findings_str:
            diagnostic_hypothesis = "Patient presents with acute respiratory symptoms; consider viral infection or bronchitis."
            confidence = "Medium"

        explanation = f"Synthesis of findings:\n{combined_findings_str}\n\nOverall diagnostic reasoning based on the query '{main_query}': {diagnostic_hypothesis}"

        return {
            "diagnostic_hypothesis": diagnostic_hypothesis,
            "confidence": confidence,
            "explanation": explanation
        }

class MultimodalDiagnosticAssistant:
    def __init__(self):
        self.preprocessing_module = InputPreprocessingModule()
        self.decomposition_module = ProblemDecompositionModule()
        self.image_solver = MockImageAnalysisModel()
        self.text_solver = MockTextAnalysisModel()
        self.synthesis_module = IntegrationSynthesisModule()
        logger.info("MultimodalDiagnosticAssistant initialized.")

    def diagnose(self, main_query: str, input_data: InputData) -> Dict[str, Any]:
        logger.info(f"Starting diagnosis for main query: '{main_query}'")

        processed_image = None
        if input_data.image_path:
            processed_image = self.preprocessing_module.load_image(input_data.image_path)
            if processed_image is None:
                logger.warning(f"Skipping image analysis due to loading failure for {input_data.image_path}")

        processed_ehr_text = self.preprocessing_module.preprocess_text(input_data.ehr_text)
        processed_lab_results = self.preprocessing_module.preprocess_text(input_data.lab_results)

        sub_questions = self.decomposition_module.decompose_query(main_query, input_data)
        sub_question_answers = {}

        for sq in sub_questions:
            logger.info(f"Solving sub-question [{sq.id}] ({sq.modality}): {sq.query}")
            if sq.modality == "image" and processed_image:
                dummy_input = torch.randn(1, 3, 224, 224)
                _ = self.image_solver(dummy_input)
                answer = self.image_solver.analyze(processed_image)
                sub_question_answers[sq.id] = answer
                sq.status = "solved"
            elif sq.modality == "text":
                text_to_analyze = ""
                if "ehr" in sq.query.lower() and processed_ehr_text:
                    text_to_analyze = processed_ehr_text
                elif "lab results" in sq.query.lower() and processed_lab_results:
                    text_to_analyze = processed_lab_results
                elif "general" in sq.type.lower() and (processed_ehr_text or processed_lab_results):
                    text_to_analyze = f"{processed_ehr_text or ''} {processed_lab_results or ''}".strip()
                
                if text_to_analyze:
                    answer = self.text_solver.analyze(text_to_analyze)
                    sub_question_answers[sq.id] = answer
                    sq.status = "solved"
                else:
                    sub_question_answers[sq.id] = "No relevant text data found for this sub-question."
                    sq.status = "skipped"
            else:
                sub_question_answers[sq.id] = f"Cannot solve sub-question of modality '{sq.modality}' or missing data."
                sq.status = "unsolvable"

        final_diagnosis = self.synthesis_module.synthesize_findings(sub_question_answers, main_query)
        logger.info("Diagnosis process completed.")
        return final_diagnosis

if __name__ == "__main__":
    try:
        dummy_image = Image.new('RGB', (800, 600), color = 'red')
        dummy_image.save("dummy_xray.png")
        logger.info("Created dummy_xray.png for demonstration.")
    except Exception as e:
        logger.error(f"Could not create dummy image: {e}. Please ensure Pillow is installed.")

    assistant = MultimodalDiagnosticAssistant()

    patient_data_1 = InputData(
        image_path="dummy_xray.png",
        ehr_text="Patient presents with persistent cough, high fever for 3 days, and general fatigue. No known allergies. History of asthma.",
        lab_results="WBC: 12.5 x10^9/L (High), CRP: 45 mg/L (High), Procalcitonin: 0.8 ng/mL (Elevated)."
    )
    query_1 = "Evaluate the patient's respiratory condition based on visual (X-ray) and textual data (EHR, lab results) to suggest a primary diagnosis."
    result_1 = assistant.diagnose(query_1, patient_data_1)
    print("\n--- Diagnosis Result 1 ---")
    print(f"Main Query: {query_1}")
    print(f"Diagnostic Hypothesis: {result_1['diagnostic_hypothesis']}")
    print(f"Confidence: {result_1['confidence']}")
    print(f"Explanation:\n{result_1['explanation']}")

    print("\n" + "="*50 + "\n")

    patient_data_2 = InputData(
        lab_results="D-dimer: 850 ng/mL (Elevated), Troponin: Normal, BNP: Normal, Glucose: 105 mg/dL."
    )
    query_2 = "Analyze recent lab results to identify any significant health concerns, particularly cardiovascular or thrombotic risks."
    result_2 = assistant.diagnose(query_2, patient_data_2)
    print("\n--- Diagnosis Result 2 ---")
    print(f"Main Query: {query_2}")
    print(f"Diagnostic Hypothesis: {result_2['diagnostic_hypothesis']}")
    print(f"Confidence: {result_2['confidence']}")
    print(f"Explanation:\n{result_2['explanation']}")

    print("\n" + "="*50 + "\n")

    patient_data_3 = InputData(
        ehr_text="Patient reports severe headache, sensitivity to light, and nausea for 24 hours. No fever. History of migraines."
    )
    query_3 = "Assess the patient's neurological symptoms and history to suggest a potential cause."
    result_3 = assistant.diagnose(query_3, patient_data_3)
    print("\n--- Diagnosis Result 3 ---")
    print(f"Main Query: {query_3}")
    print(f"Diagnostic Hypothesis: {result_3['diagnostic_hypothesis']}")
    print(f"Confidence: {result_3['confidence']}")
    print(f"Explanation:\n{result_3['explanation']}")

    if os.path.exists("dummy_xray.png"):
        os.remove("dummy_xray.png")
        logger.info("Removed dummy_xray.png.")
