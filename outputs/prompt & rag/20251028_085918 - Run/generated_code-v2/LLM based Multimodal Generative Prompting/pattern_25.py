import cv2
import numpy as np
import pandas as pd
import spacy
import librosa
from transformers import pipeline, AutoProcessor, Wav2Vec2ForCTC
import gradio as gr

# Placeholder for a medical knowledge base
class MedicalKnowledgeBase:
    def __init__(self):
        self.disease_symptoms = {
            "pneumonia": ["cough", "fever", "shortness of breath", "chest pain"],
            "bronchitis": ["cough", "mucus", "fatigue"],
            "fracture": ["pain", "swelling", "bruising"]
        }
        self.lab_result_indicators = {
            "inflammation": ["CRP high", "WBC high"],
            "infection": ["WBC high", "ESR high"]
        }

    def query_symptoms(self, symptoms):
        matching_diseases = []
        for disease, known_symptoms in self.disease_symptoms.items():
            if any(s in known_symptoms for s in symptoms):
                matching_diseases.append(disease)
        return matching_diseases
    
    def query_lab_results(self, lab_results_text):
        indicators_found = []
        for indicator, patterns in self.lab_result_indicators.items():
            if any(p in lab_results_text for p in patterns):
                indicators_found.append(indicator)
        return indicators_found


class ImagePreprocessor:
    def preprocess_image(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        image = cv2.resize(image, (224, 224))
        image = image / 255.0  # Normalize
        return image

class EHRPreprocessor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def preprocess_text(self, text):
        doc = self.nlp(text)
        tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
        cleaned_text = " ".join(tokens)
        return cleaned_text

class AudioPreprocessor:
    def __init__(self):
        # Initialize a pre-trained Speech-to-Text model
        self.stt_processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
        self.stt_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

    def preprocess_audio(self, audio_path):
        y, sr = librosa.load(audio_path, sr=16000) # Load audio at 16kHz
        input_values = self.stt_processor(y, return_tensors="pt", sampling_rate=sr).input_values
        logits = self.stt_model(input_values).logits
        predicted_ids = np.argmax(logits.detach().numpy(), axis=-1)
        transcription = self.stt_processor.batch_decode(predicted_ids)[0]
        return transcription

class ImageAnalysisSolver:
    def __init__(self):
        # Placeholder for a medical image analysis model (e.g., a pre-trained CNN)
        # In a real scenario, this would load a PyTorch or TensorFlow model
        pass

    def analyze_image(self, preprocessed_image, sub_question):
        # Simulate image analysis based on sub_question
        if "pneumonia" in sub_question.lower() and np.mean(preprocessed_image) < 0.5: # Simple heuristic
            return "Possible signs of pneumonia detected in chest X-ray."
        elif "fracture" in sub_question.lower() and np.max(preprocessed_image) > 0.8: # Simple heuristic
            return "Potential fracture indicated in the image."
        return "No specific abnormalities detected in image for: " + sub_question

class NLPSolver:
    def __init__(self):
        # Placeholder for an NLP model, e.g., for entity recognition or classification
        self.nlp_pipeline = pipeline("ner", model="dslim/bert-base-NER") # Example NER model

    def process_text(self, preprocessed_text, sub_question):
        # Simulate NLP processing based on sub_question
        if "symptoms" in sub_question.lower():
            entities = self.nlp_pipeline(preprocessed_text)
            symptoms = [entity['word'] for entity in entities if entity['entity'].startswith('B-MISC') or entity['entity'].startswith('I-MISC')] # Simplified
            return f"Identified symptoms: {', '.join(symptoms) if symptoms else 'None'}."
        elif "lab results" in sub_question.lower():
            return f"Analyzed lab results for indicators: {preprocessed_text}."
        return "NLP analysis for " + sub_question + ": " + preprocessed_text

class SpeechToTextNLPSolver(NLPSolver):
    def __init__(self):
        super().__init__()

    def process_transcription(self, transcription, sub_question):
        return self.process_text(transcription, sub_question)

class DDCoTController:
    def __init__(self):
        self.image_preprocessor = ImagePreprocessor()
        self.ehr_preprocessor = EHRPreprocessor()
        self.audio_preprocessor = AudioPreprocessor()
        self.image_solver = ImageAnalysisSolver()
        self.nlp_solver = NLPSolver()
        self.stt_nlp_solver = SpeechToTextNLPSolver()
        self.knowledge_base = MedicalKnowledgeBase()

    def decompose_problem(self, patient_data):
        sub_questions = []
        if patient_data.get("image_path"):
            sub_questions.append("Analyze chest X-ray for pneumonia signs.")
            sub_questions.append("Check image for any fractures.")
        if patient_data.get("ehr_text"):
            sub_questions.append("Identify key symptoms from EHR text.")
            sub_questions.append("Evaluate lab results for inflammation markers from EHR.")
        if patient_data.get("audio_path"):
            sub_questions.append("Transcribe and analyze doctor's audio description for relevant medical terms.")
        return sub_questions

    def route_and_solve_subquestion(self, sub_question, patient_data):
        answer = ""
        if "image" in sub_question.lower() and patient_data.get("image_path"):
            preprocessed_image = self.image_preprocessor.preprocess_image(patient_data["image_path"])
            answer = self.image_solver.analyze_image(preprocessed_image, sub_question)
        elif ("ehr" in sub_question.lower() or "symptoms" in sub_question.lower() or "lab results" in sub_question.lower()) and patient_data.get("ehr_text"):
            preprocessed_text = self.ehr_preprocessor.preprocess_text(patient_data["ehr_text"])
            answer = self.nlp_solver.process_text(preprocessed_text, sub_question)
        elif ("audio" in sub_question.lower() or "transcribe" in sub_question.lower()) and patient_data.get("audio_path"):
            transcription = self.audio_preprocessor.preprocess_audio(patient_data["audio_path"])
            answer = self.stt_nlp_solver.process_transcription(transcription, sub_question)
        else:
            answer = f"Could not process sub-question '{sub_question}' due to missing data or unrecognized type."
        return answer

    def synthesize_answers(self, sub_question_answers):
        final_assessment = "Comprehensive Diagnostic Assessment:\n"
        potential_diagnoses = set()
        identified_symptoms = []
        identified_lab_indicators = []

        for sq, ans in sub_question_answers.items():
            final_assessment += f"- {sq}: {ans}\n"
            if "pneumonia" in ans.lower():
                potential_diagnoses.add("Pneumonia")
            if "fracture" in ans.lower():
                potential_diagnoses.add("Fracture")
            if "symptoms:" in ans.lower():
                symptoms_str = ans.split("symptoms:")[1].strip().replace('.', '')
                identified_symptoms.extend([s.strip() for s in symptoms_str.split(',') if s.strip()])
            if "lab results for indicators:" in ans.lower():
                lab_str = ans.split("lab results for indicators:")[1].strip().replace('.', '')
                identified_lab_indicators.extend(self.knowledge_base.query_lab_results(lab_str))

        # Use knowledge base to refine diagnoses
        kb_diagnoses_from_symptoms = self.knowledge_base.query_symptoms(identified_symptoms)
        for diag in kb_diagnoses_from_symptoms:
            potential_diagnoses.add(diag)
        
        if potential_diagnoses:
            final_assessment += "\nPotential Diagnoses (based on integrated evidence): " + ", ".join(list(potential_diagnoses)) + "\n"
        else:
            final_assessment += "\nNo strong potential diagnoses identified from current evidence.\n"

        if identified_symptoms:
            final_assessment += f"Identified Symptoms: {', '.join(set(identified_symptoms))}.\n"
        if identified_lab_indicators:
            final_assessment += f"Identified Lab Indicators: {', '.join(set(identified_lab_indicators))}.\n"

        final_assessment += "\nSuggested Next Steps: Consult a specialist for definitive diagnosis, consider further specific imaging or lab tests based on preliminary findings."
        return final_assessment

    def diagnose_patient(self, image_path=None, ehr_text=None, audio_path=None):
        patient_data = {
            "image_path": image_path,
            "ehr_text": ehr_text,
            "audio_path": audio_path
        }

        sub_questions = self.decompose_problem(patient_data)
        sub_question_answers = {}

        for sq in sub_questions:
            answer = self.route_and_solve_subquestion(sq, patient_data)
            sub_question_answers[sq] = answer
        
        final_assessment = self.synthesize_answers(sub_question_answers)
        return final_assessment, sub_question_answers


def run_diagnosis(image_file, ehr_text, audio_file):
    controller = DDCoTController()
    image_path = image_file.name if image_file else None
    audio_path = audio_file.name if audio_file else None
    
    final_assessment, sub_question_answers = controller.diagnose_patient(
        image_path=image_path,
        ehr_text=ehr_text,
        audio_path=audio_path
    )

    sub_questions_output = "\nSub-Question Breakdown:\n"
    for sq, ans in sub_question_answers.items():
        sub_questions_output += f"- {sq}: {ans}\n"
    
    return final_assessment, sub_questions_output

# Gradio Interface
if __name__ == "__main__":
    inputs = [
        gr.File(label="Upload Medical Image (X-ray, MRI, CT)", type="filepath"),
        gr.Textbox(label="Enter EHR Text (Symptoms, History, Lab Results)", lines=10),
        gr.File(label="Upload Doctor's Audio Description", type="filepath")
    ]
    outputs = [
        gr.Textbox(label="Final Diagnostic Assessment"),
        gr.Textbox(label="Detailed Sub-Question Analysis")
    ]

    gr.Interface(
        fn=run_diagnosis,
        inputs=inputs,
        outputs=outputs,
        title="Multimodal Medical Diagnostic Assistant (DDCoT)",
        description="Upload medical images, EHR text, and/or audio descriptions to get a decomposed diagnostic assessment."
    ).launch()
