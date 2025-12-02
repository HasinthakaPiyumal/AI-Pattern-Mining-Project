import torch
from PIL import Image
from transformers import pipeline
import cv2 # For video processing placeholder
import os


class ImageAnalysisModule:
    def __init__(self):
        # Placeholder for a medical image analysis model
        # In a real scenario, you'd load a pre-trained CNN model (e.g., ResNet, DenseNet)
        # trained on medical imaging datasets.
        # Example using a dummy model or a simple feature extractor
        self.model = lambda x: {"anomalies_detected": ["lesion_like_area", "inflammation_pattern"], "confidence": 0.85}
        # For a real implementation:
        # self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        # self.model.eval()
        print("Image Analysis Module initialized.")

    def analyze_image(self, image_path: str):
        # Preprocess image (e.g., resize, normalize)
        # For a real model, this would involve torch.transforms
        try:
            image = Image.open(image_path).convert("RGB")
            # Dummy analysis
            result = self.model(image)
            print(f"Image analysis complete for {image_path}.")
            return result
        except FileNotFoundError:
            return {"error": f"Image file not found at {image_path}"}
        except Exception as e:
            return {"error": f"Error analyzing image: {e}"}


class TextAnalysisModule:
    def __init__(self):
        # Placeholder for an NLP model for medical text analysis
        # Using Hugging Face pipeline for demonstration
        try:
            self.summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
            self.qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
            print("Text Analysis Module initialized with Hugging Face pipelines.")
        except Exception as e:
            print(f"Could not load Hugging Face models, using dummy functions: {e}. Please ensure 'transformers' is installed.")
            self.summarizer = lambda text, max_length, min_length, do_sample: [{"summary_text": "Dummy summary of medical text."}]
            self.qa_model = lambda question, context: {"answer": "Dummy answer to medical question."}


    def analyze_text(self, text_data: str, sub_question: str):
        # Process medical text (lab results, symptoms, history)
        if "summarize" in sub_question.lower():
            # Example: Summarize patient history
            summary = self.summarizer(text_data, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
            print("Text summarization complete.")
            return {"text_summary": summary}
        elif "indicate" in sub_question.lower() or "what do" in sub_question.lower():
            # Example: Answer questions about lab results
            answer = self.qa_model(question=sub_question, context=text_data)['answer']
            print("Text QA complete.")
            return {"text_insight": answer}
        else:
            return {"text_insight": "Generic text analysis performed: " + text_data[:100] + "..."}


class VideoAnalysisModule:
    def __init__(self):
        # Placeholder for a video analysis model (e.g., for gait analysis)
        self.model = lambda video_path: {"gait_abnormalities": ["slight_limp_left_leg"], "severity": "moderate"}
        print("Video Analysis Module initialized.")

    def analyze_video(self, video_path: str):
        # Open video file and process frames
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {"error": f"Could not open video file at {video_path}"}

            frames_processed = 0
            # In a real scenario, you'd pass frames to a deep learning model
            # For this example, we'll just simulate processing a few frames
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames_processed += 1
                if frames_processed > 50: # Limit processing for example
                    break

            cap.release()
            result = self.model(video_path)
            result["frames_processed"] = frames_processed
            print(f"Video analysis complete for {video_path}.")
            return result
        except ImportError:
            return {"error": "OpenCV (cv2) not installed. Please install with 'pip install opencv-python'."}
        except Exception as e:
            return {"error": f"Error analyzing video: {e}"}


class MultimodalDiagnosticAssistant:
    def __init__(self):
        self.image_module = ImageAnalysisModule()
        self.text_module = TextAnalysisModule()
        self.video_module = VideoAnalysisModule()
        self.history_of_insights = {} # To store intermediate results

    def decompose_task(self, patient_data: dict):
        """
        Decomposes the complex diagnostic task into distinct sub-questions based on modality.
        patient_data expected format:
        {
            "image_path": "path/to/xray.png",
            "lab_results": "patient_lab_results_text",
            "symptoms_history": "patient_symptoms_and_history_text",
            "video_path": "path/to/gait_video.mp4" (optional)
        }
        """
        sub_questions = []
        if patient_data.get("image_path") and os.path.exists(patient_data["image_path"]):
            sub_questions.append({"modality": "image", "question": "What are the key visual anomalies in the provided medical image?", "data": patient_data["image_path"]})
        if patient_data.get("lab_results"):
            sub_questions.append({"modality": "text", "question": "What do the lab results indicate about inflammation markers and other key parameters?", "data": patient_data["lab_results"]})
        if patient_data.get("symptoms_history"):
            sub_questions.append({"modality": "text", "question": "Summarize the patient's reported symptoms and medical history for relevant conditions.", "data": patient_data["symptoms_history"]})
        if patient_data.get("video_path") and os.path.exists(patient_data["video_path"]):
            sub_questions.append({"modality": "video", "question": "Are there any video cues (e.g., gait analysis) that suggest neurological issues or movement disorders?", "data": patient_data["video_path"]})

        print(f"Decomposed task into {len(sub_questions)} sub-questions.")
        return sub_questions

    def process_sub_question(self, sub_question: dict):
        """
        Routes a sub-question to the appropriate specialized AI module and processes it.
        """
        modality = sub_question["modality"]
        data = sub_question["data"]
        question_text = sub_question["question"]
        insight = None

        if modality == "image":
            print(f"Processing image sub-question: '{question_text}' for {data}")
            insight = self.image_module.analyze_image(data)
        elif modality == "text":
            print(f"Processing text sub-question: '{question_text}' for {'lab results' if 'lab results' in question_text.lower() else 'symptoms/history'}")
            insight = self.text_module.analyze_text(data, question_text)
        elif modality == "video":
            print(f"Processing video sub-question: '{question_text}' for {data}")
            insight = self.video_module.analyze_video(data)
        else:
            insight = {"error": f"Unknown modality: {modality}"}

        self.history_of_insights[modality] = self.history_of_insights.get(modality, []) + [insight]
        return insight

    def synthesize_insights(self):
        """
        Aggregates and synthesizes insights from all modules to provide a comprehensive diagnosis.
        This part would typically involve a more sophisticated reasoning engine or another LLM.
        """
        print("\nSynthesizing all collected insights...")
        final_diagnosis_summary = "Based on the multimodal analysis:\n"
        potential_diagnoses = []
        recommended_next_steps = []

        # Example of how to combine insights (simplified)
        for modality, insights_list in self.history_of_insights.items():
            final_diagnosis_summary += f"\n--- {modality.upper()} Insights ---\n"
            for insight in insights_list:
                if "error" in insight:
                    final_diagnosis_summary += f"  Error in {modality} analysis: {insight['error']}\n"
                elif modality == "image":
                    final_diagnosis_summary += f"  Image analysis detected anomalies: {insight.get('anomalies_detected', 'None')}\n"
                    if "lesion_like_area" in insight.get('anomalies_detected', []):
                        potential_diagnoses.append("Consider further investigation for possible tumor/growth.")
                        recommended_next_steps.append("Order follow-up high-resolution imaging (e.g., MRI with contrast).")
                elif modality == "text":
                    if "text_summary" in insight:
                        final_diagnosis_summary += f"  Patient history summary: {insight['text_summary']}\n"
                        if "fever" in insight['text_summary'].lower() and "cough" in insight['text_summary'].lower():
                            potential_diagnoses.append("Possible respiratory infection.")
                    if "text_insight" in insight:
                        final_diagnosis_summary += f"  Specific text insight: {insight['text_insight']}\n"
                        if "high inflammation markers" in insight['text_insight'].lower():
                            potential_diagnoses.append("Systemic inflammation present.")
                            recommended_next_steps.append("Investigate underlying inflammatory causes.")
                elif modality == "video":
                    final_diagnosis_summary += f"  Video analysis detected gait abnormalities: {insight.get('gait_abnormalities', 'None')}\n"
                    if "limp" in str(insight.get('gait_abnormalities', [])): # Convert to string for broader search
                        potential_diagnoses.append("Possible neurological or musculoskeletal issue affecting gait.")
                        recommended_next_steps.append("Referral to neurologist or orthopedist.")

        # Consolidate and refine
        final_diagnosis_summary += "\n--- Consolidated Assessment ---\n"
        final_diagnosis_summary += "Potential Differential Diagnoses: " + ", ".join(list(set(potential_diagnoses))) + "\n"
        final_diagnosis_summary += "Recommended Next Steps: " + ", ".join(list(set(recommended_next_steps))) + "\n"

        # A more advanced synthesis might involve an LLM to generate narrative diagnosis
        # For example:
        # llm_response = self.llm_for_synthesis(f"Synthesize the following medical insights: {self.history_of_insights}")
        # final_diagnosis_summary += llm_response

        print("Synthesis complete.")
        return {
            "comprehensive_assessment": final_diagnosis_summary,
            "potential_diagnoses": list(set(potential_diagnoses)),
            "recommended_next_steps": list(set(recommended_next_steps))
        }

    def diagnose(self, patient_data: dict):
        """
        Orchestrates the entire Duty Distinct Chain of Thought process.
        """
        print("Starting multimodal diagnostic process...")
        self.history_of_insights = {} # Reset for new diagnosis

        # 1. Decompose the task
        sub_questions = self.decompose_task(patient_data)

        # 2. Process each sub-question sequentially
        for sq in sub_questions:
            _ = self.process_sub_question(sq) # We store results in history_of_insights

        # 3. Synthesize all insights
        final_output = self.synthesize_insights()

        print("Multimodal diagnostic process finished.")
        return final_output

# Helper functions for simulating data loading (for testing purposes)
def create_dummy_files():
    # Create a dummy image file
    try:
        Image.new('RGB', (60, 30), color = 'red').save('dummy_xray.png')
        print("Created dummy_xray.png")
    except ImportError:
        print("Pillow not installed, cannot create dummy image. Please install with 'pip install Pillow'")
    except Exception as e:
        print(f"Error creating dummy image: {e}")

    # Create dummy text files
    with open('dummy_lab_results.txt', 'w') as f:
        f.write("Patient Lab Results:\nWhite Blood Cell Count: 12.5 (High)\nC-Reactive Protein: 15 mg/L (High)\nGlucose: 90 mg/dL (Normal)")
    print("Created dummy_lab_results.txt")

    with open('dummy_symptoms_history.txt', 'w') as f:
        f.write("Patient reports fever for 3 days, persistent cough, and general fatigue. No significant medical history apart from childhood asthma.")
    print("Created dummy_symptoms_history.txt")

    # Create a dummy video file (using OpenCV for a minimal one)
    try:
        fourcc = cv2.VideoWriter_fourcc(*'MP4V') # Or XVID
        out = cv2.VideoWriter('dummy_gait.mp4', fourcc, 20.0, (640, 480))
        for i in range(30): # 30 frames for a 1.5 second video at 20fps
            # Create a simple colored frame (e.g., gradually changing color)
            b = int(255 * (i / 29.0))
            g = 0
            r = int(255 - b)
            frame_color = (b, g, r) # OpenCV uses BGR
            frame = (255 * (i / 29.0) * (255, 0, 0)).astype("uint8") # Simple color change
            frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA) # Ensure correct size
            frame[:] = frame_color # Set entire frame to color
            out.write(frame)
        out.release()
        print("Created dummy_gait.mp4")
    except Exception as e:
        print(f"Error creating dummy video (requires OpenCV and numpy): {e}. Please install with 'pip install opencv-python numpy'")


# Example Usage (demonstrates the flow)
if __name__ == "__main__":
    print("\n--- Running Example Usage ---")
    create_dummy_files()

    assistant = MultimodalDiagnosticAssistant()

    patient_data = {
        "image_path": "dummy_xray.png",
        "lab_results": open("dummy_lab_results.txt").read(),
        "symptoms_history": open("dummy_symptoms_history.txt").read(),
        "video_path": "dummy_gait.mp4"
    }

    # Remove data paths if dummy files were not created successfully
    if not os.path.exists("dummy_gait.mp4"):
        print("Warning: dummy_gait.mp4 not found, proceeding without video analysis.")
        patient_data.pop("video_path", None)

    if not os.path.exists("dummy_xray.png"):
        print("Warning: dummy_xray.png not found, proceeding without image analysis.")
        patient_data.pop("image_path", None)

    if not os.path.exists("dummy_lab_results.txt"):
        print("Warning: dummy_lab_results.txt not found.")
        patient_data.pop("lab_results", None)

    if not os.path.exists("dummy_symptoms_history.txt"):
        print("Warning: dummy_symptoms_history.txt not found.")
        patient_data.pop("symptoms_history", None)

    diagnosis_result = assistant.diagnose(patient_data)
    print("\n--- FINAL DIAGNOSIS RESULT ---")
    print(diagnosis_result["comprehensive_assessment"])
    print(f"\nPotential Diagnoses: {diagnosis_result['potential_diagnoses']}")
    print(f"Recommended Next Steps: {diagnosis_result['recommended_next_steps']}")

    # Clean up dummy files
    print("\n--- Cleaning up dummy files ---")
    for f in ["dummy_xray.png", "dummy_lab_results.txt", "dummy_symptoms_history.txt", "dummy_gait.mp4"]:
        if os.path.exists(f):
            os.remove(f)
            print(f"Cleaned up {f}")