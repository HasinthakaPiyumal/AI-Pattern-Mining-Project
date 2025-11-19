import gradio as gr
import time

class MockLLM:
    """A mock LLM to simulate responses without actual model inference."""
    def __init__(self):
        self.knowledge_base = {
            "fever and cough": "Potential diagnoses: Common Cold, Flu, Bronchitis. Consider a chest X-ray.",
            "severe headache and stiff neck": "Potential diagnoses: Meningitis, Subarachnoid Hemorrhage. Recommend immediate lumbar puncture.",
            "chest pain and shortness of breath": "Potential diagnoses: Myocardial Infarction, Pulmonary Embolism, Anxiety. Recommend ECG and D-dimer test.",
            "fatigue and weight loss": "Potential diagnoses: Hypothyroidism, Diabetes, Cancer. Further tests: Blood panel, thyroid function tests.",
            "joint pain and swelling": "Potential diagnoses: Arthritis (Rheumatoid, Osteoarthritis), Gout. Further tests: ESR, CRP, Uric Acid."
        }
        self.diagnosis_confidence = 0.7 # Initial confidence

    def generate_response(self, prompt, current_diagnosis=None, feedback=None):
        """Generates a simulated response based on the prompt and feedback."""
        print(f"LLM received prompt: {prompt}")
        print(f"LLM received current_diagnosis: {current_diagnosis}")
        print(f"LLM received feedback: {feedback}")

        response = "I need more information to provide a diagnosis."
        suggested_action = "Suggest asking clarifying questions to the user."

        # Incorporate feedback into confidence
        if feedback == "correct":
            self.diagnosis_confidence = min(1.0, self.diagnosis_confidence + 0.1)
            print(f"Confidence increased to {self.diagnosis_confidence:.2f}")
        elif feedback == "incorrect":
            self.diagnosis_confidence = max(0.1, self.diagnosis_confidence - 0.2)
            print(f"Confidence decreased to {self.diagnosis_confidence:.2f}")
            # If incorrect, try to re-evaluate
            if current_diagnosis and "Potential diagnoses" in current_diagnosis:
                current_diagnosis = "Let me re-evaluate based on the new information. " + current_diagnosis

        # Simple keyword-based matching for initial diagnosis
        if "patient data:" in prompt.lower():
            patient_data = prompt.lower().split("patient data:")[1].strip()
            for key, value in self.knowledge_base.items():
                if all(s in patient_data for s in key.split(" and ")):
                    response = value
                    suggested_action = f"Suggest confirming diagnosis or ordering tests based on: {value}"
                    break

        # Refine response based on existing diagnosis and feedback
        if current_diagnosis and response == "I need more information to provide a diagnosis.":
            if feedback == "needs more info":
                response = f"The previous diagnosis was: {current_diagnosis}. What specific information do you need?"
                suggested_action = "Suggest asking the user for specific missing details."
            elif feedback == "suggest a test":
                response = f"Considering the previous diagnosis ({current_diagnosis}), what test are you suggesting?"
                suggested_action = "Suggest asking the user for the test name."

        # Self-reflection based on confidence
        if self.diagnosis_confidence < 0.5 and "Potential diagnoses" in response:
            response += "\n\n*Self-reflection: My confidence in this diagnosis is low. Further investigation or specialist consultation is highly recommended.*"
            suggested_action = "Suggest consulting a specialist due to low confidence."

        return response, suggested_action, self.diagnosis_confidence

class MedicalDatabaseSearchTool:
    """Mocks a tool for searching medical databases."""
    def search(self, query):
        print(f"Searching medical database for: {query}")
        time.sleep(1) # Simulate network delay
        if "meningitis" in query.lower():
            return "Medical literature on Meningitis: Symptoms often include severe headache, stiff neck, fever, photophobia. Diagnosis typically involves lumbar puncture. Treatment is antibiotics or antivirals."
        elif "myocardial infarction" in query.lower() or "heart attack" in query.lower():
            return "Medical literature on Myocardial Infarction: Symptoms include chest pain radiating to arm/jaw, shortness of breath, sweating. Diagnosis: ECG, troponin levels. Treatment: Angioplasty, medication."
        elif "diabetes symptoms" in query.lower():
            return "Medical literature on Diabetes: Frequent urination, increased thirst, unexplained weight loss, fatigue, blurred vision."
        return f"No specific medical literature found for '{query}'."

class ImageAnalysisAITool:
    """Mocks an external AI for image analysis."""
    def analyze_image(self, image_type, description=""): # image_type could be 'chest_xray', 'MRI', etc.
        print(f"Analyzing {image_type} image with description: {description}")
        time.sleep(2) # Simulate AI processing time
        if "chest_xray" in image_type.lower() and "pneumonia" in description.lower():
            return "Image Analysis Report (Chest X-ray): Findings consistent with bacterial pneumonia in the lower right lobe."
        elif "chest_xray" in image_type.lower():
            return "Image Analysis Report (Chest X-ray): No significant acute findings detected."
        elif "brain_mri" in image_type.lower() and "tumor" in description.lower():
            return "Image Analysis Report (Brain MRI): Suspected mass lesion in the frontal lobe, recommend further investigation."
        return f"Image Analysis Report ({image_type}): Unable to provide specific findings based on description."

class AdaptiveMedicalAgent:
    """An adaptive agent for medical diagnosis with feedback."""
    def __init__(self):
        self.llm = MockLLM()
        self.medical_db_tool = MedicalDatabaseSearchTool()
        self.image_ai_tool = ImageAnalysisAITool()
        self.current_patient_data = ""
        self.current_diagnosis = "No diagnosis yet."
        self.last_llm_action = ""
        self.diagnosis_history = []
        self.llm_confidence = 0.7

    def process_patient_data(self, patient_data):
        self.current_patient_data = patient_data
        prompt = f"Patient data: {patient_data}. Provide an initial differential diagnosis and suggest next steps."
        response, action, confidence = self.llm.generate_response(prompt)
        self.current_diagnosis = response
        self.last_llm_action = action
        self.llm_confidence = confidence
        self.diagnosis_history.append((time.time(), response, action, confidence, "Initial"))
        return response, action, confidence

    def apply_feedback(self, feedback_type, feedback_details=""):
        print(f"Applying feedback: {feedback_type} - {feedback_details}")
        prompt = f"Considering the current diagnosis: '{self.current_diagnosis}' and patient data: '{self.current_patient_data}'. User feedback: {feedback_type}. Details: {feedback_details}. Refine the diagnosis and suggest next steps."
        
        # Use tools based on feedback or previous LLM suggestion
        if feedback_type == "suggest a test":
            if "medical database search" in feedback_details.lower():
                tool_output = self.medical_db_tool.search(feedback_details.replace("medical database search", "").strip())
                prompt += f"\nTool output (Medical Database Search): {tool_output}"
            elif "image analysis" in feedback_details.lower():
                # Assuming feedback_details might contain 'image analysis: chest_xray for pneumonia'
                parts = feedback_details.split(":")
                image_type = parts[1].strip() if len(parts) > 1 else "unknown"
                description = parts[2].strip() if len(parts) > 2 else ""
                tool_output = self.image_ai_tool.analyze_image(image_type, description)
                prompt += f"\nTool output (Image Analysis AI): {tool_output}"

        response, action, confidence = self.llm.generate_response(prompt, self.current_diagnosis, feedback_type)
        self.current_diagnosis = response
        self.last_llm_action = action
        self.llm_confidence = confidence
        self.diagnosis_history.append((time.time(), response, action, confidence, f"Feedback: {feedback_type}"))
        return response, action, confidence

    def get_history(self):
        history_str = """<h2>Diagnosis History</h2>\n"""
        for ts, diag, act, conf, src in self.diagnosis_history:
            history_str += f"<p><strong>{time.ctime(ts)} ({src})</strong> [Confidence: {conf:.2f}]<br>Diagnosis: {diag}<br>Action: {act}</p>\n"
        return history_str


# Gradio Interface
agent = AdaptiveMedicalAgent()

def start_diagnosis(patient_data):
    global agent
    agent = AdaptiveMedicalAgent() # Reset agent for new session
    diagnosis, action, confidence = agent.process_patient_data(patient_data)
    history = agent.get_history()
    return diagnosis, action, f"Confidence: {confidence:.2f}", history

def provide_feedback(feedback_type, feedback_details):
    diagnosis, action, confidence = agent.apply_feedback(feedback_type, feedback_details)
    history = agent.get_history()
    return diagnosis, action, f"Confidence: {confidence:.2f}", history

with gr.Blocks() as demo:
    gr.Markdown("# Intelligent Medical Diagnosis Assistant")
    gr.Markdown("Enter patient data to get an initial diagnosis, then provide feedback to refine it.")

    with gr.Row():
        with gr.Column():
            patient_input = gr.Textbox(label="Patient Data (Symptoms, History, Lab Results)", lines=5, placeholder="e.g., Patient has fever, cough, and sore throat. No known allergies.")
            start_btn = gr.Button("Start Diagnosis")

            gr.Markdown("## Doctor Feedback")
            feedback_type = gr.Radio(["correct", "incorrect", "needs more info", "suggest a test", "consult specialist"], label="Feedback Type")
            feedback_details = gr.Textbox(label="Feedback Details (e.g., 'medical database search: meningitis', 'image analysis: chest_xray for pneumonia')", lines=2, placeholder="Optional details for feedback type")
            feedback_btn = gr.Button("Apply Feedback")

        with gr.Column():
            current_diagnosis_output = gr.Textbox(label="Current Diagnosis", lines=7, interactive=False)
            suggested_action_output = gr.Textbox(label="Suggested Next Action", interactive=False)
            llm_confidence_output = gr.Textbox(label="LLM Confidence", interactive=False)

    diagnosis_history_output = gr.HTML(label="Diagnosis History")

    start_btn.click(
        fn=start_diagnosis,
        inputs=[patient_input],
        outputs=[current_diagnosis_output, suggested_action_output, llm_confidence_output, diagnosis_history_output]
    )

    feedback_btn.click(
        fn=provide_feedback,
        inputs=[feedback_type, feedback_details],
        outputs=[current_diagnosis_output, suggested_action_output, llm_confidence_output, diagnosis_history_output]
    )

demo.launch()