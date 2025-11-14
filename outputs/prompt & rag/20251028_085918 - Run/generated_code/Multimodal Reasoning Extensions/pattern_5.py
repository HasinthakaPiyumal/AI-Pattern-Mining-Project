import torch
from PIL import Image, ImageDraw, ImageFont
import networkx as nx
import numpy as np
import io
import base64

# Mocking Hugging Face transformers and SentenceTransformers for demonstration
# In a real application, you would load pre-trained models.
class MockTextEncoder:
    def __init__(self):
        pass

    def encode(self, text):
        # Simulate embedding generation
        print(f"Encoding text: {text[:50]}...")
        return np.random.rand(768) # Example embedding dimension

class MockImageEncoder:
    def __init__(self):
        pass

    def encode(self, image):
        # Simulate embedding generation
        print(f"Encoding image of size: {image.size}")
        return np.random.rand(1024) # Example embedding dimension

class MockLLM:
    def __init__(self):
        pass

    def generate(self, prompt, max_new_tokens=50):
        # Simulate LLM response
        print(f"LLM processing prompt: {prompt[:100]}...")
        if "decompose" in prompt.lower():
            return "Sub-question 1: What are the key symptoms mentioned? Sub-question 2: What abnormalities are visible in the X-ray? Sub-question 3: Based on these, what are potential diagnoses?"
        elif "visual findings" in prompt.lower():
            return "The X-ray shows a diffuse haziness in the lower right lung field, suggesting consolidation or pleural effusion."
        elif "diagnosis" in prompt.lower() or "summary" in prompt.lower():
            return "Considering the patient's cough and fever, and the X-ray showing right lower lobe consolidation, a likely diagnosis is bacterial pneumonia. Further tests like sputum culture are recommended."
        else:
            return f"Mock LLM response to: {prompt[:50]}..."

class MockVQAorCaptioningModel:
    def __init__(self):
        pass

    def generate_caption(self, image, question=None):
        print(f"Generating caption for image of size: {image.size}")
        if question:
            return f"Based on the image and question '{question}', there appears to be a lung abnormality."
        return "An X-ray image showing the thoracic cavity."


text_encoder = MockTextEncoder()
image_encoder = MockImageEncoder()
llm_model = MockLLM()
vqa_captioning_model = MockVQAorCaptioningModel()

class MedicalDiagnosisAssistant:
    def __init__(self):
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.llm = llm_model
        self.vqa_captioning_model = vqa_captioning_model
        self.reasoning_graph = nx.DiGraph()
        self.current_diagnosis_context = {}

    def _load_data(self, medical_text: str, image_file):
        # Simulate loading text and image
        text_data = medical_text
        image_data = Image.open(image_file).convert("RGB")
        return text_data, image_data

    def _encode_multimodal_data(self, text_data, image_data):
        text_embedding = self.text_encoder.encode(text_data)
        image_embedding = self.image_encoder.encode(image_data)
        return text_embedding, image_embedding

    def _decompose_problem(self, main_query: str):
        prompt = f"Decompose the following medical diagnostic query into actionable sub-questions: {main_query}"
        sub_questions_str = self.llm.generate(prompt)
        return [q.strip() for q in sub_questions_str.split(". ") if q.strip()]

    def _build_reasoning_graph(self, sub_questions, text_embedding, image_embedding, patient_text_data):
        self.reasoning_graph.clear()
        self.reasoning_graph.add_node("Start", type="initial_state", text="Patient data received")
        self.reasoning_graph.add_node("Patient_Text_Embed", type="embedding", modality="text", embedding=text_embedding, raw_data=patient_text_data)
        self.reasoning_graph.add_node("Image_Embed", type="embedding", modality="image", embedding=image_embedding)
        self.reasoning_graph.add_edge("Start", "Patient_Text_Embed")
        self.reasoning_graph.add_edge("Start", "Image_Embed")

        current_node = "Start"
        for i, sq in enumerate(sub_questions):
            sq_node_name = f"SubQuestion_{i+1}"
            self.reasoning_graph.add_node(sq_node_name, type="sub_question", question=sq)
            self.reasoning_graph.add_edge(current_node, sq_node_name)
            current_node = sq_node_name
            self.current_diagnosis_context[sq_node_name] = sq # Store for reasoning

        # Add a node for initial findings integration
        self.reasoning_graph.add_node("Integrated_Findings_Initial", type="integration", text="Initial integration of text and image embeddings")
        self.reasoning_graph.add_edge("Patient_Text_Embed", "Integrated_Findings_Initial")
        self.reasoning_graph.add_edge("Image_Embed", "Integrated_Findings_Initial")
        self.reasoning_graph.add_edge(current_node, "Integrated_Findings_Initial") # Link last sub-question to initial findings

        print("Reasoning graph initialized with sub-questions.")

    def _reason_multimodally(self, image_data):
        # This simulates traversing the graph and reasoning
        nodes_to_process = list(nx.topological_sort(self.reasoning_graph))
        explanations = []
        annotated_images = []

        for node_id in nodes_to_process:
            node_data = self.reasoning_graph.nodes[node_id]

            if node_data.get("type") == "sub_question":
                question = node_data["question"]
                print(f"Processing sub-question: {question}")

                # Simulate LLM reasoning based on context
                prompt_text_reasoning = f"Given patient medical record data and the current question: '{question}', what are the relevant text findings? Current context: {self.current_diagnosis_context}"
                text_finding = self.llm.generate(prompt_text_reasoning)
                explanations.append(f"**Reasoning for '{question}' (Text):** {text_finding}")
                self.reasoning_graph.nodes[node_id]["text_finding"] = text_finding

                # Simulate VQA/Image Captioning for visual findings
                prompt_image_reasoning = f"Describe visual findings in the image relevant to: {question}"
                visual_finding = self.vqa_captioning_model.generate_caption(image_data, question=question)
                explanations.append(f"**Reasoning for '{question}' (Image):** {visual_finding}")
                self.reasoning_graph.nodes[node_id]["image_finding"] = visual_finding

                # Simulate image annotation based on visual finding (simple rectangle)
                if "lung abnormality" in visual_finding.lower() or "haziness" in visual_finding.lower():
                    annotated_img = image_data.copy()
                    draw = ImageDraw.Draw(annotated_img)
                    # Example: Draw a red rectangle at a fixed location for demonstration
                    draw.rectangle([(50, 50), (200, 200)], outline="red", width=3)
                    try:
                        font = ImageFont.truetype("arial.ttf", 20)
                    except IOError:
                        font = ImageFont.load_default()
                    draw.text((50, 20), "Possible Abnormality", fill="red", font=font)
                    annotated_images.append((f"Annotated Image for '{question}'", annotated_img))


            elif node_data.get("type") == "integration":
                print("Integrating findings...")
                # Here, a more complex LLM call would integrate all gathered findings
                integration_prompt = f"Integrate the following text findings: {self.current_diagnosis_context.get('Patient_Text_Embed', '')} and image findings: {self.current_diagnosis_context.get('Image_Embed', '')} to form a coherent understanding."
                integrated_summary = self.llm.generate(integration_prompt)
                explanations.append(f"**Integrated Summary:** {integrated_summary}")
                self.reasoning_graph.nodes[node_id]["integrated_summary"] = integrated_summary
                self.current_diagnosis_context["Integrated_Findings_Summary"] = integrated_summary

        return explanations, annotated_images

    def _diagnose_and_explain(self):
        final_prompt = f"Based on all gathered findings and integrated reasoning: {self.current_diagnosis_context}, provide a final diagnosis and a step-by-step explanation."
        final_output = self.llm.generate(final_prompt, max_new_tokens=200) # Increased tokens for a detailed explanation
        return final_output

    def run_diagnosis_workflow(self, medical_text: str, image_file):
        # 1. Input Layer
        text_data, image_data = self._load_data(medical_text, image_file)
        self.current_diagnosis_context["Patient_Text_Raw"] = medical_text
        self.current_diagnosis_context["Image_Raw"] = image_data

        # 2. Multimodal Encoders
        text_embedding, image_embedding = self._encode_multimodal_data(text_data, image_data)
        self.current_diagnosis_context["Patient_Text_Embed"] = "Text embedding generated."
        self.current_diagnosis_context["Image_Embed"] = "Image embedding generated."

        # 3. Problem Decomposition Module
        main_query = "Diagnose the patient based on their medical records and X-ray."
        sub_questions = self._decompose_problem(main_query)
        self.current_diagnosis_context["Sub_Questions"] = sub_questions

        # 4. Reasoning Graph Construction Module
        self._build_reasoning_graph(sub_questions, text_embedding, image_embedding, text_data)

        # 5. Multimodal Reasoning Engine
        reasoning_explanations, annotated_images = self._reason_multimodally(image_data)
        self.current_diagnosis_context["Reasoning_Explanations"] = reasoning_explanations

        # 6. Diagnosis and Explainability Module
        final_diagnosis_and_explanation = self._diagnose_and_explain()

        # Prepare results for UI
        image_outputs = []
        for title, img in annotated_images:
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            image_outputs.append((title, f"data:image/png;base64,{img_str}"))

        full_explanation = "\n".join(reasoning_explanations) + "\n\n" + final_diagnosis_and_explanation

        return full_explanation, image_outputs

# --- Gradio UI Integration ---
import gradio as gr

def gradio_interface(medical_text, image_file):
    assistant = MedicalDiagnosisAssistant()
    explanation, image_results = assistant.run_diagnosis_workflow(medical_text, image_file)
    return explanation, [img_src for title, img_src in image_results]

if __name__ == "__main__":
    print("Starting Medical Diagnosis Assistant. Please wait for Gradio to launch...")
    iface = gr.Interface(
        fn=gradio_interface,
        inputs=[
            gr.Textbox(label="Patient Medical Records (Text)", lines=10, value="Patient presents with a persistent cough, fever of 101.5°F, and shortness of breath for 3 days. No known allergies. Recent travel history: none."),
            gr.Image(type="filepath", label="Medical Image (e.g., X-ray)")
        ],
        outputs=[
            gr.Markdown(label="Diagnosis and Explanation"),
            gr.Gallery(label="Annotated Images", columns=2)
        ],
        title="Multimodal Medical Diagnosis Assistant",
        description="Upload patient text records and a medical image (e.g., X-ray) to receive a structured diagnosis and explanation." 
    )
    iface.launch()

