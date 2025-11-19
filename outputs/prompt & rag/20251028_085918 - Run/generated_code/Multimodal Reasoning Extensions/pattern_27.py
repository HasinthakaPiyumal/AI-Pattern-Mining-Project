import pandas as pd
import networkx as nx
from PIL import Image
import matplotlib.pyplot as plt
import io
import base64

class MultimodalDiagnosticAssistant:
    def __init__(self):
        pass

    # 1. Data Ingestion and Preprocessing Module
    def _load_image(self, image_input):
        if isinstance(image_input, str):
            # Assume image_input is a path for simulation
            try:
                return Image.open(image_input)
            except FileNotFoundError:
                print(f"Simulating loading image from {image_input}")
                # Create a dummy image if path doesn't exist for simulation
                return Image.new("RGB", (256, 256), color = "red")
        elif isinstance(image_input, Image.Image):
            return image_input
        else:
            # For simplicity, assume base64 string or bytes for direct processing
            try:
                img_bytes = base64.b64decode(image_input)
                return Image.open(io.BytesIO(img_bytes))
            except Exception as e:
                print(f"Could not load image: {e}. Returning a dummy image.")
                return Image.new("RGB", (256, 256), color = "red")

    def _preprocess_image(self, image):
        # Placeholder for image preprocessing (e.g., resizing, normalization)
        # In a real application, torchvision transforms or similar would be used
        # print("Simulating image preprocessing.")
        return image.resize((224, 224))

    def _load_text(self, text_input):
        return str(text_input)

    def _preprocess_text(self, text):
        # Placeholder for NLP preprocessing (tokenization, cleaning)
        # In a real application, transformers tokenizers or spacy would be used
        # print("Simulating text preprocessing.")
        return text.lower()

    def _load_structured_data(self, data_input):
        if isinstance(data_input, pd.DataFrame):
            return data_input
        else:
            # Simulate loading from a dictionary or list of dicts
            # print("Simulating loading structured data.")
            return pd.DataFrame(data_input)

    def _preprocess_structured_data(self, df):
        # Placeholder for structured data preprocessing (e.g., normalization)
        # print("Simulating structured data preprocessing.")
        return df

    # 2. Multimodal Feature Extraction Module
    def _extract_image_features(self, processed_image):
        # Placeholder for actual image feature extraction (e.g., ViT, CNNs)
        # In a real application, this would use models from transformers or torchvision
        # print("Simulating image feature extraction.")
        # Return a conceptual feature vector (e.g., mean pixel values as a proxy)
        if processed_image.mode == "RGB":
            r, g, b = processed_image.split()
            return [sum(list(r.getdata())) / len(list(r.getdata())), 
                    sum(list(g.getdata())) / len(list(g.getdata())), 
                    sum(list(b.getdata())) / len(list(b.getdata()))]
        else:
            # Handle grayscale or other modes simply
            return [sum(list(processed_image.getdata())) / len(list(processed_image.getdata()))]

    def _extract_text_features(self, processed_text):
        # Placeholder for actual text feature extraction (e.g., BERT embeddings)
        # In a real application, this would use models from transformers or sentence-transformers
        # print("Simulating text feature extraction.")
        return [len(processed_text), processed_text.count("disease"), processed_text.count("symptom")] # Conceptual features

    def _process_structured_data_features(self, processed_df):
        # Placeholder for processing structured data into features
        # print("Simulating structured data feature processing.")
        return processed_df.mean(numeric_only=True).tolist() if not processed_df.empty else [] # Conceptual features

    # 3. Structured Reasoning Engine Module
    def _decompose_problem(self, query, patient_data):
        # Simulates Chain-of-Thought by breaking down the diagnostic query
        # print("Decomposing problem into sub-questions (Chain-of-Thought).")
        sub_questions = [
            f"Analyze image for abnormalities related to ",
            f"Correlate text symptoms with potential conditions",
            f"Evaluate lab results for indicators",
            f"Synthesize findings to propose diagnosis"
        ]
        return sub_questions

    def _construct_thought_graph(self, features):
        # Uses networkx to build a conceptual graph of evidence and reasoning steps
        # print("Constructing thought graph (Multimodal Graph-of-Thought).")
        G = nx.DiGraph()
        
        # Add feature nodes
        if "image" in features and features["image"]:
            G.add_node("Image_Features", type="visual", data=features["image"])
        if "text" in features and features["text"]:
            G.add_node("Text_Features", type="linguistic", data=features["text"])
        if "structured" in features and features["structured"]:
            G.add_node("Structured_Features", type="numerical", data=features["structured"])

        # Add conceptual reasoning steps and connect them
        G.add_node("Analyze_Visual_Evidence", type="reasoning")
        G.add_node("Analyze_Linguistic_Evidence", type="reasoning")
        G.add_node("Analyze_Numerical_Evidence", type="reasoning")
        G.add_node("Integrate_Evidence", type="reasoning")
        G.add_node("Formulate_Diagnosis", type="reasoning")

        if "Image_Features" in G:
            G.add_edge("Image_Features", "Analyze_Visual_Evidence", relation="provides_input")
        if "Text_Features" in G:
            G.add_edge("Text_Features", "Analyze_Linguistic_Evidence", relation="provides_input")
        if "Structured_Features" in G:
            G.add_edge("Structured_Features", "Analyze_Numerical_Evidence", relation="provides_input")

        G.add_edge("Analyze_Visual_Evidence", "Integrate_Evidence", relation="contributes_to")
        G.add_edge("Analyze_Linguistic_Evidence", "Integrate_Evidence", relation="contributes_to")
        G.add_edge("Analyze_Numerical_Evidence", "Integrate_Evidence", relation="contributes_to")
        G.add_edge("Integrate_Evidence", "Formulate_Diagnosis", relation="leads_to")

        return G

    def _integrate_evidence(self, thought_graph, features):
        # Placeholder for conceptually integrating evidence from the graph
        # In a real system, this would involve complex graph neural networks or reasoning algorithms
        # print("Integrating multimodal evidence.")
        integrated_summary = {
            "image_insights": f"Visual analysis suggests {features.get('image_description', 'no clear abnormalities')}.",
            "text_insights": f"Textual analysis highlights symptoms like {features.get('text_symptoms', 'none specified')}.",
            "structured_insights": f"Lab results show deviations in {features.get('structured_deviations', 'normal ranges')}."
        }
        return integrated_summary

    def _generate_visual_insight(self, original_image, reasoning_step):
        # Simulates generating an intermediate visual step (e.g., highlighting)
        # In a real application, this would use OpenCV or matplotlib for actual image manipulation
        # print(f"Generating visual insight for step: {reasoning_step}")
        
        if original_image is None:
            return None, "No original image to annotate."

        # Create a conceptual annotated image
        fig, ax = plt.subplots(1)
        ax.imshow(original_image)
        ax.set_title(f"Visual Insight: {reasoning_step}")
        ax.axis("off")
        
        # Simulate a highlight (e.g., a red rectangle)
        if "abnormalities" in reasoning_step or "highlight" in reasoning_step:
            rect = plt.Rectangle((50, 50), 100, 100, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            explanation = "Simulated highlight of a potential area of interest."
        else:
            explanation = "No specific visual anomaly highlighted for this step."

        # Save plot to a bytes object and then encode to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig) # Close the plot to prevent it from displaying directly
        buf.seek(0)
        encoded_image = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        return encoded_image, explanation

    def _synthesize_diagnosis(self, thought_graph, integrated_evidence, reasoning_steps):
        # Formulates the final diagnosis and an interpretable explanation
        # print("Synthesizing final diagnosis and explanation.")
        
        diagnosis_options = [
            "Possible Viral Infection",
            "Suspected Bacterial Pneumonia",
            "Early Stage Neoplasm",
            "Inflammatory Condition",
            "Metabolic Disorder"
        ]
        
        # Simple heuristic for diagnosis based on evidence
        final_diagnosis = diagnosis_options[0] # Default
        explanation = ""
        
        if "Visual analysis suggests" in integrated_evidence.get("image_insights", "") and "abnormalities" in integrated_evidence["image_insights"]:
            final_diagnosis = diagnosis_options[2] # Neoplasm if visual abnormalities
            explanation += integrated_evidence["image_insights"] + " "
            
        if "textual analysis highlights symptoms like fever" in integrated_evidence.get("text_insights", ""):
            final_diagnosis = diagnosis_options[0] # Viral if fever
            explanation += integrated_evidence["text_insights"] + " "
            
        if "lab results show deviations in inflammation markers" in integrated_evidence.get("structured_insights", ""):
            final_diagnosis = diagnosis_options[3] # Inflammatory if inflammation markers
            explanation += integrated_evidence["structured_insights"] + " "
            
        if not explanation:
            explanation = "Based on the integrated multimodal data, a preliminary assessment suggests further investigation is warranted. No definitive diagnosis can be made at this stage without more specific clinical data."
            
        full_explanation = f"Proposed Diagnosis: {final_diagnosis}.\n\nReasoning Path:\n"
        for i, step in enumerate(reasoning_steps):
            full_explanation += f"  Step {i+1}: {step}\n"
        full_explanation += f"\nDetailed Evidence Integration:\n- Visual: {integrated_evidence.get('image_insights', 'N/A')}\n"
        full_explanation += f"- Textual: {integrated_evidence.get('text_insights', 'N/A')}\n"
        full_explanation += f"- Structured: {integrated_evidence.get('structured_insights', 'N/A')}\n"
        full_explanation += f"\nFurther Analysis: {explanation}"

        return final_diagnosis, full_explanation

    # 4. Orchestration and Output Module
    def diagnose(self, image_input, text_input, structured_data_input, diagnostic_query):
        print("Starting diagnostic process...")

        # 1. Data Ingestion and Preprocessing
        original_image = self._load_image(image_input)
        processed_image = self._preprocess_image(original_image)
        processed_text = self._preprocess_text(self._load_text(text_input))
        processed_structured_data = self._preprocess_structured_data(self._load_structured_data(structured_data_input))

        # 2. Multimodal Feature Extraction
        image_features = self._extract_image_features(processed_image)
        text_features = self._extract_text_features(processed_text)
        structured_features = self._process_structured_data_features(processed_structured_data)

        all_features = {
            "image": image_features,
            "text": text_features,
            "structured": structured_features,
            "image_description": "some visual abnormalities" if sum(image_features) > 100 else "no significant visual findings",
            "text_symptoms": "fever, cough" if "fever" in processed_text else "mild discomfort",
            "structured_deviations": "elevated CRP, low WBC" if structured_features and structured_features[0] > 10 else "normal ranges"
        }

        # 3. Structured Reasoning Engine
        reasoning_steps = self._decompose_problem(diagnostic_query, all_features)
        thought_graph = self._construct_thought_graph(all_features)
        integrated_evidence = self._integrate_evidence(thought_graph, all_features)
        
        # Generate visual insights for a specific step, if applicable
        visual_insights = []
        for step in reasoning_steps:
            if "image for abnormalities" in step:
                encoded_img, visual_exp = self._generate_visual_insight(original_image, step)
                if encoded_img:
                    visual_insights.append({"step": step, "image": encoded_img, "explanation": visual_exp})
            
        final_diagnosis, full_explanation = self._synthesize_diagnosis(thought_graph, integrated_evidence, reasoning_steps)

        print("Diagnostic process completed.")
        return {
            "proposed_diagnosis": final_diagnosis,
            "detailed_explanation": full_explanation,
            "reasoning_steps": reasoning_steps,
            "visual_insights": visual_insights
        }

if __name__ == "__main__":
    # Example Usage
    assistant = MultimodalDiagnosticAssistant()

    # Simulate inputs
    # For image, you can provide a path to a dummy image or create one in memory
    # For demonstration, we will create a dummy image file if it doesn't exist
    dummy_image_path = "dummy_xray.png"
    try:
        Image.open(dummy_image_path)
    except FileNotFoundError:
        dummy_img = Image.new("RGB", (400, 300), color = "blue")
        dummy_img.save(dummy_image_path)

    patient_text_report = "Patient presents with persistent cough, mild fever, and fatigue for 3 days. No severe respiratory distress. History of seasonal allergies."
    patient_lab_results = pd.DataFrame({
        "Test": ["CRP", "WBC", "Hemoglobin"],
        "Value": [15.2, 12.5, 14.1],
        "Unit": ["mg/L", "x10^9/L", "g/dL"]
    })
    diagnostic_query = "What is the most likely diagnosis based on all available data?"

    result = assistant.diagnose(
        image_input=dummy_image_path,
        text_input=patient_text_report,
        structured_data_input=patient_lab_results,
        diagnostic_query=diagnostic_query
    )

    print("\n--- Diagnostic Result ---")
    print(f"Proposed Diagnosis: {result['proposed_diagnosis']}")
    print(f"\nDetailed Explanation:\n{result['detailed_explanation']}")
    print(f"\nReasoning Steps: {result['reasoning_steps']}")
    
    if result['visual_insights']:
        print("\nVisual Insights Generated:")
        for insight in result['visual_insights']:
            print(f"  - Step: {insight['step']}")
            print(f"    Explanation: {insight['explanation']}")
            # You could decode and display insight['image'] here if running in an environment that supports image display
            # For example, save to file:
            # with open("visual_insight.png", "wb") as f:
            #     f.write(base64.b64decode(insight['image']))
            # print("    (Image content encoded in base64)")

    print("\n--- Second Example: Different Scenario ---")
    patient_text_report_2 = "Patient reports sudden severe chest pain, shortness of breath, and dizziness. No fever. Elevated troponin levels expected."
    patient_lab_results_2 = pd.DataFrame({
        "Test": ["Troponin", "CK-MB", "BP"],
        "Value": [25.0, 50.0, 140/90],
        "Unit": ["ng/mL", "ng/mL", "mmHg"]
    })

    result_2 = assistant.diagnose(
        image_input=None, # No image for this case
        text_input=patient_text_report_2,
        structured_data_input=patient_lab_results_2,
        diagnostic_query="Evaluate for cardiac event."
    )
    print(f"Proposed Diagnosis (Scenario 2): {result_2['proposed_diagnosis']}")
    print(f"\nDetailed Explanation (Scenario 2):\n{result_2['detailed_explanation']}")