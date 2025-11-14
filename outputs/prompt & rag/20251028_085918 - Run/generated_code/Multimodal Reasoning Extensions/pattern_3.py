import streamlit as st
import cv2
import numpy as np
from PIL import Image
import networkx as nx
import io
import base64

# --- 1. Data Ingestion and Preprocessing Layer ---
def preprocess_image(image_file):
    """Loads and preprocesses an image for feature extraction."""
    if image_file:
        image = Image.open(image_file).convert("RGB")
        # Convert PIL Image to OpenCV format (numpy array)
        img_np = np.array(image)
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Basic preprocessing: resize
        target_size = (224, 224) # Common input size for many CNNs
        resized_img = cv2.resize(img_cv, target_size, interpolation=cv2.INTER_AREA)

        # Normalize (simple example, would be model-specific)
        normalized_img = resized_img / 255.0
        return normalized_img, img_cv # Return original CV2 for annotation too
    return None, None

def preprocess_text(text):
    """Performs basic text preprocessing (conceptual)."""
    if text:
        # In a real scenario, this would involve NLTK/spaCy for tokenization,
        # stemming/lemmatization, stop-word removal, etc.
        processed_tokens = text.lower().split()
        return " ".join(processed_tokens)
    return ""

# --- 2. Multimodal Feature Extraction Layer (Mocks) ---
# In a real application, these would be sophisticated deep learning models.
# For this example, we return dummy features.

class ImageFeatureExtractor:
    def extract(self, processed_image):
        """Mocks image feature extraction."""
        if processed_image is not None:
            # Simulate features as a flattened array
            return np.random.rand(512) # Example feature vector size
        return np.array([])

class TextFeatureExtractor:
    def extract(self, processed_text):
        """Mocks text feature extraction."""
        if processed_text:
            # Simulate features as a fixed-size embedding
            return np.random.rand(768) # Example feature vector size (e.g., BERT-like)
        return np.array([])

# --- 3. Structured Reasoning Layer ---

class ThoughtGraph:
    """Represents the reasoning graph using networkx."""
    def __init__(self):
        self.graph = nx.DiGraph()
        self.node_id_counter = 0

    def add_node(self, label, node_type="concept", attributes=None):
        node_id = self.node_id_counter
        self.node_id_counter += 1
        self.graph.add_node(node_id, label=label, type=node_type, **(attributes if attributes else {}))
        return node_id

    def add_edge(self, u_id, v_id, relation_type="associated_with", weight=1.0):
        self.graph.add_edge(u_id, v_id, type=relation_type, weight=weight)

    def get_graph_visualization(self):
        """Generates a simple representation of the graph for display."""
        nodes = []
        edges = []
        for node_id, data in self.graph.nodes(data=True):
            nodes.append(f"{node_id} [{data['type']}] : {data['label']}")
        for u, v, data in self.graph.edges(data=True):
            edges.append(f"{u} --({data['type']})--> {v}")
        return "\n".join(nodes + edges)


class LLMClient:
    """Mocks a Large Language Model client for generating responses."""
    def query(self, prompt, temperature=0.7):
        """Simulates an LLM response based on the prompt."""
        if "abnormalities in the lung scan" in prompt.lower():
            return "Initial image analysis suggests a subtle consolidation in the right lower lobe, possibly indicative of early pneumonia or inflammation. Further correlation with textual symptoms is recommended."
        elif "patient's key symptoms" in prompt.lower():
            return "Key symptoms identified from patient notes include persistent cough, mild fever, and fatigue. No significant history of chronic lung disease reported."
        elif "synthesize findings" in prompt.lower():
            return (f"Based on the visual evidence of right lower lobe consolidation and the textual symptoms of cough, fever, and fatigue, "
                    f"the most probable diagnosis is **Pneumonia**. "
                    f"Differential diagnoses include bronchitis or atypical infection. "
                    f"Further examination with blood tests (CBC, inflammatory markers) and sputum culture is suggested for definitive diagnosis and treatment planning.")
        else:
            return f"LLM placeholder response for: '{prompt[:100]}...'"

class MultimodalReasoningEngine:
    """Orchestrates the multimodal reasoning process."""
    def __init__(self, image_extractor, text_extractor, llm_client):
        self.image_extractor = image_extractor
        self.text_extractor = text_extractor
        self.llm_client = llm_client
        self.thought_graph = ThoughtGraph()

    def decompose_problem(self, query):
        """Simulates problem decomposition."""
        st.info(f"Decomposing problem: '{query}'")
        # In a real LangChain/Llama-Index setup, an agent would generate these.
        return [
            "Analyze medical images for any abnormalities.",
            "Extract key symptoms and medical history from patient text.",
            "Correlate visual findings with textual symptoms.",
            "Synthesize findings for a diagnostic suggestion and explanation."
        ]

    def generate_annotated_image(self, original_image_cv2, bounding_boxes, labels):
        """Draws bounding boxes and labels on an image."""
        annotated_img = original_image_cv2.copy()
        for i, (box, label) in enumerate(zip(bounding_boxes, labels)):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_img, label, (x1, y1 - 10 if y1 - 10 > 10 else y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return annotated_img

    def reason(self, original_image_cv2, processed_image, processed_text, initial_query):
        """Performs structured multimodal reasoning."""
        st.subheader("Reasoning Steps:")
        self.thought_graph = ThoughtGraph() # Reset graph for new diagnosis

        # --- Step 1: Feature Extraction ---
        st.write("1. Extracting Multimodal Features...")
        image_features = self.image_extractor.extract(processed_image)
        text_features = self.text_extractor.extract(processed_text)

        # Add initial features/observations to graph
        img_feat_node = self.thought_graph.add_node("Image Features", type="observation")
        text_feat_node = self.thought_graph.add_node("Text Features", type="observation")

        # --- Step 2: Problem Decomposition & Sub-question Answering ---
        sub_questions = self.decompose_problem(initial_query)

        visual_findings = ""
        textual_symptoms = ""
        annotated_img_data = None

        for sq in sub_questions:
            st.write(f"- Sub-question: {sq}")
            if "abnormalities in the lung scan" in sq.lower():
                # Simulate image analysis and annotation
                llm_response_image = self.llm_client.query(sq)
                visual_findings = llm_response_image

                # Mock bounding box generation
                # In a real system, this would come from a detection model
                bounding_boxes = [(50, 50, 150, 150), (100, 100, 200, 200)] # Example boxes
                labels = ["Consolidation", "Possible Nodule"]

                annotated_img = self.generate_annotated_image(original_image_cv2, bounding_boxes, labels)
                is_success, buffer = cv2.imencode(".png", annotated_img)
                if is_success:
                    annotated_img_data = base64.b64encode(buffer).decode("utf-8")
                    st.image(annotated_img, caption="Intermediate Visual Output: Annotated Image", use_column_width=True)
                    st.markdown(f"**Visual Analysis (LLM Summary):** {visual_findings}")
                else:
                    st.warning("Could not encode annotated image.")

                vis_node = self.thought_graph.add_node("Visual Findings", type="finding", attributes={"description": visual_findings})
                self.thought_graph.add_edge(img_feat_node, vis_node, relation_type="derived_from")


            elif "key symptoms and medical history" in sq.lower():
                llm_response_text = self.llm_client.query(sq)
                textual_symptoms = llm_response_text
                st.markdown(f"**Textual Analysis (LLM Summary):** {textual_symptoms}")

                text_node = self.thought_graph.add_node("Textual Symptoms", type="finding", attributes={"description": textual_symptoms})
                self.thought_graph.add_edge(text_feat_node, text_node, relation_type="derived_from")

            elif "correlate visual findings with textual symptoms" in sq.lower():
                # This step would involve a complex LLM prompt that combines both summaries
                correlation_prompt = f"Given visual findings: '{visual_findings}' and textual symptoms: '{textual_symptoms}', what are the key correlations and potential diagnostic implications?"
                correlation_result = self.llm_client.query(correlation_prompt)
                st.markdown(f"**Correlation (LLM):** {correlation_result}")
                corr_node = self.thought_graph.add_node("Correlated Findings", type="inference", attributes={"description": correlation_result})
                if 'vis_node' in locals(): self.thought_graph.add_edge(vis_node, corr_node, relation_type="correlates_with")
                if 'text_node' in locals(): self.thought_graph.add_edge(text_node, corr_node, relation_type="correlates_with")


        st.subheader("Thought Graph (Simplified Representation):")
        st.code(self.thought_graph.get_graph_visualization())

        # --- Step 3: Diagnostic Output Generation ---
        st.write("4. Synthesizing for Diagnostic Suggestion and Explanation...")
        final_synthesis_prompt = (f"Synthesize the following information for a medical diagnosis:\n\n"
                                  f"Visual Findings: {visual_findings}\n"
                                  f"Textual Symptoms: {textual_symptoms}\n"
                                  f"Provide a primary diagnosis, differential diagnoses, confidence scores (mock), explanation of reasoning, and suggested further examinations.")

        final_diagnosis_output = self.llm_client.query(final_synthesis_prompt)
        st.success("Reasoning complete!")
        return final_diagnosis_output, annotated_img_data

# --- Streamlit UI Layer ---
def main():
    st.set_page_config(layout="wide", page_title="Multimodal Medical Diagnostic Assistant")

    st.title("👨‍⚕️ Multimodal Medical Diagnostic Assistant")
    st.markdown("""
        This assistant uses structured reasoning to help doctors diagnose complex conditions by integrating
        medical images and patient textual data.
    """)

    st.sidebar.header("Patient Data Input")

    # Image Upload
    uploaded_image_file = st.sidebar.file_uploader("Upload Medical Image (X-ray, CT, MRI)", type=["png", "jpg", "jpeg", "dcm"])

    # Text Input
    patient_text = st.sidebar.text_area(
        "Enter Patient Medical Notes (Symptoms, History, Lab Results)",
        "Patient presents with persistent cough for 3 days, mild fever (100.2 F), and general fatigue. No significant prior medical history. Recent travel to a densely populated area. Oxygen saturation 97% on room air. No wheezing on auscultation.",
        height=200
    )

    if st.sidebar.button("Diagnose Patient"):
        if not uploaded_image_file and not patient_text:
            st.error("Please upload an image or enter patient text to proceed with diagnosis.")
        else:
            with st.spinner("Processing and reasoning... This may take a moment."):
                processed_image, original_image_cv2 = preprocess_image(uploaded_image_file)
                processed_text = preprocess_text(patient_text)

                if uploaded_image_file:
                    st.subheader("Uploaded Medical Image:")
                    st.image(uploaded_image_file, caption="Original Image", use_column_width=True)

                st.subheader("Processed Patient Text:")
                st.write(processed_text if processed_text else "No text provided.")

                # Initialize components
                image_extractor = ImageFeatureExtractor()
                text_extractor = TextFeatureExtractor()
                llm_client = LLMClient()
                reasoning_engine = MultimodalReasoningEngine(image_extractor, text_extractor, llm_client)

                # Perform reasoning
                initial_query = "Diagnose the patient based on provided medical image and textual information."
                final_diagnosis_output, annotated_img_data = reasoning_engine.reason(
                    original_image_cv2, processed_image, processed_text, initial_query
                )

                st.subheader("--- Final Diagnostic Suggestion ---")
                st.markdown(final_diagnosis_output)

                if annotated_img_data:
                    st.markdown("---")
                    st.subheader("Intermediate Annotated Image (for reasoning guidance):")
                    st.image(f"data:image/png;base64,{annotated_img_data}", caption="Annotated Visual Output", use_column_width=True)

    st.sidebar.markdown("---")
    st.sidebar.info("This is a simplified demonstration of a Multimodal Structured Reasoning pattern. "
                    "Actual medical diagnostic systems require extensive training on vast datasets, "
                    "rigorous validation, and regulatory approval.")

if __name__ == "__main__":
    main()
