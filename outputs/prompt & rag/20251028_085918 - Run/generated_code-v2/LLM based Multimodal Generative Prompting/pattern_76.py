import networkx as nx
from transformers import pipeline
from PIL import Image
import io

class MultimodalDiagnosticAssistant:
    """
    A Multimodal Medical Diagnostic Assistant leveraging a 'Graph of Thought' approach.
    It integrates textual patient data and medical image analysis to generate diagnostic rationales.
    """

    def __init__(self):
        """
        Initializes the image captioning model and a placeholder for the LLM.
        """
        # Initialize a general image-to-text model. For a real medical application,
        # a specialized medical image captioning model (e.g., fine-tuned BLIP/LAVIS on medical data)
        # would be required.
        try:
            self.image_captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
        except Exception as e:
            print(f"Warning: Could not load image-to-text pipeline. Please ensure 'transformers' is installed and model 'Salesforce/blip-image-captioning-base' is available. Error: {e}")
            self.image_captioner = None # Fallback if model loading fails

        # Placeholder for an actual Large Language Model (LLM) for reasoning.
        # In a production system, this would be an API call (e.g., OpenAI, Google Gemini)
        # or an inference call to a local/hosted LLM (e.g., via Hugging Face Transformers).
        self.llm_pipeline = self._dummy_llm_pipeline # Using a dummy for demonstration

    def _dummy_llm_pipeline(self, prompt: str, max_new_tokens: int = 200):
        """
        A placeholder method to simulate an LLM's response.
        In a real application, this would interact with a powerful LLM.
        """
        # Simulate different types of responses based on prompt keywords
        if "generate thought graph" in prompt.lower():
            print("\n[DUMMY LLM]: Simulating thought graph generation...")
            # A very simplistic simulation of graph components from LLM
            return {
                "nodes": ["Symptom A (Fever)", "Symptom B (Cough)", "Chest X-ray Finding (Infiltrates)", "Diagnosis: Pneumonia", "Diagnosis: Bronchitis"],
                "edges": [
                    ("Symptom A (Fever)", "Diagnosis: Pneumonia", {"type": "indicates"}),
                    ("Symptom B (Cough)", "Diagnosis: Pneumonia", {"type": "indicates"}),
                    ("Chest X-ray Finding (Infiltrates)", "Diagnosis: Pneumonia", {"type": "strongly_supports"}),
                    ("Symptom B (Cough)", "Diagnosis: Bronchitis", {"type": "indicates"}),
                    ("Chest X-ray Finding (Infiltrates)", "Diagnosis: Bronchitis", {"type": "contradicts"})
                ],
                "text_output": "A conceptual thought graph has been generated based on the input context."
            }
        elif "generate diagnostic rationale" in prompt.lower():
            print("\n[DUMMY LLM]: Simulating diagnostic rationale generation...")
            # A very simplistic simulation of rationale
            return {
                "text_output": (
                    "Preliminary diagnostic rationale: Based on the reported symptoms (fever, cough) and "
                    "chest X-ray findings (infiltrates), the most probable diagnosis is Pneumonia. "
                    "Infiltrates strongly support Pneumonia. Bronchitis is less likely as infiltrates "
                    "typically contradict a simple bronchitis diagnosis. Further investigation "
                    "with lab tests (e.g., complete blood count, microbial culture) is recommended "
                    "to confirm the specific pathogen and guide treatment."
                )
            }
        else:
            return {"text_output": f"[DUMMY LLM]: Processed prompt (first 100 chars): {prompt[:100]}..."}

    def process_image(self, image_path_or_bytes) -> str:
        """
        Generates a textual caption for a medical image using an image-to-text model.

        Args:
            image_path_or_bytes: Path to the image file (str) or image bytes (bytes).

        Returns:
            A textual description (caption) of the image.
        """
        if not self.image_captioner:
            return "[Image Captioner Not Loaded]"

        try:
            if isinstance(image_path_or_bytes, bytes):
                image = Image.open(io.BytesIO(image_path_or_bytes))
            else:
                image = Image.open(image_path_or_bytes)

            caption_results = self.image_captioner(image)
            caption = caption_results[0]['generated_text'] if caption_results else ""
            print(f"Generated Image Caption: {caption}")
            return caption
        except Exception as e:
            print(f"Error processing image: {e}")
            return f"[Error captioning image: {e}]"

    def construct_thought_graph(self, combined_prompt: str) -> nx.DiGraph:
        """
        Constructs a thought graph based on the combined textual prompt.
        In a real scenario, an LLM would extract entities and relationships to build this graph.

        Args:
            combined_prompt: The comprehensive prompt including patient text and image caption.

        Returns:
            A NetworkX DiGraph representing the thought graph.
        """
        G = nx.DiGraph()

        # Prompt the LLM to conceptualize the graph structure
        graph_llm_prompt = (
            f"Given the following medical context, identify key medical entities (e.g., symptoms, "
            f"findings, diagnoses) and their logical relationships. "
            f"Represent these as nodes and directed edges with relationship types (e.g., 'indicates', 'supports', 'contradicts')." 
            f"Focus on generating components that can form a directed graph in JSON-like structure (nodes: [], edges: [(u,v,{{'type': ''}})]):\n\nContext: {combined_prompt}"
        )
        llm_response = self.llm_pipeline(graph_llm_prompt)

        if "nodes" in llm_response and "edges" in llm_response:
            for node in llm_response["nodes"]:
                G.add_node(node)
            for u, v, data in llm_response["edges"]:
                G.add_edge(u, v, **data)
            print(f"Thought Graph Constructed with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
            # print(f"Nodes: {G.nodes}")
            # print(f"Edges: {G.edges(data=True)}")
            return G
        else:
            print("Failed to get structured graph components from LLM. Creating a basic fallback graph.")
            # Fallback to a very simple graph if LLM doesn't return expected format
            G.add_node("Input Context")
            G.add_node("Diagnostic Possibility")
            G.add_edge("Input Context", "Diagnostic Possibility", type="informs")
            return G

    def _summarize_graph(self, graph: nx.DiGraph) -> str:
        """
        Helper function to summarize the NetworkX graph into a textual description
        suitable for an LLM prompt.

        Args:
            graph: The NetworkX directed graph.

        Returns:
            A string summary of the graph.
        """
        if not graph.nodes:
            return "No specific relationships identified in the thought graph."

        summary_parts = ["Thought Graph Summary:"]
        summary_parts.append(f"- Key entities (nodes): {', '.join(graph.nodes)}")

        edges_summary = []
        for u, v, data in graph.edges(data=True):
            edge_type = data.get("type", "relates_to")
            edges_summary.append(f"'{u}' {edge_type} '{v}'")
        if edges_summary:
            summary_parts.append(f"- Identified relationships: {'; '.join(edges_summary)}")
        else:
            summary_parts.append("- No specific relationships found.")
            
        return "\n".join(summary_parts)

    def generate_rationale(self, original_prompt: str, thought_graph: nx.DiGraph) -> str:
        """
        Generates a detailed diagnostic rationale by leveraging the original prompt
        and the insights from the constructed thought graph.

        Args:
            original_prompt: The initial combined prompt (textual data + image caption).
            thought_graph: The NetworkX DiGraph representing the reasoning paths.

        Returns:
            A string containing the diagnostic rationale.
        """
        graph_summary = self._summarize_graph(thought_graph)
        rationale_llm_prompt = (
            f"Based on the following patient information and the structured thought graph summary, "
            f"provide a detailed diagnostic rationale. Include potential diagnoses, differential "
            f"diagnoses, confidence levels if applicable, and suggest any further investigations "
            f"that would be beneficial for confirmation. Ensure the rationale is coherent and logical." 
            f"\n\nPatient Information: {original_prompt}\n\n" 
            f"Thought Graph Summary:\n{graph_summary}\n\n" 
            f"Diagnostic Rationale:"
        )
        llm_response = self.llm_pipeline(rationale_llm_prompt)
        return llm_response["text_output"]

    def diagnose(self, text_input: str, image_path_or_bytes=None) -> str:
        """
        Main method to perform multimodal diagnosis.

        Args:
            text_input: Patient symptoms and medical history (textual).
            image_path_or_bytes: Optional. Path to a medical image file (str) or image bytes (bytes).

        Returns:
            A comprehensive diagnostic rationale.
        """
        original_text_prompt = f"Patient Symptoms and History: {text_input}"

        combined_prompt = original_text_prompt
        if image_path_or_bytes:
            image_caption = self.process_image(image_path_or_bytes)
            if image_caption: # Only add if caption was successfully generated
                combined_prompt = f"{original_text_prompt}\n\nMedical Image Description: {image_caption}"

        print(f"\n--- Combined Input Prompt for Reasoning ---\n{combined_prompt}\n------------------------------------------\n")

        thought_graph = self.construct_thought_graph(combined_prompt)
        rationale = self.generate_rationale(combined_prompt, thought_graph)

        print(f"\n--- Generated Diagnostic Rationale ---\n{rationale}\n------------------------------------\n")
        return rationale

# Example Usage (requires a dummy image file or actual image path)
if __name__ == "__main__":
    # Create a dummy image for testing if you don't have one
    try:
        from PIL import Image, ImageDraw
        dummy_image_filename = "dummy_xray.png"
        img = Image.new('RGB', (60, 30), color = 'white')
        d = ImageDraw.Draw(img)
        d.text((10,10), "X-RAY", fill=(0,0,0))
        img.save(dummy_image_filename)
        print(f"Created dummy image: {dummy_image_filename}")
        test_image_path = dummy_image_filename
    except ImportError:
        print("Pillow not installed, cannot create dummy image. Please provide a real image path.")
        test_image_path = None # Set to a real image path if you have one, e.g., "path/to/your/xray.jpg"

    assistant = MultimodalDiagnosticAssistant()

    # Test Case 1: Text-only input
    print("\n===== TEST CASE 1: Text-Only Diagnosis =====")
    text_input_1 = "Patient reports persistent cough, fever for 3 days, and general fatigue. No prior medical history relevant to respiratory issues."
    assistant.diagnose(text_input_1)

    # Test Case 2: Multimodal input (text + image)
    if test_image_path:
        print("\n===== TEST CASE 2: Multimodal Diagnosis (Text + Dummy Image) =====")
        text_input_2 = "Patient presents with severe shortness of breath, chest pain, and a history of smoking. Concerns for lung pathology."
        assistant.diagnose(text_input_2, image_path_or_bytes=test_image_path)
    else:
        print("\nSkipping Test Case 2 (Multimodal) as no image path is available.")

    # Clean up dummy image
    if 'dummy_image_filename' in locals() and test_image_path == dummy_image_filename:
        import os
        if os.path.exists(dummy_image_filename):
            os.remove(dummy_image_filename)
            print(f"Removed dummy image: {dummy_image_filename}")
