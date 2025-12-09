from PIL import Image
import networkx as nx

class MultimodalInputProcessor:
    def __init__(self):
        pass

    def load_image(self, image_path):
        try:
            img = Image.open(image_path).convert("RGB")
            return img
        except FileNotFoundError:
            print(f"Error: Image file not found at {image_path}")
            return None
        except Exception as e:
            print(f"Error loading image: {e}")
            return None

    def get_text_input(self, symptoms, question):
        return f"Patient symptoms: {symptoms}\nPatient question: {question}"

class ImageCaptioningModule:
    def __init__(self):
        # In a real application, this would load a pre-trained model
        pass

    def generate_image_caption(self, image):
        # This is a placeholder for a real image captioning model.
        # In a production system, you'd use a model from transformers like BLIP or BLIP-2.
        if image:
            # Simulate a caption based on a simple logic or generic medical terms
            return "Image shows signs of lung pathology, possibly consolidation or infiltrates."
        return "No image provided or captioning failed."

class ThoughtGraphConstructor:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.keywords = {
            "symptoms": ["fever", "cough", "headache", "fatigue", "chest pain", "shortness of breath"],
            "findings": ["infiltrates", "consolidation", "effusion", "fracture", "mass"],
            "conditions": ["pneumonia", "bronchitis", "influenza", "tuberculosis", "cancer", "fracture"]
        }

    def add_entity(self, entity_name, entity_type="unknown"):
        if entity_name not in self.graph:
            self.graph.add_node(entity_name, type=entity_type)

    def add_relationship(self, source, target, relationship_type="associated_with"):
        self.graph.add_edge(source, target, type=relationship_type)

    def construct_graph_from_text(self, text):
        # Simple keyword-based entity extraction and relationship creation for prototype
        text_lower = text.lower()

        detected_symptoms = [s for s in self.keywords["symptoms"] if s in text_lower]
        detected_findings = [f for f in self.keywords["findings"] if f in text_lower]
        detected_conditions = [c for c in self.keywords["conditions"] if c in text_lower]

        # Add detected entities as nodes
        for s in detected_symptoms: self.add_entity(s, "symptom")
        for f in detected_findings: self.add_entity(f, "finding")
        for c in detected_conditions: self.add_entity(c, "condition")

        # Create simple relationships (very basic for prototype)
        for s in detected_symptoms:
            for c in detected_conditions:
                if s in text_lower and c in text_lower:
                    self.add_relationship(s, c, "indicates_potential")
            for f in detected_findings:
                if s in text_lower and f in text_lower:
                    self.add_relationship(s, f, "related_to")
        
        for f in detected_findings:
            for c in detected_conditions:
                if f in text_lower and c in text_lower:
                    self.add_relationship(f, c, "suggests_diagnosis")

    def get_graph_representation(self):
        # Convert graph to a string format for LLM input
        graph_str = "Thought Graph:\n"
        for node, data in self.graph.nodes(data=True):
            graph_str += f"  Node: {node} (Type: {data.get('type', 'unknown')})\n"
        for u, v, data in self.graph.edges(data=True):
            graph_str += f"  Edge: {u} --({data.get('type', 'unknown')})--> {v}\n"
        return graph_str

class LLMIntegration:
    def __init__(self):
        # In a real application, this would involve loading and configuring an LLM
        pass

    def generate_llm_response(self, combined_prompt, graph_representation):
        # This is a placeholder for actual LLM interaction.
        # The LLM would process the combined_prompt and graph_representation
        # to generate a rationale and diagnosis.

        simulated_rationale = (
            f"Based on the provided information and the constructed thought graph:\n"
            f"Patient presented with symptoms and image findings. The graph highlights potential connections. "
            f"For instance, if 'cough' and 'infiltrates' were detected, the graph might connect these to 'pneumonia'.\n"
            f"Actual LLM would analyze this in detail: {combined_prompt}\n"
            f"Graph insights used: {graph_representation}\n"
        )

        simulated_diagnosis = "Potential Diagnosis: Further investigation recommended, but considering symptoms and image findings, conditions like Pneumonia or Bronchitis might be relevant."
        simulated_answer = "Please consult a medical professional for a definitive diagnosis and treatment plan."

        return {
            "rationale": simulated_rationale,
            "diagnosis": simulated_diagnosis,
            "answer": simulated_answer
        }

def medical_diagnostic_assistant(image_path, patient_text_symptoms, patient_question):
    print("Starting Medical Diagnostic Assistant...")

    # 1. Multimodal Input Processing
    input_processor = MultimodalInputProcessor()
    image = input_processor.load_image(image_path)
    text_input = input_processor.get_text_input(patient_text_symptoms, patient_question)
    print("\n--- Input Processed ---")
    print(f"Text Input: {text_input}")

    # 2. Image Captioning
    captioning_module = ImageCaptioningModule()
    image_caption = captioning_module.generate_image_caption(image)
    print("\n--- Image Captioning ---")
    print(f"Image Caption: {image_caption}")

    # Combine all textual data for graph construction and LLM input
    combined_text_for_graph = f"{text_input}\nImage description: {image_caption}"
    combined_llm_prompt = f"Original patient request: {text_input}\nImage observation: {image_caption}\n"

    # 3. Thought Graph Construction
    graph_constructor = ThoughtGraphConstructor()
    graph_constructor.construct_graph_from_text(combined_text_for_graph)
    thought_graph_representation = graph_constructor.get_graph_representation()
    print("\n--- Thought Graph Constructed ---")
    print(thought_graph_representation)

    # 4. Rationale and Diagnosis Generation (LLM Integration)
    llm_integrator = LLMIntegration()
    llm_output = llm_integrator.generate_llm_response(combined_llm_prompt, thought_graph_representation)
    print("\n--- LLM Response ---")
    print(f"Rationale: {llm_output['rationale']}")
    print(f"Diagnosis: {llm_output['diagnosis']}")
    print(f"Answer: {llm_output['answer']}")

    print("\nMedical Diagnostic Assistant finished.")
    return llm_output

if __name__ == "__main__":
    # Example Usage:
    # Create a dummy image file for testing if you don't have one
    try:
        dummy_img = Image.new('RGB', (60, 30), color = 'red')
        dummy_img.save('dummy_xray.png')
        print("Created dummy_xray.png for testing.")
    except Exception as e:
        print(f"Could not create dummy image: {e}. Please ensure Pillow is installed.")

    image_path = "dummy_xray.png"  # Replace with a real path to an X-ray or MRI image
    patient_symptoms = "Patient has a persistent cough, shortness of breath, and fever for 3 days."
    patient_question = "What could be the possible diagnosis and recommended next steps?"

    result = medical_diagnostic_assistant(image_path, patient_symptoms, patient_question)

    print("\n--- Final Result from Assistant ---")
    print(f"Diagnosis: {result['diagnosis']}")
    print(f"Rationale: {result['rationale']}")
    print(f"Answer: {result['answer']}")

    # Another example with different inputs
    print("\n\n--- Second Example --- ")
    image_path_2 = "dummy_xray.png"  # Still using dummy image
    patient_symptoms_2 = "Severe headache and fatigue, no fever. Recent fall."
    patient_question_2 = "Is there any indication of head injury?"

    result_2 = medical_diagnostic_assistant(image_path_2, patient_symptoms_2, patient_question_2)

    print("\n--- Final Result from Assistant (Second Example) ---")
    print(f"Diagnosis: {result_2['diagnosis']}")
    print(f"Rationale: {result_2['rationale']}")
    print(f"Answer: {result_2['answer']}")
