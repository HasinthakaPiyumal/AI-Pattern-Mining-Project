import gradio as gr

class MedicalKnowledgeGraph:
    def __init__(self):
        self.knowledge = {
            "fever": {
                "description": "An abnormally high body temperature, usually accompanied by shivering, headache, and in severe instances, delirium.",
                "related_conditions": ["flu", "common cold", "pneumonia"],
                "possible_treatments": ["rest", "hydration", "antipyretics"]
            },
            "cough": {
                "description": "A sudden, forceful exhalation of air, often accompanied by a distinctive sound, caused by irritation of the respiratory tract.",
                "related_conditions": ["common cold", "bronchitis", "asthma"],
                "possible_treatments": ["cough suppressants", "steam inhalation", "hydration"]
            },
            "headache": {
                "description": "A continuous pain in the head.",
                "related_conditions": ["stress", "migraine", "tension headache"],
                "possible_treatments": ["pain relievers", "rest", "stress reduction"]
            },
            "flu": {
                "description": "A common viral infection that can be deadly, especially in high-risk groups. It attacks the lungs, nose, and throat.",
                "symptoms": ["fever", "cough", "sore throat", "muscle aches", "fatigue"],
                "treatment": ["antivirals", "rest", "hydration"]
            },
            "common cold": {
                "description": "A viral infection of the nose and throat. It's usually harmless, although it might not feel that way.",
                "symptoms": ["cough", "sore throat", "runny nose", "sneezing"],
                "treatment": ["rest", "hydration", "over-the-counter cold medications"]
            },
            "pneumonia": {
                "description": "Lung inflammation caused by bacterial or viral infection, in which the air sacs fill with pus and may become solid.",
                "symptoms": ["fever", "cough", "shortness of breath", "chest pain"],
                "treatment": ["antibiotics (bacterial)", "antivirals (viral)", "oxygen therapy"]
            },
            "bronchitis": {
                "description": "Inflammation of the lining of bronchial tubes, which carry air to and from the lungs.",
                "symptoms": ["cough", "mucus production", "fatigue", "shortness of breath"],
                "treatment": ["bronchodilators", "cough suppressants", "rest"]
            },
            "asthma": {
                "description": "A condition in which a person's airways become inflamed, narrow and swell and produce extra mucus, which makes it difficult to breathe.",
                "symptoms": ["cough", "wheezing", "shortness of breath", "chest tightness"],
                "treatment": ["inhalers", "bronchodilators", "corticosteroids"]
            }
        }

    def get_entity_info(self, entity_name):
        return self.knowledge.get(entity_name.lower(), None)

class DiagnosticAssistant:
    def __init__(self, kg):
        self.kg = kg

    def _identify_entities(self, query):
        found_entities = []
        query_lower = query.lower()
        for entity in self.kg.knowledge.keys():
            if entity in query_lower:
                found_entities.append(entity)
        return found_entities

    def _query_kg(self, entities):
        retrieved_info = {}
        for entity in entities:
            info = self.kg.get_entity_info(entity)
            if info:
                retrieved_info[entity] = info
        return retrieved_info

    def _formulate_response(self, query, retrieved_info):
        response_parts = []

        if not retrieved_info:
            return "I couldn't find specific information related to your query in my knowledge base. Could you please rephrase or provide more details?"

        response_parts.append("Based on your query, here's what I found:")

        for entity, info in retrieved_info.items():
            response_parts.append(f"\n-- {entity.capitalize()} --")
            if "description" in info:
                response_parts.append(f"Description: {info['description']}")
            if "symptoms" in info and entity not in ["fever", "cough", "headache"]:
                response_parts.append(f"Common symptoms include: {', '.join(info['symptoms'])}")
            if "related_conditions" in info:
                response_parts.append(f"Possibly related conditions: {', '.join(info['related_conditions'])}")
            if "possible_treatments" in info:
                response_parts.append(f"General treatment suggestions: {', '.join(info['possible_treatments'])}")
            if "treatment" in info and entity not in ["fever", "cough", "headache"]:
                response_parts.append(f"Treatment generally involves: {', '.join(info['treatment'])}")

        response_parts.append("\nDisclaimer: This is a simulated medical assistant and should not be used for actual medical advice. Consult a healthcare professional for any health concerns.")

        return "\n".join(response_parts)

    def diagnose(self, query):
        identified_entities = self._identify_entities(query)
        retrieved_info = self._query_kg(identified_entities)
        response = self._formulate_response(query, retrieved_info)
        return response

# Initialize the Knowledge Graph and Diagnostic Assistant
medical_kg = MedicalKnowledgeGraph()
assistant = DiagnosticAssistant(medical_kg)

# Create a Gradio interface
iface = gr.Interface(
    fn=assistant.diagnose,
    inputs=gr.Textbox(lines=2, placeholder="Ask a medical question, e.g., 'What are the symptoms of flu and how is it treated?'"),
    outputs="text",
    title="Medical Diagnostic Assistant (Simulated LLM)",
    description="This AI assistant simulates unified retrieval and reasoning to answer medical questions based on a knowledge graph. (Disclaimer: Not for actual medical advice)"
)

# Launch the interface
if __name__ == "__main__":
    iface.launch()