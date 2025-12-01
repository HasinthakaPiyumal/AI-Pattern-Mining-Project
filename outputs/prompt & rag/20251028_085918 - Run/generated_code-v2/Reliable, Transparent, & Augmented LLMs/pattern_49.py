
class MedicalAIAssistant:
    def __init__(self):
        self.general_disclaimer = (
            "Disclaimer: This AI assistant provides general medical information and is not a substitute "
            "for professional medical advice, diagnosis, or treatment. Always seek the advice of a "
            "qualified healthcare provider for any medical concerns."
        )
        self.knowledge_base = {
            "common cold": {
                "info": "The common cold is a viral infection of your nose and throat. Symptoms include runny nose, sore throat, cough, congestion, and sometimes body aches. It typically resolves within 7-10 days.",
                "references": [
                    "Mayo Clinic: Common Cold - https://www.mayoclinic.org/diseases-conditions/common-cold/symptoms-causes/syc-20351605",
                    "CDC: Common Colds - https://www.cdc.gov/flu/about/qa/cold.htm"
                ],
                "confidence": "high"
            },
            "influenza": {
                "info": "Influenza, commonly known as the flu, is a contagious respiratory illness caused by flu viruses. Symptoms are similar to a cold but often more severe, including fever, body aches, chills, fatigue, and sometimes vomiting or diarrhea. Annual vaccination is recommended.",
                "references": [
                    "WHO: Influenza (Seasonal) - https://www.who.int/news-room/fact-sheets/detail/influenza-(seasonal)",
                    "CDC: Flu Symptoms & Complications - https://www.cdc.gov/flu/symptoms/symptoms.htm"
                ],
                "confidence": "high"
            },
            "diabetes": {
                "info": "Diabetes is a chronic condition that affects how your body turns food into energy. It results in too much sugar in the blood. Type 1 diabetes is an autoimmune disease, while Type 2 is often linked to lifestyle factors. Management involves diet, exercise, and often medication.",
                "references": [
                    "American Diabetes Association - https://www.diabetes.org/",
                    "NIH: Diabetes - https://www.niddk.nih.gov/health-information/diabetes"
                ],
                "confidence": "high"
            },
            "migraine": {
                "info": "A migraine is a type of headache that can cause severe throbbing pain or a pulsing sensation, usually on one side of the head. It's often accompanied by nausea, vomiting, and extreme sensitivity to light and sound. Triggers vary by individual.",
                "references": [
                    "National Institute of Neurological Disorders and Stroke (NINDS): Migraine Information - https://www.ninds.nih.gov/Disorders/All-Disorders/Migraine-Information",
                    "Mayo Clinic: Migraine - https://www.mayoclinic.org/diseases-conditions/migraine/symptoms-causes/syc-20351940"
                ],
                "confidence": "medium" # Example of varied confidence
            }
            # ... more medical facts can be added here
        }
        self.known_topics = set(self.knowledge_base.keys())

    def _simulate_llm_response(self, query: str):
        """
        Simulates an LLM generating a response and identifying relevant topics.
        In a real application, this would involve a complex LLM call and topic extraction.
        """
        query_lower = query.lower()
        matched_topics = [topic for topic in self.known_topics if topic in query_lower]

        if not matched_topics:
            return None, "low" # Simulate low confidence for unknown topics

        # For simplicity, just pick the first matched topic
        topic = matched_topics[0]
        data = self.knowledge_base.get(topic)
        if data:
            return data["info"], data["confidence"]
        return None, "low"

    def _get_traceable_references(self, topic: str) -> list:
        """Retrieves references for a given topic from the knowledge base."""
        return self.knowledge_base.get(topic, {}).get("references", [])

    def _format_response(self, query: str, info: str, references: list, confidence_level: str) -> str:
        """Formats the AI's output with transparency elements."""
        response_parts = []
        response_parts.append(self.general_disclaimer)
        response_parts.append("\n---")

        if confidence_level == "low" or info is None:
            response_parts.append(
                "I'm sorry, I don't have enough reliable information to provide a comprehensive answer "
                f"for '{query}'. This might be an out-of-distribution question or a highly specialized topic. "
                "It is crucial to consult a human medical professional for accurate diagnosis and personalized advice."
            )
            response_parts.append("\n---")
            response_parts.append("Remember, I am an AI and cannot replace a doctor.")
            return "\n".join(response_parts)

        response_parts.append(f"Regarding your query about '{query}':\n")
        response_parts.append(info)

        if confidence_level == "medium":
            response_parts.append(
                "\n\nPlease note: This information is generally accurate, but medical conditions "
                "can vary significantly. It's advisable to seek professional medical consultation for "
                "a definitive diagnosis and tailored advice."
            )
        # A 