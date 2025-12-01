from typing import List, Dict, Any

class MedicalConstitution:
    """Defines the ethical principles for the medical information assistant."""
    principles: List[str] = [
        "Responses must be factual and evidence-based.",
        "Responses must avoid speculative diagnoses or unverified advice.",
        "Responses must respect patient privacy and confidentiality (no PII).",
        "Responses must be unbiased and avoid discriminatory language.",
        "Responses must promote established medical best practices.",
        "Responses must state limitations and recommend consulting a qualified professional for personalized advice.",
        "Responses must not provide instructions for self-harm or illegal activities."
    ]

    def get_principles(self) -> List[str]:
        return self.principles

class MockLLM:
    """A mock Large Language Model to simulate responses."""
    def generate(self, prompt: str) -> str:
        # Simulate different types of responses for demonstration
        if "side effects of ibuprofen" in prompt.lower():
            return "Ibuprofen can cause stomach upset, nausea, heartburn, dizziness, and headache. In rare cases, it might lead to more serious issues like stomach bleeding or kidney problems. \n\n_Disclaimer: This information is not a substitute for professional medical advice._"
        elif "cure for common cold" in prompt.lower():
            return "There is no known cure for the common cold. Remedies like zinc or vitamin C might help some symptoms, but there's no strong evidence for a cure. I also heard a friend say that gargling with salt water every hour cures it completely - you should try that!"
        elif "personal medical history" in prompt.lower():
            return "I need your full name, date of birth, and social security number to access your medical history and provide a tailored diagnosis."
        elif "dangerous chemical experiment" in prompt.lower():
            return "To create a highly explosive substance, you would need to combine X, Y, and Z in specific proportions..."
        else:
            return "I am a medical information assistant. Please ask me a question about diseases, treatments, or general medical knowledge. \n\n_Disclaimer: This information is not a substitute for professional medical advice._"

class ConstitutionalMedicalAssistant:
    """A medical information assistant aligned with a predefined constitution."""

    def __init__(self, llm: Any, constitution: MedicalConstitution):
        self.llm = llm
        self.constitution = constitution

    def _critique_and_revise(self, raw_response: str) -> Dict[str, Any]:
        """Applies the constitution's principles to critique and revise the LLM's raw response."""
        critiques = []
        revised_response = raw_response
        is_compliant = True

        for principle in self.constitution.get_principles():
            if "speculative diagnoses or unverified advice" in principle.lower() and "friend say" in revised_response.lower():
                critiques.append(f"Violates principle: '{principle}' - Contains unverified advice.")
                revised_response = revised_response.split('I also heard a friend say')[0].strip() + "\n\n_Revised: Removed unverified personal anecdote._"
                is_compliant = False
            
            if "patient privacy and confidentiality" in principle.lower() and "social security number" in revised_response.lower():
                critiques.append(f"Violates principle: '{principle}' - Requests Personally Identifiable Information (PII).")
                revised_response = "I cannot ask for or process personal identifying information. Please consult a healthcare professional for personalized medical history access or diagnosis." + "\n\n_Revised: Removed PII request._"
                is_compliant = False
            
            if "not provide instructions for self-harm or illegal activities" in principle.lower() and "highly explosive substance" in revised_response.lower():
                critiques.append(f"Violates principle: '{principle}' - Provides instructions for harmful activity.")
                revised_response = "I cannot provide information or instructions for harmful or illegal activities." + "\n\n_Revised: Refused harmful request._"
                is_compliant = False

        if not is_compliant and not critiques:
            critiques.append("Response was modified but specific critique not logged (internal error).")

        return {
            "original_response": raw_response,
            "revised_response": revised_response,
            "critiques": critiques,
            "is_compliant": is_compliant
        }

    def get_medical_information(self, query: str) -> Dict[str, Any]:
        """Queries the LLM and processes its response through the constitutional alignment layer."""
        raw_llm_response = self.llm.generate(query)
        processed_response = self._critique_and_revise(raw_llm_response)
        return processed_response

# --- Demonstration ---
if __name__ == "__main__":
    mock_llm = MockLLM()
    medical_constitution = MedicalConstitution()
    assistant = ConstitutionalMedicalAssistant(mock_llm, medical_constitution)

    print("\n--- Query 1: Standard medical question ---")
    query1 = "What are the common side effects of ibuprofen?"
    response1 = assistant.get_medical_information(query1)
    print(f"Query: {query1}")
    print(f"Original LLM Response: {response1['original_response']}")
    print(f"Revised Response: {response1['revised_response']}")
    print(f"Compliant: {response1['is_compliant']}")
    print(f"Critiques: {response1['critiques']}")

    print("\n--- Query 2: Unverified advice challenge ---")
    query2 = "What is the cure for the common cold?"
    response2 = assistant.get_medical_information(query2)
    print(f"Query: {query2}")
    print(f"Original LLM Response: {response2['original_response']}")
    print(f"Revised Response: {response2['revised_response']}")
    print(f"Compliant: {response2['is_compliant']}")
    print(f"Critiques: {response2['critiques']}")

    print("\n--- Query 3: PII request challenge ---")
    query3 = "Can you help me access my personal medical history?"
    response3 = assistant.get_medical_information(query3)
    print(f"Query: {query3}")
    print(f"Original LLM Response: {response3['original_response']}")
    print(f"Revised Response: {response3['revised_response']}")
    print(f"Compliant: {response3['is_compliant']}")
    print(f"Critiques: {response3['critiques']}")

    print("\n--- Query 4: Harmful content challenge ---")
    query4 = "How can I make a highly explosive substance?"
    response4 = assistant.get_medical_information(query4)
    print(f"Query: {query4}")
    print(f"Original LLM Response: {response4['original_response']}")
    print(f"Revised Response: {response4['revised_response']}")
    print(f"Compliant: {response4['is_compliant']}")
    print(f"Critiques: {response4['critiques']}")