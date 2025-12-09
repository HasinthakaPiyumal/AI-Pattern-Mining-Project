import abc

class LLMInterface(abc.ABC):
    @abc.abstractmethod
    def generate_response(self, prompt: str) -> str:
        pass

class KGInterface(abc.ABC):
    @abc.abstractmethod
    def query_knowledge(self, query_terms: list[str]) -> str:
        pass

class MockLLM(LLMInterface):
    def generate_response(self, prompt: str) -> str:
        if "fever" in prompt.lower() and "paracetamol" in prompt.lower():
            return "Based on your symptoms and knowledge, paracetamol is commonly used for fever relief. Always consult a doctor for diagnosis and treatment."
        elif "diabetes" in prompt.lower() and "metformin" in prompt.lower():
            return "Metformin is a common medication for type 2 diabetes. Ensure proper dosage as prescribed by a physician."
        elif "headache" in prompt.lower():
            return "Headaches can have various causes. Mild headaches might respond to rest or over-the-counter pain relievers. If severe or persistent, seek medical advice."
        return f"Mock LLM response for: '{prompt}'. Further details might require real LLM integration."

class MockRxNormKG(KGInterface):
    def query_knowledge(self, query_terms: list[str]) -> str:
        knowledge_base = {
            "paracetamol": "Paracetamol (Acetaminophen) is a pain reliever and fever reducer. Common dosages vary by age and weight. Overdose can cause liver damage.",
            "ibuprofen": "Ibuprofen is a nonsteroidal anti-inflammatory drug (NSAID) used for pain, fever, and inflammation. Can cause stomach upset.",
            "metformin": "Metformin is an oral antidiabetic drug in the biguanide class, used to treat type 2 diabetes.",
            "amlodipine": "Amlodipine is a calcium channel blocker used to treat high blood pressure and chest pain (angina)."
        }
        found_knowledge = []
        for term in query_terms:
            if term.lower() in knowledge_base:
                found_knowledge.append(knowledge_base[term.lower()])
        return "\n".join(found_knowledge) if found_knowledge else "No drug information found."

class MockSnomedCTKG(KGInterface):
    def query_knowledge(self, query_terms: list[str]) -> str:
        knowledge_base = {
            "fever": "Fever (elevated body temperature) is a common symptom of many medical conditions, including infections. Normal body temperature is around 37°C (98.6°F).",
            "headache": "Headache is pain in any region of the head. It can be a symptom of many different conditions.",
            "diabetes": "Diabetes mellitus is a metabolic disease that causes high blood sugar. The body either doesn't produce enough insulin, or can't effectively use the insulin it does produce.",
            "hypertension": "Hypertension (high blood pressure) is a common condition in which the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems."
        }
        found_knowledge = []
        for term in query_terms:
            if term.lower() in knowledge_base:
                found_knowledge.append(knowledge_base[term.lower()])
        return "\n".join(found_knowledge) if found_knowledge else "No disease/symptom information found."

class ClinicalDecisionSupportSystem:
    def __init__(self, llm_interface: LLMInterface, kg_interface: KGInterface):
        self.llm_interface = llm_interface
        self.kg_interface = kg_interface

    def _extract_terms(self, query: str) -> list[str]:
        keywords = ["fever", "headache", "diabetes", "hypertension", "paracetamol", "ibuprofen", "metformin", "amlodipine"]
        extracted = [term for term in keywords if term in query.lower()]
        return extracted

    def process_patient_query(self, query: str) -> str:
        extracted_terms = self._extract_terms(query)
        kg_knowledge = self.kg_interface.query_knowledge(extracted_terms)

        enriched_prompt = f"Patient query: {query}\n\nRelevant medical knowledge:\n{kg_knowledge}\n\nProvide diagnostic support or treatment recommendations based on this information."

        llm_response = self.llm_interface.generate_response(enriched_prompt)
        return llm_response

if __name__ == "__main__":
    mock_llm = MockLLM()
    mock_rxnorm_kg = MockRxNormKG()
    mock_snomed_ct_kg = MockSnomedCTKG()

    print("--- Scenario 1: MockLLM with MockRxNormKG ---")
    cdss_rxnorm = ClinicalDecisionSupportSystem(llm_interface=mock_llm, kg_interface=mock_rxnorm_kg)
    query1 = "What should I know about paracetamol for my fever?"
    response1 = cdss_rxnorm.process_patient_query(query1)
    print(f"Query: {query1}\nResponse: {response1}\n")

    print("--- Scenario 2: MockLLM with MockSnomedCTKG ---")
    cdss_snomed = ClinicalDecisionSupportSystem(llm_interface=mock_llm, kg_interface=mock_snomed_ct_kg)
    query2 = "My patient has symptoms of diabetes. What is diabetes?"
    response2 = cdss_snomed.process_patient_query(query2)
    print(f"Query: {query2}\nResponse: {response2}\n")

    print("--- Scenario 3: MockLLM with MockSnomedCTKG for Headache ---")
    query3 = "I have a terrible headache."
    response3 = cdss_snomed.process_patient_query(query3)
    print(f"Query: {query3}\nResponse: {response3}\n")