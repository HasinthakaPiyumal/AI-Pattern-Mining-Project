from typing import List, Dict
from knowledge_base import query_medical_knowledge_base

class MedicalTools:
    """A collection of tools for the Medical Diagnostic and Treatment Recommendation System."""

    @staticmethod
    def calculate_dosage(drug_name: str, patient_age: int, patient_weight_kg: float) -> str:
        """Calculates a recommended dosage for a given drug based on patient age and weight (simplified).

        Args:
            drug_name (str): The name of the drug.
            patient_age (int): The age of the patient in years.
            patient_weight_kg (float): The weight of the patient in kilograms.

        Returns:
            str: A string indicating the recommended dosage or a message if not found.
        """
        drug_name_lower = drug_name.lower()
        if "acetaminophen" in drug_name_lower or "paracetamol" in drug_name_lower:
            if patient_age < 12:
                return f"For {drug_name}, consult a pediatrician for dosage. Typical pediatric dosage is 10-15 mg/kg every 4-6 hours."
            else:
                return "For Acetaminophen (adult), typical dosage: 325-650 mg every 4-6 hours. Max 4000 mg/24h."
        elif "ibuprofen" in drug_name_lower:
            if patient_age < 12:
                return f"For {drug_name}, consult a pediatrician for dosage. Typical pediatric dosage is 5-10 mg/kg every 6-8 hours."
            else:
                return "For Ibuprofen (adult), typical dosage: 200-400 mg every 4-6 hours. Max 1200 mg/24h without medical supervision."
        else:
            return f"Dosage information for {drug_name} not readily available in this tool. Please refer to medical guidelines."

    @staticmethod
    def check_drug_interactions(drugs: List[str]) -> str:
        """Simulates checking for potential drug interactions.

        Args:
            drugs (List[str]): A list of drug names.

        Returns:
            str: A message indicating potential interactions or no known interactions (simulated).
        """
        drugs_lower = [d.lower() for d in drugs]

        # Simulate some common interactions
        if "warfarin" in drugs_lower and ("ibuprofen" in drugs_lower or "aspirin" in drugs_lower):
            return "WARNING: Potential increased risk of bleeding with Warfarin and NSAIDs (e.g., Ibuprofen, Aspirin). Consult a physician."
        if len(drugs) > 2 and "antidepressant" in " ".join(drugs_lower) and "triptan" in " ".join(drugs_lower):
            return "WARNING: Potential risk of Serotonin Syndrome with combined use of certain antidepressants and triptans. Consult a physician."
        if len(drugs) > 1:
            return f"Simulated check: No major known interactions for {', '.join(drugs)} based on this tool. Always consult a pharmacist or drug interaction database."
        else:
            return "Please provide at least two drugs to check for interactions."

    @staticmethod
    def retrieve_medical_guidelines(condition: str) -> List[str]:
        """Retrieves relevant medical guidelines/information for a given condition from the knowledge base.

        Args:
            condition (str): The medical condition to search for.

        Returns:
            List[str]: A list of relevant guideline snippets or an empty list if not found.
        """
        print(f"Searching knowledge base for guidelines on: {condition}")
        results = query_medical_knowledge_base(condition, k=5)
        if results:
            return [f"Guideline for {condition}: {res}" for res in results]
        else:
            return [f"No specific guidelines found for {condition} in the current knowledge base."]

# Example of how to use the tools (for testing purposes)
if __name__ == "__main__":
    print("--- Dosage Calculation ---")
    print(MedicalTools.calculate_dosage("Acetaminophen", 30, 70.0))
    print(MedicalTools.calculate_dosage("Ibuprofen", 5, 20.0))
    print(MedicalTools.calculate_dosage("Amoxicillin", 40, 80.0))

    print("\n--- Drug Interaction Check ---")
    print(MedicalTools.check_drug_interactions(["Warfarin", "Ibuprofen"]))
    print(MedicalTools.check_drug_interactions(["Fluoxetine (Antidepressant)", "Sumatriptan (Triptan)", "Aspirin"]))
    print(MedicalTools.check_drug_interactions(["Metformin", "Lisinopril"]))

    print("\n--- Medical Guideline Retrieval (requires knowledge_base.py to be initialized) ---")
    from knowledge_base import initialize_medical_knowledge_base
    initialize_medical_knowledge_base() # Ensure DB is ready
    guidelines = MedicalTools.retrieve_medical_guidelines("Diabetes treatment")
    for g in guidelines:
        print(f"- {g}")
    guidelines = MedicalTools.retrieve_medical_guidelines("Heart attack symptoms")
    for g in guidelines:
        print(f"- {g}")
    guidelines = MedicalTools.retrieve_medical_guidelines("Influenza")
    for g in guidelines:
        print(f"- {g}")
