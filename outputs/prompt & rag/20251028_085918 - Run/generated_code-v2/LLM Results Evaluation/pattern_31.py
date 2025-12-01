class KnowledgeConsolidator:
    """
    Simulates a module responsible for consolidating external medical evidence.
    In a real application, this would interface with EHR systems, medical databases,
    research papers, etc.
    """

    def get_evidence(self, patient_symptoms: str, patient_id: str = None) -> str:
        """
        Retrieves and consolidates relevant medical evidence based on symptoms and patient ID.

        Args:
            patient_symptoms (str): The symptoms described by the physician.
            patient_id (str): Optional patient identifier to fetch EHR.

        Returns:
            str: A consolidated string of relevant medical evidence.
        """
        evidence = f"Patient presented with symptoms: {patient_symptoms}.\n"
        
        if patient_id:
            # Simulate fetching patient EHR
            if patient_id == "P001":
                evidence += "EHR for P001: Age 65, male, history of hypertension, recent travel to Southeast Asia. Lab results from yesterday show elevated CRP and D-dimer.\n"
            elif patient_id == "P002":
                evidence += "EHR for P002: Age 30, female, no significant medical history, experiencing sudden onset severe headache and photophobia.\n"
            else:
                evidence += f"No detailed EHR found for patient ID: {patient_id}.\n"
        
        # Simulate fetching general medical knowledge/research
        if "fever" in patient_symptoms.lower() and "cough" in patient_symptoms.lower():
            evidence += "Relevant research: Consider viral respiratory infections (e.g., influenza, COVID-19) and bacterial pneumonia. Differential includes atypical pathogens. Recent outbreaks should be considered.\n"
        elif "headache" in patient_symptoms.lower() and "photophobia" in patient_symptoms.lower():
            evidence += "Relevant research: Migraine, meningitis, subarachnoid hemorrhage are common differentials. Neuroimaging and CSF analysis might be indicated.\n"
        else:
            evidence += "General medical knowledge: Always consider common causes first. Recent guidelines emphasize patient history and differential diagnosis approach.\n"
            
        return "--- External Medical Evidence ---\n" + evidence + "---------------------------------"

