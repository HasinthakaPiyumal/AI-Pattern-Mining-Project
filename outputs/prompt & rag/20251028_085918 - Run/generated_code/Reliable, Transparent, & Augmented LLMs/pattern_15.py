import random
import time

class SimulatedLLM:
    """Simulates a Large Language Model for interpreting patient input."""
    def process_input(self, symptoms: str, medical_history: str) -> dict:
        print(f"LLM: Processing symptoms: '{symptoms}' and history: '{medical_history}'")
        # Simulate LLM's understanding and initial plan
        # In a real scenario, this would involve NLP, entity extraction, etc.
        potential_conditions = ["Common Cold", "Influenza", "Allergy", "Bronchitis", "Pneumonia", "Migraine"]
        random_condition = random.choice(potential_conditions)
        
        # Simulate LLM suggesting tools to use
        if "chest pain" in symptoms.lower() or "shortness of breath" in symptoms.lower():
            tool_suggestions = ["medical_database", "diagnostic_imaging"]
        elif "drug" in medical_history.lower() or "medication" in symptoms.lower():
            tool_suggestions = ["medical_database", "drug_interaction_database"]
        else:
            tool_suggestions = ["medical_database"]
            
        return {
            "initial_understanding": f"Based on input, considering {random_condition} and related conditions.",
            "suggested_tools": tool_suggestions,
            "query_for_tools": {
                "symptoms": symptoms,
                "history": medical_history,
                "potential_condition": random_condition
            }
        }

class MedicalDatabaseAPI:
    """Simulates an API for a medical knowledge base."""
    def get_info(self, query: dict) -> dict:
        print(f"MedicalDB: Querying for: {query.get('potential_condition')}")
        time.sleep(0.5) # Simulate API call delay
        
        condition = query.get('potential_condition', 'Unknown Condition')
        if condition == "Common Cold":
            return {"data": f"Info on {condition}: Viral infection, symptoms include runny nose, cough, sore throat. Treatment: rest, fluids.", "confidence": 0.8}
        elif condition == "Influenza":
            return {"data": f"Info on {condition}: Viral infection, symptoms include fever, body aches, fatigue. Treatment: antivirals, rest.", "confidence": 0.85}
        elif condition == "Pneumonia":
            return {"data": f"Info on {condition}: Lung infection, symptoms include cough with phlegm, fever, shortness of breath. Often requires antibiotics.", "confidence": 0.9}
        else:
            return {"data": f"Info on {condition}: General information not specifically found. May require further investigation.", "confidence": 0.5}

class DiagnosticImagingAnalysis:
    """Simulates a diagnostic imaging analysis service."""
    def analyze_images(self, patient_id: str, imaging_type: str) -> dict:
        print(f"ImagingAnalysis: Analyzing {imaging_type} for patient {patient_id}")
        time.sleep(1) # Simulate complex analysis
        
        # Simulate different results based on patient_id for demonstration
        if patient_id == "P001": # Example for a patient with potential issue
            return {"data": "Chest X-ray shows minor consolidation in lower left lung lobe. Suggests possible early pneumonia.", "confidence": 0.92}
        else:
            return {"data": f"{imaging_type} for patient {patient_id} appears normal.", "confidence": 0.75}

class DrugInteractionDatabaseAPI:
    """Simulates an API for a drug interaction checker."""
    def check_interactions(self, drugs: list) -> dict:
        print(f"DrugInteractionDB: Checking interactions for drugs: {', '.join(drugs)}")
        time.sleep(0.7) # Simulate API call delay
        
        if "aspirin" in [d.lower() for d in drugs] and "warfarin" in [d.lower() for d in drugs]:
            return {"data": "Severe interaction: Increased risk of bleeding with Aspirin and Warfarin.", "confidence": 0.95, "interaction_found": True}
        elif len(drugs) > 1: # Generic interaction for multiple drugs
            return {"data": "Multiple medications detected. Consult a pharmacist for potential interactions.", "confidence": 0.7, "interaction_found": True}
        else:
            return {"data": "No significant interactions found for the single drug(s) provided.", "confidence": 0.8, "interaction_found": False}

class ToolOrchestrationEngine:
    """Orchestrates LLM and external tools for diagnosis and explanation."""
    def __init__(self):
        self.llm = SimulatedLLM()
        self.medical_db = MedicalDatabaseAPI()
        self.imaging_analysis = DiagnosticImagingAnalysis()
        self.drug_interaction_db = DrugInteractionDatabaseAPI()
        self.tools = {
            "medical_database": self.medical_db.get_info,
            "diagnostic_imaging": self.imaging_analysis.analyze_images,
            "drug_interaction_database": self.drug_interaction_db.check_interactions,
        }

    def diagnose(self, patient_id: str, symptoms: str, medical_history: str) -> dict:
        reasoning_path = []
        overall_confidence = 0.0
        
        # Step 1: LLM initial interpretation
        llm_response = self.llm.process_input(symptoms, medical_history)
        reasoning_path.append(f"LLM Initial Understanding: {llm_response['initial_understanding']}")
        
        tool_results = []
        tool_confidences = []
        
        # Step 2: Orchestrate external tools based on LLM suggestions
        for tool_name in llm_response['suggested_tools']:
            if tool_name == "medical_database":
                result = self.tools[tool_name](llm_response['query_for_tools'])
                tool_results.append(f"Medical Database: {result['data']}")
                tool_confidences.append(result['confidence'])
                reasoning_path.append(f"Tool Used: Medical Database (Confidence: {result['confidence']:.2f})")
            elif tool_name == "diagnostic_imaging":
                # Simulate dynamic parameters for imaging
                imaging_type = "Chest X-ray" if "chest pain" in symptoms.lower() else "General Scan"
                result = self.tools[tool_name](patient_id, imaging_type)
                tool_results.append(f"Diagnostic Imaging: {result['data']}")
                tool_confidences.append(result['confidence'])
                reasoning_path.append(f"Tool Used: Diagnostic Imaging (Type: {imaging_type}, Confidence: {result['confidence']:.2f})")
            elif tool_name == "drug_interaction_database":
                # Simple drug extraction for simulation
                drugs = [word.strip(',') for word in medical_history.lower().split() if word.lower() in ["aspirin", "warfarin", "paracetamol"]]
                if not drugs: # Try to get drugs from symptoms if not in history
                    drugs = [word.strip(',') for word in symptoms.lower().split() if word.lower() in ["aspirin", "warfarin", "paracetamol"]]
                if drugs:
                    result = self.tools[tool_name](drugs)
                    tool_results.append(f"Drug Interaction Database: {result['data']}")
                    tool_confidences.append(result['confidence'])
                    reasoning_path.append(f"Tool Used: Drug Interaction Database (Drugs: {', '.join(drugs)}, Confidence: {result['confidence']:.2f})")
                else:
                    tool_results.append("Drug Interaction Database: No specific drugs found for interaction check.")
                    tool_confidences.append(0.6) # Default confidence for not finding drugs
                    reasoning_path.append("Tool Used: Drug Interaction Database (No specific drugs found).")

        # Step 3: Synthesize results and determine final diagnosis/abstention
        synthesized_output = "\n".join(tool_results)
        
        if tool_confidences:
            overall_confidence = sum(tool_confidences) / len(tool_confidences)
        else:
            overall_confidence = 0.5 # Default if no tools were used or returned confidence

        final_diagnosis = "" 
        abstain = False
        explanation = ""
        
        if overall_confidence < 0.7:
            abstain = True
            final_diagnosis = "Uncertain Diagnosis"
            explanation = "The system is unable to provide a high-confidence diagnosis based on the available information and tool outputs. Further human expert review is strongly recommended."
        else:
            # Simple heuristic for diagnosis based on tool outputs
            if "pneumonia" in synthesized_output.lower() and overall_confidence > 0.8:
                final_diagnosis = "Probable Pneumonia"
                explanation = "Based on medical database information and imaging analysis, there is a strong indication of pneumonia."
            elif "influenza" in synthesized_output.lower():
                final_diagnosis = "Likely Influenza"
                explanation = "Based on symptoms and medical database, influenza is a likely cause."
            elif "common cold" in synthesized_output.lower():
                final_diagnosis = "Common Cold"
                explanation = "Symptoms are consistent with a common cold."
            elif "interaction" in synthesized_output.lower() and any(t.get('interaction_found', False) for t in [r for r in tool_results if isinstance(r, dict)]): # This part needs correction. tool_results is list of strings
                 final_diagnosis = "Potential Drug Interaction Issue"
                 explanation = "A significant drug interaction has been identified that requires immediate attention."
            else:
                final_diagnosis = "Undetermined Condition"
                explanation = "The system has processed the information, but a definitive diagnosis cannot be made at this time. Further investigation is needed."

        reasoning_path.append(f"Final Synthesis: Combined results from tools to reach a conclusion.")
        reasoning_path.append(f"Overall Confidence: {overall_confidence:.2f}")
        reasoning_path.append(f"Decision: {'Abstain' if abstain else 'Provide Diagnosis'}")

        return {
            "diagnosis": final_diagnosis,
            "reasoning_path": reasoning_path,
            "confidence": overall_confidence,
            "abstain": abstain,
            "explanation": explanation,
            "tool_raw_outputs": tool_results
        }

class HumanAIInteractionLayer:
    """Handles user input, displays results, and collects feedback."""
    def get_patient_input(self):
        print("\n--- Intelligent Medical Diagnostic Assistant ---")
        patient_id = input("Enter Patient ID (e.g., P001): ")
        symptoms = input("Describe patient symptoms: ")
        medical_history = input("Provide patient medical history (e.g., existing conditions, medications): ")
        return patient_id, symptoms, medical_history

    def display_diagnosis(self, result: dict):
        print("\n--- Diagnosis Result ---")
        print(f"Diagnosis: {result['diagnosis']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Abstention: {result['abstain']}")
        print(f"Explanation: {result['explanation']}")
        print("\nReasoning Path:")
        for step in result['reasoning_path']:
            print(f"  - {step}")
        print("\nTool Raw Outputs:")
        for output in result['tool_raw_outputs']:
            print(f"  - {output}")

    def collect_feedback(self) -> str:
        feedback = input("\nWas this diagnosis helpful or accurate? (Yes/No/Details): ")
        # In a real system, this feedback would be more structured and stored.
        return feedback

class EvaluationModule:
    """Simulates AI-assisted evaluation of the diagnostic assistant."""
    def evaluate_diagnosis(self, patient_case: dict, agent_output: dict) -> dict:
        print("\n--- Running Evaluation Module ---")
        # In a real scenario, this would compare agent_output with a 'ground_truth_diagnosis' in patient_case
        # and potentially use an LLM or expert system to assess reasoning coherence, accuracy, etc.
        
        # Simulate a simple evaluation based on confidence and whether it abstained appropriately
        evaluation_score = agent_output['confidence'] * 100
        evaluation_comments = []
        
        if agent_output['abstain']:
            evaluation_comments.append("Agent abstained due to low confidence, indicating appropriate caution.")
            evaluation_score *= 0.8 # Slightly penalize if abstention wasn't strictly necessary but was cautious
        
        if "Pneumonia" in agent_output['diagnosis'] and "consolidation" in " ".join(agent_output['tool_raw_outputs']).lower():
             evaluation_comments.append("Diagnosis aligns with imaging findings, good evidence.")
             evaluation_score = min(100, evaluation_score * 1.1) # Boost for strong evidence
        elif "interaction" in agent_output['diagnosis'].lower():
            evaluation_comments.append("Correctly identified potential drug interaction.")
            evaluation_score = min(100, evaluation_score * 1.05)
        
        if agent_output['confidence'] < 0.6 and not agent_output['abstain']:
            evaluation_comments.append("Warning: Low confidence diagnosis provided without abstention.")
            evaluation_score *= 0.7 # Penalize for low confidence and not abstaining

        return {
            "score": round(evaluation_score, 2),
            "comments": evaluation_comments,
            "timestamp": time.ctime()
        }

# Main application flow
if __name__ == "__main__":
    orchestrator = ToolOrchestrationEngine()
    ui = HumanAIInteractionLayer()
    evaluator = EvaluationModule()

    # Simulate a diagnostic session
    patient_id, symptoms, medical_history = ui.get_patient_input()
    diagnosis_result = orchestrator.diagnose(patient_id, symptoms, medical_history)
    ui.display_diagnosis(diagnosis_result)
    feedback = ui.collect_feedback()

    # Simulate storing feedback and performing evaluation
    print(f"\nUser Feedback Recorded: {feedback}")
    
    # For evaluation, we would ideally have a ground truth for the patient case.
    # Here, we'll pass the input symptoms and medical history as part of the 'patient_case'.
    mock_patient_case = {
        "id": patient_id,
        "symptoms": symptoms,
        "medical_history": medical_history,
        "ground_truth_diagnosis": "" # In a real system, this would be populated
    }
    evaluation_report = evaluator.evaluate_diagnosis(mock_patient_case, diagnosis_result)
    print("\n--- Evaluation Report ---")
    print(f"Evaluation Score: {evaluation_report['score']}")
    for comment in evaluation_report['comments']:
        print(f"- {comment}")
    print(f"Generated On: {evaluation_report['timestamp']}")

    print("\nIntelligent Medical Diagnostic Assistant session concluded.")
