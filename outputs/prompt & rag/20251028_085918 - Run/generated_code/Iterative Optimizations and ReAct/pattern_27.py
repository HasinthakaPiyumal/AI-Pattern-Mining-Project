class PatientRecord:
    def __init__(self, patient_id, age, gender, symptoms, lab_results=None, diagnoses=None, treatments=None, outcome=None):
        self.patient_id = patient_id
        self.age = age
        self.gender = gender
        self.symptoms = list(symptoms)
        self.lab_results = lab_results if lab_results is not None else {}
        self.diagnoses = diagnoses if diagnoses is not None else []
        self.treatments = treatments if treatments is not None else []
        self.outcome = outcome

    def update_record(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                if isinstance(getattr(self, key), list) and isinstance(value, list):
                    getattr(self, key).extend(value)
                elif isinstance(getattr(self, key), dict) and isinstance(value, dict):
                    getattr(self, key).update(value)
                else:
                    setattr(self, key, value)

class MedicalKnowledgeBase:
    def __init__(self):
        self._guidelines = {
            "fever": {"treatment": "antipyretics", "diagnosis_steps": ["check_infection", "check_inflammation"]},
            "cough": {"treatment": "cough_syrup", "diagnosis_steps": ["check_allergies", "check_respiratory_infection"]},
            "hypertension": {"treatment": "lifestyle_changes_medication", "diagnosis_steps": ["monitor_bp", "cardiac_assessment"]}
        }
        self._drug_interactions = {
            ("drug_A", "drug_B"): "contraindicated",
            ("drug_C", "drug_D"): "monitor_closely"
        }
        self._disease_info = {
            "common_cold": {"symptoms": ["cough", "sore_throat", "runny_nose"]},
            "flu": {"symptoms": ["fever", "cough", "body_ache"]}
        }

    def get_guideline(self, symptom_or_diagnosis):
        return self._guidelines.get(symptom_or_diagnosis.lower())

    def check_drug_interaction(self, drug1, drug2):
        return self._drug_interactions.get(tuple(sorted((drug1, drug2))))

    def get_disease_details(self, disease_name):
        return self._disease_info.get(disease_name.lower())

class ACDSSAgent:
    def __init__(self, knowledge_base):
        self.knowledge_base = knowledge_base
        self.internal_state = {
            "learning_rules": {},
            "confidence_scores": {},
            "past_recommendations": []
        }

    def _reasoning_engine(self, patient_data):
        possible_diagnoses = []
        recommended_treatments = []

        for symptom in patient_data.symptoms:
            guideline = self.knowledge_base.get_guideline(symptom)
            if guideline:
                recommended_treatments.append(guideline["treatment"])
                possible_diagnoses.extend(guideline["diagnosis_steps"])

        # Simple simulation of diagnostic logic
        if "fever" in patient_data.symptoms and patient_data.lab_results.get("white_blood_cells") and patient_data.lab_results["white_blood_cells"] > 10000:
            possible_diagnoses.append("bacterial_infection_suspected")
            recommended_treatments.append("antibiotics_consideration")

        # Filter unique diagnoses and treatments
        possible_diagnoses = list(set(d.replace('_suspected', '').replace('_consideration', '') for d in possible_diagnoses))
        recommended_treatments = list(set(t.replace('_consideration', '') for t in recommended_treatments))

        initial_diagnosis = possible_diagnoses[0] if possible_diagnoses else "unknown"
        initial_treatment = recommended_treatments[0] if recommended_treatments else "symptomatic_relief"

        return {"diagnosis": initial_diagnosis, "treatment": initial_treatment}

    def _process_feedback(self, feedback_type, feedback_data, patient_data):
        if feedback_type == "user_override":
            print(f"Agent received user override: {feedback_data}")
            # Example: If user overrides, reduce confidence in previous recommendation type
            last_rec = self.internal_state["past_recommendations"][-1] if self.internal_state["past_recommendations"] else None
            if last_rec and last_rec["diagnosis"] == feedback_data["original_diagnosis"]:
                self.internal_state["confidence_scores"][last_rec["diagnosis"]] = self.internal_state["confidence_scores"].get(last_rec["diagnosis"], 1.0) * 0.8
            patient_data.update_record(diagnoses=[feedback_data["new_diagnosis"]], treatments=[feedback_data["new_treatment"]])

        elif feedback_type == "patient_outcome":
            print(f"Agent received patient outcome feedback: {feedback_data}")
            # Example: Adjust learning rules based on outcome
            if feedback_data["success"]:
                self.internal_state["learning_rules"]["last_successful_treatment"] = patient_data.treatments[-1] if patient_data.treatments else None
                # Increase confidence for treatments that led to success
                for t in patient_data.treatments:
                    self.internal_state["confidence_scores"][t] = self.internal_state["confidence_scores"].get(t, 0.5) * 1.2
            else:
                self.internal_state["learning_rules"]["last_failed_treatment"] = patient_data.treatments[-1] if patient_data.treatments else None
                # Decrease confidence for treatments that led to failure
                for t in patient_data.treatments:
                    self.internal_state["confidence_scores"][t] = self.internal_state["confidence_scores"].get(t, 0.5) * 0.7
            patient_data.update_record(outcome=feedback_data["description"])

        elif feedback_type == "tool_output":
            print(f"Agent received tool output: {feedback_data}")
            # Example: Integrate lab results into patient record
            if feedback_data["tool_name"] == "lab_test":
                patient_data.update_record(lab_results=feedback_data["results"])

        elif feedback_type == "self_reflection":
            print(f"Agent performed self-reflection: {feedback_data}")
            # Example: Adjust confidence based on internal consistency checks
            if not feedback_data["consistent"]:
                print("Self-reflection identified inconsistency. Triggering self-correction.")
                # A real system would log this and potentially request more data or human review

    def _adapt_and_learn(self):
        # Conceptual adaptation: Adjusting internal 'rules' or 'weights'
        # In a real system, this would involve updating ML models, RL policies, etc.
        if "last_successful_treatment" in self.internal_state["learning_rules"]:
            print(f"Agent notes success with treatment: {self.internal_state['learning_rules']['last_successful_treatment']}")
        if "last_failed_treatment" in self.internal_state["learning_rules"]:
            print(f"Agent notes failure with treatment: {self.internal_state['learning_rules']['last_failed_treatment']}")

    def _self_correct(self, current_recommendation, patient_data):
        refined_recommendation = current_recommendation.copy()

        # Example of self-correction: check drug interactions for current treatments
        current_treatments = patient_data.treatments
        for i in range(len(current_treatments)):
            for j in range(i + 1, len(current_treatments)):
                interaction = self.knowledge_base.check_drug_interaction(current_treatments[i], current_treatments[j])
                if interaction == "contraindicated":
                    print(f"Self-correction: Identified contraindicated drugs {current_treatments[i]} and {current_treatments[j]}.")
                    refined_recommendation["treatment"] = f"Reconsider treatment due to interaction between {current_treatments[i]} and {current_treatments[j]}"
                    return refined_recommendation # Return after first critical correction

        # Example: if initial diagnosis has low confidence, suggest more tests
        if self.internal_state["confidence_scores"].get(current_recommendation["diagnosis"], 1.0) < 0.6:
            print(f"Self-correction: Low confidence in diagnosis '{current_recommendation['diagnosis']}'. Suggesting more diagnostics.")
            refined_recommendation["additional_steps"] = ["order_advanced_imaging", "consult_specialist"]

        return refined_recommendation

    def _simulate_tool_call(self, tool_name, patient_id, **kwargs):
        print(f"Simulating tool call: {tool_name} for patient {patient_id} with args {kwargs}")
        if tool_name == "lab_test":
            # Simulate lab results based on symptoms
            if "fever" in kwargs.get("symptoms", []):
                return {"tool_name": "lab_test", "results": {"white_blood_cells": 12000, "crp": 8.5}}
            return {"tool_name": "lab_test", "results": {"white_blood_cells": 7000, "crp": 2.1}}
        elif tool_name == "pharmacy_check":
            # Simulate pharmacy system check
            drug = kwargs.get("drug")
            if drug == "antibiotics_consideration" and "allergies" in kwargs.get("patient_history", []):
                return {"tool_name": "pharmacy_check", "status": "allergy_alert", "details": f"Patient allergic to {drug}"}
            return {"tool_name": "pharmacy_check", "status": "ok", "details": f"{drug} available"}
        return {"tool_name": tool_name, "status": "not_implemented", "details": "Simulated tool output"}

    def generate_recommendation(self, patient_data):
        print(f"\n--- Generating Recommendation for Patient {patient_data.patient_id} ---")
        initial_rec = self._reasoning_engine(patient_data)
        print(f"Initial Recommendation: {initial_rec}")

        # Simulate tool calls based on initial recommendation or patient data
        if "antibiotics_consideration" in initial_rec["treatment"]:
            tool_output = self._simulate_tool_call("pharmacy_check", patient_data.patient_id, drug="antibiotics_consideration", patient_history=patient_data.symptoms)
            self._process_feedback("tool_output", tool_output, patient_data)

        # Self-correction after initial recommendation and tool feedback
        current_recommendation = initial_rec.copy()
        if "antibiotics_consideration" in current_recommendation["treatment"] and any("allergy_alert" in f["details"] for f in patient_data.lab_results.values() if isinstance(f, dict) and "status" in f):
             print("Agent observes allergy alert from tool output, adjusting treatment.")
             current_recommendation["treatment"] = current_recommendation["treatment"].replace("antibiotics_consideration", "alternative_antibiotics_consideration")

        final_recommendation = self._self_correct(current_recommendation, patient_data)

        self.internal_state["past_recommendations"].append({
            "patient_id": patient_data.patient_id,
            "diagnosis": final_recommendation["diagnosis"],
            "treatment": final_recommendation["treatment"]
        })
        return final_recommendation

    def provide_feedback(self, feedback_type, feedback_data, patient_data):
        self._process_feedback(feedback_type, feedback_data, patient_data)
        self._adapt_and_learn()


# --- Simulation and Interaction Layer ---

def simulate_patient_progression(patient_record, days_passed):
    print(f"Simulating {days_passed} days of patient progression...")
    # In a real system, this would update based on actual medical outcomes.
    if patient_record.outcome and "improved" in patient_record.outcome.lower():
        patient_record.update_record(outcome="further_improved")
    elif patient_record.outcome and "worsened" in patient_record.outcome.lower():
        patient_record.update_record(outcome="critically_worsened")
    else:
        patient_record.update_record(outcome="stable")
    print(f"Patient {patient_record.patient_id} status: {patient_record.outcome}")

def main_simulation():
    print("Initializing ACDSS System...")
    knowledge_base = MedicalKnowledgeBase()
    agent = ACDSSAgent(knowledge_base)

    # --- Scenario 1: Initial Diagnosis and Treatment ---
    print("\n--- Scenario 1: Initial Diagnosis and Treatment ---")
    patient1 = PatientRecord("P001", 45, "male", ["fever", "cough"], lab_results={
        "white_blood_cells": 9000,
        "crp": 5.0
    })
    initial_recommendation = agent.generate_recommendation(patient1)
    print(f"Final Recommendation for P001: {initial_recommendation}")

    # Simulate healthcare professional acting on recommendation and patient outcome
    print("\n--- Feedback Loop 1: Treatment Success ---")
    agent.provide_feedback(
        "user_override",
        {"original_diagnosis": initial_recommendation["diagnosis"], "new_diagnosis": "viral_infection", "new_treatment": "rest_hydration"},
        patient1
    )
    simulate_patient_progression(patient1, 3)
    agent.provide_feedback(
        "patient_outcome",
        {"success": True, "description": "Patient responded well to rest and hydration, fever subsided."},
        patient1
    )

    # --- Scenario 2: Iterative Refinement and Self-Correction ---
    print("\n--- Scenario 2: Iterative Refinement and Self-Correction ---")
    patient2 = PatientRecord("P002", 60, "female", ["hypertension", "headache"], lab_results={
        "blood_pressure": "160/100",
        "cholesterol": 220
    })
    rec_p2_1 = agent.generate_recommendation(patient2)
    print(f"Recommendation 1 for P002: {rec_p2_1}")

    # Simulate tool call (e.g., further lab tests requested by agent)
    lab_tool_output = agent._simulate_tool_call("lab_test", patient2.patient_id, symptoms=patient2.symptoms)
    agent.provide_feedback("tool_output", lab_tool_output, patient2)

    # Agent generates new recommendation with new lab data
    rec_p2_2 = agent.generate_recommendation(patient2)
    print(f"Recommendation 2 for P002 (after tool output): {rec_p2_2}")

    # Simulate a negative outcome and agent's self-correction
    print("\n--- Feedback Loop 2: Treatment Failure and Self-Correction ---")
    simulate_patient_progression(patient2, 7)
    agent.provide_feedback(
        "patient_outcome",
        {"success": False, "description": "Patient's blood pressure remained high, experiencing dizziness."},
        patient2
    )
    # After negative feedback, agent might refine its approach in next recommendation call
    rec_p2_3 = agent.generate_recommendation(patient2)
    print(f"Recommendation 3 for P002 (after negative outcome): {rec_p2_3}")
    if "additional_steps" in rec_p2_3:
        print(f"Agent suggests additional steps: {rec_p2_3['additional_steps']}")

    # --- Scenario 3: Drug Interaction Detection ---
    print("\n--- Scenario 3: Drug Interaction Detection ---")
    patient3 = PatientRecord("P003", 70, "male", ["fever"], treatments=["drug_A", "drug_B"])
    rec_p3 = agent.generate_recommendation(patient3)
    print(f"Recommendation for P003: {rec_p3}")
    if "Reconsider treatment" in rec_p3["treatment"]:
        print("Agent successfully identified and flagged a drug interaction.")

if __name__ == "__main__":
    main_simulation()