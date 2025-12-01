import pandas as pd
from datetime import datetime

class Perceiver:
    def __init__(self):
        pass

    def process_user_feedback(self, user_feedback):
        keywords = []
        if "pain" in user_feedback.lower():
            keywords.append("pain")
        if "tired" in user_feedback.lower():
            keywords.append("tired")
        if "better" in user_feedback.lower():
            keywords.append("feeling better")
        return {"user_keywords": keywords, "raw_text": user_feedback}

    def process_wearable_data(self, wearable_data):
        df = pd.DataFrame(wearable_data)
        summary = {
            "avg_glucose": df["glucose"].mean() if not df.empty else None,
            "max_bp_sys": df["blood_pressure_systolic"].max() if not df.empty else None,
            "min_bp_dia": df["blood_pressure_diastolic"].min() if not df.empty else None,
            "avg_heart_rate": df["heart_rate"].mean() if not df.empty else None,
            "glucose_anomalies": df[df["glucose"] > 180]["glucose"].tolist() if not df.empty else [], # Example high glucose
            "bp_anomalies": df[(df["blood_pressure_systolic"] > 140) | (df["blood_pressure_diastolic"] > 90)].to_dict('records') if not df.empty else []
        }
        return summary

    def process_ehr_data(self, ehr_data):
        summary = {
            "latest_a1c": ehr_data.get("lab_results", {}).get("hba1c"),
            "current_medications": ehr_data.get("medications", []),
            "critical_lab_flags": []
        }
        if summary["latest_a1c"] is not None and summary["latest_a1c"] > 7.0:
            summary["critical_lab_flags"].append("High HbA1c")
        return summary

    def integrate_knowledge(self, medical_knowledge_updates):
        return {"new_knowledge_available": bool(medical_knowledge_updates), "updates": medical_knowledge_updates}

    def perceive(self, user_feedback, wearable_data, ehr_data, medical_knowledge_updates):
        processed_user = self.process_user_feedback(user_feedback)
        processed_wearable = self.process_wearable_data(wearable_data)
        processed_ehr = self.process_ehr_data(ehr_data)
        processed_knowledge = self.integrate_knowledge(medical_knowledge_updates)

        summarized_feedback = {
            "timestamp": datetime.now().isoformat(),
            "user_summary": processed_user,
            "wearable_summary": processed_wearable,
            "ehr_summary": processed_ehr,
            "knowledge_summary": processed_knowledge
        }
        return summarized_feedback

class Controller:
    def __init__(self):
        pass

    def evaluate_situation(self, summarized_feedback):
        evaluation = "Normal. "
        if summarized_feedback["user_summary"]["user_keywords"]:
            evaluation += f"User mentioned: {', '.join(summarized_feedback['user_summary']['user_keywords'])}. "
        
        if summarized_feedback["wearable_summary"]["glucose_anomalies"]:
            evaluation += f"High glucose readings detected: {summarized_feedback['wearable_summary']['glucose_anomalies']}. "
        if summarized_feedback["wearable_summary"]["bp_anomalies"]:
            evaluation += f"Blood pressure anomalies detected. "
        
        if "High HbA1c" in summarized_feedback["ehr_summary"]["critical_lab_flags"]:
            evaluation += "EHR shows high HbA1c. "

        return evaluation.strip()

    def generate_advice(self, summarized_feedback):
        advice = "Maintain your current routine."
        if summarized_feedback["wearable_summary"]["glucose_anomalies"]:
            advice = "Consider checking your diet and activity. Consult your doctor if high readings persist."
        elif "High HbA1c" in summarized_feedback["ehr_summary"]["critical_lab_flags"]:
            advice = "Your recent HbA1c is high. It's crucial to review your medication and lifestyle with your doctor."
        elif "pain" in summarized_feedback["user_summary"]["user_keywords"]:
            advice = "Please describe your pain in more detail, or consider contacting your healthcare provider."
        return advice

    def flag_for_intervention(self, summarized_feedback):
        if summarized_feedback["wearable_summary"]["glucose_anomalies"] and len(summarized_feedback["wearable_summary"]["glucose_anomalies"]) > 2:
            return True, "Multiple high glucose readings. Recommend immediate medical review."
        if summarized_feedback["wearable_summary"]["bp_anomalies"] and len(summarized_feedback["wearable_summary"]["bp_anomalies"]) > 1:
            return True, "Consistent blood pressure anomalies. Recommend medical review."
        if "High HbA1c" in summarized_feedback["ehr_summary"]["critical_lab_flags"]:
            return True, "Critically high HbA1c. Recommend doctor's appointment."
        return False, "No immediate intervention required."

# Data Simulation/Input Layer
def simulate_data():
    user_feedback_1 = "I've been feeling a bit tired lately and my sugar seems high."
    user_feedback_2 = "I feel much better today."
    user_feedback_3 = "I have a headache."

    wearable_data_1 = [
        {"timestamp": "2023-10-26T08:00:00", "glucose": 150, "blood_pressure_systolic": 120, "blood_pressure_diastolic": 80, "heart_rate": 70},
        {"timestamp": "2023-10-26T12:00:00", "glucose": 210, "blood_pressure_systolic": 130, "blood_pressure_diastolic": 85, "heart_rate": 75},
        {"timestamp": "2023-10-26T18:00:00", "glucose": 195, "blood_pressure_systolic": 145, "blood_pressure_diastolic": 92, "heart_rate": 80}
    ]
    wearable_data_2 = [
        {"timestamp": "2023-10-27T08:00:00", "glucose": 110, "blood_pressure_systolic": 118, "blood_pressure_diastolic": 78, "heart_rate": 68}
    ]

    ehr_data_1 = {
        "patient_id": "P001",
        "lab_results": {"hba1c": 8.2, "creatinine": 0.9},
        "medications": ["Metformin", "Lisinopril"]
    }
    ehr_data_2 = {
        "patient_id": "P001",
        "lab_results": {"hba1c": 6.8, "creatinine": 0.95},
        "medications": ["Metformin", "Lisinopril", "Aspirin"]
    }

    medical_knowledge_updates_1 = {"new_guideline": "ADA recommends new glucose targets for elderly patients."}
    medical_knowledge_updates_2 = {}

    return user_feedback_1, wearable_data_1, ehr_data_1, medical_knowledge_updates_1, \
           user_feedback_2, wearable_data_2, ehr_data_2, medical_knowledge_updates_2, \
           user_feedback_3

if __name__ == "__main__":
    perceiver = Perceiver()
    controller = Controller()

    uf1, wd1, ehr1, mku1, uf2, wd2, ehr2, mku2, uf3 = simulate_data()

    print("\n--- Scenario 1: Initial feedback with high readings ---")
    summarized_data_1 = perceiver.perceive(uf1, wd1, ehr1, mku1)
    print("Perceiver Output:", summarized_data_1)
    evaluation_1 = controller.evaluate_situation(summarized_data_1)
    advice_1 = controller.generate_advice(summarized_data_1)
    flag_1, msg_1 = controller.flag_for_intervention(summarized_data_1)
    print("Controller Evaluation:", evaluation_1)
    print("Controller Advice:", advice_1)
    print(f"Intervention Flagged: {flag_1}, Message: {msg_1}")

    print("\n--- Scenario 2: Improved readings ---")
    summarized_data_2 = perceiver.perceive(uf2, wd2, ehr2, mku2)
    print("Perceiver Output:", summarized_data_2)
    evaluation_2 = controller.evaluate_situation(summarized_data_2)
    advice_2 = controller.generate_advice(summarized_data_2)
    flag_2, msg_2 = controller.flag_for_intervention(summarized_data_2)
    print("Controller Evaluation:", evaluation_2)
    print("Controller Advice:", advice_2)
    print(f"Intervention Flagged: {flag_2}, Message: {msg_2}")

    print("\n--- Scenario 3: User headache ---")
    summarized_data_3 = perceiver.perceive(uf3, [], {}, {})
    print("Perceiver Output:", summarized_data_3)
    evaluation_3 = controller.evaluate_situation(summarized_data_3)
    advice_3 = controller.generate_advice(summarized_data_3)
    flag_3, msg_3 = controller.flag_for_intervention(summarized_data_3)
    print("Controller Evaluation:", evaluation_3)
    print("Controller Advice:", advice_3)
    print(f"Intervention Flagged: {flag_3}, Message: {msg_3}")