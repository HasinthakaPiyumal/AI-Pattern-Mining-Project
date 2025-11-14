
import json

class PatientMemory:
    def __init__(self, data_path="./data/patient_data.json"):
        self.data_path = data_path
        self.patients = self._load_data()

    def _load_data(self):
        try:
            with open(self.data_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def _save_data(self):
        with open(self.data_path, 'w') as f:
            json.dump(self.patients, f, indent=4)

    def get_patient_history(self, patient_id: str) -> dict:
        """Retrieves the full medical history for a given patient ID."""
        return self.patients.get(patient_id, {})

    def update_patient_history(self, patient_id: str, new_data: dict):
        """Updates or adds new data to a patient's record."""
        if patient_id not in self.patients:
            self.patients[patient_id] = {"id": patient_id, "history": []}
        self.patients[patient_id]["history"].append(new_data)
        self._save_data()
        return {"status": "success", "patient_id": patient_id}

class MedicalKnowledgeBase:
    def __init__(self, data_path="./data/medical_knowledge.json"):
        self.data_path = data_path
        self.knowledge = self._load_data()

    def _load_data(self):
        try:
            with open(self.data_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def search_knowledge(self, query: str) -> str:
        """Searches the medical knowledge base for relevant information."""
        query = query.lower()
        results = []
        for topic, info in self.knowledge.items():
            if query in topic.lower() or query in info.lower():
                results.append(f"Topic: {topic}\nInfo: {info}")
        return "\n---\n".join(results) if results else "No relevant information found."
