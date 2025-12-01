from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from abc import ABC, abstractmethod


# 1. Data Models (using Pydantic)
class PatientData(BaseModel):
    patient_id: str
    symptoms: List[str]
    medical_history: List[str]
    lab_results: Dict[str, str]
    imaging_reports: Dict[str, str]
    current_medications: List[str]


class AgentMessage(BaseModel):
    sender: str
    recipient: str
    message_type: str
    content: Any
    timestamp: datetime = datetime.now()


class Diagnosis(BaseModel):
    agent_name: str
    diagnosis: str
    confidence: float
    evidence: List[str]


class TreatmentRecommendation(BaseModel):
    agent_name: str
    recommendation: str
    dosage: Optional[str]
    duration: Optional[str]
    potential_interactions: Optional[List[str]]


# 2. Core Components
class BaseAgent(ABC):
    def __init__(self, name: str, role: str, coordinator: "CoordinatorAgent"):
        self.name = name
        self.role = role
        self.coordinator = coordinator
        self.inbox: List[AgentMessage] = []

    def receive_message(self, message: AgentMessage):
        self.inbox.append(message)

    def send_message(self, recipient: str, message_type: str, content: Any):
        message = AgentMessage(
            sender=self.name,
            recipient=recipient,
            message_type=message_type,
            content=content,
        )
        self.coordinator.route_message(message)

    @abstractmethod
    def _process_task(self, patient_data: PatientData) -> List[Union[Diagnosis, TreatmentRecommendation]]:
        pass

    def process_inbox(self):
        processed_messages = []
        for message in self.inbox:
            if message.message_type == "task_assignment":
                # Assuming content is PatientData for task assignments
                results = self._process_task(PatientData(**message.content))
                for result in results:
                    self.send_message("CoordinatorAgent", "finding", result.dict())
            processed_messages.append(message)
        self.inbox = [m for m in self.inbox if m not in processed_messages]


class GeneralPractitionerAgent(BaseAgent):
    def __init__(self, coordinator: "CoordinatorAgent"):
        super().__init__("GeneralPractitionerAgent", "General Practitioner", coordinator)

    def _process_task(self, patient_data: PatientData) -> List[Union[Diagnosis, TreatmentRecommendation]]:
        initial_diagnosis = "General check-up indicates need for specialist consultation."
        if "fever" in patient_data.symptoms:
            initial_diagnosis = "Patient presents with fever, likely infection. Recommend blood tests."

        return [
            Diagnosis(
                agent_name=self.name,
                diagnosis=initial_diagnosis,
                confidence=0.7,
                evidence=patient_data.symptoms,
            )
        ]


class RadiologyAgent(BaseAgent):
    def __init__(self, coordinator: "CoordinatorAgent"):
        super().__init__("RadiologyAgent", "Radiologist", coordinator)

    def _process_task(self, patient_data: PatientData) -> List[Union[Diagnosis, TreatmentRecommendation]]:
        if "MRI_brain" in patient_data.imaging_reports:
            report = patient_data.imaging_reports["MRI_brain"]
            if "lesion detected" in report:
                return [
                    Diagnosis(
                        agent_name=self.name,
                        diagnosis="Brain lesion detected based on MRI.",
                        confidence=0.9,
                        evidence=[report],
                    )
                ]
        return []


class PathologyAgent(BaseAgent):
    def __init__(self, coordinator: "CoordinatorAgent"):
        super().__init__("PathologyAgent", "Pathologist", coordinator)

    def _process_task(self, patient_data: PatientData) -> List[Union[Diagnosis, TreatmentRecommendation]]:
        if "blood_test" in patient_data.lab_results:
            result = patient_data.lab_results["blood_test"]
            if "high cholesterol" in result:
                return [
                    Diagnosis(
                        agent_name=self.name,
                        diagnosis="High cholesterol detected from blood test.",
                        confidence=0.8,
                        evidence=[result],
                    )
                ]
        return []


class SpecialistAgent(BaseAgent):
    def __init__(self, specialist_type: str, coordinator: "CoordinatorAgent"):
        super().__init__(f"{specialist_type}Agent", specialist_type, coordinator)
        self.specialist_type = specialist_type

    def _process_task(self, patient_data: PatientData) -> List[Union[Diagnosis, TreatmentRecommendation]]:
        if self.specialist_type == "Cardiology" and "chest pain" in patient_data.symptoms:
            return [
                Diagnosis(
                    agent_name=self.name,
                    diagnosis="Potential cardiac issue. Further investigation recommended.",
                    confidence=0.85,
                    evidence=patient_data.symptoms,
                ),
                TreatmentRecommendation(
                    agent_name=self.name,
                    recommendation="Referral to cardiologist.",
                    dosage=None,
                    duration=None,
                    potential_interactions=None,
                ),
            ]
        return []


class PharmacologyAgent(BaseAgent):
    def __init__(self, coordinator: "CoordinatorAgent"):
        super().__init__("PharmacologyAgent", "Pharmacologist", coordinator)

    def _process_task(self, patient_data: PatientData) -> List[Union[Diagnosis, TreatmentRecommendation]]:
        recommendations = []
        if "high cholesterol" in str(patient_data.lab_results.values()):
            recommendations.append(
                TreatmentRecommendation(
                    agent_name=self.name,
                    recommendation="Suggest statin therapy.",
                    dosage="20mg daily",
                    duration="Long term",
                    potential_interactions=["grapefruit juice"],
                )
            )
        if "lesion detected" in str(patient_data.imaging_reports.values()):
            recommendations.append(
                TreatmentRecommendation(
                    agent_name=self.name,
                    recommendation="Consider medication for lesion management.",
                    dosage="As prescribed by specialist",
                    duration="Varies",
                    potential_interactions=None,
                )
            )
        return recommendations


class CoordinatorAgent(BaseAgent):
    def __init__(self, name: str = "CoordinatorAgent", role: str = "Coordinator"):
        super().__init__(name, role, self)  # Coordinator acts as its own coordinator for message routing
        self.agents: Dict[str, BaseAgent] = {}
        self.patient_cases: Dict[str, dict] = {}

    def register_agent(self, agent: BaseAgent):
        self.agents[agent.name] = agent

    def route_message(self, message: AgentMessage):
        if message.recipient in self.agents:
            self.agents[message.recipient].receive_message(message)
        elif message.recipient == self.name:  # Message for the coordinator itself
            self.receive_message(message)
        else:
            print(f"Error: Recipient {message.recipient} not found.")

    def start_diagnosis_workflow(self, patient_data: PatientData):
        print(f"\n--- Starting diagnosis workflow for patient {patient_data.patient_id} ---")
        self.patient_cases[patient_data.patient_id] = {
            "patient_data": patient_data,
            "diagnoses": [],
            "treatment_recommendations": [],
            "status": "in_progress",
        }

        # Initial assessment by GP
        print("Coordinator: Assigning initial assessment to GP Agent.")
        self.send_message(
            "GeneralPractitionerAgent", "task_assignment", patient_data.dict()
        )

        self._process_all_agents_inboxes()

        # Collect initial findings and assign to relevant specialists
        for message in self.inbox:
            if message.message_type == "finding" and message.sender == "GeneralPractitionerAgent":
                self.patient_cases[patient_data.patient_id]["diagnoses"].append(
                    Diagnosis(**message.content)
                )
                print(f"Coordinator received GP finding: {message.content}")

        # Assign tasks to other agents based on patient data
        print("Coordinator: Assigning tasks to Radiology, Pathology, and Specialists.")
        self.send_message("RadiologyAgent", "task_assignment", patient_data.dict())
        self.send_message("PathologyAgent", "task_assignment", patient_data.dict())
        self.send_message("CardiologyAgent", "task_assignment", patient_data.dict())
        # Potentially other specialists here

        self._process_all_agents_inboxes()

        # Collect all findings and recommendations
        for message in self.inbox:
            if message.message_type == "finding":
                if "diagnosis" in message.content:
                    self.patient_cases[patient_data.patient_id]["diagnoses"].append(
                        Diagnosis(**message.content)
                    )
                elif "recommendation" in message.content:
                    self.patient_cases[patient_data.patient_id]["treatment_recommendations"].append(
                        TreatmentRecommendation(**message.content)
                    )
                print(f"Coordinator received finding from {message.sender}: {message.content}")
        self.inbox = [] # Clear coordinator's inbox after processing

        # Final aggregation and potential conflict resolution (simplified)
        final_diagnoses = self.patient_cases[patient_data.patient_id]["diagnoses"]
        final_treatments = self.patient_cases[patient_data.patient_id]["treatment_recommendations"]

        print("\n--- Final Patient Summary ---")
        print(f"Patient ID: {patient_data.patient_id}")
        print("Diagnoses:")
        for diag in final_diagnoses:
            print(f"  - [{diag.agent_name}] {diag.diagnosis} (Confidence: {diag.confidence})")
        print("Treatment Recommendations:")
        for treat in final_treatments:
            print(f"  - [{treat.agent_name}] {treat.recommendation}")
        print(f"--- Workflow for {patient_data.patient_id} Complete ---")
        self.patient_cases[patient_data.patient_id]["status"] = "completed"

    def _process_all_agents_inboxes(self):
        # A simple mechanism to ensure messages are processed after being sent.
        # In a real-time system, this would be event-driven or threaded.
        for agent_name, agent_obj in self.agents.items():
            if agent_name != self.name: # Don't process coordinator's inbox here
                agent_obj.process_inbox()
        self.process_inbox() # Process coordinator's own inbox as well


# Example Usage:
if __name__ == "__main__":
    # Initialize Coordinator
    coordinator = CoordinatorAgent()

    # Initialize Agents and Register with Coordinator
    gp_agent = GeneralPractitionerAgent(coordinator)
    radiology_agent = RadiologyAgent(coordinator)
    pathology_agent = PathologyAgent(coordinator)
    cardiology_agent = SpecialistAgent("Cardiology", coordinator)
    pharmacology_agent = PharmacologyAgent(coordinator)

    coordinator.register_agent(gp_agent)
    coordinator.register_agent(radiology_agent)
    coordinator.register_agent(pathology_agent)
    coordinator.register_agent(cardiology_agent)
    coordinator.register_agent(pharmacology_agent)

    # Create a Patient Case
    patient1_data = PatientData(
        patient_id="P001",
        symptoms=["fever", "headache", "chest pain"],
        medical_history=["hypertension"],
        lab_results={
            "blood_test": "elevated white blood cell count, high cholesterol"
        },
        imaging_reports={
            "MRI_brain": "no significant findings",
            "X_ray_chest": "minor lung congestion"
        },
        current_medications=["Lisinopril"]
    )

    patient2_data = PatientData(
        patient_id="P002",
        symptoms=["fatigue"],
        medical_history=[],
        lab_results={
            "blood_test": "normal"
        },
        imaging_reports={
            "MRI_brain": "lesion detected"
        },
        current_medications=[]
    )

    # Start the workflow for Patient 1
    coordinator.start_diagnosis_workflow(patient1_data)

    # Start the workflow for Patient 2
    coordinator.start_diagnosis_workflow(patient2_data)
