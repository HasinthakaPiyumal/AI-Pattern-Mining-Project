import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import random


class PatientData(BaseModel):
    patient_id: str
    age: int
    gender: str
    medical_history: List[str]
    current_medications: List[str]


class EligibilityCriteria(BaseModel):
    min_age: int
    max_age: int
    required_conditions: Optional[List[str]] = None
    excluded_conditions: Optional[List[str]] = None


class EligiblePatient(BaseModel):
    patient_id: str
    assigned_arm: Optional[str] = None


class AdverseEvent(BaseModel):
    event_id: str
    patient_id: str
    description: str
    severity: str
    date_reported: str


class StatisticalResult(BaseModel):
    analysis_type: str
    results: Dict[str, Any]
    conclusion: str


class ClinicalReport(BaseModel):
    report_id: str
    trial_name: str
    sections: Dict[str, Any]
    generated_date: str


class PatientDataIngestionTool:
    def __init__(self):
        self.patient_database = pd.DataFrame(columns=list(PatientData.model_fields.keys()))

    def ingest_data(self, patient_info: PatientData) -> bool:
        try:
            new_data = pd.DataFrame([patient_info.model_dump()])
            self.patient_database = pd.concat([self.patient_database, new_data], ignore_index=True)
            return True
        except Exception:
            return False

    def get_all_patient_data(self) -> List[PatientData]:
        return [PatientData(**row.to_dict()) for _, row in self.patient_database.iterrows()]


class EligibilityScreeningTool:
    def screen_patient(self, patient_data: PatientData, criteria: EligibilityCriteria) -> bool:
        if not (criteria.min_age <= patient_data.age <= criteria.max_age):
            return False

        if criteria.required_conditions:
            if not all(cond in patient_data.medical_history for cond in criteria.required_conditions):
                return False

        if criteria.excluded_conditions:
            if any(cond in patient_data.medical_history for cond in criteria.excluded_conditions):
                return False
        return True


class TreatmentAssignmentTool:
    def assign_treatment(self, eligible_patients: List[EligiblePatient], treatment_arms: List[str]) -> List[EligiblePatient]:
        assigned_patients = []
        for patient in eligible_patients:
            patient.assigned_arm = random.choice(treatment_arms)
            assigned_patients.append(patient)
        return assigned_patients


class AdverseEventMonitoringTool:
    def __init__(self):
        self.adverse_events = []

    def record_event(self, event: AdverseEvent) -> bool:
        self.adverse_events.append(event)
        return True

    def get_events_for_patient(self, patient_id: str) -> List[AdverseEvent]:
        return [event for event in self.adverse_events if event.patient_id == patient_id]

    def get_all_events(self) -> List[AdverseEvent]:
        return self.adverse_events


class StatisticalAnalysisTool:
    def perform_analysis(self, patient_assignments: List[EligiblePatient], adverse_events: List[AdverseEvent]) -> StatisticalResult:
        num_patients = len(patient_assignments)
        num_events = len(adverse_events)
        event_rate = num_events / num_patients if num_patients > 0 else 0

        assignments_df = pd.DataFrame([p.model_dump() for p in patient_assignments])
        events_df = pd.DataFrame([e.model_dump() for e in adverse_events])

        results = {
            "total_patients": num_patients,
            "total_adverse_events": num_events,
            "adverse_event_rate": event_rate,
        }

        if not assignments_df.empty:
            results["patients_per_arm"] = assignments_df["assigned_arm"].value_counts().to_dict()

        conclusion = "Preliminary analysis completed." if num_patients < 100 else "Comprehensive analysis completed."
        return StatisticalResult(
            analysis_type="Interim Safety Analysis",
            results=results,
            conclusion=conclusion
        )


class ReportGenerationTool:
    def generate_report(self, trial_name: str, statistical_results: StatisticalResult, patient_data: List[PatientData], adverse_events: List[AdverseEvent]) -> ClinicalReport:
        sections = {
            "summary": f"Report for clinical trial: {trial_name}.",
            "statistical_findings": statistical_results.model_dump(),
            "patient_count": len(patient_data),
            "adverse_events_summary": len(adverse_events)
        }
        return ClinicalReport(
            report_id=f"REPORT-{random.randint(1000, 9999)}",
            trial_name=trial_name,
            sections=sections,
            generated_date=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        )


class AIController:
    def __init__(self):
        self.patient_ingestion_tool = PatientDataIngestionTool()
        self.eligibility_screening_tool = EligibilityScreeningTool()
        self.treatment_assignment_tool = TreatmentAssignmentTool()
        self.adverse_event_monitoring_tool = AdverseEventMonitoringTool()
        self.statistical_analysis_tool = StatisticalAnalysisTool()
        self.report_generation_tool = ReportGenerationTool()

        self.current_trial_patients: List[EligiblePatient] = []
        self.all_patients_data: List[PatientData] = []
        self.trial_adverse_events: List[AdverseEvent] = []
        self.trial_name: Optional[str] = None

    def start_new_trial(self, trial_name: str, eligibility_criteria: EligibilityCriteria, treatment_arms: List[str]):
        self.trial_name = trial_name
        self.eligibility_criteria = eligibility_criteria
        self.treatment_arms = treatment_arms
        self.current_trial_patients = []
        self.all_patients_data = []
        self.trial_adverse_events = []
        print(f"\n--- Started New Clinical Trial: {trial_name} ---")

    def process_new_patient_data(self, patient_data: PatientData):
        print(f"Ingesting patient data for {patient_data.patient_id}...")
        if self.patient_ingestion_tool.ingest_data(patient_data):
            self.all_patients_data = self.patient_ingestion_tool.get_all_patient_data()
            print(f"Patient {patient_data.patient_id} data ingested.")
            return True
        return False

    def screen_and_assign_patients(self):
        print("Screening and assigning eligible patients...")
        eligible_patients_data = []
        for patient in self.all_patients_data:
            if self.eligibility_screening_tool.screen_patient(patient, self.eligibility_criteria):
                eligible_patients_data.append(EligiblePatient(patient_id=patient.patient_id))

        if eligible_patients_data:
            self.current_trial_patients = self.treatment_assignment_tool.assign_treatment(eligible_patients_data, self.treatment_arms)
            print(f"Assigned {len(self.current_trial_patients)} patients to treatment arms.")
        else:
            print("No eligible patients found for assignment.")
        return self.current_trial_patients

    def record_adverse_event(self, event: AdverseEvent):
        print(f"Recording adverse event for patient {event.patient_id}...")
        if self.adverse_event_monitoring_tool.record_event(event):
            self.trial_adverse_events = self.adverse_event_monitoring_tool.get_all_events()
            print("Adverse event recorded.")
            return True
        return False

    def perform_and_report_analysis(self, analysis_type: str = "Interim"):
        if not self.trial_name:
            print("No trial active. Start a trial first.")
            return None

        print(f"Performing {analysis_type} statistical analysis...")
        statistical_results = self.statistical_analysis_tool.perform_analysis(
            self.current_trial_patients, self.trial_adverse_events
        )
        print("Statistical analysis complete.")

        print("Generating clinical report...")
        clinical_report = self.report_generation_tool.generate_report(
            self.trial_name,
            statistical_results,
            self.all_patients_data,
            self.trial_adverse_events
        )
        print("Report generated.")
        return clinical_report


if __name__ == "__main__":
    controller = AIController()

    # Define trial parameters
    trial_criteria = EligibilityCriteria(
        min_age=18,
        max_age=65,
        required_conditions=["Hypertension"],
        excluded_conditions=["Diabetes"]
    )
    treatment_arms = ["Placebo", "Drug A", "Drug B"]

    controller.start_new_trial("Phase 3 Hypertension Study", trial_criteria, treatment_arms)

    # Simulate patient data ingestion
    patient1_data = PatientData(
        patient_id="P001", age=45, gender="Male", medical_history=["Hypertension", "Asthma"], current_medications=["Lisinopril"]
    )
    patient2_data = PatientData(
        patient_id="P002", age=70, gender="Female", medical_history=["Hypertension"], current_medications=["Amlodipine"]
    )
    patient3_data = PatientData(
        patient_id="P003", age=30, gender="Female", medical_history=["Headache"], current_medications=[]
    )
    patient4_data = PatientData(
        patient_id="P004", age=55, gender="Male", medical_history=["Hypertension", "Diabetes"], current_medications=["Metformin"]
    )
    patient5_data = PatientData(
        patient_id="P005", age=40, gender="Male", medical_history=["Hypertension", "Cholesterol"], current_medications=["Atorvastatin"]
    )

    controller.process_new_patient_data(patient1_data)
    controller.process_new_patient_data(patient2_data)
    controller.process_new_patient_data(patient3_data)
    controller.process_new_patient_data(patient4_data)
    controller.process_new_patient_data(patient5_data)

    # Screen and assign patients to treatment arms
    assigned_patients = controller.screen_and_assign_patients()
    print("\nAssigned Patients Details:")
    for p in assigned_patients:
        print(f" - Patient {p.patient_id}: Arm {p.assigned_arm}")

    # Simulate adverse events
    ae1 = AdverseEvent(
        event_id="AE001", patient_id="P001", description="Mild headache", severity="Mild", date_reported="2023-10-26"
    )
    ae2 = AdverseEvent(
        event_id="AE002", patient_id="P005", description="Dizziness", severity="Moderate", date_reported="2023-10-27"
    )
    controller.record_adverse_event(ae1)
    controller.record_adverse_event(ae2)

    # Perform interim analysis and generate report
    interim_report = controller.perform_and_report_analysis("Interim")
    if interim_report:
        print("\n--- Interim Report ---")
        print(f"Report ID: {interim_report.report_id}")
        print(f"Trial Name: {interim_report.trial_name}")
        print(f"Summary: {interim_report.sections['summary']}")
        print(f"Statistical Findings: {interim_report.sections['statistical_findings']['results']}")
        print(f"Conclusion: {interim_report.sections['statistical_findings']['conclusion']}")

    # Simulate more events or data over time, then final analysis
    ae3 = AdverseEvent(
        event_id="AE003", patient_id="P001", description="Rash", severity="Moderate", date_reported="2023-11-15"
    )
    controller.record_adverse_event(ae3)

    final_report = controller.perform_and_report_analysis("Final")
    if final_report:
        print("\n--- Final Report ---")
        print(f"Report ID: {final_report.report_id}")
        print(f"Trial Name: {final_report.trial_name}")
        print(f"Summary: {final_report.sections['summary']}")
        print(f"Statistical Findings: {final_report.sections['statistical_findings']['results']}")
        print(f"Conclusion: {final_report.sections['statistical_findings']['conclusion']}")
        print(f"Total Patients in Data: {final_report.sections['patient_count']}")
        print(f"Total Adverse Events: {final_report.sections['adverse_events_summary']}")
