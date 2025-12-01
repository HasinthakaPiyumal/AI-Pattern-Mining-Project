import streamlit as st
import openai
import json
import re
import os
from pydantic import BaseModel, Field
from typing import List, Optional

# --- 1. Pydantic Models for Structured Output ---
class Medication(BaseModel):
    name: str = Field(..., description="Name of the medication")
    dosage: str = Field(..., description="Dosage of the medication (e.g., '10mg', 'once daily')")
    frequency: str = Field(..., description="How often the medication should be taken")
    start_date: Optional[str] = Field(None, description="Start date for the medication (e.g., '2023-10-26')")
    end_date: Optional[str] = Field(None, description="End date for the medication (e.g., '2023-11-26')")

class Appointment(BaseModel):
    type: str = Field(..., description="Type of appointment (e.g., 'Follow-up', 'Consultation')")
    date: str = Field(..., description="Date of the appointment (e.g., '2023-11-15')")
    time: str = Field(..., description="Time of the appointment (e.g., '10:00 AM')")
    department: Optional[str] = Field(None, description="Department for the appointment")
    doctor: Optional[str] = Field(None, description="Doctor for the appointment")

class DietaryRecommendation(BaseModel):
    restriction_type: str = Field(..., description="Type of dietary restriction or recommendation (e.g., 'Low Sodium', 'Diabetic Diet')")
    specific_foods_to_avoid: Optional[List[str]] = Field(None, description="List of specific foods to avoid")
    recommendations: Optional[str] = Field(None, description="General dietary recommendations")

class ExercisePlan(BaseModel):
    type: str = Field(..., description="Type of exercise (e.g., 'Walking', 'Strength Training')")
    duration: str = Field(..., description="Duration of the exercise (e.g., '30 minutes')")
    frequency: str = Field(..., description="Frequency of the exercise (e.g., '3 times a week')")

class CarePlan(BaseModel):
    patient_id: str
    diagnosis: str
    medications: List[Medication] = []
    appointments: List[Appointment] = []
    dietary_recommendations: List[DietaryRecommendation] = []
    exercise_plans: List[ExercisePlan] = []
    notes: Optional[str] = None

# --- 2. LLM Interaction Function ---
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_natural_language_care_plan(patient_data: dict) -> str:
    prompt = f"""Generate a detailed personalized patient care plan in natural language for the following patient. 
    Organize the plan into clear sections: 'Medication Schedule:', 'Appointment Schedule:', 'Dietary Recommendations:', 'Exercise Plan:', and 'General Notes:'.

    Patient ID: {patient_data['patient_id']}
    Diagnosis: {patient_data['diagnosis']}
    Medical History Keywords: {patient_data['medical_history']}
    Patient Preferences/Concerns: {patient_data['preferences']}

    Care Plan:
    """

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful medical assistant that generates comprehensive patient care plans."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating care plan from LLM: {e}")
        return ""

# --- 3. Post-processing / Structured Extraction Function (Rule-based/Regex) ---
def extract_structured_care_plan(patient_id: str, diagnosis: str, nl_plan: str) -> CarePlan:
    medications = []
    appointments = []
    dietary_recommendations = []
    exercise_plans = []
    general_notes = ""

    # Split the plan into sections based on defined headings
    sections = re.split(r'(Medication Schedule:|Appointment Schedule:|Dietary Recommendations:|Exercise Plan:|General Notes:)', nl_plan)
    
    current_section = None
    section_content = {}

    for i in range(1, len(sections)):
        if i % 2 == 1: # This is a heading
            current_section = sections[i].strip().replace(':', '')
        else: # This is content for the previous heading
            if current_section:
                section_content[current_section] = sections[i].strip()

    # Parse Medication Schedule
    if "Medication Schedule" in section_content:
        meds_text = section_content["Medication Schedule"]
        for line in meds_text.split('\n'):
            if line.strip():
                med_match = re.search(r"- (.+?): (.+?), (.+?)(?: from (.+?) to (.+?))?", line.strip())
                if med_match:
                    name = med_match.group(1).strip()
                    dosage = med_match.group(2).strip()
                    frequency = med_match.group(3).strip()
                    start_date = med_match.group(4).strip() if med_match.group(4) else None
                    end_date = med_match.group(5).strip() if med_match.group(5) else None
                    medications.append(Medication(name=name, dosage=dosage, frequency=frequency, start_date=start_date, end_date=end_date))

    # Parse Appointment Schedule
    if "Appointment Schedule" in section_content:
        appts_text = section_content["Appointment Schedule"]
        for line in appts_text.split('\n'):
            if line.strip():
                appt_match = re.search(r"- (.+?): (.+?) at (.+?)(?: with (.+?))?(?: in (.+?))?", line.strip())
                if appt_match:
                    type_ = appt_match.group(1).strip()
                    date = appt_match.group(2).strip()
                    time = appt_match.group(3).strip()
                    doctor = appt_match.group(4).strip() if appt_match.group(4) else None
                    department = appt_match.group(5).strip() if appt_match.group(5) else None
                    appointments.append(Appointment(type=type_, date=date, time=time, doctor=doctor, department=department))

    # Parse Dietary Recommendations
    if "Dietary Recommendations" in section_content:
        diet_text = section_content["Dietary Recommendations"]
        # Simple extraction for demo purposes, can be improved
        restriction_match = re.search(r"Restriction Type: (.+)", diet_text)
        avoid_match = re.search(r"Foods to Avoid: (.+)", diet_text)
        recommendations_match = re.search(r"General Recommendations: (.+)", diet_text)

        restriction_type = restriction_match.group(1).strip() if restriction_match else "General"
        specific_foods_to_avoid = [f.strip() for f in avoid_match.group(1).split(',')] if avoid_match else []
        recommendations_str = recommendations_match.group(1).strip() if recommendations_match else diet_text.strip()
        
        dietary_recommendations.append(DietaryRecommendation(
            restriction_type=restriction_type,
            specific_foods_to_avoid=specific_foods_to_avoid if specific_foods_to_avoid else None,
            recommendations=recommendations_str
        ))

    # Parse Exercise Plan
    if "Exercise Plan" in section_content:
        exercise_text = section_content["Exercise Plan"]
        for line in exercise_text.split('\n'):
            if line.strip():
                exercise_match = re.search(r"- (.+?): Duration (.+?), Frequency (.+)", line.strip())
                if exercise_match:
                    type_ = exercise_match.group(1).strip()
                    duration = exercise_match.group(2).strip()
                    frequency = exercise_match.group(3).strip()
                    exercise_plans.append(ExercisePlan(type=type_, duration=duration, frequency=frequency))

    # General Notes
    if "General Notes" in section_content:
        general_notes = section_content["General Notes"]

    return CarePlan(
        patient_id=patient_id,
        diagnosis=diagnosis,
        medications=medications,
        appointments=appointments,
        dietary_recommendations=dietary_recommendations,
        exercise_plans=exercise_plans,
        notes=general_notes
    )

# --- 4. Streamlit UI ---
st.set_page_config(layout="wide", page_title="Patient Care Plan Generator")
st.title("🩺 Personalized Patient Care Plan Generator")
st.markdown("Generate structured care plans from natural language input for EHR integration.")

# Input fields
st.sidebar.header("Patient Information")
p_id = st.sidebar.text_input("Patient ID", "PAT001")
diagnosis_input = st.sidebar.text_area("Diagnosis", "Type 2 Diabetes, Hypertension")
medical_history_input = st.sidebar.text_area("Medical History Keywords", "Previous heart attack, allergic to penicillin")
preferences_input = st.sidebar.text_area("Patient Preferences/Concerns", "Prefers home exercises, concerned about medication side effects")

if st.sidebar.button("Generate Care Plan"):
    if not openai.api_key:
        st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    else:
        with st.spinner("Generating natural language care plan..."):
            patient_data = {
                "patient_id": p_id,
                "diagnosis": diagnosis_input,
                "medical_history": medical_history_input,
                "preferences": preferences_input,
            }
            
            # Generate natural language plan
            nl_care_plan = generate_natural_language_care_plan(patient_data)

            if nl_care_plan:
                st.subheader("Natural Language Care Plan")
                st.write(nl_care_plan)
                
                with st.spinner("Extracting structured data..."):
                    # Extract structured data
                    structured_care_plan = extract_structured_care_plan(p_id, diagnosis_input, nl_care_plan)

                    st.subheader("Structured Care Plan (JSON)")
                    json_output = structured_care_plan.json(indent=2)
                    st.code(json_output, language="json")

                    # Mock EHR Integration: Save to file
                    output_filename = f"care_plan_{p_id}.json"
                    try:
                        with open(output_filename, "w") as f:
                            f.write(json_output)
                        st.success(f"Structured care plan saved to {output_filename} (Mock EHR Integration)")
                    except Exception as e:
                        st.error(f"Could not save care plan file: {e}")

                    # Optional: Display pydantic validation
                    # st.subheader("Pydantic Model Validation Status")
                    # try:
                    #     CarePlan.parse_raw(json_output)
                    #     st.success("Pydantic model successfully validated the structured output.")
                    # except ValidationError as ve:
                    #     st.error(f"Pydantic validation error: {ve}")

st.sidebar.markdown("--- Source Code --- ")
st.sidebar.link_button("GitHub Repo", "https://github.com/your_repo_link_here")
