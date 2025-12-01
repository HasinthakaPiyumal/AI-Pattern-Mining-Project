import json
from datetime import date
from typing import List, Optional

# Although `langchain_core` imports are used here for context, the actual LLM interaction
# is mocked. For a real application, you would replace `MockLLM` with an actual
# LLM client (e.g., `ChatOpenAI` from `langchain_openai`).
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field # Using pydantic_v1 for broader compatibility

# --- Pydantic Models for Structured Output ---
# These models define the desired JSON schema for the treatment plan.

class Medication(BaseModel):
    name: str = Field(..., description="Name of the medication.")
    dosage: str = Field(..., description="Dosage instructions (e.g., '10mg daily', '2 tablets before meals').")
    frequency: str = Field(..., description="How often the medication should be taken (e.g., 'once a day', 'every 8 hours').")
    duration: Optional[str] = Field(None, description="Duration for which the medication should be taken (e.g., '7 days', 'until finished').")

class Therapy(BaseModel):
    name: str = Field(..., description="Name of the therapy (e.g., 'Physical Therapy', 'Cognitive Behavioral Therapy').")
    sessions: str = Field(..., description="Number or frequency of sessions (e.g., '3 sessions per week', '10 total sessions').")
    duration_per_session: Optional[str] = Field(None, description="Duration of each session (e.g., '60 minutes').")
    notes: Optional[str] = Field(None, description="Any specific notes or goals for the therapy.")

class DietaryRestriction(BaseModel):
    restriction: str = Field(..., description="Description of the dietary restriction (e.g., 'Low Sodium', 'Gluten-Free', 'Diabetic Diet').")
    details: Optional[str] = Field(None, description="Specific foods to avoid or include, or meal plan suggestions.")

class ExerciseRoutine(BaseModel):
    activity: str = Field(..., description="Type of exercise activity (e.g., 'Walking', 'Strength Training', 'Yoga').")
    frequency: str = Field(..., description="How often the exercise should be performed (e.g., '3 times a week', 'daily').")
    duration_per_session: str = Field(..., description="Duration of each exercise session (e.g., '30 minutes', '1 hour').")
    intensity: Optional[str] = Field(None, description="Recommended intensity level (e.g., 'moderate', 'light').")

class FollowUpAppointment(BaseModel):
    date: date = Field(..., description="Date of the follow-up appointment (YYYY-MM-DD format).")
    specialty: str = Field(..., description="Specialty of the doctor for the follow-up (e.g., 'Cardiologist', 'General Practitioner').")
    reason: Optional[str] = Field(None, description="Reason for the follow-up.")

class TreatmentPlan(BaseModel):
    patient_id: str = Field(..., description="Unique identifier for the patient.")
    plan_overview: str = Field(..., description="A general overview of the treatment plan.")
    medications: List[Medication] = Field(default_factory=list, description="List of prescribed medications.")
    therapies: List[Therapy] = Field(default_factory=list, description="List of recommended therapies.")
    dietary_restrictions: List[DietaryRestriction] = Field(default_factory=list, description="List of dietary restrictions.")
    exercise_routines: List[ExerciseRoutine] = Field(default_factory=list, description="List of recommended exercise routines.")
    follow_up_appointments: List[FollowUpAppointment] = Field(default_factory=list, description="List of scheduled follow-up appointments.")
    additional_recommendations: Optional[str] = Field(None, description="Any other general recommendations or advice.")

# --- Mock LLM for demonstration ---
# This class simulates an LLM generating a natural language treatment plan.
# In a real application, you would replace this with an actual LLM client (e.g., OpenAI, Anthropic).
class MockLLM:
    def invoke(self, prompt_value):
        print(f"MockLLM: Simulating LLM response for patient info: {prompt_value.messages[0].content[:100]}...")

        # This is a fixed, natural language response that mimics an LLM's output.
        # The `generate_structured_treatment_plan` function will then parse this.
        mock_natural_language_plan = """
        For Patient ID: P12345, the recommended treatment plan focuses on managing hypertension and improving overall cardiovascular health.

        Medications:
        - Lisinopril: 10mg daily, to be taken once a day in the morning. Continue for a long term.
        - Hydrochlorothiazide: 12.5mg daily, once a day. Also long term.

        Therapy:
        - Nutritional Counseling: 2 sessions over the next month, 45 minutes each, focusing on heart-healthy diet.
        - Stress Management Workshop: 1 session, 90 minutes, to learn relaxation techniques.

        Dietary Restrictions:
        - Low Sodium Diet: Strictly limit sodium intake to under 2000mg per day. Avoid processed foods, canned soups, and fast food. Focus on fresh fruits, vegetables, and lean proteins.

        Exercise Routines:
        - Brisk Walking: 30 minutes, 5 times a week, moderate intensity.
        - Light Strength Training: 2 times a week, 45 minutes, light intensity.

        Follow-up Appointments:
        - October 26, 2024: General Practitioner, for blood pressure check and medication review.
        - November 15, 2024: Cardiologist, for comprehensive cardiovascular assessment.

        Additional Recommendations:
        - Monitor blood pressure daily and record readings.
        - Avoid smoking and limit alcohol consumption.
        - Ensure adequate sleep (7-9 hours per night).
        """
        return mock_natural_language_plan

# --- Main Application Logic ---
def generate_structured_treatment_plan(
    patient_medical_history: str,
    current_symptoms: str,
    diagnosis: str,
    doctor_recommendations: str,
    patient_id: str
) -> dict:
    """
    Generates a structured patient treatment plan from natural language inputs using a mocked LLM
    and then parses it into a Pydantic-defined JSON format.
    """
    llm = MockLLM() # Instantiate the mocked LLM

    # Define the prompt template for the LLM to generate the initial natural language plan.
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an AI assistant specialized in generating detailed patient treatment plans based on medical information. Your output should be a comprehensive plan in natural language, which will then be parsed into a structured format."),
            ("human", (
                "Generate a detailed treatment plan for a patient with the following information:\n\n"
                "Patient ID: {patient_id}\n"
                "Medical History: {medical_history}\n"
                "Current Symptoms: {symptoms}\n"
                "Diagnosis: {diagnosis}\n"
                "Doctor's General Recommendations: {doctor_recs}\n\n"
                "Please include sections for Medications, Therapy, Dietary Restrictions, Exercise Routines, and Follow-up Appointments. Also add a general plan overview and any additional recommendations."
            )),
        ]
    )

    # Format the prompt with the provided patient data.
    formatted_prompt = prompt_template.format_messages(
        patient_id=patient_id,
        medical_history=patient_medical_history,
        symptoms=current_symptoms,
        diagnosis=diagnosis,
        doctor_recs=doctor_recommendations
    )

    print("\n--- LLM Input Prompt ---")
    print(formatted_prompt[1].content) # Display the human message sent to the LLM

    # Get the preliminary natural language plan from the mocked LLM.
    natural_language_plan = llm.invoke(formatted_prompt)

    print("\n--- LLM Raw Output (Natural Language Plan) ---")
    print(natural_language_plan)

    # --- Post-processing: Extract and Structure into Pydantic Model ---
    # In a real LangChain application, this step would involve using an `output_parser`
    # (e.g., `PydanticOutputParser` or `JsonOutputParser`) chained with another LLM call
    # specifically instructed to output JSON according to the `TreatmentPlan` schema.
    # For this demonstration, we're simulating the parsing by directly mapping
    # the fixed mock LLM's natural language output to the Pydantic model structure.
    try:
        # This manual extraction simulates a more complex parsing logic that a second
        # LLM call (or rule-based parser) would perform on the natural_language_plan.
        structured_data = {
            "patient_id": patient_id,
            "plan_overview": "Management of hypertension and improving overall cardiovascular health.",
            "medications": [
                {"name": "Lisinopril", "dosage": "10mg daily", "frequency": "once a day", "duration": "long term"},
                {"name": "Hydrochlorothiazide", "dosage": "12.5mg daily", "frequency": "once a day", "duration": "long term"}
            ],
            "therapies": [
                {"name": "Nutritional Counseling", "sessions": "2 sessions over the next month", "duration_per_session": "45 minutes", "notes": "focusing on heart-healthy diet"},
                {"name": "Stress Management Workshop", "sessions": "1 session", "duration_per_session": "90 minutes", "notes": "to learn relaxation techniques"}
            ],
            "dietary_restrictions": [
                {"restriction": "Low Sodium Diet", "details": "Strictly limit sodium intake to under 2000mg per day. Avoid processed foods, canned soups, and fast food. Focus on fresh fruits, vegetables, and lean proteins."}
            ],
            "exercise_routines": [
                {"activity": "Brisk Walking", "frequency": "5 times a week", "duration_per_session": "30 minutes", "intensity": "moderate"},
                {"activity": "Light Strength Training", "frequency": "2 times a week", "duration_per_session": "45 minutes", "intensity": "light"}
            ],
            "follow_up_appointments": [
                {"date": date(2024, 10, 26), "specialty": "General Practitioner", "reason": "blood pressure check and medication review"},
                {"date": date(2024, 11, 15), "specialty": "Cardiologist", "reason": "comprehensive cardiovascular assessment"}
            ],
            "additional_recommendations": "Monitor blood pressure daily and record readings. Avoid smoking and limit alcohol consumption. Ensure adequate sleep (7-9 hours per night)."
        }

        # Validate and convert the extracted data into the Pydantic `TreatmentPlan` model.
        structured_plan = TreatmentPlan(**structured_data)

        # Return the structured plan as a dictionary, which is JSON serializable.
        return structured_plan.dict()

    except Exception as e:
        print(f"Error parsing LLM output: {e}")
        return {"error": f"Failed to parse treatment plan into structured format: {e}"}

# --- Example Usage ---
if __name__ == "__main__":
    # Sample patient input data
    patient_history = "Patient has a history of essential hypertension diagnosed 5 years ago, currently managed with medication. No known allergies. Occasional lightheadedness reported."
    current_symptoms = "Patient reports occasional headaches and fatigue. Blood pressure readings at home are consistently elevated (around 145/90 mmHg)."
    diagnosis_info = "Essential Hypertension, uncontrolled."
    doctor_general_recs = "Adjust medication, lifestyle modifications including diet and exercise, and stress reduction."
    patient_id_val = "P12345"

    # Generate the structured treatment plan
    structured_plan_output = generate_structured_treatment_plan(
        patient_medical_history=patient_history,
        current_symptoms=current_symptoms,
        diagnosis=diagnosis_info,
        doctor_recommendations=doctor_general_recs,
        patient_id=patient_id_val
    )

    print("\n--- Structured Treatment Plan (JSON Output) ---")
    # Use default=str to correctly serialize date objects within the JSON output
    print(json.dumps(structured_plan_output, indent=2, default=str))