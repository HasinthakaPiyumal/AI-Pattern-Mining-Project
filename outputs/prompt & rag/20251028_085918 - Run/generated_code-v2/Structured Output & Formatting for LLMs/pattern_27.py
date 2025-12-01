def generate_treatment_plan_nl(patient_info: dict) -> str:
    """
    Simulates an LLM generating a personalized treatment plan in natural language.
    In a real application, this would be an actual LLM call.
    """
    patient_name = patient_info.get("name", "Patient")
    diagnosis = patient_info.get("diagnosis", "general condition")

    plan = f"""
    Based on {patient_name}\'s diagnosis of {diagnosis}, here is a personalized treatment plan:

    Medication:
    - Drug Name: Atorvastatin, Dosage: 20mg, Frequency: Once daily in the evening, Duration: Ongoing
    - Drug Name: Metformin, Dosage: 500mg, Frequency: Twice daily with meals, Duration: 6 months

    Therapy:
    - Type: Physical Therapy, Focus: Lower back strengthening, Schedule: 3 times a week for 8 weeks
    - Type: Nutritional Counseling, Focus: Diabetic diet education, Schedule: Bi-weekly for 1 month

    Appointments:
    - Type: Follow-up with Cardiologist, Date: 2024-03-15, Time: 10:00 AM
    - Type: Blood Work, Date: 2024-03-08, Time: 08:00 AM, Instructions: Fasting required

    Lifestyle Recommendations:
    - Exercise: Moderate intensity for 30 minutes, 5 days a week
    - Diet: Low-sugar, high-fiber diet
    - Stress Management: Practice mindfulness daily

    Important Notes:
    Please monitor blood sugar levels regularly and report any unusual symptoms to your doctor.
    """
    return plan