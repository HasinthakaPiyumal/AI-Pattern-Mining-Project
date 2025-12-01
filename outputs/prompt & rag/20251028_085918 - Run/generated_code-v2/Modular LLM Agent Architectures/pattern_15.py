import os
from dotenv import load_dotenv
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import PromptTemplate

import gradio as gr

load_dotenv()

# --- 1. Simulated Data Storage ---
patients = {
    "patient_1": {
        "name": "Alice Smith",
        "age": 45,
        "conditions": ["Hypertension"],
        "medications": {
            "Lisinopril": {"dosage": "10mg", "frequency": "daily", "adherence": []},
            "Aspirin": {"dosage": "81mg", "frequency": "daily", "adherence": []}
        },
        "appointments": [
            {"id": "app_001", "date": "2024-07-20", "time": "10:00", "doctor_id": "doc_001", "reason": "Annual Checkup", "status": "scheduled"},
            {"id": "app_002", "date": "2024-06-15", "time": "14:00", "doctor_id": "doc_002", "reason": "Follow-up", "status": "completed"}
        ]
    }
}

doctors = {
    "doc_001": {"name": "Dr. Emily White", "specialty": "Cardiology"},
    "doc_002": {"name": "Dr. John Brown", "specialty": "General Practice"}
}

# --- 2. Tool Definitions ---
def get_appointment_info(patient_id: str, date: str = None) -> str:
    if patient_id not in patients:
        return "Patient not found."
    
    patient_appointments = patients[patient_id]["appointments"]
    if not patient_appointments:
        return "No appointments found for this patient."

    if date:
        filtered_appointments = [app for app in patient_appointments if app["date"] == date]
    else:
        filtered_appointments = patient_appointments

    if not filtered_appointments:
        return f"No appointments found for {date}."

    info = []
    for app in filtered_appointments:
        doctor_name = doctors.get(app["doctor_id"], {}).get("name", "Unknown Doctor")
        info.append(f"ID: {app['id']}, Date: {app['date']}, Time: {app['time']}, Doctor: {doctor_name}, Reason: {app['reason']}, Status: {app['status']}")
    return "\n".join(info)

def book_appointment(patient_id: str, doctor_id: str, date: str, time: str, reason: str) -> str:
    if patient_id not in patients:
        return "Patient not found."
    if doctor_id not in doctors:
        return "Doctor not found."

    # Simple check for availability - not robust for real-world
    for app in patients[patient_id]["appointments"]:
        if app["date"] == date and app["time"] == time and app["doctor_id"] == doctor_id:
            return "An appointment with this doctor at this time on this date is already booked."

    new_app_id = f"app_{len(patients[patient_id]['appointments']) + 1:03d}"
    new_appointment = {
        "id": new_app_id,
        "date": date,
        "time": time,
        "doctor_id": doctor_id,
        "reason": reason,
        "status": "scheduled"
    }
    patients[patient_id]["appointments"].append(new_appointment)
    doctor_name = doctors.get(doctor_id, {}).get("name", "Unknown Doctor")
    return f"Appointment {new_app_id} successfully booked for {patients[patient_id]['name']} with {doctor_name} on {date} at {time} for {reason}."

def get_medication_details(patient_id: str, medication_name: str) -> str:
    if patient_id not in patients:
        return "Patient not found."
    
    meds = patients[patient_id]["medications"]
    med_name_lower = medication_name.lower()
    for med in meds:
        if med.lower() == med_name_lower:
            details = meds[med]
            return f"{med}: Dosage - {details['dosage']}, Frequency - {details['frequency']}."
    return f"Medication '{medication_name}' not found for patient {patients[patient_id]['name']}."

def log_medication_adherence(patient_id: str, medication_name: str, date: str, adhered: bool) -> str:
    if patient_id not in patients:
        return "Patient not found."

    meds = patients[patient_id]["medications"]
    med_name_lower = medication_name.lower()
    for med in meds:
        if med.lower() == med_name_lower:
            meds[med]["adherence"].append({"date": date, "adhered": adhered})
            status = "adhered to" if adhered else "missed"
            return f"Logged that {patients[patient_id]['name']} {status} {medication_name} on {date}."
    return f"Medication '{medication_name}' not found for patient {patients[patient_id]['name']}."

def get_health_summary(patient_id: str) -> str:
    if patient_id not in patients:
        return "Patient not found."
    
    patient = patients[patient_id]
    summary_parts = [
        f"Health Summary for {patient['name']} (ID: {patient_id}):",
        f"Age: {patient['age']}",
        f"Conditions: {', '.join(patient['conditions']) if patient['conditions'] else 'None'}"
    ]

    medication_list = []
    for med, details in patient['medications'].items():
        adherence_count = sum(1 for entry in details['adherence'] if entry['adhered'])
        total_logs = len(details['adherence'])
        adherence_rate = f"{(adherence_count / total_logs * 100):.1f}%" if total_logs > 0 else "N/A"
        medication_list.append(f"- {med} ({details['dosage']}, {details['frequency']}). Adherence: {adherence_rate}")
    if medication_list:
        summary_parts.append("Medications:")
        summary_parts.extend(medication_list)
    else:
        summary_parts.append("Medications: None listed.")

    appointment_list = []
    for app in patient['appointments']:
        doctor_name = doctors.get(app["doctor_id"], {}).get("name", "Unknown Doctor")
        appointment_list.append(f"- {app['date']} {app['time']} with {doctor_name} for {app['reason']} ({app['status']})")
    if appointment_list:
        summary_parts.append("Appointments:")
        summary_parts.extend(appointment_list)
    else:
        summary_parts.append("Appointments: No past or upcoming appointments.")
    
    return "\n".join(summary_parts)

# --- 3. Langchain Integration ---
llm = ChatOpenAI(model="gpt-4o", temperature=0)

tools = [
    Tool(
        name="GetAppointmentInfo",
        func=get_appointment_info,
        description="Useful for getting information about a patient's appointments. Takes patient_id (string) and optional date (string YYYY-MM-DD) as input."
    ),
    Tool(
        name="BookAppointment",
        func=book_appointment,
        description="Useful for booking a new appointment for a patient. Takes patient_id (string), doctor_id (string), date (string YYYY-MM-DD), time (string HH:MM), and reason (string) as input."
    ),
    Tool(
        name="GetMedicationDetails",
        func=get_medication_details,
        description="Useful for getting details about a specific medication for a patient. Takes patient_id (string) and medication_name (string) as input."
    ),
    Tool(
        name="LogMedicationAdherence",
        func=log_medication_adherence,
        description="Useful for logging whether a patient adhered to their medication on a specific date. Takes patient_id (string), medication_name (string), date (string YYYY-MM-DD), and adhered (boolean) as input."
    ),
    Tool(
        name="GetHealthSummary",
        func=get_health_summary,
        description="Useful for getting a comprehensive health summary for a patient, including conditions, medications, and appointments. Takes patient_id (string) as input."
    ),
]

prompt_template = PromptTemplate.from_template(
    """You are a helpful and empathetic Personalized Healthcare Navigator AI. You assist patients in managing their healthcare journey.
    You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Previous conversation history:
    {chat_history}

    New input: {input}
    {agent_scratchpad}"""
)

memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=5)

agent = create_react_agent(llm, tools, prompt_template)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=memory, handle_parsing_errors=True)

# --- 4. Gradio Interface ---
def predict(message, history):
    try:
        response = agent_executor.invoke({"input": message, "chat_history": history})
        return response["output"]
    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == "__main__":
    gr.ChatInterface(
        predict,
        chatbot=gr.Chatbot(height=400),
        textbox=gr.Textbox(placeholder="Ask me about your health, appointments, or medications.", container=False, scale=7),
        title="Personalized Healthcare Navigator",
        description="I am an AI agent designed to help you manage your healthcare journey. I can retrieve appointment info, book appointments, check medication details, log adherence, and provide a health summary. (Note: Using patient_1 as default patient for demo)",
        theme="soft"
    ).launch(share=False)
