import os
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.tools import Tool
from langchain.schema import Document

load_dotenv()

llm = OpenAI(temperature=0.7)

patient_records_db = {
    "P1001": {
        "demographics": {"name": "Alice Smith", "age": 45, "gender": "Female"},
        "history": ["Hypertension (diagnosed 2010)", "Type 2 Diabetes (diagnosed 2015)"],
        "allergies": ["Penicillin"],
        "medications": ["Lisinopril 10mg daily", "Metformin 500mg twice daily"],
        "diagnoses": ["Hypertension", "Type 2 Diabetes"],
        "appointments": []
    },
    "P1002": {
        "demographics": {"name": "Bob Johnson", "age": 60, "gender": "Male"},
        "history": ["Coronary Artery Disease (CABG 2018)"],
        "allergies": [],
        "medications": ["Aspirin 81mg daily", "Atorvastatin 20mg daily"],
        "diagnoses": ["Coronary Artery Disease"],
        "appointments": []
    }
}

def get_patient_records_func(patient_id: str) -> str:
    if patient_id in patient_records_db:
        return str(patient_records_db[patient_id])
    return f"No record found for patient ID: {patient_id}"

def update_patient_record_func(patient_id: str, field: str, value: str) -> str:
    if patient_id in patient_records_db:
        if field in patient_records_db[patient_id]:
            patient_records_db[patient_id][field] = value
            return f"Patient {patient_id} {field} updated to {value}"
        elif field in patient_records_db[patient_id]["demographics"]:
            patient_records_db[patient_id]["demographics"][field] = value
            return f"Patient {patient_id} demographics {field} updated to {value}"
        return f"Field '{field}' not found for patient {patient_id}."
    return f"No record found for patient ID: {patient_id}"

def add_new_diagnosis_func(patient_id: str, diagnosis: str) -> str:
    if patient_id in patient_records_db:
        if "diagnoses" not in patient_records_db[patient_id]:
            patient_records_db[patient_id]["diagnoses"] = []
        patient_records_db[patient_id]["diagnoses"].append(diagnosis)
        return f"Added new diagnosis '{diagnosis}' for patient {patient_id}."
    return f"No record found for patient ID: {patient_id}"

def prescribe_medication_func(patient_id: str, medication: str, dosage: str, instructions: str) -> str:
    if patient_id in patient_records_db:
        if "medications" not in patient_records_db[patient_id]:
            patient_records_db[patient_id]["medications"] = []
        patient_records_db[patient_id]["medications"].append(f"{medication} {dosage} ({instructions})")
        return f"Prescribed {medication} to patient {patient_id}."
    return f"No record found for patient ID: {patient_id}"

medical_docs = [
    "Hypertension, also known as high blood pressure, is a long-term medical condition in which the blood pressure in the arteries is persistently elevated.",
    "Type 2 diabetes is a chronic condition that affects the way your body processes blood sugar (glucose).",
    "Lisinopril is an ACE inhibitor used to treat high blood pressure and heart failure.",
    "Metformin is a first-line medication for the treatment of type 2 diabetes, particularly in people who are overweight.",
    "Coronary Artery Disease (CAD) is a condition caused by plaque buildup in the walls of the arteries that supply blood to the heart.",
    "Aspirin is commonly used as an analgesic, anti-inflammatory, and antiplatelet drug.",
    "Atorvastatin is a statin medication used to lower cholesterol and triglycerides in the blood.",
    "Common symptoms of a heart attack include chest pain, shortness of breath, pain radiating to the left arm, and lightheadedness."
]
documents = [Document(page_content=doc) for doc in medical_docs]
embeddings = OpenAIEmbeddings()
medical_vectorstore = Chroma.from_documents(documents, embeddings)
medical_knowledge_qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=medical_vectorstore.as_retriever())

def search_medical_database_func(query: str) -> str:
    return medical_knowledge_qa.invoke({"query": query})["result"]

appointments_db = {}
appointment_id_counter = 0

def schedule_appointment_func(patient_id: str, doctor_id: str, date_time: str, reason: str) -> str:
    global appointment_id_counter
    appointment_id_counter += 1
    appt_id = f"A{appointment_id_counter:04d}"
    appointments_db[appt_id] = {
        "patient_id": patient_id,
        "doctor_id": doctor_id,
        "date_time": date_time,
        "reason": reason,
        "status": "scheduled"
    }
    if patient_id in patient_records_db:
        patient_records_db[patient_id]["appointments"].append(appt_id)
    return f"Appointment {appt_id} scheduled for patient {patient_id} with doctor {doctor_id} on {date_time} for '{reason}'."

def reschedule_appointment_func(appointment_id: str, new_date_time: str) -> str:
    if appointment_id in appointments_db:
        old_date_time = appointments_db[appointment_id]["date_time"]
        appointments_db[appointment_id]["date_time"] = new_date_time
        return f"Appointment {appointment_id} rescheduled from {old_date_time} to {new_date_time}."
    return f"Appointment ID {appointment_id} not found."

def cancel_appointment_func(appointment_id: str) -> str:
    if appointment_id in appointments_db:
        patient_id = appointments_db[appointment_id]["patient_id"]
        del appointments_db[appointment_id]
        if patient_id in patient_records_db and appointment_id in patient_records_db[patient_id]["appointments"]:
            patient_records_db[patient_id]["appointments"].remove(appointment_id)
        return f"Appointment {appointment_id} cancelled."
    return f"Appointment ID {appointment_id} not found."

tools = [
    Tool(
        name="GetPatientRecords",
        func=get_patient_records_func,
        description="Useful for retrieving a patient's complete medical record given their patient ID. Input should be a patient ID (e.g., 'P1001')."
    ),
    Tool(
        name="UpdatePatientRecord",
        func=update_patient_record_func,
        description="Useful for updating a specific field in a patient's medical record. Input should be patient_id, field, value (e.g., 'P1001', 'allergies', 'Aspirin')."
    ),
    Tool(
        name="AddNewDiagnosis",
        func=add_new_diagnosis_func,
        description="Useful for adding a new diagnosis to a patient's record. Input should be patient_id, diagnosis (e.g., 'P1001', 'Acute Bronchitis')."
    ),
    Tool(
        name="PrescribeMedication",
        func=prescribe_medication_func,
        description="Useful for prescribing medication to a patient. Input should be patient_id, medication, dosage, instructions (e.g., 'P1001', 'Amoxicillin', '500mg', 'Take twice daily for 7 days')."
    ),
    Tool(
        name="SearchMedicalDatabase",
        func=search_medical_database_func,
        description="Useful for searching a comprehensive medical knowledge base for information about diseases, drugs, treatments, etc. Input should be a medical query (e.g., 'symptoms of a heart attack')."
    ),
    Tool(
        name="ScheduleAppointment",
        func=schedule_appointment_func,
        description="Useful for scheduling a new appointment for a patient. Input should be patient_id, doctor_id, date_time, reason (e.g., 'P1001', 'Dr. Lee', '2023-11-15 10:00 AM', 'Follow-up on hypertension')."
    ),
    Tool(
        name="RescheduleAppointment",
        func=reschedule_appointment_func,
        description="Useful for rescheduling an existing appointment. Input should be appointment_id, new_date_time (e.g., 'A0001', '2023-11-20 11:00 AM')."
    ),
    Tool(
        name="CancelAppointment",
        func=cancel_appointment_func,
        description="Useful for canceling an existing appointment. Input should be appointment_id (e.g., 'A0001')."
    )
]

agent_prompt = PromptTemplate.from_template("""
You are an intelligent medical assistant designed to help healthcare professionals.
You have access to patient records, a medical knowledge base, and scheduling tools.
Your goal is to assist with diagnosis, treatment recommendations, and administrative tasks.
Always try to use tools to get factual information when appropriate, and base your responses on the retrieved information.

TOOLS:
------
You have access to the following tools:

{tools}

To use a tool, please use the following format:

```
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
```

The response from the tool will be added to the conversation.

After the tool output, you can either continue using tools or provide a final answer.
If you have a definitive answer, use this format:

```
Thought: I have gathered enough information and can provide a final answer.
Final Answer: [your final answer here]
```

Begin!

Previous conversation history:
{chat_history}

New Human input: {input}
{agent_scratchpad}
""")

memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=5)

agent = create_react_agent(llm, tools, agent_prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    memory=memory,
    handle_parsing_errors=True
)

if __name__ == "__main__":
    print("Intelligent Medical Assistant - How can I help you today?")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Assistant: Goodbye!")
            break
        try:
            response = agent_executor.invoke({"input": user_input})
            print(f"Assistant: {response['output']}")
        except Exception as e:
            print(f"Assistant: An error occurred: {e}")