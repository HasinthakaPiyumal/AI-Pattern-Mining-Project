import os
from typing import Dict, Any

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import tool
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# --- 0. Configuration and Mock Data ---
# Set your OpenAI API key as an environment variable
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# Mock Medical Knowledge Base (simulating long-term memory/knowledge retrieval)
medical_knowledge_base = {
    "headache": "Common causes of headache include stress, dehydration, and lack of sleep. Mild cases can be managed with over-the-counter pain relievers. If severe or persistent, consult a doctor.",
    "fever": "Fever is often a sign of infection. Rest, hydration, and fever-reducing medication can help. Seek medical attention if fever is very high, lasts long, or is accompanied by other severe symptoms.",
    "sore throat": "Most sore throats are caused by viral infections and resolve on their own. Rest, warm liquids, and lozenges can provide relief. Bacterial infections (strep throat) require antibiotics.",
    "diabetes": "Diabetes is a chronic condition affecting blood sugar. Management includes diet, exercise, and medication. Regular monitoring and doctor consultations are crucial.",
    "hypertension": "Hypertension (high blood pressure) can lead to serious health problems. Lifestyle changes and medication can help control it. Regular check-ups are essential."
}

# --- 1. Tool Definitions ---

@tool
def medical_information_retrieval_tool(query: str) -> str:
    """Searches a medical knowledge base for information related to symptoms, conditions, or treatments."""
    query_lower = query.lower()
    for key, value in medical_knowledge_base.items():
        if key in query_lower:
            return f"Information on {key}: {value}"
    return "No specific information found for your query. Please rephrase or provide more details."

@tool
def patient_triage_tool(symptoms: str) -> str:
    """Assesses patient symptoms and provides a triage recommendation (e.g., self-care, consult doctor, emergency)."""
    symptoms_lower = symptoms.lower()
    if "chest pain" in symptoms_lower or "difficulty breathing" in symptoms_lower or "severe bleeding" in symptoms_lower:
        return "Recommendation: This requires immediate medical attention. Please go to the emergency room or call emergency services."
    elif "high fever" in symptoms_lower or "persistent vomiting" in symptoms_lower or "severe pain" in symptoms_lower:
        return "Recommendation: Please consult a doctor as soon as possible. It's advisable to book an appointment."
    elif "mild headache" in symptoms_lower or "sniffles" in symptoms_lower or "slight cough" in symptoms_lower:
        return "Recommendation: These symptoms often resolve with self-care (rest, hydration, over-the-counter medication). Monitor your condition."
    else:
        return "Recommendation: Based on the provided symptoms, it's advisable to consult a general practitioner for further assessment."

@tool
def appointment_booking_tool(patient_name: str, preferred_time: str, reason: str) -> str:
    """Simulates booking an appointment for a patient."""
    # In a real system, this would integrate with a calendar API or hospital system.
    return f"Appointment for {patient_name} regarding '{reason}' has been tentatively booked for {preferred_time}. A confirmation will be sent shortly."

# --- 2. LLM Core Initialization ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# --- 3. Memory Module Initialization (Short-term Conversational History) ---
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# --- 4. Planning Module & Agent Initialization ---
tools = [
    medical_information_retrieval_tool,
    patient_triage_tool,
    appointment_booking_tool
]

# Define the prompt for the agent
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful medical assistant. Use the available tools to answer questions, triage patients, and assist with appointments. Always prioritize patient safety."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

# Create the agent
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)

# --- 5. Main Interaction Loop ---
if __name__ == "__main__":
    print("Hello! I'm your Medical Assistant. How can I help you today? (Type 'exit' to quit)")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        try:
            response = agent_executor.invoke({"input": user_input})
            print(f"Assistant: {response['output']}")
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please ensure your OPENAI_API_KEY environment variable is set correctly.")