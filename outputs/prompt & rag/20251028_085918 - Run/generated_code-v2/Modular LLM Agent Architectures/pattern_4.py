import os
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

# --- 1. Tool Definitions ---

@tool
def medical_knowledge_base_search(query: str) -> str:
    """
    Searches a simulated medical knowledge base for information about diseases,
    symptoms, treatments, and general medical conditions.
    Input should be a specific medical query, e.g., "symptoms of common cold".
    """
    knowledge_base = {
        "common cold": "The common cold is a viral infection of your nose and throat (upper respiratory tract). Symptoms include a runny nose, sore throat, cough, congestion, slight body aches or a mild headache, sneezing, and low-grade fever. It's usually harmless, though it might not feel that way. Many types of viruses can cause a common cold. Treatment often involves rest, fluids, and over-the-counter medications.",
        "influenza": "Influenza (flu) is a contagious respiratory illness caused by influenza viruses that infect the nose, throat, and sometimes the lungs. It can cause mild to severe illness, and at times can lead to death. The flu is different from a cold. The flu usually comes on suddenly. People who have the flu often feel some or all of these symptoms: fever or feeling feverish/chills, cough, sore throat, runny or stuffy nose, muscle or body aches, headaches, and fatigue. Annual vaccination is recommended.",
        "diabetes type 2": "Type 2 diabetes is a chronic condition that affects the way your body processes blood sugar (glucose). With type 2 diabetes, your body either doesn't produce enough insulin, or it resists the effects of insulin. Symptoms can include increased thirst, frequent urination, increased hunger, unintended weight loss, fatigue, blurred vision, slow-healing sores, and frequent infections. Management involves diet, exercise, and often medication.",
        "headache": "A headache is a pain in any region of the head. Headaches may be a symptom of a wide range of conditions. Common types include tension headaches, migraines, and cluster headaches. Treatment depends on the type and severity, ranging from over-the-counter pain relievers to prescription medications and lifestyle changes.",
        "hypertension": "Hypertension, also known as high blood pressure, is a common condition in which the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease. Symptoms are often subtle. Regular monitoring and lifestyle changes are key to management, along with medication.",
        "allergies": "Allergies are a reaction by your immune system to a substance that doesn't bother most other people. These substances are called allergens. They can include certain foods, pollen, dust mites, and pet dander. Symptoms vary widely and can include sneezing, runny nose, itchy eyes, skin rashes, or in severe cases, anaphylaxis. Treatment involves avoiding allergens, antihistamines, decongestants, and sometimes immunotherapy."
    }
    return knowledge_base.get(query.lower(), "Information not found for that specific medical query. Please try a different query or be more specific.")

@tool
def drug_interaction_checker(drug1: str, drug2: str) -> str:
    """
    Checks for potential interactions between two specified drugs.
    Provides a warning if an interaction is found, otherwise indicates no known interaction.
    Input should be two drug names, e.g., "warfarin", "ibuprofen".
    """
    interactions = {
        ("warfarin", "ibuprofen"): "High risk of increased bleeding. Avoid concurrent use or monitor closely.",
        ("paracetamol", "alcohol"): "Increased risk of liver damage with excessive alcohol consumption.",
        ("lisinopril", "potassium supplements"): "Risk of hyperkalemia (high potassium levels). Monitor potassium levels.",
        ("amoxicillin", "methotrexate"): "May increase methotrexate levels, leading to toxicity. Monitor closely.",
        ("sertraline", "tramadol"): "Increased risk of serotonin syndrome. Monitor for symptoms like agitation, confusion, rapid heart rate.",
        ("omeprazole", "clopidogrel"): "May reduce the effectiveness of clopidogrel. Consider alternative PPIs.",
        ("metformin", "iodinated contrast"): "Risk of lactic acidosis in patients with renal impairment. May need to temporarily discontinue metformin."
    }
    interaction_key1 = tuple(sorted((drug1.lower(), drug2.lower())))

    if interaction_key1 in interactions:
        return f"Interaction found between {drug1} and {drug2}: {interactions[interaction_key1]}"
    else:
        return f"No common interactions found between {drug1} and {drug2} in our database."

@tool
def dosage_calculator(medication: str, weight_kg: float, age_years: int) -> str:
    """
    Calculates a sample dosage for a given medication based on patient weight and age.
    This is a highly simplified example and should NOT be used for actual medical advice.
    Returns the calculated dosage or an error if parameters are invalid.
    Input examples: medication="paracetamol", weight_kg=70.0, age_years=30.
    """
    if not (1 <= age_years <= 100 and 1 <= weight_kg <= 300):
        return "Invalid age or weight. Please provide realistic values."

    medication_dosages = {
        "paracetamol": {"adult_dose_mg_per_kg": 15, "max_daily_mg": 4000, "frequency_hours": 4},
        "ibuprofen": {"adult_dose_mg_per_kg": 10, "max_daily_mg": 2400, "frequency_hours": 6},
        "amoxicillin": {"child_dose_mg_per_kg": 20, "adult_dose_mg": 500, "frequency_hours": 8}
    }

    med_info = medication_dosages.get(medication.lower())
    if not med_info:
        return f"Dosage information not available for {medication} in this calculator."

    if age_years >= 12: # Adult dosage
        if "adult_dose_mg" in med_info:
            dose_mg = med_info["adult_dose_mg"]
        elif "adult_dose_mg_per_kg" in med_info:
            dose_mg = min(med_info["adult_dose_mg_per_kg"] * weight_kg, med_info.get("max_daily_mg", float('inf')))
        else:
            return f"Adult dosage information incomplete for {medication}."
        return (f"For {medication} (adult): Approximately {dose_mg:.0f} mg every {med_info['frequency_hours']} hours. "
                f"Maximum daily dose: {med_info.get('max_daily_mg', 'N/A')} mg. "
                "Consult a doctor for precise dosage.")
    else: # Child dosage
        if "child_dose_mg_per_kg" in med_info:
            dose_mg = med_info["child_dose_mg_per_kg"] * weight_kg
            return (f"For {medication} (child, {age_years} years, {weight_kg} kg): Approximately {dose_mg:.0f} mg every {med_info['frequency_hours']} hours. "
                    "Consult a pediatrician for precise dosage.")
        else:
            return f"Child dosage information not available for {medication}."

@tool
def schedule_appointment(patient_name: str, doctor_name: str, date: str, time: str, reason: str) -> str:
    """
    Simulates scheduling a medical appointment for a patient with a specific doctor on a given date and time.
    Returns a confirmation message.
    Input example: patient_name="John Doe", doctor_name="Smith", date="2024-12-25", time="10:00 AM", reason="Annual check-up".
    """
    # In a real system, this would interact with a calendar API or database.
    return (f"Appointment successfully scheduled for {patient_name} with Dr. {doctor_name} "
            f"on {date} at {time} for '{reason}'. A confirmation message has been sent.")

# --- 2. LLM Router/Agent ---

# IMPORTANT: Configure your LLM here.
# For OpenAI, ensure you have OPENAI_API_KEY set in your environment variables.
# Example:
try:
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o", temperature=0) # You can change model to "gpt-3.5-turbo" if preferred
except ImportError:
    print("Warning: langchain_openai not installed. Please install it (`pip install langchain-openai`) or configure another LLM provider.")
    print("Falling back to a placeholder LLM. The agent will not function without a proper LLM setup.")
    # A very basic mock LLM for structural integrity if langchain_openai is missing
    class MockLLM:
        def invoke(self, prompt_value, stop=None, callbacks=None, **kwargs):
            return "Mock LLM response: Please set up a proper LLM (e.g., OpenAI) to get actual functionality."
    llm = MockLLM()
except Exception as e:
    print(f"Error initializing OpenAI LLM: {e}. Please check your API key and network connection.")
    print("Falling back to a placeholder LLM. The agent will not function without a proper LLM setup.")
    class MockLLM:
        def invoke(self, prompt_value, stop=None, callbacks=None, **kwargs):
            return "Mock LLM response: Please set up a proper LLM (e.g., OpenAI) to get actual functionality."
    llm = MockLLM()


# List of tools available to the agent
tools = [
    medical_knowledge_base_search,
    drug_interaction_checker,
    dosage_calculator,
    schedule_appointment
]

# Define the prompt for the agent
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful and knowledgeable Medical Diagnostic Assistant. Your goal is to provide accurate information and assist with medical queries by utilizing the available tools. Always prioritize patient safety and advise consulting a medical professional for definitive diagnosis and treatment. If a query requires multiple steps, break it down and use the tools sequentially."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# Create the agent
# The create_tool_calling_agent automatically handles tool selection based on the prompt and LLM capabilities.
agent = create_tool_calling_agent(llm, tools, prompt)

# Create the agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- 3. and 4. User Interface and Response Generation ---

def run_medical_assistant():
    print("Welcome to the Medical Diagnostic Assistant!")
    print("I can help you with medical information, drug interactions, dosage calculations, and appointment scheduling.")
    print("Type 'exit' to quit.")
    print("\n---------------------------------------------------------")
    print("NOTE: This is a demonstration. Always consult a medical professional for real health concerns.")
    print("To use the full functionality, ensure 'langchain-openai' is installed and OPENAI_API_KEY is set in your environment.")
    print("---------------------------------------------------------")


    while True:
        user_query = input("\nHow can I help you today? ")
        if user_query.lower() == 'exit':
            print("Thank you for using the Medical Diagnostic Assistant. Stay healthy!")
            break

        if isinstance(llm, MockLLM):
            print("\nAssistant's Response:")
            print("LLM is not properly configured. Please set up an LLM (e.g., OpenAI) to get actual functionality.")
            continue

        try:
            # Invoke the agent with the user's query
            response = agent_executor.invoke({"input": user_query})
            print("\nAssistant's Final Response:")
            print(response["output"])
        except Exception as e:
            print(f"\nAn error occurred while processing your request: {e}")
            print("Please try rephrasing your query or contact support.")
            print("Ensure your LLM (e.g., OpenAI) is properly configured and API key is valid.")

if __name__ == "__main__":
    run_medical_assistant()