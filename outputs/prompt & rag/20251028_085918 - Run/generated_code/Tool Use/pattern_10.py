from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI # Using OpenAI for demonstration; can be replaced with other LLMs
from langchain_core.tools import tool
from typing import List, Dict
import json
import os

# Set OpenAI API Key from environment variable (for local testing)
# os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY_HERE" # Uncomment and set your key if not in env

################################################################################
# 1. Specialized Tools (Mock Implementations - replace with actual API calls)
################################################################################

@tool
def get_patient_ehr(patient_id: str) -> str:
    """Retrieves electronic health records for a given patient ID."""
    print(f"[TOOL CALL]: Retrieving EHR for patient_id: {patient_id}")
    if patient_id == "P123":
        return "EHR for P123: Diagnosis: Type 2 Diabetes, Hypertension. Medications: Metformin, Lisinopril. Last Visit: 2023-10-26."
    return "Patient EHR not found."

@tool
def check_drug_interactions(drugs: str) -> str:
    """Checks for potential drug interactions between a comma-separated list of drugs."""
    print(f"[TOOL CALL]: Checking drug interactions for drugs: {drugs}")
    if "Metformin" in drugs and "Lisinopril" in drugs:
        return "No significant interactions found between Metformin and Lisinopril for common dosages."
    elif "Warfarin" in drugs and "Aspirin" in drugs:
        return "WARNING: Increased risk of bleeding when Warfarin and Aspirin are taken together. Consult a doctor."
    return "Drug interaction check completed."

@tool
def search_medical_articles(query: str) -> str:
    """Searches a medical research article database for relevant information based on a query."""
    print(f"[TOOL CALL]: Searching medical articles for query: {query}")
    if "diabetes management" in query.lower():
        return "Found articles on dietary recommendations for diabetes, new insulin therapies, and exercise benefits."
    return "No specific articles found for the query."

@tool
def symptom_checker(symptoms: str) -> str:
    """Analyzes a list of symptoms to suggest potential conditions or next steps."""
    print(f"[TOOL CALL]: Running symptom checker for symptoms: {symptoms}")
    if "headache" in symptoms.lower() and "blurred vision" in symptoms.lower():
        return "Consider consulting a doctor for further evaluation. Could be related to blood pressure or other neurological issues."
    return "Symptoms noted. No immediate critical concerns identified based on provided data."

@tool
def get_diet_exercise_recommendations(patient_profile: str) -> str:
    """Provides personalized diet and exercise recommendations based on patient profile (e.g., conditions, preferences)."""
    print(f"[TOOL CALL]: Getting diet and exercise recommendations for profile: {patient_profile}")
    if "Type 2 Diabetes" in patient_profile:
        return "Diet: Low-carb, high-fiber diet recommended. Avoid sugary drinks. Exercise: Aim for 30 minutes of moderate-intensity activity most days of the week."
    return "General health recommendations: Balanced diet, regular exercise, adequate sleep."

# Example of how AI Tool Creation could be envisioned (conceptual)
def create_custom_health_script(goal: str) -> str:
    """Conceptually generates a custom monitoring script based on a goal. 
       In a real system, this would involve LLM generating code/configs."""
    if "blood sugar monitoring" in goal.lower():
        return "Generated Python script template for continuous glucose monitor data ingestion and anomaly detection."
    return "No custom script generated for this goal."

################################################################################
# 2. Multi-Agent Collaboration (Mock Implementation)
################################################################################

def medical_agent_1_diagnosis(patient_data: Dict, symptoms: List[str]) -> str:
    """Simulates a diagnostic agent focusing on cardiovascular aspects."""
    diagnosis = ""
    conditions = patient_data.get("conditions", [])
    vitals = patient_data.get("vitals", {})

    if "hypertension" in [c.lower() for c in conditions] and "chest pain" in [s.lower() for s in symptoms]:
        diagnosis = "Possible cardiac event or related issue. Recommend ECG and cardiologist consult."
    elif "blood pressure" in vitals and isinstance(vitals["blood pressure"], str) and '/' in vitals["blood pressure"]:
        try:
            systolic, diastolic = map(int, vitals["blood pressure"].split('/'))
            if systolic > 140 or diastolic > 90:
                diagnosis = "Elevated blood pressure detected. Advise monitoring and lifestyle changes."
        except ValueError:
            pass # Handle cases where blood pressure string is not as expected
    return f"Cardio Agent Report: {diagnosis or 'No specific cardiovascular concerns based on input.'}"

def medical_agent_2_diagnosis(patient_data: Dict, symptoms: List[str]) -> str:
    """Simulates a diagnostic agent focusing on metabolic/endocrine aspects."""
    diagnosis = ""
    conditions = patient_data.get("conditions", [])
    vitals = patient_data.get("vitals", {})

    if "type 2 diabetes" in [c.lower() for c in conditions] and "fatigue" in [s.lower() for s in symptoms]:
        diagnosis = "Potential diabetes-related fatigue. Check blood glucose levels and medication adherence."
    elif "blood sugar" in vitals and vitals["blood sugar"] > 180: # Assuming 180 mg/dL as high
        diagnosis = "Hyperglycemia detected. Suggest insulin adjustment consultation."
    return f"Metabolic Agent Report: {diagnosis or 'No specific metabolic concerns based on input.'}"

def collaborative_diagnosis(patient_data: Dict, symptoms: List[str]) -> str:
    """Orchestrates multiple specialized agents for a comprehensive diagnostic assessment.
       This simulates 'Multi-Agent Collaboration for Tool Learning'."""
    print(f"\n[AGENT COLLABORATION]: Initiating collaborative diagnosis for patient with symptoms: {symptoms}")

    report_cardio = medical_agent_1_diagnosis(patient_data, symptoms)
    report_metabolic = medical_agent_2_diagnosis(patient_data, symptoms)

    synthesized_report = f"Comprehensive Diagnostic Report:\n- {report_cardio}\n- {report_metabolic}\n\n" \
                         "Further LLM analysis would integrate these findings and suggest a holistic plan."

    return synthesized_report

@tool
def run_collaborative_diagnosis_tool(patient_data_json: str, symptoms_json: str) -> str:
    """Initiates a multi-agent collaborative diagnosis given patient data and symptoms.
       Input should be JSON strings for patient_data and symptoms."""
    try:
        patient_data = json.loads(patient_data_json)
        symptoms = json.loads(symptoms_json)
        return collaborative_diagnosis(patient_data, symptoms)
    except json.JSONDecodeError as e:
        return f"Invalid JSON input for patient_data or symptoms: {e}. Please provide valid JSON."

################################################################################
# 3. Formalism-Enhanced Reasoning / Knowledge Base (Mock Implementation)
################################################################################

medical_facts = {
    "Type 2 Diabetes": {
        "description": "A chronic condition that affects the way the body processes blood sugar (glucose).",
        "management": ["Dietary changes (low-carb, high-fiber)", "Regular exercise", "Medication (e.g., Metformin)", "Blood glucose monitoring"],
        "complications": ["Heart disease", "Kidney disease", "Nerve damage", "Eye damage"]
    },
    "Hypertension": {
        "description": "High blood pressure, a common condition in which the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease.",
        "management": ["Lifestyle modifications (diet, exercise, stress reduction)", "Medication (e.g., ACE inhibitors, diuretics)", "Regular blood pressure monitoring"],
        "complications": ["Heart attack", "Stroke", "Kidney failure"]
    },
    "Metformin": {
        "class": "Biguanide",
        "use": "Treats Type 2 Diabetes",
        "side_effects": ["Diarrhea", "Nausea", "Vomiting"],
        "interactions_notes": "Generally safe with many drugs, but caution with kidney issues."
    },
    "Lisinopril": {
        "class": "ACE inhibitor",
        "use": "Treats Hypertension and heart failure",
        "side_effects": ["Cough", "Dizziness", "Fatigue"],
        "interactions_notes": "Avoid with potassium-sparing diuretics; monitor kidney function."
    }
}

def get_medical_fact(topic: str) -> str:
    """Retrieves a fact from the simulated medical knowledge base."""
    print(f"[KNOWLEDGE BASE]: Querying for topic: {topic}")
    fact = medical_facts.get(topic)
    if fact:
        return f"Information for {topic}: {json.dumps(fact, indent=2)}"
    return f"No specific information found for {topic} in the knowledge base."

@tool
def get_medical_fact_tool(topic: str) -> str:
    """Retrieves a fact from the simulated medical knowledge base about a specific topic (e.g., disease, drug)."""
    return get_medical_fact(topic)

################################################################################
# 4. LLM Controller Class
################################################################################

class LLMController:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model_name=model_name, temperature=0)
        self.tools = [
            get_patient_ehr,
            check_drug_interactions,
            search_medical_articles,
            symptom_checker,
            get_diet_exercise_recommendations,
            run_collaborative_diagnosis_tool, # Integrated collaborative diagnosis as a tool
            get_medical_fact_tool,          # Integrated knowledge base query as a tool
        ]
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful medical assistant AI. You have access to various medical tools to provide accurate and personalized advice for chronic disease management. Always try to use the most relevant tools. If a tool requires JSON input for patient data or symptoms, ensure you provide valid JSON strings."),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )
        self.agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True, handle_parsing_errors=True)
        self.chat_history = []

    def process_query(self, query: str) -> str:
        """Processes a user query using the LLM and available tools."""
        print(f"\n[LLM CONTROLLER]: Processing query: {query}")
        try:
            result = self.agent_executor.invoke({"input": query, "chat_history": self.chat_history})
            response = result["output"]
            self.chat_history.append(HumanMessage(content=query))
            self.chat_history.append(AIMessage(content=response))
            return response
        except Exception as e:
            print(f"Error processing query: {e}")
            return "I apologize, but I encountered an error while processing your request. Please try again."

    def get_personalized_recommendation(self, patient_id: str, medical_condition: str) -> str:
        """Demonstrates a personalized learning aspect by proactively fetching relevant info."""
        print(f"\n[LLM CONTROLLER]: Generating personalized recommendation for patient {patient_id} with {medical_condition}.")
        ehr_info_raw = self.process_query(f"What is the EHR for patient {patient_id}?")
        # In a real app, parse EHR info more robustly
        ehr_summary = "(EHR details obtained)" # Simplified for demo

        diet_exercise_raw = self.process_query(f"Give diet and exercise recommendations for a patient with {medical_condition}.")
        diet_exercise_summary = "(Diet and exercise recommendations obtained)" # Simplified for demo

        return f"Based on patient {patient_id}\'s medical history {ehr_summary} and their condition ({medical_condition}), here are personalized recommendations: {diet_exercise_summary}. Full details: EHR: {ehr_info_raw}. Diet/Exercise: {diet_exercise_raw}"

################################################################################
# 5. Main Application Entry Point
################################################################################

def main():
    print("Initializing Personalized Medical Assistant...")
    controller = LLMController()

    print("\n--- Scenario 1: Basic Query and Tool Use (EHR) ---")
    response1 = controller.process_query("What are the medications and conditions for patient P123?")
    print(f"Assistant: {response1}")

    print("\n--- Scenario 2: Drug Interaction Check ---")
    response2 = controller.process_query("Are there any interactions between Metformin and Lisinopril?")
    print(f"Assistant: {response2}")

    print("\n--- Scenario 3: Symptom Analysis and Recommendation ---")
    response3 = controller.process_query("My patient P123 complains of headache and blurred vision. What should be done?")
    print(f"Assistant: {response3}")

    print("\n--- Scenario 4: Personalized Diet/Exercise Recommendation ---")
    response4 = controller.process_query("What diet and exercise advice would you give for someone with Type 2 Diabetes?")
    print(f"Assistant: {response4}")

    print("\n--- Scenario 5: Utilizing Knowledge Base (Formalism-Enhanced Reasoning) ---")
    response5 = controller.process_query("Tell me about Type 2 Diabetes.")
    print(f"Assistant: {response5}")

    print("\n--- Scenario 6: Multi-Agent Collaboration (Conceptual) ---")
    # The LLM should ideally parse the request and call the collaborative_diagnosis_tool
    patient_data_for_collab = {"patient_id": "P123", "conditions": ["Type 2 Diabetes", "Hypertension"], "vitals": {"blood pressure": "150/95", "blood sugar": 180}}
    symptoms_for_collab = ["fatigue", "chest pain"]
    # The prompt needs to guide the LLM to call the tool with JSON strings
    response6 = controller.process_query(f"Perform a comprehensive diagnostic analysis for patient P123. The patient has conditions: {patient_data_for_collab['conditions']} and symptoms: {symptoms_for_collab}. Current vitals are blood pressure {patient_data_for_collab['vitals']['blood pressure']} and blood sugar {patient_data_for_collab['vitals']['blood sugar']}. Consider these in a collaborative diagnosis.\n\nPatient Data JSON: {json.dumps(patient_data_for_collab)}\nSymptoms JSON: {json.dumps(symptoms_for_collab)}")
    print(f"Assistant: {response6}")

    print("\n--- Scenario 7: Personalized Learning (Proactive Recommendation) ---")
    response7 = controller.get_personalized_recommendation("P123", "Type 2 Diabetes")
    print(f"Assistant: {response7}")

    print("\n--- Scenario 8: AI Tool Creation (Conceptual) ---")
    print(f"Assistant (AI Tool Creation concept): {create_custom_health_script('generate a script for blood sugar monitoring')}")


if __name__ == "__main__":
    main()
