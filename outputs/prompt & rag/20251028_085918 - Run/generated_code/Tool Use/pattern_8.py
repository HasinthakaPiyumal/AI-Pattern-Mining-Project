import autogen
import json
import random
from typing import Dict, Any, List

# --- 1. Simulated APIs/Tools ---

class EHR_API:
    """Simulates an Electronic Health Record (EHR) system API."""
    def __init__(self):
        self.patient_records = {
            "patient_123": {
                "name": "Alice Smith",
                "age": 45,
                "gender": "Female",
                "conditions": ["Hypertension", "Type 2 Diabetes"],
                "medications": ["Lisinopril", "Metformin"],
                "allergies": ["Penicillin"],
                "history": "Diagnosed with hypertension 5 years ago, diabetes 2 years ago. No major surgeries."
            },
            "patient_456": {
                "name": "Bob Johnson",
                "age": 60,
                "gender": "Male",
                "conditions": ["Coronary Artery Disease"],
                "medications": ["Aspirin", "Atorvastatin"],
                "allergies": [],
                "history": "Had a myocardial infarction 3 years ago. Underwent angioplasty."
            }
        }

    def get_patient_history(self, patient_id: str) -> Dict[str, Any]:
        """Retrieves a patient's full medical history."""
        print(f"[EHR_API] Retrieving history for {patient_id}")
        return self.patient_records.get(patient_id, {"error": "Patient not found"})

    def get_current_conditions(self, patient_id: str) -> List[str]:
        """Retrieves current active conditions for a patient."""
        print(f"[EHR_API] Retrieving conditions for {patient_id}")
        record = self.patient_records.get(patient_id)
        return record["conditions"] if record else []

    def get_demographics(self, patient_id: str) -> Dict[str, Any]:
        """Retrieves demographic information for a patient."""
        print(f"[EHR_API] Retrieving demographics for {patient_id}")
        record = self.patient_records.get(patient_id)
        return {k: record[k] for k in ["name", "age", "gender"]} if record else {}

class MedicalKnowledgeBase_API:
    """Simulates a Medical Knowledge Base API (e.g., PubMed, drug databases)."""
    def __init__(self):
        self.knowledge = {
            "Hypertension": "High blood pressure. Can lead to heart disease, stroke. Management includes lifestyle changes and medication (e.g., ACE inhibitors, diuretics).",
            "Type 2 Diabetes": "Chronic condition affecting blood sugar regulation. Managed with diet, exercise, and medication (e.g., Metformin, insulin).",
            "Lisinopril": "ACE inhibitor, used for hypertension. Side effects: cough, dizziness.",
            "Metformin": "Oral medication for Type 2 Diabetes. Side effects: gastrointestinal issues.",
            "Penicillin": "Antibiotic. Common allergy causing rash, anaphylaxis.",
            "Coronary Artery Disease": "Narrowing of heart arteries, often due to plaque buildup. Can cause angina, heart attack.",
            "Aspirin": "Antiplatelet drug, used to prevent blood clots. Side effects: bleeding.",
            "Atorvastatin": "Statin, used to lower cholesterol. Side effects: muscle pain."
        }
        self.drug_interactions = {
            "Lisinopril+Potassium_Supplements": "Potential for hyperkalemia (high potassium). Monitor levels.",
            "Metformin+Iodinated_Contrast": "Risk of lactic acidosis. Temporarily discontinue Metformin before and after procedure."
        }

    def get_medical_facts(self, query: str) -> str:
        """Retrieves general medical facts about a condition or drug."""
        print(f"[MedicalKnowledgeBase_API] Searching facts for '{query}'")
        return self.knowledge.get(query, "No specific information found in knowledge base.")

    def get_condition_details(self, condition: str) -> str:
        """Provides detailed information about a specific medical condition."""
        print(f"[MedicalKnowledgeBase_API] Getting details for condition '{condition}'")
        return self.knowledge.get(condition, f"Details for {condition} not found.")

    def check_drug_interaction(self, drug1: str, drug2: str) -> str:
        """Checks for potential interactions between two drugs."""
        interaction1 = self.drug_interactions.get(f"{drug1}+{drug2}")
        interaction2 = self.drug_interactions.get(f"{drug2}+{drug1}")
        if interaction1:
            print(f"[MedicalKnowledgeBase_API] Checking drug interaction for {drug1} and {drug2}: {interaction1}")
            return interaction1
        elif interaction2:
            print(f"[MedicalKnowledgeBase_API] Checking drug interaction for {drug1} and {drug2}: {interaction2}")
            return interaction2
        else:
            print(f"[MedicalKnowledgeBase_API] No significant interaction found between {drug1} and {drug2} (simulated). ")
            return "No significant interaction found (simulated)."

    def check_drug_condition_interaction(self, drug: str, condition: str) -> str:
        """Checks for potential interactions between a drug and a patient's condition."""
        # Simplified simulation
        if drug == "Metformin" and condition == "Kidney Disease":
            return "Metformin is contraindicated in severe renal impairment due to risk of lactic acidosis."
        print(f"[MedicalKnowledgeBase_API] Checking drug-condition interaction for {drug} and {condition}. No specific interaction found (simulated).")
        return "No specific interaction found (simulated)."

def MedicalImageAnalysis_Tool(image_url: str) -> str:
    """Simulates a medical image analysis tool."
    In a real scenario, this would use computer vision models.
    """
    print(f"[MedicalImageAnalysis_Tool] Analyzing image from {image_url}")
    # Simulate a result based on a dummy URL
    if "xray_pneumonia" in image_url:
        return "Image analysis suggests presence of bilateral infiltrates, consistent with pneumonia."
    elif "mri_brain_tumor" in image_url:
        return "Image analysis indicates a focal lesion in the frontal lobe, suggestive of a tumor."
    else:
        return "Image analysis completed. No significant abnormalities detected (simulated)."

class ClinicalTrials_API:
    """Simulates an API for searching clinical trials."""
    def __init__(self):
        self.trials = {
            "Hypertension": [
                {"title": "Trial on Novel Antihypertensive Drug", "status": "Recruiting", "phases": "Phase 3"},
                {"title": "Lifestyle Intervention for Hypertension", "status": "Active, not recruiting", "phases": "Phase 2"}
            ],
            "Type 2 Diabetes": [
                {"title": "SGLT2 Inhibitors in Diabetic Patients", "status": "Recruiting", "phases": "Phase 3"},
                {"title": "AI-Powered Glucose Monitoring Trial", "status": "Recruiting", "phases": "Phase 2"}
            ],
            "Coronary Artery Disease": [
                {"title": "Gene Therapy for CAD", "status": "Recruiting", "phases": "Phase 1"}
            ]
        }

    def search_clinical_trials(self, condition: str, phase: str = None) -> List[Dict[str, Any]]:
        """Searches for clinical trials based on condition and optional phase."""
        print(f"[ClinicalTrials_API] Searching trials for '{condition}' (Phase: {phase if phase else 'Any'})")
        results = self.trials.get(condition, [])
        if phase:
            results = [t for t in results if t["phases"] == phase]
        return results

# --- 2. Supporting Components ---

class KnowledgeRetriever:
    """Simulates a retriever-aware component for augmenting LLM context."""
    def __init__(self):
        self.documents = {
            "diabetes_guidelines": "Latest guidelines for Type 2 Diabetes management emphasize early glycemic control, weight management, and cardiovascular risk reduction. Recommended medications include Metformin as first-line, followed by SGLT2 inhibitors or GLP-1 receptor agonists based on comorbidities.",
            "hypertension_treatment_options": "Treatment for hypertension typically starts with lifestyle modifications. If ineffective, medication classes like ACE inhibitors, ARBs, Thiazide diuretics, or Calcium Channel Blockers are used. Combination therapy is common.",
            "pneumonia_diagnosis": "Pneumonia diagnosis often involves chest X-ray, sputum culture, and clinical symptoms like cough, fever, and shortness of breath."
        }

    def retrieve_information(self, query: str) -> str:
        """Retrieves relevant documents based on a query."""
        print(f"[KnowledgeRetriever] Retrieving information for '{query}'")
        for key, doc in self.documents.items():
            if query.lower() in key.lower() or query.lower() in doc.lower():
                return f"Retrieved relevant document: {doc}"
        return "No highly relevant document found (simulated)."

class FactChecker:
    """A simplified component to conceptually detect hallucinations."
    In a real system, this would involve AST parsing, knowledge graph querying, etc.
    """
    def __init__(self):
        self.known_facts = {
            "Metformin treats diabetes": True,
            "Lisinopril treats hypertension": True,
            "Penicillin causes allergy": True,
            "Aspirin prevents heart attacks": True
        }

    def check_statement(self, statement: str) -> bool:
        """Conceptually checks if a statement is factually consistent."
        This is a very basic simulation.
        """
        print(f"[FactChecker] Checking statement: '{statement}'")
        for fact, truth in self.known_facts.items():
            if fact.lower() in statement.lower() and truth:
                return True
        # Simple negation check (very rudimentary)
        if "does not treat" in statement.lower() or "not prevent" in statement.lower():
            return False # Assume negations of known facts are false for demo
        return random.choice([True, False]) # Random for unknown facts in simulation

class UserProfileManager:
    """Manages user (doctor) preferences for personalized tool learning."""
    def __init__(self):
        self.profiles = {
            "Dr.Lee": {
                "preferred_drug_db": "Drugs.com",
                "focus_area": "Cardiology",
                "verbose_output": True
            },
            "Dr.Chen": {
                "preferred_drug_db": "Medscape",
                "focus_area": "Endocrinology",
                "verbose_output": False
            }
        }

    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Retrieves preferences for a given user."""
        print(f"[UserProfileManager] Getting preferences for {user_id}")
        return self.profiles.get(user_id, {})

    def update_user_preference(self, user_id: str, key: str, value: Any):
        """Updates a specific preference for a user."""
        if user_id not in self.profiles:
            self.profiles[user_id] = {}
        self.profiles[user_id][key] = value
        print(f"[UserProfileManager] Updated {key} for {user_id} to {value}")

# --- Instantiate Tools and Helpers ---
ehr_api = EHR_API()
medical_kb_api = MedicalKnowledgeBase_API()
clinical_trials_api = ClinicalTrials_API()
knowledge_retriever = KnowledgeRetriever()
fact_checker = FactChecker()
user_profile_manager = UserProfileManager()

# --- 3. AutoGen Agents Configuration ---

# LLM configuration for all agents (replace with your actual API key and model)
llm_config = {
    "config_list": [
        {
            "model": "gpt-4", # Or "gpt-3.5-turbo", or your local LLM endpoint
            "api_key": "YOUR_OPENAI_API_KEY", # Replace with your actual API key
        }
    ],
    "temperature": 0.7,
    "timeout": 120,
}

# --- Define Agents ---

# User Proxy Agent
user_proxy = autogen.UserProxyAgent(
    name="Doctor_User",
    human_input_mode="ALWAYS", # Allows interactive input from the user/doctor
    max_consecutive_auto_reply=10, # Adjust as needed
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={"last_n_messages": 3, "work_dir": "coding"},
    llm_config=llm_config,
    system_message=(
        "You are a human doctor interacting with an AI Clinical Decision Support System. "
        "You will ask medical questions, provide patient details, and evaluate the AI's recommendations. "
        "Type 'TERMINATE' when you are done with the consultation." 
    )
)

# Diagnosis Agent
dia_agent = autogen.AssistantAgent(
    name="Diagnosis_Agent",
    llm_config=llm_config,
    system_message=(
        "You are an AI Diagnosis Specialist. Your role is to analyze patient data, symptoms, and medical history "
        "to suggest potential diagnoses. You have access to EHR_API and MedicalKnowledgeBase_API. "
        "Always use the tools to gather necessary information before making a diagnosis. "
        "Present your diagnosis with supporting evidence and a confidence level. "
        "If you need more information, ask the Doctor_User clearly. "
        "After providing a diagnosis, consider passing the conversation to the Treatment_Planning_Agent. "
        "You can retrieve patient history using `ehr_api.get_patient_history(patient_id)`. "
        "You can get medical facts using `medical_kb_api.get_medical_facts(query)`. "
        "You can get condition details using `medical_kb_api.get_condition_details(condition)`."
    )
)

# Treatment Planning Agent
treatment_agent = autogen.AssistantAgent(
    name="Treatment_Planning_Agent",
    llm_config=llm_config,
    system_message=(
        "You are an AI Treatment Planning Specialist. Your role is to propose personalized treatment plans "
        "based on a confirmed diagnosis, patient's medical history, and latest clinical guidelines. "
        "You have access to MedicalKnowledgeBase_API and ClinicalTrials_API. "
        "Consider standard treatments, potential new therapies, and relevant clinical trials. "
        "Always check for drug interactions if medications are proposed. "
        "You can search clinical trials using `clinical_trials_api.search_clinical_trials(condition, phase=None)`. "
        "You can get medical facts using `medical_kb_api.get_medical_facts(query)`. "
        "You can check drug interactions using `medical_kb_api.check_drug_interaction(drug1, drug2)`. "
        "You can check drug-condition interactions using `medical_kb_api.check_drug_condition_interaction(drug, condition)`." 
    )
)

# Drug Interaction Agent
drug_agent = autogen.AssistantAgent(
    name="Drug_Interaction_Agent",
    llm_config=llm_config,
    system_message=(
        "You are an AI Drug Interaction Specialist. Your primary role is to rigorously check for any "
        "potential drug-drug or drug-condition interactions given a list of medications and patient conditions. "
        "You have access to MedicalKnowledgeBase_API. "
        "Provide clear warnings and recommendations if interactions are found. "
        "You must use `medical_kb_api.check_drug_interaction(drug1, drug2)` for drug-drug checks. "
        "You must use `medical_kb_api.check_drug_condition_interaction(drug, condition)` for drug-condition checks."
    )
)

# Coordinator Agent (LLM Controller)
coordinator = autogen.AssistantAgent(
    name="Coordinator_Agent",
    llm_config=llm_config,
    system_message=(
        "You are the central AI Coordinator for a Clinical Decision Support System. "
        "Your job is to understand the Doctor_User's query, delegate tasks to the appropriate specialized agents "
        "(Diagnosis_Agent, Treatment_Planning_Agent, Drug_Interaction_Agent), "
        "synthesize their responses, and present a comprehensive answer back to the Doctor_User. "
        "Ensure all necessary information is gathered by guiding the conversation. "
        "Use the KnowledgeRetriever and FactChecker for enhancing reasoning and verifying information. "
        "You can retrieve general information using `knowledge_retriever.retrieve_information(query)`. "
        "You can check statements for factual consistency using `fact_checker.check_statement(statement)`. "
        "You can manage user profiles for personalization using `user_profile_manager.get_user_preferences(user_id)` "
        "and `user_profile_manager.update_user_preference(user_id, key, value)`."
    )
)

# --- Register Tools with Agents ---

user_proxy.register_for_execution(ehr_api.get_patient_history, name="get_patient_history")
user_proxy.register_for_execution(ehr_api.get_current_conditions, name="get_current_conditions")
user_proxy.register_for_execution(ehr_api.get_demographics, name="get_demographics")
user_proxy.register_for_execution(medical_kb_api.get_medical_facts, name="get_medical_facts")
user_proxy.register_for_execution(medical_kb_api.get_condition_details, name="get_condition_details")
user_proxy.register_for_execution(medical_kb_api.check_drug_interaction, name="check_drug_interaction")
user_proxy.register_for_execution(medical_kb_api.check_drug_condition_interaction, name="check_drug_condition_interaction")
user_proxy.register_for_execution(MedicalImageAnalysis_Tool, name="MedicalImageAnalysis_Tool")
user_proxy.register_for_execution(clinical_trials_api.search_clinical_trials, name="search_clinical_trials")
user_proxy.register_for_execution(knowledge_retriever.retrieve_information, name="retrieve_information")
user_proxy.register_for_execution(fact_checker.check_statement, name="check_statement")
user_proxy.register_for_execution(user_profile_manager.get_user_preferences, name="get_user_preferences")
user_proxy.register_for_execution(user_profile_manager.update_user_preference, name="update_user_preference")


# For clarity, agents can call tools directly via a user proxy or if they have code execution enabled
# Here, the user_proxy serves as the execution engine for all tools for simplicity in this group chat setup.
# The system messages guide the LLMs on *when* to suggest calling these tools.

# --- 4. Multi-Agent Group Chat --- 

# Define the group chat
groupchat = autogen.GroupChat(
    agents=[user_proxy, coordinator, dia_agent, treatment_agent, drug_agent],
    messages=[],
    max_round=20,
    speaker_selection_method="auto", # Let AutoGen decide who speaks next
    allow_repeat_speaker=False,
)

# Create the manager for the group chat
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

# --- 5. Demonstration / Workflow --- 

if __name__ == "__main__":
    print("\n--- Starting Clinical Decision Support System ---\n")
    print("Doctor_User: Please enter your medical query. Type 'TERMINATE' to exit.\n")

    # Start the conversation
    user_proxy.initiate_chat(
        manager,
        message=(
            "As Dr. Lee, I have a patient (patient_123) with hypertension and type 2 diabetes. "
            "They are currently on Lisinopril and Metformin. "
            "I'd like to understand potential drug interactions and get recommendations for optimizing their treatment plan. "
            "Also, tell me about any relevant clinical trials for Type 2 Diabetes."
        )
    )

    print("\n--- End of Consultation ---\n")


