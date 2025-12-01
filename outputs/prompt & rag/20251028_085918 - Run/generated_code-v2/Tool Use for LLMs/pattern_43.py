import autogen

# --- Simulated Tools ---

def medical_knowledge_base_lookup(symptom: str) -> str:
    if "fever" in symptom.lower():
        return "Potential conditions: Infection, Influenza, Common Cold."
    return "No specific conditions found for this symptom in basic knowledge base."

def differential_diagnosis_generator(symptoms: str) -> str:
    if "fever, cough, fatigue" in symptoms.lower():
        return "Differential diagnoses: Viral infection, Bacterial pneumonia, Bronchitis."
    if "headache, stiff neck, fever" in symptoms.lower():
        return "Differential diagnoses: Meningitis, Severe migraine, Encephalitis."
    return "Could not generate a differential diagnosis based on provided symptoms."

def ehr_parser(medical_history_text: str) -> dict:
    history = {"conditions": [], "medications": [], "allergies": [], "family_history": []}
    if "diabetes" in medical_history_text.lower():
        history["conditions"].append("Diabetes Type 2")
    if "hypertension" in medical_history_text.lower():
        history["conditions"].append("Hypertension")
    if "insulin" in medical_history_text.lower():
        history["medications"].append("Insulin")
    if "penicillin allergy" in medical_history_text.lower():
        history["allergies"].append("Penicillin")
    return history

def medication_interaction_checker(medications: list) -> str:
    if "insulin" in [m.lower() for m in medications] and "beta-blocker" in [m.lower() for m in medications]:
        return "Warning: Potential interaction between Insulin and Beta-blockers (can mask hypoglycemia symptoms)."
    return "No significant drug interactions detected among the provided medications."

def image_analysis_tool(image_path: str) -> str:
    if "xray_lung_patientA.png" in image_path:
        return "Image analysis findings: Evidence of consolidation in the lower left lung lobe, suggestive of pneumonia."
    if "mri_brain_patientB.jpg" in image_path:
        return "Image analysis findings: Small lesion detected in the frontal cortex, further investigation recommended."
    return "No significant findings in the provided image."

def lab_reference_range_checker(lab_results: dict) -> dict:
    anomalies = {}
    if lab_results.get("glucose") and lab_results["glucose"] > 120:
        anomalies["glucose"] = "High (Normal: 70-100 mg/dL)"
    if lab_results.get("white_blood_cells") and lab_results["white_blood_cells"] > 10000:
        anomalies["white_blood_cells"] = "High (Normal: 4,000-10,000 cells/mcL)"
    return anomalies

def biomarker_correlator(lab_results: dict) -> str:
    if lab_results.get("white_blood_cells") and lab_results["white_blood_cells"] > 10000 and lab_results.get("crp") and lab_results["crp"] > 5:
        return "Elevated WBC and CRP strongly suggest an inflammatory or infectious process."
    return "No specific biomarker correlations identified."

def treatment_protocol_database(diagnosis: str) -> str:
    if "pneumonia" in diagnosis.lower():
        return "Treatment Protocol for Pneumonia: Antibiotics (e.g., Azithromycin or Amoxicillin), rest, hydration, oxygen therapy if needed."
    if "diabetes type 2" in diagnosis.lower():
        return "Treatment Protocol for Diabetes Type 2: Lifestyle modification, Metformin, insulin therapy if needed, regular monitoring."
    return "No specific treatment protocol found for this diagnosis."

def drug_dosage_calculator(drug: str, patient_info: dict) -> str:
    if drug.lower() == "amoxicillin" and patient_info.get("weight_kg"): 
        dosage = 25 * patient_info["weight_kg"]
        return f"Recommended Amoxicillin dosage: {dosage} mg per day, divided into 2-3 doses."
    return f"Cannot calculate dosage for {drug} without sufficient patient info."

def patient_preference_integrator(treatment_plan: str, preferences: str) -> str:
    if "oral medication" in preferences.lower() and "IV antibiotics" in treatment_plan.lower():
        return treatment_plan.replace("IV antibiotics", "oral antibiotics (if clinically appropriate)") + "\nNote: Patient prefers oral medication, adjusted plan accordingly where possible."
    return treatment_plan

# --- Agent Configuration ---

config_list = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_json={"model": ["gpt-4", "gpt-3.5-turbo", "gemini-pro"]},
)

llm_config_generic = {"config_list": config_list, "temperature": 0}

# User Proxy Agent
user_proxy = autogen.UserProxyAgent(
    name="User_Proxy",
    system_message="A human doctor who provides patient information and evaluates the diagnostic and treatment plan.",
    code_execution_config={
        "last_n_messages": 2,
        "work_dir": "coding",
        "use_docker": False
    },
    human_input_mode="ALWAYS",
)

# Symptoms Analysis Agent
symptoms_agent = autogen.AssistantAgent(
    name="Symptoms_Analysis_Agent",
    llm_config=llm_config_generic,
    system_message=(
        "You are an AI specializing in analyzing patient reported symptoms. "
        "Your goal is to identify potential conditions and suggest further diagnostic steps. "
        "You can use the medical_knowledge_base_lookup and differential_diagnosis_generator tools."    
    ),
)
user_proxy.register_for_execution(symptoms_agent)

# Medical History Agent
history_agent = autogen.AssistantAgent(
    name="Medical_History_Agent",
    llm_config=llm_config_generic,
    system_message=(
        "You are an AI specializing in parsing and interpreting patient medical history. "
        "Extract relevant conditions, medications, allergies, and family history. "
        "You can use the ehr_parser and medication_interaction_checker tools."        
    ),
)
user_proxy.register_for_execution(history_agent)

# Diagnostic Imaging Agent
imaging_agent = autogen.AssistantAgent(
    name="Diagnostic_Imaging_Agent",
    llm_config=llm_config_generic,
    system_message=(
        "You are an AI specializing in interpreting diagnostic images. "
        "Provide structured findings based on the image analysis tool results. "
        "You can use the image_analysis_tool."        
    ),
)
user_proxy.register_for_execution(imaging_agent)

# Lab Results Interpretation Agent
lab_agent = autogen.AssistantAgent(
    name="Lab_Results_Interpretation_Agent",
    llm_config=llm_config_generic,
    system_message=(
        "You are an AI specializing in interpreting laboratory test results. "
        "Flag abnormalities and correlate them with potential conditions. "
        "You can use the lab_reference_range_checker and biomarker_correlator tools."        
    ),
)
user_proxy.register_for_execution(lab_agent)

# Treatment Recommendation Agent
treatment_agent = autogen.AssistantAgent(
    name="Treatment_Recommendation_Agent",
    llm_config=llm_config_generic,
    system_message=(
        "You are an AI specializing in recommending comprehensive and personalized treatment plans. "
        "Synthesize information from all other agents and consider patient factors. "
        "You can use the treatment_protocol_database, drug_dosage_calculator, and patient_preference_integrator tools."        
    ),
)
user_proxy.register_for_execution(treatment_agent)

# Register tools for agents to use when necessary
symptoms_agent.register_for_tool_use(medical_knowledge_base_lookup, config_list=config_list)
symptoms_agent.register_for_tool_use(differential_diagnosis_generator, config_list=config_list)
history_agent.register_for_tool_use(ehr_parser, config_list=config_list)
history_agent.register_for_tool_use(medication_interaction_checker, config_list=config_list)
imaging_agent.register_for_tool_use(image_analysis_tool, config_list=config_list)
lab_agent.register_for_tool_use(lab_reference_range_checker, config_list=config_list)
lab_agent.register_for_tool_use(biomarker_correlator, config_list=config_list)
treatment_agent.register_for_tool_use(treatment_protocol_database, config_list=config_list)
treatment_agent.register_for_tool_use(drug_dosage_calculator, config_list=config_list)
treatment_agent.register_for_tool_use(patient_preference_integrator, config_list=config_list)

# --- Group Chat Setup ---

groupchat = autogen.GroupChat(
    agents=[
        user_proxy,
        symptoms_agent,
        history_agent,
        imaging_agent,
        lab_agent,
        treatment_agent,
    ],
    messages=[],
    max_round=15,
)

manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config_generic)

# --- Start Conversation ---

patient_case = """
Patient Information:
- Symptoms: Patient reports persistent fever (102 F), cough, and severe fatigue for 3 days.
- Medical History: Diagnosed with Type 2 Diabetes 5 years ago, currently on Metformin. No known allergies.
- Imaging: Chest X-ray performed (xray_lung_patientA.png).
- Lab Results: 
    - Glucose: 180 mg/dL
    - White Blood Cells: 12,000 cells/mcL
    - CRP: 15 mg/L
- Patient Preferences: Prefers oral medication if possible.
"""

user_proxy.initiate_chat(
    manager,
    message=f"Analyze the following patient case and propose a comprehensive diagnosis and treatment plan:\n{patient_case}"
)
