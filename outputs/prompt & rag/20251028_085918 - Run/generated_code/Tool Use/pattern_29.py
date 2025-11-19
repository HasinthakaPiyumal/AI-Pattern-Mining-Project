
class Tool:
    """Base class for all tools that the AI agent can use."""
    def __init__(self, name, description):
        self.name = name
        self.description = description

    def run(self, input_data):
        """Executes the tool with the given input data."""
        raise NotImplementedError("Each tool must implement its own run method.")

class MedicalKnowledgeBase(Tool):
    """A tool to access a database of medical information."""
    def __init__(self):
        super().__init__("MedicalKnowledgeBase", "Accesses a database of diseases, symptoms, and treatments.")

    def run(self, query):
        print(f"  [Tool: {self.name}] Querying for: {query}")
        # Simulate knowledge base lookup
        if "fever" in query.lower() and "cough" in query.lower():
            return "Potential diagnosis: Flu. Recommended test: Viral panel."
        if "hypertension" in query.lower():
            return "Hypertension is high blood pressure. Common medications include ACE inhibitors, beta-blockers."
        return "Information not found for the specific query."

class EHR_API(Tool):
    """A tool to retrieve patient Electronic Health Records."""
    def __init__(self):
        super().__init__("EHR_API", "Retrieves patient electronic health records.")

    def run(self, patient_id):
        print(f"  [Tool: {self.name}] Retrieving EHR for patient ID: {patient_id}")
        # Simulate EHR data retrieval
        if patient_id == "P101":
            return {"patient_id": "P101", "age": 45, "gender": "Male", "past_conditions": ["Hypertension"], "medications": ["Lisinopril"]}
        return {"error": "Patient not found."}

class LabSystem_API(Tool):
    """A tool to order and retrieve lab test results."""
    def __init__(self):
        super().__init__("LabSystem_API", "Orders and retrieves lab test results.")

    def run(self, order_details):
        print(f"  [Tool: {self.name}] Ordering lab test: {order_details}")
        # Simulate lab order and result
        if "viral panel" in order_details.lower():
            return {"test": "Viral Panel", "result": "Positive for Influenza A"}
        return {"status": "Lab order processed. Awaiting results."}

class ImageAnalyzerTool(Tool):
    """A tool to analyze medical images using AI models."""
    def __init__(self):
        super().__init__("ImageAnalyzerTool", "Analyzes medical images (e.g., X-rays, MRIs) using AI models.")

    def run(self, image_id):
        print(f"  [Tool: {self.name}] Analyzing image ID: {image_id}")
        # Simulate image analysis
        if image_id == "XRAY_001":
            return {"finding": "Mild lung inflammation detected.", "confidence": 0.85}
        return {"status": "Image analysis pending or no significant findings."}

class LLMAgent:
    """Simulates a Large Language Model agent for reasoning and planning."""
    def __init__(self, model_name="Medical LLM"):
        self.model_name = model_name
        self.context = [] # Simulating internal memory/context for the LLM

    def process_input(self, prompt):
        """Simulates LLM's understanding and initial thought process."""
        self.context.append(f"User Input: {prompt}")
        print(f"[LLM Agent] Processing input: '{prompt}'")
        # In a real scenario, this would involve calling the actual LLM API
        return f"Understood the patient's symptoms. Need to gather more information and plan diagnostic steps."

    def decide_tool_use(self, current_state, available_tools):
        """Simulates LLM deciding which tool to use based on current state."""
        print(f"[LLM Agent] Deciding which tool to use. Current state: {current_state}")
        self.context.append(f"Current State for decision: {current_state}")
        
        tool_names = [t.name.lower() for t in available_tools]

        if "patient id" in current_state.lower() or "ehr" in current_state.lower() and "ehr_api" in tool_names:
            return {"tool_name": "EHR_API", "parameters": {"patient_id": "P101"}}
        elif ("fever" in current_state.lower() or "cough" in current_state.lower()) and "medicalknowledgebase" in tool_names:
            return {"tool_name": "MedicalKnowledgeBase", "parameters": {"query": "symptoms of fever and cough"}}
        elif "need lab test" in current_state.lower() or "viral panel" in current_state.lower() and "labsystem_api" in tool_names:
            return {"tool_name": "LabSystem_API", "parameters": {"order_details": "viral panel"}}
        elif ("image_id" in current_state.lower() or "x-ray" in current_state.lower()) and "imageanalyzertool" in tool_names:
            return {"tool_name": "ImageAnalyzerTool", "parameters": {"image_id": "XRAY_001"}}
        elif "hypertension" in current_state.lower() and "medicalknowledgebase" in tool_names:
            return {"tool_name": "MedicalKnowledgeBase", "parameters": {"query": "hypertension treatment"}}

        return {"tool_name": None, "reason": "No specific tool identified for the current state based on keywords."}

    def integrate_tool_feedback(self, tool_output):
        """Simulates LLM integrating tool output into its reasoning."""
        print(f"[LLM Agent] Integrating tool feedback: {tool_output}")
        self.context.append(f"Tool Feedback: {tool_output}")
        # Based on feedback, the LLM would update its internal state, refine its plan, etc.
        return f"Feedback incorporated. Next steps will consider: {tool_output}"

    def formulate_recommendation(self):
        """Simulates LLM formulating the final recommendation."""
        print(f"[LLM Agent] Formulating final recommendation based on accumulated context.")
        final_context = " ".join(self.context)
        
        recommendation = "Based on available information, a general recommendation for symptom management is advised. Further tests may be needed."

        if "Positive for Influenza A" in final_context:
            recommendation = "Based on patient EHR, symptoms (fever, cough), and lab results (positive for Influenza A), the recommendation is antiviral medication and rest. Monitor for complications."
        elif "Mild lung inflammation detected" in final_context and "Hypertension" in final_context:
             recommendation = "Mild lung inflammation detected in X-ray. Patient has a history of hypertension. Further investigation with a CT scan might be beneficial, alongside symptomatic treatment and continued management of hypertension."
        elif "Mild lung inflammation detected" in final_context:
             recommendation = "Mild lung inflammation detected in X-ray. Further investigation with a CT scan might be beneficial, alongside symptomatic treatment."

        return recommendation

class MedicalDiagnosisAgent:
    """Orchestrates the LLM agent and specialized tools for medical diagnosis."""
    def __init__(self, llm_agent, tools):
        self.llm = llm_agent
        self.tools = {tool.name: tool for tool in tools}
        print("Medical Diagnosis Agent initialized with LLM and tools.")

    def run_diagnosis(self, patient_data):
        """Runs the complete diagnostic process for a given patient."""
        print("\n--- Starting Medical Diagnosis Process ---")
        
        # 1. Interpret patient data
        initial_llm_response = self.llm.process_input(patient_data)
        current_state = f"Patient presents with: {patient_data}. {initial_llm_response}"

        # 2. Iterative diagnosis planning and tool execution
        for step in range(4): # Simulate a few turns of agent-tool interaction
            print(f"\n--- Step {step + 1}: Agent Planning and Tool Use ---")
            
            # LLM decides which tool to use
            tool_decision = self.llm.decide_tool_use(current_state, list(self.tools.values()))
            
            if tool_decision and tool_decision["tool_name"]:
                tool_name = tool_decision["tool_name"]
                tool_params = tool_decision["parameters"]
                
                print(f"[Agent] Decided to use tool: {tool_name} with parameters: {tool_params}")
                
                # Execute the chosen tool
                chosen_tool = self.tools.get(tool_name)
                if chosen_tool:
                    tool_output = chosen_tool.run(**tool_params)
                    print(f"  [Tool Output] {tool_output}")
                    
                    # LLM integrates tool feedback
                    integration_response = self.llm.integrate_tool_feedback(tool_output)
                    current_state = f"{current_state}\nTool {tool_name} output: {tool_output}. Agent integration: {integration_response}"
                else:
                    print(f"[Agent Error] Tool '{tool_name}' not found.")
                    break
            else:
                print(f"[Agent] No suitable tool found or decision made. Reason: {tool_decision.get('reason', 'Unknown.')}")
                break

        # 3. Formulate final recommendation
        final_recommendation = self.llm.formulate_recommendation()
        print("\n--- Diagnosis Process Complete ---")
        print(f"\nFinal Medical Recommendation: {final_recommendation}")
        return final_recommendation

# Main execution block to demonstrate the Adaptive Tool-Augmented AI Agent
if __name__ == "__main__":
    # Initialize specialized tools
    medical_kb_tool = MedicalKnowledgeBase()
    ehr_api_tool = EHR_API()
    lab_api_tool = LabSystem_API()
    image_analyzer_tool = ImageAnalyzerTool()

    # List of available tools for the agent
    available_tools = [medical_kb_tool, ehr_api_tool, lab_api_tool, image_analyzer_tool]

    # Initialize the central LLM Agent
    llm_core_agent = LLMAgent()

    # Initialize the Medical Diagnosis Agent, orchestrating the LLM and tools
    medical_diagnosis_system = MedicalDiagnosisAgent(llm_core_agent, available_tools)

    # --- Scenario 1: Patient with flu-like symptoms and X-ray ---
    print("\n----- Running Scenario 1: Flu-like symptoms with X-ray and history -----")
    patient_symptoms_1 = "Patient ID P101, 45-year-old male with fever, cough, and general malaise for 3 days. Has a history of hypertension. Also has a recent chest X-ray taken, image ID XRAY_001."
    medical_diagnosis_system.run_diagnosis(patient_symptoms_1)

    # Reset LLM context for a new scenario (simulating a fresh patient interaction)
    llm_core_agent.context = []

    # --- Scenario 2: Patient with only general symptoms ---
    print("\n\n----- Running Scenario 2: General symptoms only -----")
    patient_symptoms_2 = "Patient presents with headache and fatigue. No other significant symptoms mentioned."
    medical_diagnosis_system.run_diagnosis(patient_symptoms_2)
