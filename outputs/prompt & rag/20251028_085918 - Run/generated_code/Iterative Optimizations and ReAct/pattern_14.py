
import json
import time
from datetime import datetime

# --- 1. Tool Integration Layer --- 

class HealthcareTools:
    """Simulates various tools for healthcare data retrieval and actions."""

    def __init__(self):
        self.ehr_data_store = {
            "patient_001": {
                "medical_history": ["Type 2 Diabetes", "Hypertension"],
                "diagnoses": {"diabetes": "diagnosed_2020", "hypertension": "diagnosed_2021"},
                "medications": ["Metformin", "Lisinopril"],
                "allergies": ["Penicillin"]
            }
        }
        self.wearable_data_store = {
            "patient_001": {
                "heart_rate": [], # (timestamp, value)
                "blood_pressure": [], # (timestamp, systolic, diastolic)
                "glucose_levels": [], # (timestamp, value)
                "activity": [] # (timestamp, steps)
            }
        }
        self.feedback_logs = {}
        self.doctor_alerts = []

    def get_ehr_data(self, patient_id):
        """Simulates fetching relevant medical history, diagnoses, and medication from EHRs."""
        print(f"[Tool] Fetching EHR data for {patient_id}...")
        return self.ehr_data_store.get(patient_id, {"error": "Patient not found in EHR."})

    def get_wearable_data(self, patient_id, data_type, limit=5):
        """Simulates retrieving real-time health metrics from wearable devices."""
        print(f"[Tool] Retrieving latest {limit} {data_type} data for {patient_id}...")
        data = self.wearable_data_store.get(patient_id, {}).get(data_type, [])
        return data[-limit:] # Return last 'limit' entries

    def make_dietary_recommendation(self, health_goals, dietary_restrictions):
        """Generates personalized dietary advice based on goals and restrictions."""
        print(f"[Tool] Generating dietary recommendation for goals: {health_goals}, restrictions: {dietary_restrictions}...")
        recommendation = f"Consider a diet low in processed sugars and sodium. Focus on whole grains, lean proteins, and plenty of vegetables. Given {', '.join(dietary_restrictions)}, avoid those ingredients." # Simple logic
        return {"type": "dietary", "recommendation": recommendation}

    def suggest_exercise_plan(self, health_goals, physical_limitations):
        """Creates tailored exercise routines."""
        print(f"[Tool] Suggesting exercise plan for goals: {health_goals}, limitations: {physical_limitations}...")
        plan = f"Start with light cardio (e.g., walking 30 min/day) and incorporate strength training 2-3 times a week. Avoid high-impact activities if {', '.join(physical_limitation)}." # Simple logic
        return {"type": "exercise", "plan": plan}

    def alert_doctor(self, patient_id, anomaly_details):
        """Simulates alerting a healthcare professional in case of critical anomalies."""
        print(f"[Tool] !!! ALERTING DOCTOR for {patient_id}: {anomaly_details} !!!")
        self.doctor_alerts.append({
            "timestamp": datetime.now().isoformat(),
            "patient_id": patient_id,
            "anomaly": anomaly_details
        })
        return {"status": "Doctor alerted", "details": anomaly_details}

    def provide_educational_content(self, topic):
        """Offers relevant information about the chronic disease."""
        print(f"[Tool] Providing educational content on topic: {topic}...")
        content = f"Understanding {topic}: {topic} is a complex condition... (simplified educational text)"
        return {"type": "educational", "topic": topic, "content": content}

    def log_feedback(self, patient_id, action_type, feedback_type, feedback_content):
        """Records patient feedback on actions."""
        if patient_id not in self.feedback_logs:
            self.feedback_logs[patient_id] = []
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "action_type": action_type,
            "feedback_type": feedback_type,
            "content": feedback_content
        }
        self.feedback_logs[patient_id].append(log_entry)
        print(f"[Tool] Logged feedback for {patient_id} on {action_type}: {feedback_content}")
        return {"status": "Feedback logged"}

    def record_wearable_data(self, patient_id, data_type, value1, value2=None):
        """Simulates recording new wearable data."""
        timestamp = datetime.now().isoformat()
        if patient_id not in self.wearable_data_store:
            self.wearable_data_store[patient_id] = {
                "heart_rate": [], "blood_pressure": [], "glucose_levels": [], "activity": []
            }
        if data_type == "blood_pressure" and value2 is not None:
            self.wearable_data_store[patient_id][data_type].append((timestamp, value1, value2))
            print(f"[Tool] Recorded {data_type}: {value1}/{value2} for {patient_id}")
        else:
            self.wearable_data_store[patient_id][data_type].append((timestamp, value1))
            print(f"[Tool] Recorded {data_type}: {value1} for {patient_id}")
        return {"status": "Data recorded"}


# --- 2. Core LLM Agent (`AdaptiveHealthcareAgent`) --- 

class AdaptiveHealthcareAgent:
    """An adaptive AI agent for personalized chronic disease management."""

    def __init__(self, patient_id, tools: HealthcareTools):
        self.patient_id = patient_id
        self.tools = tools
        self.memory = {
            "patient_id": patient_id,
            "current_health_status": {},
            "preferences": {"diet": "vegetarian", "exercise_intensity": "moderate"},
            "past_recommendations": [],
            "learned_insights": []
        }
        self.system_prompt = (
            f"You are an Adaptive Healthcare Navigator for patient {self.patient_id}. "
            "Your goal is to provide personalized, adaptive guidance for chronic disease management. "
            "You have access to various tools to gather data and provide recommendations. "
            "Continuously reason, act, process feedback, and refine your approach."
        )

    def _simulate_llm_response(self, prompt, context):
        """A simple function to simulate LLM behavior based on prompt and context."
        In a real application, this would call an actual LLM (e.g., via OpenAI API, Hugging Face transformers).
        """
        print(f"[LLM] Simulating LLM response for: {prompt[:100]}...")
        response = {"thought": "", "action": None, "plan": []}

        # Simple rule-based simulation of LLM decision making
        if "initial assessment" in prompt.lower() or not context.get("current_health_status"): # Initial state or need for data
            response["thought"] = "Initial assessment needed. Gathering EHR and latest wearable data."
            response["plan"] = [
                {"tool": "get_ehr_data", "args": {"patient_id": self.patient_id}},
                {"tool": "get_wearable_data", "args": {"patient_id": self.patient_id, "data_type": "heart_rate"}},
                {"tool": "get_wearable_data", "args": {"patient_id": self.patient_id, "data_type": "blood_pressure"}},
                {"tool": "get_wearable_data", "args": {"patient_id": self.patient_id, "data_type": "glucose_levels"}}
            ]
        elif "high" in str(context.get("current_health_status")).lower() and "blood_pressure" in str(context.get("current_health_status")).lower():
            response["thought"] = "Blood pressure is high. Need to recommend actions and potentially alert doctor."
            response["plan"] = [
                {"tool": "make_dietary_recommendation", "args": {"health_goals": "lower blood pressure", "dietary_restrictions": self.memory["preferences"]["diet"]}},
                {"tool": "suggest_exercise_plan", "args": {"health_goals": "lower blood pressure", "physical_limitations": []}},
                {"tool": "alert_doctor", "args": {"patient_id": self.patient_id, "anomaly_details": "Elevated blood pressure detected."}}
            ]
        elif "low glucose" in str(context.get("current_health_status")).lower():
            response["thought"] = "Glucose levels are low. Provide immediate guidance and log."
            response["plan"] = [
                {"tool": "provide_educational_content", "args": {"topic": "managing hypoglycemia"}},
                {"tool": "log_feedback", "args": {"patient_id": self.patient_id, "action_type": "hypoglycemia_guidance", "feedback_type": "system_action", "feedback_content": "Provided guidance for low glucose."}}
            ]
        elif "patient feedback" in prompt.lower() and "not effective" in prompt.lower():
            response["thought"] = "Patient feedback indicates previous recommendation was not effective. Need to refine and try a new approach."
            response["plan"] = [
                {"tool": "make_dietary_recommendation", "args": {"health_goals": "stable blood sugar, alternative diet", "dietary_restrictions": [self.memory["preferences"]["diet"], "new approach"]}}
            ]
        elif "general guidance" in prompt.lower():
            response["thought"] = "Providing general health guidance."
            response["plan"] = [
                {"tool": "make_dietary_recommendation", "args": {"health_goals": "general health", "dietary_restrictions": self.memory["preferences"]["diet"]}},
                {"tool": "suggest_exercise_plan", "args": {"health_goals": "general fitness", "physical_limitations": []}}
            ]
        else:
            response["thought"] = "Analyzing current context and patient input to form a plan."
            response["plan"] = [
                {"tool": "provide_educational_content", "args": {"topic": "chronic disease management"}}
            ]

        return response

    def _dynamic_plan(self, current_situation, goals):
        """Generates and adapts action plans driven by the simulated LLM."""
        prompt = f"Given the current situation: {current_situation} and goals: {goals}, what is the best plan of action?"
        context = {"current_health_status": current_situation, "goals": goals, "memory": self.memory}
        llm_output = self._simulate_llm_response(prompt, context)
        print(f"[Agent] Dynamic Plan Generated: {llm_output['thought']}")
        return llm_output.get("plan", [])

    def _self_evaluate(self, action_outcome, expected_outcome, criteria):
        """Internally evaluates the effectiveness of actions and recommendations."""
        print(f"[Agent] Self-evaluating action outcome: {action_outcome} against expected: {expected_outcome}...")
        evaluation = {"effective": False, "reason": ""}

        if "success" in str(action_outcome).lower() and "target" in str(expected_outcome).lower():
            evaluation["effective"] = True
            evaluation["reason"] = "Action achieved the desired outcome."
        elif "high" in str(action_outcome).lower() and "blood_pressure" in str(action_outcome).lower() and "normalize" in str(expected_outcome).lower():
            evaluation["effective"] = False
            evaluation["reason"] = "Blood pressure remains high despite intervention."
        else:
            evaluation["reason"] = "Outcome is not explicitly matching expected criteria, further monitoring needed."

        return evaluation

    def _refine_strategy(self, evaluation_result):
        """Refines the agent's understanding and future plans based on evaluation and feedback."""
        print(f"[Agent] Refining strategy based on evaluation: {evaluation_result['reason']}")
        if not evaluation_result["effective"]:
            insight = f"Learned that previous strategy was ineffective because: {evaluation_result['reason']}. Need to try alternative approaches or gather more data."
            self.memory["learned_insights"].append(insight)
            print(f"[Agent] Updated learned insights: {insight}")
            return True # Indicates refinement happened, might trigger re-planning
        return False # No significant refinement needed

    def process_patient_input(self, patient_input):
        """Main method to process patient input and drive the adaptive agentic loop."
        This simulates a single turn of interaction.
        """
        print(f"\n--- Processing Patient Input for {self.patient_id}: '{patient_input}' ---")

        # 1. Reason & Plan
        current_situation = self.memory["current_health_status"]
        goals = f"Address patient input '{patient_input}' and manage chronic conditions."

        # Include patient input in the prompt for LLM to consider
        llm_prompt_with_input = f"Patient says: '{patient_input}'. {self.system_prompt} Current health: {current_situation}. Goals: {goals}. What's the next best plan?"
        
        plan = self._dynamic_plan(current_situation, goals) # LLM-driven planning

        if not plan:
            print("[Agent] No specific plan generated. Providing general advice.")
            plan = [{"tool": "provide_educational_content", "args": {"topic": "general chronic disease self-management"}}]

        # 2. Act through integrated tools
        action_outcomes = []
        for step in plan:
            tool_name = step["tool"]
            tool_args = step["args"]
            print(f"[Agent] Executing tool: {tool_name} with args: {tool_args}")
            tool_func = getattr(self.tools, tool_name, None)
            if tool_func:
                try:
                    outcome = tool_func(**tool_args)
                    action_outcomes.append({"tool": tool_name, "outcome": outcome})
                    # Update memory based on data retrieval tools
                    if tool_name == "get_ehr_data":
                        self.memory["current_health_status"].update({"ehr": outcome})
                    elif tool_name.startswith("get_wearable_data"):
                        data_type = tool_args.get("data_type")
                        if data_type:
                            self.memory["current_health_status"][data_type] = outcome

                    self.memory["past_recommendations"].append(outcome)
                except Exception as e:
                    print(f"[Agent Error] Tool {tool_name} failed: {e}")
                    action_outcomes.append({"tool": tool_name, "error": str(e)})
            else:
                print(f"[Agent Error] Unknown tool: {tool_name}")
                action_outcomes.append({"tool": tool_name, "error": "Tool not found"})
        
        # 3. Process Feedback (simulated internal feedback for simplicity, but could be external)
        # For this demo, let's simulate some feedback for evaluation
        simulated_expected_outcome = "normalize blood pressure" if "high blood pressure" in str(current_situation).lower() else "maintain health"
        
        overall_evaluation = {"effective": True, "reason": "All actions executed.", "tool_evaluations": []}
        for outcome_entry in action_outcomes:
            tool_eval = self._self_evaluate(outcome_entry, simulated_expected_outcome, criteria="general health improvement")
            overall_evaluation["tool_evaluations"].append(tool_eval)
            if not tool_eval["effective"]:
                overall_evaluation["effective"] = False
                overall_evaluation["reason"] = f"One or more actions were ineffective: {tool_eval['reason']}"

        # 4. Iterative Self-Refinement
        if self._refine_strategy(overall_evaluation):
            print("[Agent] Strategy refined. Considering re-planning with new insights.")
            # In a full loop, this would lead to another planning phase
        else:
            print("[Agent] Current strategy seems effective or no major refinement needed.")

        print("--- Processing Complete ---")
        return {"actions_taken": action_outcomes, "memory_snapshot": self.memory}

    def run_scenario(self):
        """Demonstrates the adaptive agent's behavior through a simple scenario."""
        print("\n===== Starting Healthcare Navigator Scenario =====")

        # Initial data entry (simulated wearable input)
        self.tools.record_wearable_data(self.patient_id, "heart_rate", 75)
        self.tools.record_wearable_data(self.patient_id, "blood_pressure", 135, 85)
        self.tools.record_wearable_data(self.patient_id, "glucose_levels", 120)
        self.tools.record_wearable_data(self.patient_id, "activity", 5000)

        # Scenario 1: Initial assessment and recommendations
        self.process_patient_input("I want to improve my overall health and manage my diabetes.")
        time.sleep(1) # Simulate time passing

        # Scenario 2: High blood pressure detected from new data
        self.tools.record_wearable_data(self.patient_id, "blood_pressure", 150, 95) # High reading
        self.process_patient_input("I've been feeling a bit stressed lately.")
        time.sleep(1)

        # Scenario 3: Patient gives negative feedback on diet recommendation
        self.tools.log_feedback(self.patient_id, "dietary_recommendation", "negative", "The last diet plan was too restrictive and hard to follow.")
        self.process_patient_input("The diet plan you gave me was not effective for me.")
        time.sleep(1)

        # Scenario 4: General check-up, agent should provide educational content or routine advice
        self.process_patient_input("Just checking in, how am I doing?")
        time.sleep(1)

        print("\n===== Scenario Complete =====")
        print("\nFinal Agent Memory Snapshot:")
        print(json.dumps(self.memory, indent=2))
        print("\nDoctor Alerts:")
        print(json.dumps(self.tools.doctor_alerts, indent=2))
        print("\nPatient Feedback Logs:")
        print(json.dumps(self.tools.feedback_logs, indent=2))

# --- Main Execution --- 
if __name__ == "__main__":
    # Initialize tools
    healthcare_tools = HealthcareTools()

    # Initialize the agent for a specific patient
    patient_id = "patient_001"
    agent = AdaptiveHealthcareAgent(patient_id, healthcare_tools)

    # Run the predefined scenario
    agent.run_scenario()
