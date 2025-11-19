import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import random

# Mock Database/Data Storage
patient_data = {
    "user_id_1": {
        "name": "Alice Johnson",
        "age": 45,
        "conditions": ["Type 2 Diabetes", "Hypertension"],
        "medications": ["Metformin", "Lisinopril"],
        "glucose_readings": [
            {"date": (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"), "value": 145},
            {"date": (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d"), "value": 160},
            {"date": (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d"), "value": 152},
        ],
        "blood_pressure_readings": [
            {"date": (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"), "systolic": 135, "diastolic": 85},
        ]
    }
}

medical_guidelines = [
    "For Type 2 Diabetes, aim for fasting glucose below 130 mg/dL.",
    "Normal blood pressure is typically below 120/80 mmHg.",
    "Metformin and Lisinopril can be safely taken together for diabetes and hypertension.",
    "High blood sugar can lead to fatigue, increased thirst, and frequent urination.",
    "Regular exercise and a balanced diet are crucial for managing diabetes and hypertension."
]

# Mock Tool Definitions
class HealthcareTools:
    def get_glucose_reading(self, user_id: str, date: str = None) -> dict:
        if user_id in patient_data:
            readings = patient_data[user_id].get("glucose_readings", [])
            if date:
                for r in readings:
                    if r["date"] == date:
                        return {"status": "success", "data": r}
                return {"status": "error", "message": f"No glucose reading found for {date}"}
            elif readings:
                return {"status": "success", "data": readings[-1]} 
            return {"status": "error", "message": "No glucose readings available"}
        return {"status": "error", "message": "User not found"}

    def suggest_meal_plan(self, user_id: str, dietary_restrictions: list = None) -> dict:
        if user_id in patient_data:
            restrictions = dietary_restrictions if dietary_restrictions else []
            if "Type 2 Diabetes" in patient_data[user_id]["conditions"]:
                restrictions.append("low sugar")
            
            if "Hypertension" in patient_data[user_id]["conditions"]:
                restrictions.append("low sodium")

            suggested_meals = {
                "low sugar": ["Oatmeal with berries and nuts", "Grilled chicken salad", "Lentil soup"],
                "low sodium": ["Baked salmon with vegetables", "Quinoa and black bean bowl", "Fresh fruit salad"],
                "general": ["Whole grain pasta with lean protein", "Vegetable stir-fry"]
            }
            
            plan = []
            if restrictions:
                for r in restrictions:
                    if r in suggested_meals:
                        plan.append(f"For your {r} needs: {random.choice(suggested_meals[r])}")
            else:
                plan.append(f"General healthy meal: {random.choice(suggested_meals['general'])}")
            
            return {"status": "success", "data": "\n".join(plan)}
        return {"status": "error", "message": "User not found"}

    def check_medication_interaction(self, med1: str, med2: str) -> dict:
        known_interactions = {
            frozenset({"Metformin", "Alcohol"}): "Increased risk of lactic acidosis",
            frozenset({"Lisinopril", "Potassium Supplements"}): "Increased risk of hyperkalemia",
        }
        interaction_key = frozenset({med1, med2})
        if interaction_key in known_interactions:
            return {"status": "warning", "message": known_interactions[interaction_key]}
        return {"status": "success", "message": "No known interaction found between these two medications."}

tools = HealthcareTools()

# Mock LLM Agent and RAG
class AdaptiveHealthcareAgent:
    def __init__(self, user_id):
        self.user_id = user_id
        self.memory = []
        self.feedback_log = []

    def _retrieve_info(self, query):
        relevant_info = []
        query_lower = query.lower()
        for guideline in medical_guidelines:
            if any(keyword in guideline.lower() for keyword in query_lower.split()):
                relevant_info.append(guideline)
        return relevant_info

    def _decide_action(self, user_input: str) -> tuple:
        if "glucose" in user_input.lower() or "blood sugar" in user_input.lower():
            return "get_glucose_reading", {}
        elif "meal plan" in user_input.lower() or "diet" in user_input.lower():
            return "suggest_meal_plan", {}
        elif "medication interaction" in user_input.lower():
            parts = user_input.lower().split("medication interaction between ")
            if len(parts) > 1:
                meds_str = parts[1].replace(" and ", ",").strip()
                meds = [m.strip() for m in meds_str.split(",") if m.strip()]
                if len(meds) == 2:
                    return "check_medication_interaction", {"med1": meds[0].capitalize(), "med2": meds[1].capitalize()}
            return "unknown", {}
        else:
            return "general_query", {}

    def process_query(self, user_input: str):
        st.session_state.conversation.append(f"**User:** {user_input}")

        action, args = self._decide_action(user_input)
        agent_response = ""
        tool_output = None

        if action == "get_glucose_reading":
            tool_output = tools.get_glucose_reading(self.user_id)
            if tool_output["status"] == "success":
                agent_response = f"Your latest glucose reading was {tool_output['data']['value']} mg/dL on {tool_output['data']['date']}."
                
                # Self-correction/Feedback Integration
                if tool_output['data']['value'] > 130:
                    agent_response += " This is higher than the recommended fasting level. Consider adjusting your diet or consulting your doctor. "
                    relevant_guidelines = self._retrieve_info("Type 2 Diabetes glucose")
                    if relevant_guidelines:
                        agent_response += f"\n\nRelevant guideline: {relevant_guidelines[0]}"

            else:
                agent_response = f"Could not retrieve glucose reading: {tool_output['message']}"
        
        elif action == "suggest_meal_plan":
            user_conditions = patient_data.get(self.user_id, {}).get("conditions", [])
            dietary_restrictions = []
            if "Type 2 Diabetes" in user_conditions: dietary_restrictions.append("low sugar")
            if "Hypertension" in user_conditions: dietary_restrictions.append("low sodium")
            
            tool_output = tools.suggest_meal_plan(self.user_id, dietary_restrictions)
            if tool_output["status"] == "success":
                agent_response = f"Here's a personalized meal plan suggestion:\n{tool_output['data']}"
            else:
                agent_response = f"Could not suggest meal plan: {tool_output['message']}"

        elif action == "check_medication_interaction":
            if "med1" in args and "med2" in args:
                tool_output = tools.check_medication_interaction(args["med1"], args["med2"])
                if tool_output["status"] == "warning":
                    agent_response = f"**Warning:** {tool_output['message']}. Please consult your doctor.\n"
                elif tool_output["status"] == "success":
                    agent_response = f"Good news! {tool_output['message']}.\n"
            else:
                agent_response = "Please specify two medications to check for interactions. E.g., 'check medication interaction between Metformin and Lisinopril'."

        else: 
            # General query with RAG
            relevant_info = self._retrieve_info(user_input)
            agent_response = "I'm a healthcare assistant. How can I help you today?" # Default response
            if relevant_info:
                agent_response = f"Based on medical guidelines, here's some information: {relevant_info[0]}"
                if len(relevant_info) > 1:
                    agent_response += f"\nAdditional information: {relevant_info[1]}"
            else:
                agent_response = "I couldn't find specific information for that query in my current knowledge base. Can you rephrase or ask something else?"
            
        st.session_state.conversation.append(f"**Assistant:** {agent_response}")
        return agent_response

    def record_feedback(self, query: str, response: str, rating: str, comment: str = None):
        feedback_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response": response,
            "rating": rating,
            "comment": comment
        }
        self.feedback_log.append(feedback_entry)
        st.success("Thank you for your feedback!")


# Streamlit UI
st.set_page_config(page_title="Personalized Healthcare Assistant", layout="centered")
st.title("🩺 Personalized Healthcare Assistant")
st.markdown("Ask me about your health data, get meal suggestions, or check medication interactions!")

# Initialize session state for conversation history and agent
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "agent" not in st.session_state:
    st.session_state.agent = AdaptiveHealthcareAgent(user_id="user_id_1") # Hardcode for demo

user_query = st.text_input("Your query:", key="user_input")

if st.button("Send Query") and user_query:
    with st.spinner("Processing..."):
        st.session_state.agent.process_query(user_query)
    st.session_state.user_input = "" # Clear input box

# Display conversation
for msg in st.session_state.conversation:
    st.markdown(msg)

st.subheader("Give Feedback")
if st.session_state.conversation:
    last_query = st.session_state.conversation[-2].replace("**User:** ", "") if len(st.session_state.conversation) >= 2 else ""
    last_response = st.session_state.conversation[-1].replace("**Assistant:** ", "")
    
    feedback_rating = st.radio(
        "Was the last response helpful?",
        ("Very Helpful", "Helpful", "Neutral", "Not Helpful", "Incorrect"),
        key="feedback_rating"
    )
    feedback_comment = st.text_area("Optional: Add a comment", key="feedback_comment")

    if st.button("Submit Feedback"):
        st.session_state.agent.record_feedback(last_query, last_response, feedback_rating, feedback_comment)

st.subheader("Patient Data (Mock)")
st.json(patient_data["user_id_1"])

# To run this Streamlit app, save it as healthcare_assistant.py and run 'streamlit run healthcare_assistant.py' in your terminal.