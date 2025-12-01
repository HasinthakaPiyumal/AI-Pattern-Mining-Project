import streamlit as st
import re
from pydantic import BaseModel
from typing import List, Dict, Any

class QueryEntities(BaseModel):
    drugs: List[str] = []
    symptoms: List[str] = []
    conditions: List[str] = []
    patient_id: str = None
    query_type: str = "general_qa"

def check_drug_interactions(drugs: List[str]) -> Dict[str, Any]:
    if "ibuprofen" in drugs and "warfarin" in drugs:
        return {"drug_interactions": f"Potential severe interaction between Ibuprofen and Warfarin: increased bleeding risk.", "severity": "severe"}
    elif "paracetamol" in drugs and "alcohol" in drugs:
        return {"drug_interactions": f"Potential moderate interaction between Paracetamol and Alcohol: increased liver risk with chronic heavy use.", "severity": "moderate"}
    else:
        return {"drug_interactions": "No significant interactions found for the specified drugs.", "severity": "none"}

def search_medical_papers(query: str) -> Dict[str, Any]:
    if "diabetes type 2" in query.lower():
        return {
            "papers": [
                {"title": "Advances in Type 2 Diabetes Management", "abstract": "Recent breakthroughs in oral hypoglycemic agents and lifestyle interventions."},
                {"title": "Role of GLP-1 Agonists in Diabetes Treatment", "abstract": "A review of efficacy and safety profiles of GLP-1 receptor agonists."}
            ]
        }
    elif "hypertension treatment" in query.lower():
        return {
            "papers": [
                {"title": "Guidelines for Essential Hypertension Management", "abstract": "Updates on first-line pharmacotherapy and non-pharmacological approaches."}
            ]
        }
    else:
        return {"papers": [{"title": "Generic Medical Research Paper", "abstract": "A general overview of medical research relevant to health."}]}

def get_ehr_data(patient_id: str) -> Dict[str, Any]:
    if patient_id == "PAT001":
        return {
            "patient_id": "PAT001",
            "name": "Jane Doe",
            "age": 45,
            "conditions": ["Type 2 Diabetes", "Hypertension"],
            "medications": ["Metformin", "Lisinopril"],
            "allergies": ["Penicillin"]
        }
    else:
        return {"patient_id": patient_id, "data": "Patient record not found or access denied."}

KNOWLEDGE_BASE = [
    {"topic": "Diabetes", "snippet": "Type 2 diabetes is a chronic condition that affects the way your body processes blood sugar (glucose)."},
    {"topic": "Hypertension", "snippet": "Hypertension, or high blood pressure, is a common condition in which the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease."},
    {"topic": "Ibuprofen", "snippet": "Ibuprofen is a nonsteroidal anti-inflammatory drug (NSAID) used for pain and inflammation."},
    {"topic": "Warfarin", "snippet": "Warfarin is an anticoagulant (blood thinner) used to prevent blood clots."}
]

def retrieve_knowledge(query: str) -> List[str]:
    relevant_snippets = []
    query_lower = query.lower()
    for item in KNOWLEDGE_BASE:
        if any(keyword in query_lower for keyword in item["topic"].lower().split() + item["snippet"].lower().split()):
            relevant_snippets.append(item["snippet"])
    return list(set(relevant_snippets))

class MockLLM:
    def synthesize_response(self, query: str, tool_outputs: Dict[str, Any], knowledge_snippets: List[str], user_type: str) -> str:
        response = f"Hello {'patient' if user_type == 'Patient' else 'professional'}. Here's the synthesized information based on your request:\n\n"
        response += f"Your original query was: \"{query}\"\n\n"
        
        if tool_outputs:
            response += "Findings from external medical tools:\n"
            if "drug_interactions" in tool_outputs:
                response += f"- Drug Interactions: {tool_outputs['drug_interactions']['drug_interactions']} (Severity: {tool_outputs['drug_interactions']['severity']})\n"
            if "papers" in tool_outputs and tool_outputs["papers"]:
                response += "- Relevant Research Papers:\n"
                for paper in tool_outputs["papers"]:
                    response += f"  - Title: {paper['title']}\n    Abstract: {paper['abstract']}\n"
            if "patient_data" in tool_outputs and "name" in tool_outputs["patient_data"]:
                 response += f"- Patient Data for {tool_outputs['patient_data']['name']} (ID: {tool_outputs['patient_data']['patient_id']}):\n"
                 response += f"  - Age: {tool_outputs['patient_data']['age']}\n"
                 response += f"  - Conditions: {\
                     ', '.join(tool_outputs['patient_data']['conditions'])
                 }\n"
                 response += f"  - Medications: {\
                     ', '.join(tool_outputs['patient_data']['medications'])
                 }\n"
                 response += f"  - Allergies: {\
                     ', '.join(tool_outputs['patient_data']['allergies'])
                 }\n"
            elif "patient_data" in tool_outputs:
                 response += f"- Patient Data: {tool_outputs['patient_data']['data']}\n"

        if knowledge_snippets:
            response += "\nGeneral medical knowledge relevant to your query:\n"
            for snippet in knowledge_snippets:
                response += f"- {snippet}\n"

        response += "\n\nDisclaimer: This information is for educational purposes only and should not replace professional medical advice."
        if user_type == "Patient":
             response += " Consult a healthcare provider for any medical concerns."
        else:
             response += " Always refer to official guidelines and patient records for clinical decisions."

        return response

def run_qa_system(query: str, user_type: str) -> str:
    entities = QueryEntities()

    drug_matches = re.findall(r"(ibuprofen|warfarin|paracetamol|aspirin)", query, re.IGNORECASE)
    entities.drugs = list(set([d.lower() for d in drug_matches]))

    patient_id_match = re.search(r"patient\s*id:\s*(\w+)", query, re.IGNORECASE)
    if patient_id_match:
        entities.patient_id = patient_id_match.group(1).upper()
        entities.query_type = "ehr_lookup"
    elif any(d in query.lower() for d in ["drug interaction", "medication interaction", "what happens if i take"]):
        entities.query_type = "drug_interaction"
    elif any(kw in query.lower() for kw in ["research", "paper", "study", "latest findings"]):
        entities.query_type = "research_search"
    else:
        entities.query_type = "general_qa"

    st.sidebar.write(f"Detected Query Type: {entities.query_type}")
    st.sidebar.write(f"Extracted Drugs: {entities.drugs}")
    st.sidebar.write(f"Extracted Patient ID: {entities.patient_id}")

    tool_outputs = {}
    if entities.query_type == "drug_interaction" and entities.drugs:
        tool_outputs["drug_interactions"] = check_drug_interactions(entities.drugs)
    if entities.query_type == "research_search":
        tool_outputs["papers"] = search_medical_papers(query)
    if entities.query_type == "ehr_lookup" and entities.patient_id:
        tool_outputs["patient_data"] = get_ehr_data(entities.patient_id)

    knowledge_snippets = retrieve_knowledge(query)

    llm = MockLLM()
    final_response = llm.synthesize_response(query, tool_outputs, knowledge_snippets, user_type)
    
    return final_response

st.set_page_config(layout="wide")
st.title("Medical Information Synthesis and Patient Q&A System")

st.sidebar.header("System Configuration")
user_type = st.sidebar.radio("I am a:", ("Patient", "Healthcare Professional"))
st.sidebar.markdown("---")

query = st.text_area("Enter your medical query here:", height=150)

if st.button("Get Answer"):
    if query:
        with st.spinner("Synthesizing your answer..."):
            response = run_qa_system(query, user_type)
            st.markdown("## Synthesized Response:")
            st.write(response)
    else:
        st.warning("Please enter a medical query.")

st.sidebar.markdown("---")
st.sidebar.info("This is a demonstration system. Always consult a qualified healthcare professional for medical advice.")