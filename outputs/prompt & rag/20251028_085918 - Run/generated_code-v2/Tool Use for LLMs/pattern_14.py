import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool


@tool
def lab_result_analyzer(lab_results: dict) -> dict:
    if lab_results.get("blood_glucose") and lab_results["blood_glucose"] > 100:
        return {"status": "abnormal", "interpretation": "Elevated blood glucose suggests hyperglycemia."}
    if lab_results.get("white_blood_cell_count") and lab_results["white_blood_cell_count"] > 10000:
        return {"status": "abnormal", "interpretation": "Elevated WBC count suggests infection."}
    return {"status": "normal", "interpretation": "Lab results within normal limits."}

@tool
def drug_interaction_database(medications: list) -> dict:
    interactions = []
    if "Amoxicillin" in medications and "Warfarin" in medications:
        interactions.append({"drug1": "Amoxicillin", "drug2": "Warfarin", "severity": "high", "advice": "Increased risk of bleeding. Monitor INR closely."})
    if "Metformin" in medications and "Cimetidine" in medications:
        interactions.append({"drug1": "Metformin", "drug2": "Cimetidine", "severity": "medium", "advice": "Increased Metformin levels. Monitor for lactic acidosis."})
    if not interactions:
        return {"interactions": [], "message": "No significant drug interactions found."}
    return {"interactions": interactions, "message": "Potential drug interactions found."}

@tool
def medical_image_analysis(image_description: dict) -> dict:
    image_type = image_description.get("image_type")
    region = image_description.get("region")
    if image_type == "X-ray" and region == "chest":
        return {"findings": "Consolidation in lower left lobe, suggestive of pneumonia.", "confidence": 0.85}
    if image_type == "MRI" and region == "brain":
        return {"findings": "Small white matter lesions, possibly age-related or demyelinating.", "confidence": 0.70}
    return {"findings": "No specific abnormalities detected in the provided image description.", "confidence": 0.95}

llm = ChatOpenAI(model="gpt-4", temperature=0)

tools = [lab_result_analyzer, drug_interaction_database, medical_image_analysis]

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a medical diagnostic assistant. Your goal is to help doctors analyze complex patient cases by using available medical tools. Think step by step."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

st.set_page_config(page_title="Medical Diagnostic Assistant", layout="wide")
st.title("Medical Diagnostic Assistant with Tool-Integrated Reasoning")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Describe the patient's symptoms, history, or lab results..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Analyzing..."):
        response = agent_executor.invoke({"input": prompt, "chat_history": []})
        full_response = response["output"]

    with st.chat_message("assistant"):
        st.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})