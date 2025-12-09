import streamlit as st
import networkx as nx


class MedicalKnowledgeGraph:
    def __init__(self):
        self.graph = nx.Graph()
        self.populate_graph()

    def populate_graph(self):
        # Diseases
        self.graph.add_node("Influenza", type="disease")
        self.graph.add_node("Common Cold", type="disease")
        self.graph.add_node("Pneumonia", type="disease")
        self.graph.add_node("Diabetes Type 2", type="disease")
        self.graph.add_node("Hypertension", type="disease")
        self.graph.add_node("Migraine", type="disease")

        # Symptoms
        self.graph.add_node("Fever", type="symptom")
        self.graph.add_node("Cough", type="symptom")
        self.graph.add_node("Sore Throat", type="symptom")
        self.graph.add_node("Headache", type="symptom")
        self.graph.add_node("Fatigue", type="symptom")
        self.graph.add_node("Shortness of Breath", type="symptom")
        self.graph.add_node("High Blood Sugar", type="symptom")
        self.graph.add_node("High Blood Pressure", type="symptom")
        self.graph.add_node("Nausea", type="symptom")
        self.graph.add_node("Vomiting", type="symptom")
        self.graph.add_node("Runny Nose", type="symptom")
        self.graph.add_node("Body Aches", type="symptom")

        # Treatments/Drugs (simplified)
        self.graph.add_node("Antivirals", type="treatment")
        self.graph.add_node("Antibiotics", type="treatment")
        self.graph.add_node("Pain Relievers", type="treatment")
        self.graph.add_node("Insulin", type="treatment")
        self.graph.add_node("Antihypertensives", type="treatment")
        self.graph.add_node("Rest and Fluids", type="treatment")

        # Relationships (Disease-Symptom)
        self.graph.add_edge("Influenza", "Fever", relation="associated_with")
        self.graph.add_edge("Influenza", "Cough", relation="associated_with")
        self.graph.add_edge("Influenza", "Body Aches", relation="associated_with")
        self.graph.add_edge("Influenza", "Fatigue", relation="associated_with")

        self.graph.add_edge("Common Cold", "Cough", relation="associated_with")
        self.graph.add_edge("Common Cold", "Sore Throat", relation="associated_with")
        self.graph.add_edge("Common Cold", "Runny Nose", relation="associated_with")

        self.graph.add_edge("Pneumonia", "Fever", relation="associated_with")
        self.graph.add_edge("Pneumonia", "Cough", relation="associated_with")
        self.graph.add_edge("Pneumonia", "Shortness of Breath", relation="associated_with")

        self.graph.add_edge("Diabetes Type 2", "High Blood Sugar", relation="associated_with")
        self.graph.add_edge("Diabetes Type 2", "Fatigue", relation="associated_with")

        self.graph.add_edge("Hypertension", "High Blood Pressure", relation="associated_with")
        self.graph.add_edge("Hypertension", "Headache", relation="can_cause")

        self.graph.add_edge("Migraine", "Headache", relation="associated_with")
        self.graph.add_edge("Migraine", "Nausea", relation="associated_with")
        self.graph.add_edge("Migraine", "Vomiting", relation="associated_with")

        # Relationships (Disease-Treatment)
        self.graph.add_edge("Influenza", "Antivirals", relation="treated_by")
        self.graph.add_edge("Influenza", "Rest and Fluids", relation="treated_by")
        self.graph.add_edge("Common Cold", "Rest and Fluids", relation="treated_by")
        self.graph.add_edge("Common Cold", "Pain Relievers", relation="treated_by")
        self.graph.add_edge("Pneumonia", "Antibiotics", relation="treated_by")
        self.graph.add_edge("Diabetes Type 2", "Insulin", relation="treated_by")
        self.graph.add_edge("Hypertension", "Antihypertensives", relation="treated_by")
        self.graph.add_edge("Migraine", "Pain Relievers", relation="treated_by")

    def retrieve_knowledge(self, query_keywords):
        retrieved_facts = set()
        for keyword in query_keywords:
            for node in self.graph.nodes:
                if keyword.lower() in node.lower():
                    for neighbor in self.graph.neighbors(node):
                        relation = self.graph[node][neighbor]["relation"]
                        fact = f"Fact: {node} {relation.replace('_', ' ')} {neighbor}."
                        retrieved_facts.add(fact)

                    # Also add facts where the node is the neighbor
                    for u, v, data in self.graph.edges(data=True):
                        if keyword.lower() in v.lower():
                             relation = data["relation"]
                             fact = f"Fact: {u} {relation.replace('_', ' ')} {v}."
                             retrieved_facts.add(fact)

        return list(retrieved_facts)


class MockLLM:
    def __init__(self):
        pass

    def generate_response(self, prompt):
        # Simulate LLM reasoning based on keywords in the prompt
        diagnoses = []
        reasoning_steps = []

        if "fever" in prompt.lower() and "cough" in prompt.lower() and "body aches" in prompt.lower():
            diagnoses.append("Influenza")
            reasoning_steps.append("Step 1: Patient presents with Fever, Cough, and Body Aches. Fact: Influenza is associated with Fever, Cough, and Body Aches.")
            if "antivirals" in prompt.lower():
                reasoning_steps.append("Step 2: Antivirals are a known treatment for Influenza. Fact: Influenza treated by Antivirals.")

        if "cough" in prompt.lower() and "sore throat" in prompt.lower() and "runny nose" in prompt.lower():
            diagnoses.append("Common Cold")
            reasoning_steps.append("Step 1: Patient presents with Cough, Sore Throat, and Runny Nose. Fact: Common Cold is associated with Cough, Sore Throat, and Runny Nose.")
            if "rest and fluids" in prompt.lower():
                reasoning_steps.append("Step 2: Rest and Fluids are recommended for Common Cold. Fact: Common Cold treated by Rest and Fluids.")

        if "fever" in prompt.lower() and "shortness of breath" in prompt.lower():
            diagnoses.append("Pneumonia")
            reasoning_steps.append("Step 1: Patient presents with Fever and Shortness of Breath. Fact: Pneumonia is associated with Fever and Shortness of Breath.")
            if "antibiotics" in prompt.lower():
                reasoning_steps.append("Step 2: Antibiotics are a common treatment for Pneumonia. Fact: Pneumonia treated by Antibiotics.")

        if "high blood sugar" in prompt.lower():
            diagnoses.append("Diabetes Type 2")
            reasoning_steps.append("Step 1: Patient presents with High Blood Sugar. Fact: Diabetes Type 2 is associated with High Blood Sugar.")
            if "insulin" in prompt.lower():
                reasoning_steps.append("Step 2: Insulin is used to treat Diabetes Type 2. Fact: Diabetes Type 2 treated by Insulin.")

        if "high blood pressure" in prompt.lower():
            diagnoses.append("Hypertension")
            reasoning_steps.append("Step 1: Patient presents with High Blood Pressure. Fact: Hypertension is associated with High Blood Pressure.")
            if "antihypertensives" in prompt.lower():
                reasoning_steps.append("Step 2: Antihypertensives are used to treat Hypertension. Fact: Hypertension treated by Antihypertensives.")

        if not diagnoses and not reasoning_steps: # Default if no specific match
            diagnoses.append("Undetermined/Further Investigation Needed")
            reasoning_steps.append("No direct diagnostic match found based on current knowledge. Recommend further tests and specialist consultation.")

        return {"diagnoses": list(set(diagnoses)), "reasoning": reasoning_steps}


class PromptEngineer:
    def __init__(self):
        pass

    def construct_prompt(self, symptoms, history, lab_results, retrieved_knowledge):
        prompt = ""
        prompt += "You are a medical diagnostic assistant. Your task is to analyze patient information, retrieve relevant medical facts, and provide a step-by-step diagnostic reasoning process, concluding with potential diagnoses. Base your reasoning *only* on the provided facts and patient data.\n\n"
        prompt += "Patient Information:\n"
        prompt += f"Symptoms: {symptoms}\n"
        prompt += f"Medical History: {history}\n"
        prompt += f"Lab Results: {lab_results}\n\n"

        if retrieved_knowledge:
            prompt += "Retrieved Medical Facts (Grounding Knowledge):\n"
            for fact in retrieved_knowledge:
                prompt += f"- {fact}\n"
            prompt += "\n"

        prompt += "Please provide a Chain-of-Thought reasoning, explicitly referencing the 'Retrieved Medical Facts' where applicable, and then state your proposed diagnoses.\n\n"
        prompt += "Chain-of-Thought Reasoning:\n"

        return prompt


# Streamlit UI
st.set_page_config(layout="wide")
st.title("🧠 Clinical Diagnosis Assistant with Knowledge-Grounded Reasoning")
st.markdown("This assistant uses a Medical Knowledge Graph and Chain-of-Thought reasoning to help diagnose complex patient cases, reducing hallucinations by grounding LLM responses in factual medical knowledge.")

# Initialize components
kg = MedicalKnowledgeGraph()
mock_llm = MockLLM()
prompt_engineer = PromptEngineer()

with st.sidebar:
    st.header("Patient Data Input")
    patient_symptoms = st.text_area("Symptoms (e.g., fever, cough, headache)", height=100)
    patient_history = st.text_area("Medical History", height=150)
    patient_lab_results = st.text_area("Lab Results", height=100)

    process_button = st.button("Get Diagnosis")

st.subheader("Diagnostic Output")

if process_button:
    if not patient_symptoms and not patient_history and not patient_lab_results:
        st.warning("Please enter some patient information to get a diagnosis.")
    else:
        # 1. Knowledge Retrieval
        query_keywords = []
        query_keywords.extend(patient_symptoms.lower().split(', '))
        query_keywords.extend(patient_history.lower().split(', '))
        query_keywords.extend(patient_lab_results.lower().split(', '))
        query_keywords = [kw.strip() for kw in query_keywords if kw.strip()]

        retrieved_facts = kg.retrieve_knowledge(query_keywords)
        
        st.markdown("### Retrieved Knowledge:")
        if retrieved_facts:
            for fact in retrieved_facts:
                st.write(f"- {fact}")
        else:
            st.write("No specific medical facts retrieved based on provided input keywords.")
        st.markdown("--- ")

        # 2. Prompt Engineering
        llm_prompt = prompt_engineer.construct_prompt(
            patient_symptoms,
            patient_history,
            patient_lab_results,
            retrieved_facts
        )

        # For demonstration, show the constructed prompt to see the grounding in action
        with st.expander("View LLM Prompt"): # Removed the comment for actual code
            st.text(llm_prompt)

        # 3. LLM Reasoning (using Mock LLM)
        st.markdown("### LLM Reasoning Process (Chain-of-Thought):")
        llm_output = mock_llm.generate_response(llm_prompt)

        for step in llm_output["reasoning"]:
            st.write(f"- {step}")
        st.markdown("--- ")

        # 4. Proposed Diagnoses
        st.markdown("### Proposed Diagnoses:")
        for diagnosis in llm_output["diagnoses"]:
            st.success(f"**{diagnosis}**")

else:
    st.info("Enter patient details in the sidebar and click 'Get Diagnosis' to begin.")


