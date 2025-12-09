import networkx as nx
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool
import gradio as gr

class MedicalKnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self._populate_sample_data()

    def _populate_sample_data(self):
        diseases = ["Influenza", "Common Cold", "Diabetes Type 2", "Hypertension", "Migraine"]
        symptoms = ["Fever", "Cough", "Sore Throat", "Runny Nose", "Fatigue", "Headache", "Blurred Vision", "Increased Thirst", "Frequent Urination", "High Blood Pressure", "Nausea", "Light Sensitivity"]
        treatments = ["Rest", "Fluids", "Paracetamol", "Insulin", "Metformin", "ACE Inhibitors", "Beta-blockers", "Triptans"]
        drugs = ["Paracetamol", "Insulin", "Metformin", "Lisinopril", "Metoprolol", "Sumatriptan"]
        
        for d in diseases: self.graph.add_node(d, type="disease")
        for s in symptoms: self.graph.add_node(s, type="symptom")
        for t in treatments: self.graph.add_node(t, type="treatment")
        for dr in drugs: self.graph.add_node(dr, type="drug")

        self.add_relation("Influenza", "has_symptom", "Fever")
        self.add_relation("Influenza", "has_symptom", "Cough")
        self.add_relation("Influenza", "has_symptom", "Fatigue")
        self.add_relation("Influenza", "treatable_by", "Rest")
        self.add_relation("Influenza", "treatable_by", "Fluids")
        self.add_relation("Influenza", "treatable_by", "Paracetamol")

        self.add_relation("Common Cold", "has_symptom", "Cough")
        self.add_relation("Common Cold", "has_symptom", "Sore Throat")
        self.add_relation("Common Cold", "has_symptom", "Runny Nose")
        self.add_relation("Common Cold", "treatable_by", "Rest")
        self.add_relation("Common Cold", "treatable_by", "Fluids")
        self.add_relation("Common Cold", "treatable_by", "Paracetamol")

        self.add_relation("Diabetes Type 2", "has_symptom", "Increased Thirst")
        self.add_relation("Diabetes Type 2", "has_symptom", "Frequent Urination")
        self.add_relation("Diabetes Type 2", "has_symptom", "Blurred Vision")
        self.add_relation("Diabetes Type 2", "treatable_by", "Insulin")
        self.add_relation("Diabetes Type 2", "treatable_by", "Metformin")

        self.add_relation("Hypertension", "has_symptom", "High Blood Pressure")
        self.add_relation("Hypertension", "treatable_by", "ACE Inhibitors")
        self.add_relation("Hypertension", "treatable_by", "Beta-blockers")

        self.add_relation("Migraine", "has_symptom", "Headache")
        self.add_relation("Migraine", "has_symptom", "Nausea")
        self.add_relation("Migraine", "has_symptom", "Light Sensitivity")
        self.add_relation("Migraine", "treatable_by", "Triptans")

        self.add_relation("Paracetamol", "is_drug", "Paracetamol")
        self.add_relation("Insulin", "is_drug", "Insulin")
        self.add_relation("Metformin", "is_drug", "Metformin")
        self.add_relation("Lisinopril", "is_drug", "ACE Inhibitors")
        self.add_relation("Metoprolol", "is_drug", "Beta-blockers")
        self.add_relation("Sumatriptan", "is_drug", "Triptans")

        self.add_relation("Metformin", "interacts_with", "Contrast Dye")
        self.add_relation("ACE Inhibitors", "interacts_with", "Potassium Supplements")

    def add_relation(self, source, relation_type, target):
        if source in self.graph and target in self.graph:
            self.graph.add_edge(source, target, type=relation_type)

    def get_neighbors(self, entity, relation_type=None, node_type=None):
        results = []
        if entity not in self.graph:
            return []
        
        for neighbor in self.graph.neighbors(entity):
            edge_data = self.graph.get_edge_data(entity, neighbor)
            if edge_data and (relation_type is None or edge_data.get("type") == relation_type):
                if node_type is None or (neighbor in self.graph and self.graph.nodes[neighbor].get("type") == node_type):
                    results.append(neighbor)
        return list(set(results))

    def get_diseases_by_symptom(self, symptom: str) -> list:
        symptom = symptom.replace("_", " ").title()
        if symptom not in self.graph:
            return []
        
        diseases = []
        for u, v, data in self.graph.edges(data=True):
            if v == symptom and data.get("type") == "has_symptom":
                diseases.append(u)
        return diseases

    def get_symptoms_by_disease(self, disease: str) -> list:
        disease = disease.replace("_", " ").title()
        return self.get_neighbors(disease, relation_type="has_symptom", node_type="symptom")

    def get_treatments_by_disease(self, disease: str) -> list:
        disease = disease.replace("_", " ").title()
        return self.get_neighbors(disease, relation_type="treatable_by", node_type="treatment")

    def get_drug_interactions(self, drug: str) -> list:
        drug = drug.replace("_", " ").title()
        return self.get_neighbors(drug, relation_type="interacts_with")


llm = ChatOpenAI(model="gpt-4o", temperature=0)

medical_kg = MedicalKnowledgeGraph()

tools = [
    Tool(
        name="get_diseases_by_symptom",
        func=medical_kg.get_diseases_by_symptom,
        description="""Use this tool to find potential diseases associated with a given symptom.\n        Input should be a single symptom string (e.g., "Fever", "Headache").\n        Returns a list of diseases."""
    ),
    Tool(
        name="get_symptoms_by_disease",
        func=medical_kg.get_symptoms_by_disease,
        description="""Use this tool to find common symptoms for a specific disease.\n        Input should be a single disease name string (e.g., "Influenza", "Diabetes Type 2").\n        Returns a list of symptoms."""
    ),
    Tool(
        name="get_treatments_by_disease",
        func=medical_kg.get_treatments_by_disease,
        description="""Use this tool to find general treatments or management strategies for a specific disease.\n        Input should be a single disease name string (e.g., "Influenza", "Hypertension").\n        Returns a list of treatments."""
    ),
    Tool(
        name="get_drug_interactions",
        func=medical_kg.get_drug_interactions,
        description="""Use this tool to find known interactions for a specific drug.\n        Input should be a single drug name string (e.g., "Metformin", "Lisinopril").\n        Returns a list of interacting substances or drugs."""
    )
]

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are a medical diagnosis and treatment recommendation assistant.\n         You have access to a medical knowledge graph and can query it to gather information about diseases, symptoms, treatments, and drug interactions.\n         Your goal is to provide accurate diagnoses and personalized treatment recommendations based on the patient's symptoms and any additional context.\n         Always strive to use the knowledge graph tools to gather relevant information before making a recommendation.\n         Explain your reasoning and cite the information you found in the knowledge graph.\n         If you cannot find sufficient information, state that you need more details or that the information is not in your current knowledge base.\n         When asked for a diagnosis, first identify potential diseases based on symptoms, then confirm symptoms for those diseases, and finally suggest treatments.\n         Be cautious and never provide definitive medical advice without a disclaimer that this is for informational purposes only and not a substitute for professional medical consultation.\n         """),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def medical_consultation(user_input):
    try:
        response = agent_executor.invoke({"input": user_input})
        return response["output"]
    except Exception as e:
        return f"An error occurred: {str(e)}. Please ensure your OpenAI API key is set correctly."

iface = gr.Interface(
    fn=medical_consultation,
    inputs=gr.Textbox(lines=3, placeholder="Describe the patient's symptoms or ask a medical question here..."),
    outputs="text",
    title="LLM-KG Medical Assistant (Tight-Coupling Paradigm)",
    description="This system leverages an LLM to interactively explore a medical knowledge graph for diagnosis and treatment recommendations. For informational purposes only, not medical advice."
)

if __name__ == "__main__":
    iface.launch()