import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain import hub
from langchain.tools import tool

load_dotenv()

# --- 1. External Knowledge Retrieval Tools ---

@tool
def medical_search_engine(query: str) -> str:
    """Searches real-time medical literature, clinical trials, and research papers for the given query. """
    # In a real application, this would integrate with PubMed, Google Scholar, or a specialized medical search API.
    # For this prototype, we return a simulated result.
    if "Ehlers-Danlos Syndrome" in query:
        return "Found recent research on gene therapy for Ehlers-Danlos Syndrome published in The New England Journal of Medicine (2023)."
    elif "Marfan Syndrome treatment" in query:
        return "Clinical trial data suggests positive outcomes for Losartan in early-stage Marfan Syndrome patients (2022 review)."
    return f"Simulated search result for '{query}': No specific recent updates found, general information available."

@tool
def rare_disease_database(disease_name: str) -> str:
    """Queries a structured database for information on rare diseases, symptoms, genetic markers, and established treatments."""
    # In a real application, this would connect to Orphanet, OMIM, or a custom medical database.
    # For this prototype, we use a simple dictionary.
    rare_diseases_data = {
        "Ehlers-Danlos Syndrome": {
            "symptoms": "Hypermobility, skin hyperextensibility, tissue fragility",
            "genetics": "COL5A1, COL5A2, COL3A1 genes",
            "treatment": "Symptomatic management, physical therapy, pain control",
            "prevalence": "1 in 5,000 to 1 in 20,000"
        },
        "Marfan Syndrome": {
            "symptoms": "Tall stature, long limbs, heart problems (aortic dilation), lens dislocation",
            "genetics": "FBN1 gene",
            "treatment": "Beta-blockers, Losartan, aortic surgery, regular cardiovascular monitoring",
            "prevalence": "1 in 5,000 to 1 in 10,000"
        },
        "Huntington's Disease": {
            "symptoms": "Involuntary movements (chorea), cognitive decline, psychiatric problems",
            "genetics": "HTT gene",
            "treatment": "Symptomatic, tetrabenazine for chorea, supportive care",
            "prevalence": "3 to 7 per 100,000 people of European descent"
        }
    }
    info = rare_diseases_data.get(disease_name, {})
    if info:
        return f"Found information for {disease_name}: Symptoms: {info['symptoms']}, Genetics: {info['genetics']}, Treatment: {info['treatment']}, Prevalence: {info['prevalence']}."
    return f"Information for {disease_name} not found in rare disease database."

@tool
def medical_knowledge_graph(entities: str) -> str:
    """Traverses a medical knowledge graph to find relationships between symptoms, diseases, drugs, and genes. """
    # In a real application, this would query a graph database like Neo4j or a knowledge graph API.
    # For this prototype, we simulate some relationships.
    if "hypermobility" in entities.lower() and "skin hyperextensibility" in entities.lower():
        return "Knowledge graph suggests a strong link between hypermobility and skin hyperextensibility, often associated with Ehlers-Danlos Syndrome."
    elif "aortic dilation" in entities.lower() and "lens dislocation" in entities.lower():
        return "Knowledge graph indicates a direct relationship between aortic dilation and lens dislocation, key indicators for Marfan Syndrome."
    return f"Simulated knowledge graph result for '{entities}': No strong relationships found for the given entities."

# --- 2. LLM and Agent Setup ---

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.7, openai_api_key=os.getenv("OPENAI_API_KEY"))

# Define the tools available to the agent
tools = [
    medical_search_engine,
    rare_disease_database,
    medical_knowledge_graph
]

# Get the prompt for the OpenAI tools agent
prompt = hub.pull("hwchase17/openai-tools-agent")

# Create the agent
agent = create_openai_tools_agent(llm, tools, prompt)

# Create the agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- 3. Main Function to Run the System ---

def get_medical_diagnosis_and_treatment(patient_symptoms_history: str) -> str:
    """Provides a medical diagnosis and treatment recommendation for rare diseases based on patient symptoms and external knowledge. """
    print(f"\nProcessing patient symptoms: {patient_symptoms_history}")
    response = agent_executor.invoke({"input": patient_symptoms_history})
    return response["output"]

# --- Example Usage ---
if __name__ == "__main__":
    # Example 1: Symptoms suggesting Ehlers-Danlos Syndrome
    symptoms1 = "Patient presents with extreme joint hypermobility, unusually stretchy skin that bruises easily, and frequent joint dislocations. Family history includes similar connective tissue issues."
    diagnosis1 = get_medical_diagnosis_and_treatment(symptoms1)
    print("\n--- Diagnosis and Treatment Recommendation 1 ---")
    print(diagnosis1)

    # Example 2: Symptoms suggesting Marfan Syndrome
    symptoms2 = "25-year-old male with tall, slender build, abnormally long limbs and fingers. Reports recent onset of shortness of breath and chest pain. Ocular examination revealed lens dislocation. Has a history of a heart murmur."
    diagnosis2 = get_medical_diagnosis_and_treatment(symptoms2)
    print("\n--- Diagnosis and Treatment Recommendation 2 ---")
    print(diagnosis2)

    # Example 3: More general symptoms, requiring broader search
    symptoms3 = "Child with persistent fatigue, muscle weakness, and developmental delay. Parents are concerned about a possible genetic disorder."
    diagnosis3 = get_medical_diagnosis_and_treatment(symptoms3)
    print("\n--- Diagnosis and Treatment Recommendation 3 ---")
    print(diagnosis3)
