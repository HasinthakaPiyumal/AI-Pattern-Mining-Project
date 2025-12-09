import os
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF, RDFS, Namespace
from pydantic import BaseModel, Field
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain import hub

# Set OpenAI API Key. It is recommended to set this as an environment variable.
# Example: os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"
# If not set, the ChatOpenAI constructor might raise an error.

# 1. Knowledge Graph (KG) Layer
# Initialize RDFLib Graph
kg = Graph()

# Define Namespaces
MED = Namespace("http://example.org/medical#")
kg.bind("med", MED)

# Populate KG with sample medical data
# Diseases
kg.add((MED.Diabetes, RDF.type, MED.Disease))
kg.add((MED.Diabetes, RDFS.label, Literal("Diabetes Mellitus")))
kg.add((MED.Diabetes, MED.hasSymptom, MED.Polyuria))
kg.add((MED.Diabetes, MED.hasSymptom, MED.Polydipsia))
kg.add((MED.Diabetes, MED.hasSymptom, MED.WeightLoss))
kg.add((MED.Diabetes, MED.hasTreatment, MED.InsulinTherapy))
kg.add((MED.Diabetes, MED.hasTreatment, MED.DietaryChanges))

kg.add((MED.Hypertension, RDF.type, MED.Disease))
kg.add((MED.Hypertension, RDFS.label, Literal("Hypertension")))
kg.add((MED.Hypertension, MED.hasSymptom, MED.Headache))
kg.add((MED.Hypertension, MED.hasSymptom, MED.Dizziness))
kg.add((MED.Hypertension, MED.hasTreatment, MED.ACEInhibitors))
kg.add((MED.Hypertension, MED.hasTreatment, MED.LifestyleChanges))

kg.add((MED.CommonCold, RDF.type, MED.Disease))
kg.add((MED.CommonCold, RDFS.label, Literal("Common Cold")))
kg.add((MED.CommonCold, MED.hasSymptom, MED.SoreThroat))
kg.add((MED.CommonCold, MED.hasSymptom, MED.RunnyNose))
kg.add((MED.CommonCold, MED.hasTreatment, MED.Rest))
kg.add((MED.CommonCold, MED.hasTreatment, MED.Fluids))

# Symptoms
kg.add((MED.Polyuria, RDF.type, MED.Symptom))
kg.add((MED.Polydipsia, RDF.type, MED.Symptom))
kg.add((MED.WeightLoss, RDF.type, MED.Symptom))
kg.add((MED.Headache, RDF.type, MED.Symptom))
kg.add((MED.Dizziness, RDF.type, MED.Symptom))
kg.add((MED.SoreThroat, RDF.type, MED.Symptom))
kg.add((MED.RunnyNose, RDF.type, MED.Symptom))

# Treatments
kg.add((MED.InsulinTherapy, RDF.type, MED.Treatment))
kg.add((MED.DietaryChanges, RDF.type, MED.Treatment))
kg.add((MED.ACEInhibitors, RDF.type, MED.Treatment))
kg.add((MED.LifestyleChanges, RDF.type, MED.Treatment))
kg.add((MED.Rest, RDF.type, MED.Treatment))
kg.add((MED.Fluids, RDF.type, MED.Treatment))

# Drug interactions (example)
kg.add((MED.Drug_A, RDF.type, MED.Medication))
kg.add((MED.Drug_B, RDF.type, MED.Medication))
kg.add((MED.Drug_C, RDF.type, MED.Medication))
kg.add((MED.Drug_A, MED.interactsWith, MED.Drug_B))

# 2. KG Interaction Layer (Tooling)

class KGQueryInput(BaseModel):
    sparql_query: str = Field(description="A SPARQL query string to execute against the knowledge graph.")

def _query_knowledge_graph(sparql_query: str) -> str:
    results_str = []
    try:
        qres = kg.query(sparql_query)
        if not qres:
            return "No results found for the query."

        for i, row in enumerate(qres):
            row_items = []
            for k, v in row.asdict().items():
                val = str(v)
                # Attempt to extract local name from URI for readability
                if "#" in val:
                    val = val.split("#")[-1]
                elif "/" in val and ":" not in val: # Simple URI without a clear namespace separator like #
                    val = val.split("/")[-1]
                # Clean up variable names for better presentation
                cleaned_key = str(k).replace("?", "").replace("Label", "").replace("med:", "")
                row_items.append(f"{cleaned_key.capitalize()}: {val}")
            results_str.append(f"Result {i+1}: " + ", ".join(row_items))

        if not results_str:
            return "No results found for the query."
        return "\n".join(results_str)
    except Exception as e:
        return f"Error executing SPARQL query: {e}"

query_knowledge_graph_tool = Tool(
    name="query_knowledge_graph",
    func=_query_knowledge_graph,
    description="""
    Use this tool to query the medical knowledge graph using SPARQL.
    Input should be a well-formed SPARQL query string.
    
    **Example queries:**
    1.  To find diseases associated with a symptom (e.g., Polyuria):
        ```sparql
        SELECT ?diseaseLabel WHERE {
          ?disease med:hasSymptom med:Polyuria .
          ?disease rdfs:label ?diseaseLabel .
        }
        ```
    2.  To find treatments for a specific disease (e.g., Diabetes Mellitus):
        ```sparql
        SELECT ?treatmentLabel WHERE {
          ?disease rdfs:label "Diabetes Mellitus" .
          ?disease med:hasTreatment ?treatment .
          ?treatment rdfs:label ?treatmentLabel .
        }
        ```
    3.  To find symptoms of a disease (e.g., Hypertension):
        ```sparql
        SELECT ?symptomLabel WHERE {
          ?disease rdfs:label "Hypertension" .
          ?disease med:hasSymptom ?symptom .
          ?symptom rdfs:label ?symptomLabel .
        }
        ```
    4.  To find drugs that interact with a specific drug (e.g., Drug_A):
        ```sparql
        SELECT ?interactingDrugLabel WHERE {
          med:Drug_A med:interactsWith ?interactingDrug .
          ?interactingDrug rdfs:label ?interactingDrugLabel .
        }
        ```
    
    Always include a LIMIT clause if you expect many results or want to keep output concise (e.g., `LIMIT 5`).
    Be specific with your prefixes (e.g., `med:` for medical entities, `rdfs:` for labels).
    """
)

# 3. LLM Agent Layer

# Initialize the LLM
# Ensure OPENAI_API_KEY is set in your environment variables.
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Define the tools the agent can use
tools = [query_knowledge_graph_tool]

# Get the prompt for the ReAct agent from Langchain Hub
prompt = hub.pull("hwchase17/react")

# Customize the system prompt to guide the LLM agent
system_prompt = """
You are a highly intelligent and specialized Healthcare Diagnostic Assistant.
Your primary role is to assist medical professionals in diagnosing complex diseases and suggesting personalized treatment plans by dynamically interacting with a comprehensive medical Knowledge Graph.

**Important Guidelines:**
1.  **Prioritize Patient Safety:** Always preface your responses with a strong disclaimer: "Disclaimer: This AI assistant provides preliminary information for medical professionals only and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for any medical concerns."
2.  **Evidence-Based Reasoning:** Base your reasoning on information retrieved from the Knowledge Graph and the provided patient details.
3.  **Dynamic Information Retrieval:** If you need specific, up-to-date, or structured medical knowledge to answer a question or make a diagnosis, use the `query_knowledge_graph` tool.
4.  **Formulate Precise SPARQL Queries:** When using the `query_knowledge_graph` tool, formulate clear and precise SPARQL queries to retrieve exactly the information you need. Identify the specific entities (diseases, symptoms, drugs) and relationships (hasSymptom, hasTreatment, interactsWith) from the `med:` namespace.
5.  **Suggest Next Steps:** After providing potential diagnoses or treatment considerations, suggest relevant next steps such as further tests, specialist consultations, or monitoring.
6.  **Acknowledge Limitations:** If the KG does not contain enough information to provide a definitive answer, state that clearly.
7.  **Be Concise but Comprehensive:** Provide enough detail for a medical professional to understand your reasoning.

"""

# Prepend the custom system prompt to the existing ReAct prompt
prompt.messages[0].content = system_prompt + prompt.messages[0].content

# Create the ReAct agent
agent = create_react_agent(llm, tools, prompt)

# Create the AgentExecutor
# verbose=True will show the agent's thought process in the console
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# 4. User Interface Layer (CLI)

def main():
    print("Welcome to the AI-powered Healthcare Diagnostic Assistant (Prototype).")
    print("Enter patient symptoms and history. Type 'exit' to quit.")
    print("Disclaimer: This AI assistant provides preliminary information for medical professionals only and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for any medical concerns.\n")

    while True:
        user_input = input("Medical Professional Input (e.g., 'Patient has polyuria and polydipsia. What are potential diseases?'):\n> ")
        if user_input.lower() == 'exit':
            break

        try:
            # Invoke the agent with the user's input
            response = agent_executor.invoke({"input": user_input})
            print("\n" + response["output"] + "\n")
        except Exception as e:
            print(f"\nAn error occurred during agent execution: {e}\n")

if __name__ == "__main__":
    main()
