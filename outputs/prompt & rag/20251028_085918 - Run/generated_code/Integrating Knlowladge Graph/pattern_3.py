import os
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from neo4j import GraphDatabase

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import tool
from langchain_core.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI # Or replace with other LLM providers like ChatGoogleGenerativeAI

import spacy

# --- Configuration --- #
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # Ensure this is set in your environment

# Load a spaCy model for medical entity extraction. You might need to install 'en_core_web_sm' or a specialized medical model.
try:
    nlp = spacy.load("en_core_web_sm") # For general NER, consider 'en_core_med7' for better medical context
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'. Run 'python -m spacy download en_core_web_sm' if this fails.")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# --- Neo4j KG Connection --- #
class Neo4jKG:
    def __init__(self, uri, username, password):
        self._driver = GraphDatabase.driver(uri, auth=(username, password))
        self._driver.verify_connectivity()
        print("Connected to Neo4j Knowledge Graph.")

    def close(self):
        self._driver.close()

    def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        with self._driver.session() as session:
            result = session.run(query, parameters)
            return [record.data() for record in result]

kg_db = Neo4jKG(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)

# --- Helper Functions --- #
def _format_path_as_triples(path_data: List[Dict[str, Any]]) -> List[str]:
    formatted_triples = []
    for record in path_data:
        # Assuming path_data contains results like {n: ..., r: ..., m: ...}
        if 'n' in record and 'r' in record and 'm' in record:
            source = record['n'].get('name', record['n'].get('id', 'UnknownSource'))
            relation = record['r'].get('type', 'UNKNOWN_RELATION')
            target = record['m'].get('name', record['m'].get('id', 'UnknownTarget'))
            formatted_triples.append(f"({source})-[:{relation}]->({target})")
        elif 'path' in record and isinstance(record['path'], dict) and 'nodes' in record['path'] and 'relationships' in record['path']:
            # Handle more complex path representations if necessary
            nodes = record['path']['nodes']
            rels = record['path']['relationships']
            # This is a simplified example; a full implementation would iterate and reconstruct
            if len(nodes) >= 2 and len(rels) >= 1:
                source = nodes[0].get('name', nodes[0].get('id', 'UnknownSource'))
                relation = rels[0].get('type', 'UNKNOWN_RELATION')
                target = nodes[1].get('name', nodes[1].get('id', 'UnknownTarget'))
                formatted_triples.append(f"({source})-[:{relation}]->({target})")
        else:
             # Fallback for other record structures, just print the raw record for debugging
            print(f"Warning: Unexpected record structure in _format_path_as_triples: {record}")

    return formatted_triples

# --- LangChain Tools --- #
llm = ChatOpenAI(temperature=0, api_key=OPENAI_API_KEY)

@tool
def kg_query_executor(query: str) -> List[Dict[str, Any]]:
    """Executes a Cypher query against the Neo4j Knowledge Graph and returns the results.
    Input should be a valid Cypher query string.
    Example: MATCH (d:Disease)-[:HAS_SYMPTOM]->(s:Symptom {{name: 'fever'}}) RETURN d.name LIMIT 5
    """
    print(f"Executing Cypher Query: {query}")
    try:
        results = kg_db.execute_query(query)
        return results
    except Exception as e:
        return [{"error": str(e), "query": query}]

@tool
def llm_guided_kg_explorer(
    start_entity_name: str,
    start_entity_label: str,
    exploration_goal: str,
    max_steps: int = 3,
    beam_width: int = 2
) -> List[str]:
    """Explores the Knowledge Graph starting from an entity, guided by an LLM to find relevant paths.
    Uses a hybrid pruning strategy (lightweight + LLM) and returns paths as triples.

    Args:
        start_entity_name (str): The name or identifier of the starting entity (e.g., 'Hypertension').
        start_entity_label (str): The label of the starting entity (e.g., 'Disease', 'Symptom').
        exploration_goal (str): A natural language instruction for the LLM on what to explore (e.g., 'find common treatments', 'identify potential causes').
        max_steps (int): Maximum number of hops to explore in the graph.
        beam_width (int): The number of top paths to keep at each step based on LLM's guidance.

    Returns:
        List[str]: A list of relevant paths found, formatted as (source)-[:RELATION]->(target) triples.
    """
    print(f"Starting KG exploration for '{start_entity_name}' (Label: {start_entity_label}) with goal: '{exploration_goal}'")
    current_paths = [] # Stores paths as lists of (node, relation, node) tuples or similar
    # Initialize with the starting entity as a single-node path
    initial_node_query = f"MATCH (n:{start_entity_label} {{name: '{start_entity_name}'}}) RETURN n LIMIT 1"
    initial_node_result = kg_db.execute_query(initial_node_query)

    if not initial_node_result:
        return [f"Could not find starting entity: {start_entity_name} ({start_entity_label})"]

    start_node = initial_node_result[0]['n'] # Assuming 'n' is the node object
    current_paths = [[start_node]] # Each path is a list of nodes
    all_relevant_triples = set()

    for step in range(max_steps):
        next_paths_candidates = []
        for path in current_paths:
            last_node = path[-1]
            if not isinstance(last_node, dict) or 'id' not in last_node: # Check if it's a valid node object
                continue
            last_node_id = last_node['id'] # Assuming nodes have 'id' property or similar for internal use

            # Lightweight Pruning: Fetch neighbors, simple filtering can happen here based on common sense
            # For example, avoid certain types of nodes or relationships early
            neighbor_query = f"MATCH (n)-[r]-(m) WHERE id(n) = {last_node_id} RETURN n, r, m LIMIT 20" # Fetch up to 20 neighbors
            neighbors_data = kg_db.execute_query(neighbor_query)

            for record in neighbors_data:
                if 'n' in record and 'r' in record and 'm' in record:
                    # Simple lightweight pruning: filter out generic 'HAS_ATTRIBUTE' type relationships if not relevant
                    if record['r'].get('type') == 'HAS_ATTRIBUTE' and not exploration_goal.lower().startswith('attribute'):
                        continue # Example pruning rule
                    
                    # Add the new step to the path
                    extended_path_nodes = path + [record['m']]
                    next_paths_candidates.append({
                        "path_nodes": extended_path_nodes,
                        "triple": (record['n'].get('name', 'N/A'), record['r'].get('type', 'N/A'), record['m'].get('name', 'N/A'))
                    })
                    all_relevant_triples.add(f"({record['n'].get('name', 'N/A')})-[:{record['r'].get('type', 'N/A')}]->({record['m'].get('name', 'N/A')})")

        if not next_paths_candidates:
            break

        # LLM-based Pruning and Scoring (Beam Search logic)
        # Create a prompt for the LLM to rank candidate paths
        candidate_path_descriptions = []
        for i, candidate in enumerate(next_paths_candidates):
            path_desc = " -> ".join([n.get('name', n.get('id', 'Unknown')) for n in candidate['path_nodes']])
            candidate_path_descriptions.append(f"Path {i+1}: {path_desc}")

        ranking_prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are an AI assistant that ranks potential knowledge graph paths based on a given exploration goal."),
            ("human", (
                f"Given the starting entity '{start_entity_name}' and the goal '{exploration_goal}', "
                f"please rank the following paths from most relevant to least relevant. "
                f"Respond with a comma-separated list of path numbers (e.g., '2,1,3'):\n" +
                "\n".join(candidate_path_descriptions)
            ))
        ])

        try:
            ranking_chain = ranking_prompt_template | llm | StrOutputParser()
            ranking_response = ranking_chain.invoke({"start_entity_name": start_entity_name, "exploration_goal": exploration_goal, "candidate_path_descriptions": candidate_path_descriptions})
            ranked_indices = [int(x.strip()) - 1 for x in ranking_response.split(',') if x.strip().isdigit()]
            print(f"LLM ranked paths: {ranking_response}")
        except Exception as e:
            print(f"Error during LLM ranking: {e}. Falling back to random selection.")
            ranked_indices = list(range(len(next_paths_candidates)))

        # Select top 'beam_width' paths
        current_paths = []
        for idx in ranked_indices:
            if idx < len(next_paths_candidates) and len(current_paths) < beam_width:
                current_paths.append(next_paths_candidates[idx]['path_nodes'])
            elif len(current_paths) >= beam_width:
                break

    # Return all unique relevant triples found during the exploration
    return list(all_relevant_triples)


@tool
def topic_entity_extractor(text: str) -> List[Dict[str, str]]:
    """Extracts medical topic entities (diseases, symptoms, drugs) from unstructured text using spaCy and an LLM for refinement.
    Returns a list of dictionaries with 'entity' and 'label'.
    """
    doc = nlp(text)
    extracted_entities = []
    for ent in doc.ents:
        # Basic filtering/mapping for medical context
        if ent.label_ in ["ORG", "PERSON", "GPE", "PRODUCT"]:# Example: filter common non-medical entities
             # If using en_core_med7, labels like 'DISEASE', 'SYMPTOM', 'DRUG' would be more direct
             # For generic models, we might rely on the LLM more for refinement
             pass
        else:
            extracted_entities.append({"entity": ent.text, "label": ent.label_})

    # LLM refinement (Semantic Parsing for KGQA - simplified here for entity resolution)
    if extracted_entities:
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are an expert medical entity resolver. Refine the extracted entities to be precise medical terms and assign appropriate, general medical labels (e.g., 'Symptom', 'Disease', 'Drug', 'Procedure'). If an entity is not clearly medical, remove it."),
            ("human", (
                f"Refine the following list of entities extracted from patient notes: {extracted_entities}. "
                "Return a JSON array of objects with 'entity' and a refined 'label' (e.g., [{'entity': 'fever', 'label': 'Symptom'}])."
            ))
        ])
        refinement_chain = prompt_template | llm | StrOutputParser() # Output parser would need to parse JSON
        try:
            # This requires a proper JSON parser, for simplicity, we'll try to parse it directly
            refined_json_str = refinement_chain.invoke({"extracted_entities": extracted_entities})
            # Attempt to parse, fallback if not perfect JSON
            refined_entities = eval(refined_json_str) # Using eval is generally unsafe, better to use json.loads
            if isinstance(refined_entities, list):
                # Further validate the structure of each item
                validated_refined_entities = []
                for item in refined_entities:
                    if isinstance(item, dict) and 'entity' in item and 'label' in item:
                        validated_refined_entities.append(item)
                return validated_refined_entities
            else:
                print(f"LLM refinement did not return a list: {refined_json_str}")
                return extracted_entities # Fallback
        except Exception as e:
            print(f"Error during LLM entity refinement: {e}")
            return extracted_entities # Fallback
    return extracted_entities

# --- LLM-KG Agent Setup --- #
tools = [
    kg_query_executor,
    llm_guided_kg_explorer,
    topic_entity_extractor
]

# The agent prompt - incorporating KDCoT and RoG principles
AGENT_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a highly intelligent Medical Diagnosis and Treatment Recommendation AI Assistant. "
        "Your goal is to assist healthcare professionals by accurately diagnosing conditions and recommending treatments. "
        "You have access to a comprehensive Medical Knowledge Graph (KG) and various tools to interact with it. "
        "Think step-by-step (Knowledge-Driven Chain-of-Thought) and leverage the KG to ground your reasoning, reduce hallucinations, "
        "and provide up-to-date, interpretable knowledge. "
        "When exploring the KG, consider relevant paths for diagnosis or treatment options."
        "Use the following tools: {tool_names}"
    )),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

agent = create_react_agent(llm, tools, AGENT_PROMPT_TEMPLATE)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# --- FastAPI Application --- #
app = FastAPI(
    title="Medical Diagnosis and Treatment Recommendation System",
    description="An AI system leveraging LLMs and Knowledge Graphs for enhanced medical reasoning."
)

class PatientInput(BaseModel):
    patient_id: str = Field(..., example="P001")
    symptoms: str = Field(..., example="The patient reports severe headache, high fever, and stiff neck.")
    medical_history: Optional[str] = Field(None, example="No significant past medical history. Allergic to Penicillin.")
    lab_results: Optional[str] = Field(None, example="White blood cell count elevated, CRP high.")

class RecommendationOutput(BaseModel):
    patient_id: str
    diagnosis_summary: str
    potential_diagnoses: List[str]
    recommended_treatments: List[str]
    reasoning_path: List[str]
    warnings: List[str]

@app.post("/diagnose", response_model=RecommendationOutput)
async def diagnose_patient(patient_data: PatientInput):
    """Provides a medical diagnosis and treatment recommendations based on patient data.
    The system leverages LLMs and a Knowledge Graph for robust reasoning.
    """
    full_patient_narrative = (
        f"Patient ID: {patient_data.patient_id}. "
        f"Symptoms: {patient_data.symptoms}. "
        f"Medical History: {patient_data.medical_history if patient_data.medical_history else 'None'}. "
        f"Lab Results: {patient_data.lab_results if patient_data.lab_results else 'None'}."
    )

    print(f"Received diagnosis request for patient: {patient_data.patient_id}")

    # The LLM-KG agent will orchestrate the entire process
    # It will use topic_entity_extractor, then kg_query_executor, and llm_guided_kg_explorer
    # The prompt below guides the agent's initial action.
    try:
        agent_response = agent_executor.invoke({
            "input": (
                f"Analyze the following patient data to provide a potential diagnosis and treatment recommendations. "
                f"First, extract all key medical entities (symptoms, diseases, drugs) from the narrative. "
                f"Then, use the knowledge graph to explore possible diseases related to the extracted symptoms, and subsequently, "
                f"find recommended treatments for those diseases. "
                f"Patient narrative: {full_patient_narrative}"
            )
        })

        # Parse the agent's output - this is a simplification; a real system would need more structured output from the agent
        # The agent's output is typically a string, which we need to interpret.
        # For a more robust solution, the agent could be prompted to output a JSON-like structure for its final answer.
        diagnosis_summary = agent_response.get("output", "Could not generate a clear diagnosis summary.")

        # Placeholder for extracting structured info from the agent's narrative output
        # In a production system, the agent's final answer should be structured (e.g., JSON) or parsed robustly.
        potential_diagnoses = [d.strip() for d in diagnosis_summary.split('Potential Diagnoses:')[-1].split('Recommended Treatments:')[0].split(',') if d.strip()] if 'Potential Diagnoses:' in diagnosis_summary else ["Uncertain"] # Simplified parsing
        recommended_treatments = [t.strip() for t in diagnosis_summary.split('Recommended Treatments:')[-1].split(',') if t.strip()] if 'Recommended Treatments:' in diagnosis_summary else ["Consult specialist"]
        reasoning_path = _format_path_as_triples(kg_query_executor(f"MATCH p=(n)-[r]-(m) WHERE n.name CONTAINS '{patient_data.patient_id}' RETURN n,r,m LIMIT 5")) # Simplified placeholder for reasoning path
        warnings = []

        return RecommendationOutput(
            patient_id=patient_data.patient_id,
            diagnosis_summary=diagnosis_summary,
            potential_diagnoses=potential_diagnoses,
            recommended_treatments=recommended_treatments,
            reasoning_path=reasoning_path,
            warnings=warnings
        )

    except Exception as e:
        print(f"An error occurred during diagnosis: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

@app.get("/health")
async def health_check():
    """Health check endpoint to ensure the API is running and connected to KG."""
    try:
        kg_db.execute_query("MATCH (n) RETURN n LIMIT 1")
        return {"status": "healthy", "neo4j_connected": True}
    except Exception as e:
        return {"status": "unhealthy", "neo4j_connected": False, "error": str(e)}

# --- Main Execution --- #
if __name__ == "__main__":
    import uvicorn
    print("Starting Medical Diagnosis System API. Ensure Neo4j is running and environment variables are set.")
    print("To run: uvicorn medical_diagnosis_system:app --reload")
    uvicorn.run(app, host="0.0.0.0", port=8000)