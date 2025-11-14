import networkx as nx
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI # Placeholder for any LLM
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import os

# Set your OpenAI API key or configure a different LLM
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# --- 1. MedicalKnowledgeGraph Class ---
class MedicalKnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_node(self, node_id: str, node_type: str, attributes: Dict[str, Any] = None):
        if attributes is None:
            attributes = {}
        self.graph.add_node(node_id, type=node_type, **attributes)

    def add_edge(self, u: str, v: str, relation: str, attributes: Dict[str, Any] = None):
        if attributes is None:
            attributes = {}
        self.graph.add_edge(u, v, relation=relation, **attributes)

    def get_neighbors(self, node_id: str):
        if node_id not in self.graph:
            return []
        neighbors = []
        for neighbor in self.graph.neighbors(node_id):
            edge_data = self.graph.get_edge_data(node_id, neighbor)
            neighbors.append({
                "source": node_id,
                "relation": edge_data["relation"],
                "target": neighbor,
                "target_type": self.graph.nodes[neighbor].get("type")
            })
        for source, _, edge_data in self.graph.in_edges(node_id, data=True):
            neighbors.append({
                "source": source,
                "relation": edge_data["relation"],
                "target": node_id,
                "target_type": self.graph.nodes[node_id].get("type")
            })
        return neighbors

    def get_paths_between(self, start_node: str, end_node: str, max_depth: int = 3):
        # Simplified path finding for demonstration
        if start_node not in self.graph or end_node not in self.graph:
            return []
        all_paths = []
        for path in nx.all_simple_paths(self.graph, source=start_node, target=end_node, cutoff=max_depth):
            formatted_path = []
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                edge_data = self.graph.get_edge_data(u, v)
                if edge_data:
                    formatted_path.append(f"({u} --{edge_data['relation']}--> {v})")
            if formatted_path: # Only add if path has edges
                 all_paths.append(" ; ".join(formatted_path))
        return all_paths

    def get_subgraph_around_nodes(self, nodes: List[str], depth: int = 1) -> List[str]:
        subgraph_triples = set()
        for node_id in nodes:
            if node_id not in self.graph:
                continue
            # Add direct outgoing edges
            for neighbor in self.graph.neighbors(node_id):
                edge_data = self.graph.get_edge_data(node_id, neighbor)
                subgraph_triples.add(
                    (node_id, edge_data['relation'], neighbor)
                )
            # Add direct incoming edges
            for source, _, edge_data in self.graph.in_edges(node_id, data=True):
                 subgraph_triples.add(
                    (source, edge_data['relation'], node_id)
                 )
            # For deeper exploration, this would recursively call itself or use BFS/DFS
        
        # Format as list of strings (triples)
        formatted_triples = [f"({s} --{p}--> {o})" for s, p, o in subgraph_triples]
        return formatted_triples

# --- 2. Pydantic Models for Structured Output ---
class ExtractedEntities(BaseModel):
    symptoms: List[str] = Field(description="List of symptoms extracted from the patient description.")
    medical_conditions: List[str] = Field(description="List of known or suspected medical conditions.")
    other_entities: List[str] = Field(description="Other relevant medical entities (e.g., body parts, medications).")

class DiagnosticHypothesis(BaseModel):
    hypothesis: str = Field(description="The primary diagnostic hypothesis.")
    supporting_evidence_kg: List[str] = Field(description="List of relevant facts/triples from the Knowledge Graph supporting the hypothesis.")
    reasoning_steps: str = Field(description="Step-by-step reasoning process, referencing KG facts.")
    differential_diagnoses: List[str] = Field(description="List of other possible diagnoses to consider.")

# --- 3. LLMKGAgent Class ---
class LLMKGAgent:
    def __init__(self, kg: MedicalKnowledgeGraph, llm: ChatOpenAI):
        self.kg = kg
        self.llm = llm
        self._setup_chains()

    def _setup_chains(self):
        # Chain for Entity Extraction
        self.entity_extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a medical assistant tasked with extracting key medical entities from patient descriptions. Respond in JSON format according to the `ExtractedEntities` schema."),
            ("human", "Patient description: {patient_description}")
        ])
        self.entity_extraction_chain = self.entity_extraction_prompt | self.llm.with_structured_output(ExtractedEntities)

        # Chain for KG Guided Exploration (Simplified - more complex agents would be here)
        # This prompt guides the LLM to 'think' about what to explore next
        self.kg_exploration_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a medical reasoning agent exploring a knowledge graph. Given the patient's entities and available KG facts, suggest the next logical steps for exploration to narrow down a diagnosis. Focus on diseases, related symptoms, and risk factors. Provide your suggestions as a comma-separated list of entities to investigate further in the KG."),
            ("human", "Patient entities: {patient_entities}\nKnowledge Graph facts explored so far: {kg_facts}")
        ])
        self.kg_exploration_chain = self.kg_exploration_prompt | self.llm | StrOutputParser()

        # Chain for Diagnosis and Reasoning (RAG + KDCoT)
        self.diagnosis_reasoning_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a highly intelligent medical diagnostic assistant. Given the patient's symptoms and relevant medical facts from a Knowledge Graph, provide a differential diagnosis, a primary diagnostic hypothesis, supporting evidence from the KG (as a list of triples), and a step-by-step reasoning process. Respond in JSON format according to the `DiagnosticHypothesis` schema.\n            Focus on explainability and referencing the provided KG facts directly in your reasoning."),
            ("human", "Patient Symptoms: {symptoms}\nRelevant KG Facts: {kg_facts}")
        ])
        self.diagnosis_reasoning_chain = self.diagnosis_reasoning_prompt | self.llm.with_structured_output(DiagnosticHypothesis)


    def extract_entities(self, patient_description: str) -> ExtractedEntities:
        print(f"\n--- Extracting Entities from: '{patient_description}' ---")
        try:
            entities = self.entity_extraction_chain.invoke({"patient_description": patient_description})
            print(f"Extracted Entities: {entities.model_dump_json(indent=2)}")
            return entities
        except Exception as e:
            print(f"Error during entity extraction: {e}")
            return ExtractedEntities(symptoms=[], medical_conditions=[], other_entities=[])

    def explore_kg_guided(self, patient_entities: List[str], max_steps: int = 3) -> List[str]:
        print(f"\n--- KG Guided Exploration for entities: {patient_entities} ---")
        explored_facts = []
        current_focus_nodes = set(patient_entities)

        for step in range(max_steps):
            print(f"Step {step+1}: Current focus nodes: {list(current_focus_nodes)}")
            
            # Get immediate subgraph around current focus nodes
            subgraph_triples = self.kg.get_subgraph_around_nodes(list(current_focus_nodes), depth=1)
            
            # Filter out already explored facts
            new_facts = [fact for fact in subgraph_triples if fact not in explored_facts]
            if not new_facts and not current_focus_nodes: # No new facts and no new nodes to explore
                break
            
            explored_facts.extend(new_facts)
            print(f"New facts discovered: {new_facts}")

            # LLM decides next steps (simulating beam search / agentic planning)
            # We use the raw text of explored facts for the prompt, but LLM can interpret and guide
            exploration_suggestion = self.kg_exploration_chain.invoke({
                "patient_entities": ", ".join(patient_entities),
                "kg_facts": " ; ".join(explored_facts)
            })
            print(f"LLM exploration suggestion: {exploration_suggestion}")

            # Update focus nodes based on LLM's suggestion (simple parsing of comma-separated string)
            suggested_nodes = [s.strip() for s in exploration_suggestion.split(',') if s.strip()] 
            # Only consider actual nodes existing in KG for next steps, and add potentially new nodes mentioned by LLM
            next_focus_nodes = set()
            for node in suggested_nodes:
                if node in self.kg.graph.nodes: # Check if it's an existing node in the graph
                    next_focus_nodes.add(node)
                else: # If LLM suggests a new entity, try to find it in the KG or add it to be considered in future steps
                    # For this demo, we'll just check if it exists. A real system would have fuzzy matching/entity linking
                    pass 
            
            if not next_focus_nodes and new_facts: # If LLM didn't suggest new nodes, but new facts were found, continue exploring around existing nodes
                 next_focus_nodes = current_focus_nodes
            elif not next_focus_nodes: # No new nodes or facts, stop
                break
            
            current_focus_nodes = next_focus_nodes # Refine current focus for next iteration

            # Simple pruning: If no new facts or suggestions, stop.
            if not new_facts and not next_focus_nodes:
                break

        print(f"Final explored KG facts: {explored_facts}")
        return explored_facts

    def reason_and_generate_diagnosis(self, symptoms: List[str], kg_facts: List[str]) -> DiagnosticHypothesis:
        print(f"\n--- Generating Diagnosis and Reasoning ---")
        try:
            diagnosis = self.diagnosis_reasoning_chain.invoke({
                "symptoms": ", ".join(symptoms),
                "kg_facts": " ; ".join(kg_facts)
            })
            print(f"Diagnostic Output: {diagnosis.model_dump_json(indent=2)}")
            return diagnosis
        except Exception as e:
            print(f"Error during diagnosis generation: {e}")
            return DiagnosticHypothesis(
                hypothesis="Could not generate a diagnosis.",
                supporting_evidence_kg=[],
                reasoning_steps=f"An error occurred: {e}",
                differential_diagnoses=[]
            )

# --- Main Execution --- 
if __name__ == "__main__":
    # Initialize LLM (Ensure OPENAI_API_KEY is set in your environment)
    llm = ChatOpenAI(model="gpt-4o", temperature=0.1)

    # Initialize and populate Medical Knowledge Graph
    kg = MedicalKnowledgeGraph()
    kg.add_node("Severe Headache", "Symptom")
    kg.add_node("Nausea", "Symptom")
    kg.add_node("Photophobia", "Symptom")
    kg.add_node("Migraine", "Disease")
    kg.add_node("Tension Headache", "Disease")
    kg.add_node("Cluster Headache", "Disease")
    kg.add_node("Aura", "Symptom")
    kg.add_node("Stress", "RiskFactor")
    kg.add_node("Genetics", "RiskFactor")
    kg.add_node("Ibuprofen", "Treatment")
    kg.add_node("Triptans", "Treatment")
    kg.add_node("Rest", "Treatment")

    kg.add_edge("Migraine", "Severe Headache", "HAS_SYMPTOM")
    kg.add_edge("Migraine", "Nausea", "HAS_SYMPTOM")
    kg.add_edge("Migraine", "Photophobia", "HAS_SYMPTOM")
    kg.add_edge("Migraine", "Aura", "HAS_SYMPTOM")
    kg.add_edge("Stress", "Migraine", "TRIGGERS")
    kg.add_edge("Genetics", "Migraine", "PREDISPOSES")
    kg.add_edge("Triptans", "Migraine", "TREATS")
    kg.add_edge("Ibuprofen", "Migraine", "ALLEVIATES")
    kg.add_edge("Rest", "Migraine", "ALLEVIATES")

    kg.add_edge("Tension Headache", "Severe Headache", "HAS_SYMPTOM")
    kg.add_edge("Tension Headache", "Stress", "CAUSED_BY")
    kg.add_edge("Ibuprofen", "Tension Headache", "TREATS")

    kg.add_edge("Cluster Headache", "Severe Headache", "HAS_SYMPTOM")
    kg.add_edge("Cluster Headache", "Nausea", "HAS_SYMPTOM", attributes={"frequency": "less_common"})


    # Initialize LLM-KG Agent
    agent = LLMKGAgent(kg=kg, llm=llm)

    # --- Sample Diagnostic Workflow ---
    patient_description = "Patient is a 35-year-old female presenting with a severe, throbbing headache, accompanied by intense nausea and extreme sensitivity to light. She reports visual disturbances before the onset of the headache. No fever or recent trauma."

    # Step 1: Extract entities
    extracted_entities = agent.extract_entities(patient_description)
    all_relevant_entities = list(set(extracted_entities.symptoms + extracted_entities.medical_conditions + extracted_entities.other_entities))

    # Step 2: KG Guided Exploration
    # Focus nodes for initial exploration could be the extracted symptoms
    kg_facts = agent.explore_kg_guided(patient_entities=extracted_entities.symptoms)

    # Step 3: Reason and Generate Diagnosis
    final_diagnosis = agent.reason_and_generate_diagnosis(
        symptoms=extracted_entities.symptoms,
        kg_facts=kg_facts
    )

    print("\n====== Final Diagnostic Output ======")
    print(f"Primary Hypothesis: {final_diagnosis.hypothesis}")
    print(f"Differential Diagnoses: {', '.join(final_diagnosis.differential_diagnoses)}")
    print(f"Reasoning: {final_diagnosis.reasoning_steps}")
    print(f"Supporting KG Evidence:\n- " + "\n- ".join(final_diagnosis.supporting_evidence_kg))