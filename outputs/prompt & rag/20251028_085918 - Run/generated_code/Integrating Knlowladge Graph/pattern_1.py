
import streamlit as st
import spacy
import networkx as nx
from collections import deque
from typing import List, Dict, Any, Tuple

# Placeholder for LLM integration (e.g., using LangChain/OpenAI)
# In a real application, you would initialize your LLM here
# from langchain_openai import ChatOpenAI
# llm = ChatOpenAI(model="gpt-4", temperature=0.0)

class LLMAgent:
    def __init__(self, model_name: str = "gpt-4"):
        # self.llm = ChatOpenAI(model=model_name, temperature=0.0)
        self.model_name = model_name
        st.write(f"Initializing LLM Agent with placeholder for model: {self.model_name}")

    def invoke_llm(self, prompt: str) -> str:
        # Placeholder for actual LLM call
        # In a real application:
        # response = self.llm.invoke(prompt)
        # return response.content
        st.write(f"LLM Prompt:\n```\n{prompt}\n```")
        # Simulate LLM response for demonstration
        if "extract key entities" in prompt.lower():
            return "Extracted Entities: fever, cough, diabetes, headache"
        elif "diseases associated with" in prompt.lower():
            return "Potential diseases: Common Cold, Flu, Pneumonia"
        elif "diagnostic criteria for" in prompt.lower():
            return "Criteria for Common Cold: runny nose, sore throat, sneezing"
        elif "reasoning based on" in prompt.lower():
            return "Based on symptoms (fever, cough) and KG path (Symptom->Disease), probable diagnosis is Common Cold with high confidence. Recommend rest and hydration."
        return f"[LLM response to: {prompt[:50]}...]"


class MedicalKnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self._load_sample_data()

    def _load_sample_data(self):
        # Sample medical data for demonstration
        self.graph.add_nodes_from([
            "Fever", "Cough", "Headache", "Runny Nose", "Sore Throat",
            "Common Cold", "Flu", "Pneumonia", "Diabetes",
            "Paracetamol", "Ibuprofen", "Antibiotics",
            "Complication: Dehydration", "Test: Nasal Swab", "Test: Chest X-ray"
        ])

        self.graph.add_edges_from([
            ("Fever", "has_symptom_of", "Common Cold"),
            ("Cough", "has_symptom_of", "Common Cold"),
            ("Runny Nose", "has_symptom_of", "Common Cold"),
            ("Sore Throat", "has_symptom_of", "Common Cold"),
            ("Fever", "has_symptom_of", "Flu"),
            ("Cough", "has_symptom_of", "Flu"),
            ("Headache", "has_symptom_of", "Flu"),
            ("Cough", "has_symptom_of", "Pneumonia"),
            ("Fever", "has_symptom_of", "Pneumonia"),
            ("Pneumonia", "requires_test", "Test: Chest X-ray"),
            ("Common Cold", "treatment", "Paracetamol"),
            ("Flu", "treatment", "Ibuprofen"),
            ("Pneumonia", "treatment", "Antibiotics"),
            ("Common Cold", "can_lead_to", "Complication: Dehydration"),
            ("Flu", "can_lead_to", "Complication: Dehydration"),
            ("Test: Nasal Swab", "diagnoses", "Flu"),
            ("Test: Nasal Swab", "diagnoses", "Common Cold"),
            ("Diabetes", "risk_factor_for", "Pneumonia"),
        ], relation=True)

        st.write("Medical Knowledge Graph initialized with sample data.")

    def get_neighbors(self, entity: str) -> List[Tuple[str, str, str]]:
        # Returns (source, relation, target) triples
        neighbors = []
        if entity in self.graph:
            for neighbor in self.graph.neighbors(entity):
                relation = self.graph.get_edge_data(entity, neighbor).get('relation', 'unknown')
                neighbors.append((entity, relation, neighbor))
            for u, v, data in self.graph.edges(data=True):
                if v == entity:
                    relation = data.get('relation', 'unknown')
                    neighbors.append((u, relation, entity))
        return list(set(neighbors)) # Remove duplicates if any from bidirectional check

    def find_paths(self, start_entity: str, max_depth: int = 2) -> List[List[Tuple[str, str, str]]]:
        all_paths = []
        queue = deque([([start_entity], [])]) # (path_nodes, path_edges)

        while queue:
            current_nodes, current_edges = queue.popleft()
            last_node = current_nodes[-1]

            if len(current_edges) >= max_depth:
                all_paths.append(current_edges)
                continue

            for u, rel, v in self.get_neighbors(last_node):
                if v not in current_nodes: # Avoid cycles for simplicity in demo
                    new_nodes = current_nodes + [v]
                    new_edges = current_edges + [(u, rel, v)]
                    queue.append((new_nodes, new_edges))
                elif u not in current_nodes: # Handles incoming edges to start node if not already covered
                    new_nodes = current_nodes + [u]
                    new_edges = current_edges + [(u, rel, v)]
                    queue.append((new_nodes, new_edges))

            if not self.get_neighbors(last_node) and current_edges: # If no more neighbors, add the path
                 all_paths.append(current_edges)

        # Ensure unique paths
        unique_paths = []
        for path in all_paths:
            if path not in unique_paths:
                unique_paths.append(path)
        return unique_paths


class EntityExtractionModule:
    def __init__(self, model: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(model)
            st.write(f"SpaCy model '{model}' loaded successfully for entity extraction.")
        except OSError:
            st.error(f"SpaCy model '{model}' not found. Downloading... (This might take a moment)")
            spacy.cli.download(model)
            self.nlp = spacy.load(model)
            st.success(f"SpaCy model '{model}' downloaded and loaded.")

    def extract_entities(self, text: str) -> List[str]:
        doc = self.nlp(text)
        entities = [ent.text for ent in doc.ents] # Using named entities
        # Add a simple heuristic for common medical terms not always caught as NER
        medical_keywords = ["fever", "cough", "diabetes", "headache", "pneumonia", "MRI", "lab results", "symptoms"]
        for keyword in medical_keywords:
            if keyword in text.lower() and keyword not in [e.lower() for e in entities]:
                entities.append(keyword)
        return list(set(entities))


class BeamSearchPruningModule:
    def __init__(self, llm_agent: LLMAgent, kg: MedicalKnowledgeGraph, beam_width: int = 5):
        self.llm_agent = llm_agent
        self.kg = kg
        self.beam_width = beam_width

    def beam_search_and_prune(self, start_entities: List[str], max_depth: int = 2) -> List[List[Tuple[str, str, str]]]:
        st.write(f"Initiating beam search from entities: {start_entities} with max depth {max_depth} and beam width {self.beam_width}")
        all_candidate_paths = []
        for entity in start_entities:
            paths_from_entity = self.kg.find_paths(entity, max_depth)
            all_candidate_paths.extend(paths_from_entity)

        # Placeholder for LLM-guided pruning and scoring
        # In a real system, the LLM would score paths based on relevance to the query
        # For demonstration, we'll simulate a pruning step.
        st.write(f"Found {len(all_candidate_paths)} raw candidate paths. Simulating pruning...")

        # Simple pruning: prioritize paths with more medical relevance or shorter paths
        # This is a highly simplified heuristic for the demo
        pruned_paths = sorted(all_candidate_paths, key=lambda p: len(p), reverse=False)
        pruned_paths = pruned_paths[:self.beam_width] # Select top paths by a simple heuristic

        st.write(f"Pruned to {len(pruned_paths)} paths using a simple heuristic.")
        return pruned_paths


class ReasoningExplanationGenerationModule:
    def __init__(self, llm_agent: LLMAgent):
        self.llm_agent = llm_agent

    def generate_reasoning_and_explanation(self, patient_data: str, kg_paths: List[List[Tuple[str, str, str]]]) -> Dict[str, Any]:
        st.write("Generating reasoning and explanations...")
        kg_paths_str = "\n".join([" -> ".join([f"({s})-{r}->({o})" for s, r, o in path]) for path in kg_paths])

        reasoning_prompt = f"""
        Patient Data: {patient_data}
        Relevant Medical Knowledge Graph Paths:
        {kg_paths_str}

        Based on the patient data and the provided medical knowledge graph paths, perform faithful reasoning to:
        1. Identify potential differential diagnoses and their likelihoods.
        2. Explain the evidence supporting each diagnosis from the KG paths.
        3. Suggest further diagnostic tests or next steps.

        Format your response clearly, starting with 'Differential Diagnoses:', then 'Evidence:', and then 'Suggested Next Steps:'.
        """
        llm_reasoning = self.llm_agent.invoke_llm(reasoning_prompt)

        # Parse LLM response (simplified for demo)
        diagnoses = []
        evidence = []
        next_steps = []

        lines = llm_reasoning.split('\n')
        current_section = None
        for line in lines:
            if line.startswith("Differential Diagnoses:"):
                current_section = "diagnoses"
            elif line.startswith("Evidence:"):
                current_section = "evidence"
            elif line.startswith("Suggested Next Steps:"):
                current_section = "next_steps"
            elif current_section == "diagnoses" and line.strip() and not line.startswith("Based on"): # Avoid initial LLM boilerplate
                diagnoses.append(line.strip())
            elif current_section == "evidence" and line.strip():
                evidence.append(line.strip())
            elif current_section == "next_steps" and line.strip():
                next_steps.append(line.strip())

        return {
            "differential_diagnoses": diagnoses if diagnoses else ["Common Cold (placeholder)"],
            "evidence": evidence if evidence else ["Symptoms match known patterns in KG."],
            "suggested_next_steps": next_steps if next_steps else ["Monitor symptoms, rest, hydration."],
            "raw_llm_reasoning": llm_reasoning
        }


# Main application logic for Streamlit
st.title("🧠 Clinical Diagnostic Assistant (KG-Guided LLM)")
st.markdown("This assistant helps doctors diagnose complex diseases by leveraging an LLM augmented with a comprehensive medical Knowledge Graph.")

# Initialize components
@st.cache_resource
def init_components():
    llm_agent = LLMAgent()
    kg = MedicalKnowledgeGraph()
    entity_extractor = EntityExtractionModule()
    beam_search_pruner = BeamSearchPruningModule(llm_agent, kg)
    reasoning_generator = ReasoningExplanationGenerationModule(llm_agent)
    return llm_agent, kg, entity_extractor, beam_search_pruner, reasoning_generator

llm_agent, kg, entity_extractor, beam_search_pruner, reasoning_generator = init_components()

patient_input = st.text_area(
    "Enter patient data (symptoms, history, lab results):",
    "Patient presents with fever, persistent cough for 3 days, and a mild headache. No known allergies."
)

if st.button("Diagnose"):
    if not patient_input:
        st.warning("Please enter patient data to proceed with diagnosis.")
    else:
        with st.spinner("Processing patient data and consulting medical knowledge graph..."):
            st.subheader("\n--- Process Steps ---\n")

            # 1. Entity Extraction
            st.write("**Step 1: Entity Extraction**")
            extracted_entities = entity_extractor.extract_entities(patient_input)
            st.success(f"Extracted Entities: {', '.join(extracted_entities)}")

            # 2. KG Exploration (Beam Search & Pruning)
            st.write("**Step 2: KG Exploration (Beam Search & Pruning)**")
            if not extracted_entities:
                st.warning("No relevant entities extracted. Cannot perform KG exploration.")
                st.stop()

            # Use a subset of extracted entities for beam search start to avoid too much breadth
            # In a real system, LLM would help prioritize start nodes
            start_entities_for_kg = [e for e in extracted_entities if e in kg.graph.nodes()][:3] # Limit for demo
            if not start_entities_for_kg:
                st.warning("Extracted entities not found in KG. Cannot perform KG exploration.")
                st.stop()

            relevant_kg_paths = beam_search_pruner.beam_search_and_prune(start_entities_for_kg, max_depth=2)

            if relevant_kg_paths:
                st.success(f"Found {len(relevant_kg_paths)} relevant KG paths:")
                for i, path in enumerate(relevant_kg_paths):
                    st.text(f"  Path {i+1}: {' -> '.join([f"({s})-{r}->({o})" for s, r, o in path])}")
            else:
                st.info("No relevant KG paths found based on extracted entities.")

            # 3. Reasoning & Explanation Generation
            st.write("**Step 3: Reasoning & Explanation Generation**")
            diagnosis_output = reasoning_generator.generate_reasoning_and_explanation(patient_input, relevant_kg_paths)

            st.subheader("\n--- Diagnosis Results ---\n")
            st.markdown("### Differential Diagnoses:")
            for diag in diagnosis_output["differential_diagnoses"]:
                st.write(f"- {diag}")

            st.markdown("### Evidence:")
            for ev in diagnosis_output["evidence"]:
                st.write(f"- {ev}")

            st.markdown("### Suggested Next Steps:")
            for step in diagnosis_output["suggested_next_steps"]:
                st.write(f"- {step}")

            with st.expander("View Raw LLM Reasoning (for debugging)"):
                st.code(diagnosis_output["raw_llm_reasoning"])


# To run this application:
# 1. Save the code as `clinical_diagnostic_assistant.py`.
# 2. Install necessary libraries:
#    `pip install streamlit spacy networkx`
#    `python -m spacy download en_core_web_sm`
#    (If using actual LLM like OpenAI, also `pip install langchain-openai`)  
# 3. Run from your terminal: `streamlit run clinical_diagnostic_assistant.py`
