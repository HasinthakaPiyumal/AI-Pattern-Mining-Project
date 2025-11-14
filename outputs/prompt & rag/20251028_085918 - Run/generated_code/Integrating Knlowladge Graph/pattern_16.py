
import networkx as nx
import json
from typing import List, Dict, Any, Tuple, Optional
from pydantic import BaseModel, Field

# --- 1. Pydantic Models for Structured Data --- 

class Entity(BaseModel):
    id: str
    type: str
    attributes: Dict[str, Any] = Field(default_factory=dict)

class Relationship(BaseModel):
    source: str
    target: str
    type: str
    attributes: Dict[str, Any] = Field(default_factory=dict)

class KGQuery(BaseModel):
    query_type: str # e.g., "FIND_DISEASES_RELATED_TO_SYMPTOMS", "GET_DIAGNOSTIC_CRITERIA"
    entities: List[str] = Field(default_factory=list)
    relations: List[str] = Field(default_factory=list)
    filters: Dict[str, Any] = Field(default_factory=dict)

class KGResponse(BaseModel):
    results: List[Dict[str, Any]] = Field(default_factory=list)
    raw_triples: List[Tuple[str, str, str]] = Field(default_factory=list)

# --- 2. Medical Knowledge Graph (KG) Module --- 

class MedicalKnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.entity_map = {}

    def add_entity(self, entity: Entity):
        if entity.id not in self.graph:
            self.graph.add_node(entity.id, type=entity.type, **entity.attributes)
            self.entity_map[entity.id] = entity

    def add_relation(self, relation: Relationship):
        if relation.source in self.graph and relation.target in self.graph:
            self.graph.add_edge(relation.source, relation.target, type=relation.type, **relation.attributes)
        else:
            print(f"Warning: Source or target entity not found for relation: {relation}")

    def get_neighbors(self, entity_id: str, relation_type: Optional[str] = None) -> List[Tuple[str, str, str]]:
        neighbors = []
        if entity_id in self.graph:
            for neighbor in self.graph.neighbors(entity_id):
                edge_data = self.graph.get_edge_data(entity_id, neighbor)
                if relation_type is None or edge_data.get("type") == relation_type:
                    neighbors.append(self._format_triple(entity_id, edge_data.get("type", "has_relation"), neighbor))
            for source, _, edge_data in self.graph.in_edges(entity_id, data=True):
                if relation_type is None or edge_data.get("type") == relation_type:
                    neighbors.append(self._format_triple(source, edge_data.get("type", "has_relation"), entity_id))
        return neighbors

    def get_paths(self, source_id: str, target_id: str, max_depth: int = 2) -> List[List[str]]:
        paths = []
        if source_id in self.graph and target_id in self.graph:
            for path in nx.all_simple_paths(self.graph, source_id, target_id, cutoff=max_depth):
                paths.append(path)
        return paths

    def _format_triple(self, s: str, p: str, o: str) -> str:
        return f"({s}, {p}, {o})"

    def execute_query(self, query: KGQuery) -> KGResponse:
        results = []
        raw_triples = []

        if query.query_type == "FIND_DISEASES_RELATED_TO_SYMPTOMS":
            symptom_entities = [s for s in query.entities if self.entity_map.get(s) and self.entity_map[s].type == "Symptom"]
            disease_candidates = set()

            for symptom in symptom_entities:
                # Find diseases related to this symptom
                for _, target, data in self.graph.out_edges(symptom, data=True):
                    if data.get("type") == "causes" and self.entity_map.get(target) and self.entity_map[target].type == "Disease":
                        disease_candidates.add(target)
                        raw_triples.append(self._format_triple(symptom, "causes", target))
                for source, _, data in self.graph.in_edges(symptom, data=True):
                    if data.get("type") == "has_symptom" and self.entity_map.get(source) and self.entity_map[source].type == "Disease":
                        disease_candidates.add(source)
                        raw_triples.append(self._format_triple(source, "has_symptom", symptom))

            # Filter by family history if present
            family_history_diseases = query.filters.get("family_history_of", [])
            if family_history_diseases:
                filtered_candidates = set()
                for disease in disease_candidates:
                    # Check if the disease itself is in family history, or if related autoimmune disorders are
                    if disease in family_history_diseases or any(self.graph.has_edge(disease, fh_dis) and self.graph.get_edge_data(disease, fh_dis).get("type") == "is_type_of" for fh_dis in family_history_diseases):
                        filtered_candidates.add(disease)
                disease_candidates = filtered_candidates

            for disease_id in disease_candidates:
                entity_data = self.entity_map.get(disease_id)
                if entity_data:
                    results.append({"entity_id": disease_id, "type": entity_data.type, "attributes": entity_data.attributes})

        elif query.query_type == "GET_DIAGNOSTIC_CRITERIA":
            target_disease = query.entities[0] if query.entities else None
            if target_disease and self.entity_map.get(target_disease) and self.entity_map[target_disease].type == "Disease":
                # Find associated symptoms and lab markers
                for _, target, data in self.graph.out_edges(target_disease, data=True):
                    if data.get("type") == "has_symptom" or data.get("type") == "requires_lab_test":
                        entity_data = self.entity_map.get(target)
                        if entity_data:
                            results.append({"entity_id": target, "type": entity_data.type, "attributes": entity_data.attributes})
                            raw_triples.append(self._format_triple(target_disease, data.get("type"), target))

        # Add any paths found to raw_triples (simplified for this mock)
        for s in query.entities:
            for t in query.entities:
                if s != t:
                    for path in self.get_paths(s, t):
                        # Convert path nodes to triples
                        for i in range(len(path) - 1):
                            source_node = path[i]
                            target_node = path[i+1]
                            edge_data = self.graph.get_edge_data(source_node, target_node)
                            if edge_data:
                                raw_triples.append(self._format_triple(source_node, edge_data.get("type", "relates_to"), target_node))

        return KGResponse(results=results, raw_triples=list(set(raw_triples))) # Use set to avoid duplicate triples

# --- 3. Mock LLM and LLM Agent Module --- 

class MockLLM:
    def generate(self, prompt: str) -> str:
        # Simulate LLM's response based on keywords in the prompt
        if "extract medical entities" in prompt.lower():
            if "chronic fatigue, muscle weakness, joint pain. Family history of autoimmune disorders" in prompt:
                return json.dumps(["chronic fatigue", "muscle weakness", "joint pain", "autoimmune disorders"])
            elif "persistent cough and fever" in prompt:
                return json.dumps(["persistent cough", "fever"])
            else:
                return json.dumps(["symptom X", "disease Y"])
        elif "convert to structured query" in prompt.lower():
            if "chronic fatigue, muscle weakness, joint pain" in prompt:
                return json.dumps(KGQuery(query_type="FIND_DISEASES_RELATED_TO_SYMPTOMS", 
                                        entities=["chronic fatigue", "muscle weakness", "joint pain"],
                                        filters={"family_history_of": ["autoimmune disorders"]}).dict())
            elif "diagnostic criteria for Lupus" in prompt:
                return json.dumps(KGQuery(query_type="GET_DIAGNOSTIC_CRITERIA", entities=["Lupus"]).dict())
            else:
                return json.dumps(KGQuery(query_type="GENERIC_QUERY", entities=["entity A"]).dict())
        elif "propose next reasoning step" in prompt.lower():
            if "current diagnosis candidates: Lupus, Rheumatoid Arthritis, Fibromyalgia" in prompt:
                return "Explore diagnostic criteria for Lupus vs. Rheumatoid Arthritis based on lab markers like ANA and RF factor."
            elif "exploring diagnostic criteria for Lupus" in prompt.lower():
                return "Consider common treatments and prognosis for Lupus."
            else:
                return "Further explore related symptoms or genetic factors."
        elif "synthesize diagnosis and explanation" in prompt.lower():
            if "Lupus is a strong candidate" in prompt:
                return "Based on chronic fatigue, muscle weakness, joint pain, and family history of autoimmune disorders, *Lupus* is a strong candidate. Further tests for ANA and anti-dsDNA are recommended. Reasoning derived from KG facts linking these symptoms and family history to Lupus, and distinguishing it from other conditions like Rheumatoid Arthritis which typically presents with symmetric joint involvement."
            else:
                return "Diagnosis: Undetermined. More information needed."
        return "LLM Mock Response: " + prompt[:100] + "..."


class LLMAgent:
    def __init__(self, kg_module: MedicalKnowledgeGraph, llm: MockLLM):
        self.kg_module = kg_module
        self.llm = llm

    def _call_llm(self, prompt_template: str, **kwargs) -> str:
        prompt = prompt_template.format(**kwargs)
        return self.llm.generate(prompt)

    def extract_topic_entities(self, natural_language_query: str) -> List[str]:
        prompt = f"Given the patient query, extract all relevant medical entities (symptoms, diseases, treatments, conditions). Return as a JSON list. Query: {natural_language_query}"
        try:
            entities_str = self._call_llm(prompt)
            return json.loads(entities_str)
        except json.JSONDecodeError:
            return []

    def semantic_parse_kgqa(self, natural_language_query: str, entities: List[str], filters: Dict[str, Any] = None) -> KGQuery:
        filters_str = json.dumps(filters) if filters else "{}"
        prompt = f"Convert the following natural language medical query into a structured KGQuery JSON object. Identified entities: {entities}. Filters: {filters_str}. Query: {natural_language_query}"
        try:
            query_str = self._call_llm(prompt)
            return KGQuery.parse_raw(query_str)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing KGQuery from LLM: {e}")
            return KGQuery(query_type="FALLBACK_GENERIC_QUERY", entities=entities)

    def _format_kg_data_for_llm(self, kg_response: KGResponse) -> str:
        if not kg_response.raw_triples:
            return "No specific knowledge graph facts found for this step."
        triples_str = "\n".join(f"- {t}" for t in kg_response.raw_triples)
        return f"Retrieved knowledge graph facts (triples S-P-O):\n{triples_str}"

    def _hybrid_pruning_strategy(self, current_paths: List[List[str]], current_reasoning_state: str) -> List[List[str]]:
        # Placeholder for a more sophisticated pruning strategy.
        # In a real system, this might use a lightweight model or heuristics
        # to score paths based on relevance to the current reasoning goal.
        print("Applying hybrid pruning strategy (mock). Keeping all paths for now.")
        return current_paths

    def reason_iteratively(self, initial_query: str, max_steps: int = 5, beam_width: int = 2) -> Tuple[str, List[str]]:
        reasoning_steps = []
        current_state = {"diagnosis_candidates": [], "evidence": [], "history": [initial_query]}
        diagnosis = ""

        # Step 1: Topic Entity Extraction
        entities = self.extract_topic_entities(initial_query)
        reasoning_steps.append(f"Extracted key entities: {', '.join(entities)}")

        # Assume initial query indicates symptoms and possibly family history
        # A more robust system would infer this from the LLM's semantic parsing
        family_history = []
        if "family history of autoimmune disorders" in initial_query.lower():
            family_history = ["autoimmune disorders"]

        # Step 2: Semantic Parsing for initial KG query
        kg_query = self.semantic_parse_kgqa(
            initial_query, 
            entities=entities,
            filters={"family_history_of": family_history}
        )
        reasoning_steps.append(f"Semantic parsed KG query: {kg_query.dict()}")

        # Step 3: Initial KG Query Execution
        kg_response = self.kg_module.execute_query(kg_query)
        current_state["diagnosis_candidates"] = [r["entity_id"] for r in kg_response.results if r["type"] == "Disease"]
        current_state["evidence"].extend(kg_response.raw_triples)
        reasoning_steps.append(f"Initial KG results: {self._format_kg_data_for_llm(kg_response)}")
        reasoning_steps.append(f"Initial diagnosis candidates: {', '.join(current_state['diagnosis_candidates'])}")

        for step in range(max_steps):
            if not current_state["diagnosis_candidates"]:
                reasoning_steps.append("No strong diagnosis candidates remain. Ending reasoning.")
                break

            # Construct iterative prompt for LLM agent to propose next step
            current_kg_facts = self._format_kg_data_for_llm(KGResponse(raw_triples=current_state["evidence"]))
            reasoning_prompt = (
                f"You are an AI medical diagnostic assistant. Your goal is to refine the diagnosis for a patient. "
                f"Current patient query: {initial_query}\n"
                f"Current diagnosis candidates: {', '.join(current_state['diagnosis_candidates'])}\n"
                f"Past reasoning steps: {'; '.join(reasoning_steps)}\n"
                f"{current_kg_facts}\n"
                f"Based on this information, propose the single most useful next reasoning step to narrow down the diagnosis or find more evidence. "
                f"Examples: 'Explore diagnostic criteria for X', 'Find genetic links for Y', 'Compare symptoms of A and B'."
            )
            
            next_step_proposal = self._call_llm(reasoning_prompt)
            reasoning_steps.append(f"LLM proposes: {next_step_proposal}")
            print(f"[Step {step+1}] LLM proposed: {next_step_proposal}")

            # Simulate executing the proposed step against KG (simplified)
            # In a real system, `semantic_parse_kgqa` would be called again
            # to turn `next_step_proposal` into a new `KGQuery`.
            new_kg_response = KGResponse()
            if "diagnostic criteria for Lupus" in next_step_proposal.lower():
                new_kg_response = self.kg_module.execute_query(KGQuery(query_type="GET_DIAGNOSTIC_CRITERIA", entities=["Lupus"]))
            elif "diagnostic criteria for rheumatoid arthritis" in next_step_proposal.lower():
                 new_kg_response = self.kg_module.execute_query(KGQuery(query_type="GET_DIAGNOSTIC_CRITERIA", entities=["Rheumatoid Arthritis"]))
            
            if new_kg_response.raw_triples:
                current_state["evidence"].extend(new_kg_response.raw_triples)
                reasoning_steps.append(f"KG execution for '{next_step_proposal}': {self._format_kg_data_for_llm(new_kg_response)}")
            else:
                reasoning_steps.append(f"KG execution for '{next_step_proposal}': No new relevant facts found.")

            # Simulate a beam search like selection (simplified: just update candidates based on new info)
            if "Lupus" in next_step_proposal and "Rheumatoid Arthritis" in next_step_proposal: # Simulating a comparison leading to refinement
                current_state["diagnosis_candidates"] = ["Lupus"] # Assume LLM's reasoning implies Lupus is stronger

            if len(current_state["diagnosis_candidates"]) == 1:
                diagnosis = current_state["diagnosis_candidates"][0]
                reasoning_steps.append(f"Diagnosis converged to: {diagnosis}")
                break

        if not diagnosis and current_state["diagnosis_candidates"]:
            diagnosis = current_state["diagnosis_candidates"][0] # Take the top remaining candidate

        final_diagnosis_summary = diagnosis if diagnosis else "Undetermined"
        return final_diagnosis_summary, reasoning_steps

    def generate_response(self, final_diagnosis: str, reasoning_steps: List[str]) -> str:
        explanation = "\n".join([f"- {step}" for step in reasoning_steps])
        prompt = (
            f"Synthesize a medical diagnosis report for a doctor based on the following information:\n"
            f"Final Diagnosis: {final_diagnosis}\n"
            f"Reasoning Steps:\n{explanation}\n"
            f"Provide a concise diagnosis and a clear explanation, recommending further tests if appropriate."
        )
        return self._call_llm(prompt)

# --- 4. Intelligent Diagnostic Assistant Orchestrator --- 

class IntelligentDiagnosticAssistant:
    def __init__(self, kg_data: Dict[str, Any]):
        self.kg_module = MedicalKnowledgeGraph()
        self._load_kg_data(kg_data)
        self.llm = MockLLM() # Using MockLLM for demonstration
        self.llm_agent = LLMAgent(self.kg_module, self.llm)

    def _load_kg_data(self, kg_data: Dict[str, Any]):
        print("Loading KG data...")
        for entity_data in kg_data.get("entities", []):
            entity = Entity(**entity_data)
            self.kg_module.add_entity(entity)
        for relation_data in kg_data.get("relations", []):
            relation = Relationship(**relation_data)
            self.kg_module.add_relation(relation)
        print(f"KG loaded with {len(self.kg_module.graph.nodes)} entities and {len(self.kg_module.graph.edges)} relations.")

    def diagnose(self, patient_query: str) -> str:
        print(f"\n--- Starting Diagnosis for: '{patient_query}' ---")
        final_diagnosis, reasoning_steps = self.llm_agent.reason_iteratively(patient_query)
        report = self.llm_agent.generate_response(final_diagnosis, reasoning_steps)
        print("--- Diagnosis Complete ---")
        return report

# --- Example Usage --- 
if __name__ == "__main__":
    # Sample Medical KG Data
    sample_kg_data = {
        "entities": [
            {"id": "chronic fatigue", "type": "Symptom"},
            {"id": "muscle weakness", "type": "Symptom"},
            {"id": "joint pain", "type": "Symptom"},
            {"id": "rashes", "type": "Symptom"},
            {"id": "oral ulcers", "type": "Symptom"},
            {"id": "Lupus", "type": "Disease", "attributes": {"specialty": "Rheumatology", "prevalence": "rare"}},
            {"id": "Rheumatoid Arthritis", "type": "Disease", "attributes": {"specialty": "Rheumatology", "prevalence": "common"}},
            {"id": "Fibromyalgia", "type": "Disease", "attributes": {"specialty": "Rheumatology", "prevalence": "common"}},
            {"id": "autoimmune disorders", "type": "Condition"},
            {"id": "ANA test", "type": "Lab Marker"},
            {"id": "anti-dsDNA", "type": "Lab Marker"},
            {"id": "RF factor", "type": "Lab Marker"},
            {"id": "CRP", "type": "Lab Marker"},
            {"id": "genetic predisposition", "type": "Genetic Factor"}
        ],
        "relations": [
            {"source": "Lupus", "target": "chronic fatigue", "type": "has_symptom"},
            {"source": "Lupus", "target": "muscle weakness", "type": "has_symptom"},
            {"source": "Lupus", "target": "joint pain", "type": "has_symptom"},
            {"source": "Lupus", "target": "rashes", "type": "has_symptom"},
            {"source": "Lupus", "target": "oral ulcers", "type": "has_symptom"},
            {"source": "Lupus", "target": "ANA test", "type": "requires_lab_test"},
            {"source": "Lupus", "target": "anti-dsDNA", "type": "requires_lab_test"},
            {"source": "Lupus", "target": "autoimmune disorders", "type": "is_type_of"},

            {"source": "Rheumatoid Arthritis", "target": "joint pain", "type": "has_symptom", "attributes": {"characteristic": "symmetric"}},
            {"source": "Rheumatoid Arthritis", "target": "muscle weakness", "type": "has_symptom"},
            {"source": "Rheumatoid Arthritis", "target": "chronic fatigue", "type": "has_symptom"},
            {"source": "Rheumatoid Arthritis", "target": "RF factor", "type": "requires_lab_test"},
            {"source": "Rheumatoid Arthritis", "target": "CRP", "type": "requires_lab_test"},
            {"source": "Rheumatoid Arthritis", "target": "autoimmune disorders", "type": "is_type_of"},

            {"source": "Fibromyalgia", "target": "chronic fatigue", "type": "has_symptom"},
            {"source": "Fibromyalgia", "target": "muscle weakness", "type": "has_symptom"},
            {"source": "Fibromyalgia", "target": "joint pain", "type": "has_symptom", "attributes": {"characteristic": "widespread"}},

            {"source": "genetic predisposition", "target": "Lupus", "type": "predisposes_to"},
            {"source": "genetic predisposition", "target": "Rheumatoid Arthritis", "type": "predisposes_to"}
        ]
    }

    assistant = IntelligentDiagnosticAssistant(sample_kg_data)

    # Example interaction flow from the prompt
    patient_query_1 = "Patient presents with chronic fatigue, muscle weakness, and joint pain. Family history of autoimmune disorders."
    diagnosis_report_1 = assistant.diagnose(patient_query_1)
    print("\n--- Generated Diagnosis Report 1 ---")
    print(diagnosis_report_1)

    # Another example (simpler)
    patient_query_2 = "Patient has persistent cough and fever. No other significant history."
    diagnosis_report_2 = assistant.diagnose(patient_query_2)
    print("\n--- Generated Diagnosis Report 2 ---")
    print(diagnosis_report_2)

