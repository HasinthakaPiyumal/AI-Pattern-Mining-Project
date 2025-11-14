"""Python code for the Precision Medicine Recommender. """

import networkx as nx
from typing import List, Dict, Any, Tuple
import random

# --- 1. Knowledge Graph (KG) Module --- 

class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_entity(self, entity_id: str, entity_type: str, properties: Dict = None):
        if not self.graph.has_node(entity_id):
            self.graph.add_node(entity_id, type=entity_type, **(properties if properties else {}))
            # print(f"Added entity: {entity_type} {entity_id}")

    def add_relationship(self, source_id: str, target_id: str, rel_type: str, properties: Dict = None):
        if self.graph.has_node(source_id) and self.graph.has_node(target_id):
            self.graph.add_edge(source_id, target_id, type=rel_type, **(properties if properties else {}))
            # print(f"Added relationship: {source_id} -[{rel_type}]-> {target_id}")
        else:
            # print(f"Warning: Could not add relationship. Source ({source_id}) or target ({target_id}) not found.")
            pass # Suppress warnings for cleaner output in this demo

    def get_neighbors(self, entity_id: str) -> List[Tuple[str, str, str]]:
        """Returns a list of (neighbor_id, relationship_type, neighbor_type) for an entity."""
        neighbors = []
        if self.graph.has_node(entity_id):
            for neighbor in self.graph.neighbors(entity_id):
                rel_data = self.graph.get_edge_data(entity_id, neighbor)
                rel_type = rel_data.get("type", "UNKNOWN") if rel_data else "UNKNOWN"
                neighbor_type = self.graph.nodes[neighbor].get("type", "UNKNOWN")
                neighbors.append((neighbor, rel_type, neighbor_type))
        return neighbors

    def find_shortest_path(self, start_entity: str, end_entity: str) -> List[str]:
        """Finds a shortest path between two entities."""
        try:
            return nx.shortest_path(self.graph, source=start_entity, target=end_entity)
        except nx.NetworkXNoPath:
            return []
        except nx.NodeNotFound:
            return []

    def get_path_triples(self, path: List[str]) -> List[Tuple[str, str, str]]:
        """Converts a path (list of node IDs) into a list of (subject, predicate, object) triples."""
        triples = []
        if len(path) < 2: return triples

        for i in range(len(path) - 1):
            source_node_id = path[i]
            target_node_id = path[i+1]
            edge_data = self.graph.get_edge_data(source_node_id, target_node_id)
            if edge_data:
                predicate = edge_data.get("type", "has_relation_to")
                triples.append((source_node_id, predicate, target_node_id))
        return triples


# --- 2. LLM Integration Module --- 

class LLMClient:
    def __init__(self, model_name: str = "Mock_LLM"):
        self.model_name = model_name
        # In a real application, initialize connection to OpenAI, Hugging Face, or Google Vertex AI here.
        # self.llm = OpenAI(api_key="...") # Example

    def query(self, prompt: str, temperature: float = 0.7) -> str:
        """
        Mocks an LLM query. In a real scenario, this would call a deployed LLM.
        Returns a canned response or a simple manipulation of the prompt.
        """
        # print(f"\n--- LLM Query ({self.model_name}) ---")
        # print(f"Prompt: {prompt[:200]}...") # Print first 200 chars of prompt

        if "extract medical entities" in prompt.lower():
            return "Extracted entities: {'Disease': ['Breast Cancer'], 'Gene': ['BRCA1'], 'Drug': ['Tamoxifen']}"
        elif "score the relevance" in prompt.lower():
            # Simulate LLM scoring based on some keywords or random for demo
            if "BRCA1" in prompt and "Breast Cancer" in prompt: return str(random.uniform(0.8, 0.95))
            if "Tamoxifen" in prompt and "Hormone Receptor Positive" in prompt: return str(random.uniform(0.7, 0.9))
            return str(random.uniform(0.1, 0.7))
        elif "explain the reasoning" in prompt.lower():
            return f"Based on the provided facts from the Knowledge Graph: [KG_FACTS_PLACEHOLDER], the reasoning is as follows..."
        elif "generate a comprehensive recommendation" in prompt.lower():
            return f"Based on the analyzed patient data and KG insights, the recommendation is: [RECOMMENDATION_PLACEHOLDER]"
        else:
            return f"Mock LLM response to: {prompt[:100]}... (Temperature: {temperature})"

    @staticmethod
    def _format_path_to_triples(path_triples: List[Tuple[str, str, str]]) -> str:
        """Converts a list of (subject, predicate, object) triples into a natural language string for a prompt."""
        formatted_triples = []
        for s, p, o in path_triples:
            formatted_triples.append(f"({s}) --[{p}]--> ({o})")
        return " ".join(formatted_triples)

    @staticmethod
    def _generate_iterative_prompt(context: str, user_query: str, previous_steps: List[str] = None) -> str:
        """Dynamically constructs a prompt based on context and previous interactions."""
        prompt_parts = [context]
        if previous_steps:
            prompt_parts.append("Previous steps/refinements:")
            prompt_parts.extend([f"- {step}" for step in previous_steps])
        prompt_parts.append(f"Current query: {user_query}")
        return "\n".join(prompt_parts)


# --- 3. Core Reasoning Engine --- 

class PrecisionMedicineRecommender:
    def __init__(self, kg: KnowledgeGraph, llm_client: LLMClient):
        self.kg = kg
        self.llm_client = llm_client
        self.conversation_history = []

    def _log_interaction(self, role: str, content: str):
        self.conversation_history.append({"role": role, "content": content})

    def extract_entities_with_llm(self, patient_case_description: str) -> Dict[str, List[str]]:
        """Uses LLM to extract key medical entities from patient data."""
        prompt = f"Given the patient case description, extract medical entities such as diseases, genes, and drugs. " \
                 f"Return in a dictionary format like: {{'Disease': [], 'Gene': [], 'Drug': []}}.\n\nDescription: {patient_case_description}"
        llm_response = self.llm_client.query(prompt, temperature=0.1)
        self._log_interaction("user", prompt)
        self._log_interaction("llm", llm_response)
        try:
            # In a real app, parse LLM's structured output more robustly
            entities_str = llm_response.split("Extracted entities: ")[-1]
            return eval(entities_str) # Using eval for demo, dangerous in production without validation
        except Exception as e:
            print(f"Error parsing LLM entity extraction: {e}")
            return {}

    def _llm_score_path_expansion(self, current_path_triples: List[Tuple[str, str, str]], candidate_next_step: Tuple[str, str, str]) -> float:
        """Uses LLM to score the relevance of extending a path with a candidate next step."""
        current_path_str = self.llm_client._format_path_to_triples(current_path_triples)
        candidate_step_str = self.llm_client._format_path_to_triples([candidate_next_step])
        
        prompt = f"A patient has certain conditions. We are exploring a knowledge graph. " \
                 f"Current path: {current_path_str}. Candidate next step: {candidate_step_str}. " \
                 f"On a scale of 0 to 1, how relevant is this next step to finding a precision medicine recommendation for a patient? " \
                 f"Provide only the score as a float."
        
        llm_response = self.llm_client.query(prompt, temperature=0.3)
        self._log_interaction("llm_scorer_prompt", prompt)
        self._log_interaction("llm_scorer_response", llm_response)

        try:
            return float(llm_response) # Expecting a float directly from mock LLM
        except ValueError:
            return random.uniform(0.1, 0.5) # Fallback for demo


    def llm_guided_beam_search(self, start_entity_ids: List[str], beam_width: int = 3, max_depth: int = 4) -> List[List[str]]:
        """
        Performs a beam search on the KG, guided by the LLM.
        Returns a list of the top `beam_width` paths found.
        """
        if not start_entity_ids:
            return []

        # Each item in beam: (score, path_nodes_list, path_triples_list)
        initial_beam = [(1.0, [entity_id], []) for entity_id in start_entity_ids if self.kg.graph.has_node(entity_id)]
        current_beam = initial_beam
        all_found_paths = []

        for depth in range(max_depth):
            if not current_beam: break
            next_beam = []
            for score, path_nodes, path_triples in current_beam:
                current_node = path_nodes[-1]
                
                # Consider all neighbors as potential expansions
                for neighbor_id, rel_type, _ in self.kg.get_neighbors(current_node):
                    if neighbor_id not in path_nodes: # Avoid cycles for simplicity in demo
                        new_path_nodes = path_nodes + [neighbor_id]
                        new_path_triples = path_triples + [(current_node, rel_type, neighbor_id)]
                        
                        # LLM guidance: Score the relevance of this path expansion
                        llm_relevance_score = self._llm_score_path_expansion(new_path_triples, (current_node, rel_type, neighbor_id))
                        new_score = score * llm_relevance_score # Combine scores (simple product)
                        next_beam.append((new_score, new_path_nodes, new_path_triples))
            
            # Sort and prune the beam
            next_beam = sorted(next_beam, key=lambda x: x[0], reverse=True)[:beam_width]
            current_beam = next_beam
            all_found_paths.extend([path_nodes for score, path_nodes, path_triples in current_beam])
            
        # Remove duplicates and return unique paths
        unique_paths = []
        for path in all_found_paths:
            if path not in unique_paths:
                unique_paths.append(path)
        return unique_paths

    def _retrieve_kg_facts(self, paths: List[List[str]]) -> List[Dict[str, Any]]:
        """Retrieves structured facts from the KG based on identified paths."""
        facts = []
        for path in paths:
            path_triples = self.kg.get_path_triples(path)
            for s, p, o in path_triples:
                fact = {
                    "subject_id": s, "subject_type": self.kg.graph.nodes[s].get("type"),
                    "predicate": p,
                    "object_id": o, "object_type": self.kg.graph.nodes[o].get("type")
                }
                facts.append(fact)
        return facts

    def rag_generate_response(self, query: str, kg_facts: List[Dict[str, Any]]) -> str:
        """
        Retrieval-Augmented Generation: Grounds LLM response with relevant KG facts.
        """
        formatted_facts = []
        for fact in kg_facts:
            formatted_facts.append(f"({fact['subject_id']}:{fact['subject_type']}) -[{fact['predicate']}]-> ({fact['object_id']}:{fact['object_type']})")
        
        facts_str = "\n- ".join(formatted_facts)
        if facts_str: facts_str = "Relevant facts from Knowledge Graph:\n- " + facts_str

        prompt = f"Given the following patient query and relevant facts from a medical knowledge graph, " \
                 f"generate a comprehensive and factual response. Ensure your response is grounded in the provided facts " \
                 f"and avoid hallucination.\n\nPatient Query: {query}\n\n{facts_str}\n\nRecommendation:"
        
        llm_response = self.llm_client.query(prompt, temperature=0.7)
        self._log_interaction("user", prompt)
        self._log_interaction("llm", llm_response)
        return llm_response.replace("[RECOMMENDATION_PLACEHOLDER]", f"Based on KG facts {facts_str}. A precision medicine recommendation is...")

    def kd_cot_reasoning(self, initial_query: str, kg_facts: List[Dict[str, Any]], max_steps: int = 3) -> str:
        """
        Knowledge-Driven Chain-of-Thought: Guides LLM to explain reasoning step-by-step
        by embedding KG retrieval.
        """
        reasoning_steps = []
        current_context = f"Initial query: {initial_query}\nRelevant KG facts: {self._format_kg_facts_for_cot(kg_facts)}"

        for step in range(max_steps):
            prompt = f"You are a medical reasoning assistant. Based on the following context, " \
                     f"provide the next logical step in determining a precision medicine recommendation. " \
                     f"Focus on interpreting the KG facts and connect them to the initial query. " \
                     f"\n\nContext:\n{current_context}\n\nStep {step + 1}:"
            
            llm_step_response = self.llm_client.query(prompt, temperature=0.5)
            self._log_interaction("llm_cot_prompt", prompt)
            self._log_interaction("llm_cot_response", llm_step_response)
            reasoning_steps.append(llm_step_response)
            current_context += f"\n\nStep {step + 1}: {llm_step_response}"
            
            # Simulate further KG retrieval based on LLM's intermediate step (simplified)
            if "investigate" in llm_step_response.lower() or "find more" in llm_step_response.lower():
                simulated_new_facts = [{
                    "subject_id": "NewResearch", "subject_type": "ResearchPaper",
                    "predicate": "suggests_treatment_for",
                    "object_id": "ResistantTumor", "object_type": "Disease"
                }]
                current_context += f"\n(Simulated new KG facts retrieved based on Step {step + 1}: {self._format_kg_facts_for_cot(simulated_new_facts)})"

        final_reasoning = "\n".join(reasoning_steps)
        return f"**Reasoning Chain:**\n{final_reasoning}\n\n**Final Summary of Reasoning:** Based on the step-by-step analysis, a comprehensive understanding of the patient's case and relevant KG knowledge has been developed, leading to a precision medicine recommendation."

    def _format_kg_facts_for_cot(self, kg_facts: List[Dict[str, Any]]) -> str:
        formatted_facts = []
        for fact in kg_facts:
            formatted_facts.append(f"({fact['subject_id']}:{fact['subject_type']}) --[{fact['predicate']}]--> ({fact['object_id']}:{fact['object_type']})")
        return " ; ".join(formatted_facts)


    def hybrid_pruning_strategy(self, candidate_paths: List[List[str]]) -> List[List[str]]:
        """
        Placeholder for a hybrid pruning strategy.
        In a real scenario, this would combine LLM scores with a lightweight model's heuristics
        to filter and rank candidate paths.
        """
        # For demonstration, randomly prune some paths or keep all if less than 5.
        if len(candidate_paths) > 5:
            return random.sample(candidate_paths, 5)
        return candidate_paths

    def semantic_parse_query(self, natural_language_query: str) -> Dict[str, Any]:
        """
        Placeholder for Semantic Parsing for KGQA.
        Converts a natural language query into a structured KG query (e.g., Cypher, Gremlin).
        """
        # In a real system, use an LLM fine-tuned for semantic parsing or a dedicated parser.
        if "treatment for BRCA1" in natural_language_query.lower():
            return {"query_type": "find_treatment", "entity": "BRCA1", "entity_type": "Gene"}
        elif "trials for breast cancer" in natural_language_query.lower():
            return {"query_type": "find_clinical_trials", "disease": "Breast Cancer"}
        return {"query_type": "unsupported", "raw_query": natural_language_query}

    def recommend_treatment(self, patient_case_description: str, refinement_queries: List[str] = None) -> Dict[str, Any]:
        """
        Orchestrates the entire precision medicine recommendation process.
        """
        print("\n--- Starting Precision Medicine Recommendation ---")
        self._log_interaction("system", "Starting recommendation process.")

        # 1. LLM-based Topic Entity Extraction
        print("\n1. Extracting entities from patient case...")
        extracted_entities = self.extract_entities_with_llm(patient_case_description)
        print(f"   Extracted entities: {extracted_entities}")
        self._log_interaction("system", f"Extracted entities: {extracted_entities}")

        start_entities = []
        for entity_list in extracted_entities.values():
            start_entities.extend(entity_list)
        
        if not start_entities:
            return {"error": "No relevant entities extracted. Cannot proceed with KG exploration.", "recommendation": None}

        # Ensure extracted entities exist in the KG for demonstration
        for entity in start_entities:
            if not self.kg.graph.has_node(entity):
                # Add a mock node if it doesn't exist to allow path finding for demo
                self.kg.add_entity(entity, "Concept", {"source": "LLM_Extraction"})
                print(f"   (Demo) Added missing entity \'{entity}\' to KG for exploration.")

        # 2. LLM-Guided Beam Search for KG Exploration
        print("\n2. Exploring Knowledge Graph with LLM-Guided Beam Search...")
        relevant_paths = self.llm_guided_beam_search(start_entities, beam_width=5, max_depth=3)
        print(f"   Found {len(relevant_paths)} relevant paths. Example path: {relevant_paths[0] if relevant_paths else 'N/A'}")
        self._log_interaction("system", f"Found {len(relevant_paths)} KG paths.")

        # 3. Hybrid Pruning Strategy (if many paths are found)
        if relevant_paths:
            print("\n3. Applying Hybrid Pruning Strategy...")
            pruned_paths = self.hybrid_pruning_strategy(relevant_paths)
            print(f"   Pruned paths (demo): {len(pruned_paths)}")
            self._log_interaction("system", f"Pruned paths to: {len(pruned_paths)}.")
            relevant_paths = pruned_paths

        # 4. Retrieve KG Facts for RAG and KDCoT
        print("\n4. Retrieving relevant KG facts...")
        kg_facts = self._retrieve_kg_facts(relevant_paths)
        print(f"   Retrieved {len(kg_facts)} facts. Example: {kg_facts[0] if kg_facts else 'N/A'}")
        self._log_interaction("system", f"Retrieved {len(kg_facts)} KG facts.")

        # 5. Knowledge-Driven Chain-of-Thought (KDCoT)
        print("\n5. Generating Knowledge-Driven Chain-of-Thought reasoning...")
        kd_cot_explanation = self.kd_cot_reasoning(patient_case_description, kg_facts)
        print("   KDCoT Explanation generated.")
        self._log_interaction("system", "KDCoT explanation generated.")

        # 6. Retrieval-Augmented Generation (RAG) for KGs
        print("\n6. Generating final recommendation with RAG...")
        final_recommendation_text = self.rag_generate_response(patient_case_description, kg_facts)
        print("   Final recommendation generated.")
        self._log_interaction("system", "Final recommendation generated.")

        # 7. Iterative Prompting for refinement (simulated)
        if refinement_queries:
            print("\n7. Processing refinement queries via Iterative Prompting...")
            for query_idx, refinement_query in enumerate(refinement_queries):
                context = f"Current recommendation: {final_recommendation_text}\nKDCoT Reasoning: {kd_cot_explanation}"
                iterative_prompt = self.llm_client._generate_iterative_prompt(
                    context=context,
                    user_query=refinement_query,
                    previous_steps=[h["content"] for h in self.conversation_history if h["role"] == "user"]
                )
                print(f"   Refinement Query {query_idx+1}: '{refinement_query}'")
                refined_response = self.llm_client.query(iterative_prompt, temperature=0.6)
                print(f"   Refined Response: {refined_response[:100]}...")
                final_recommendation_text = refined_response # Update with refined response for demo
                self._log_interaction("user", iterative_prompt)
                self._log_interaction("llm", refined_response)

        print("\n--- Precision Medicine Recommendation Complete ---")
        return {
            "recommendation": final_recommendation_text,
            "reasoning_explanation": kd_cot_explanation,
            "extracted_entities": extracted_entities,
            "kg_paths_found": relevant_paths,
            "kg_facts_used": kg_facts,
            "conversation_history": self.conversation_history
        }

# --- Demo Usage --- 
if __name__ == "__main__":
    # 1. Initialize Knowledge Graph and populate with mock data
    kg = KnowledgeGraph()
    
    # Diseases
    kg.add_entity("Breast Cancer", "Disease", {"description": "Malignancy originating from breast tissue"})
    kg.add_entity("Lung Cancer", "Disease")
    kg.add_entity("Ovarian Cancer", "Disease")
    kg.add_entity("Hormone Receptor Positive Breast Cancer", "Disease")

    # Genes
    kg.add_entity("BRCA1", "Gene", {"function": "Tumor suppressor"})
    kg.add_entity("HER2", "Gene", {"function": "Growth factor receptor"})
    kg.add_entity("TP53", "Gene")

    # Drugs
    kg.add_entity("Tamoxifen", "Drug", {"class": "Selective estrogen receptor modulator (SERM)"})
    kg.add_entity("Herceptin", "Drug", {"class": "Monoclonal antibody"})
    kg.add_entity("Chemotherapy", "TreatmentClass")
    kg.add_entity("Radiation Therapy", "TreatmentClass")
    kg.add_entity(" immunotherapy", "TreatmentClass")

    # Clinical Trials (mock IDs)
    kg.add_entity("CT_BRCA1_001", "ClinicalTrial", {"phase": 3, "status": "Recruiting"})
    kg.add_entity("CT_HER2_002", "ClinicalTrial", {"phase": 2, "status": "Active, not recruiting"})

    # Relationships
    kg.add_relationship("BRCA1", "Breast Cancer", "ASSOCIATED_WITH")
    kg.add_relationship("BRCA1", "Ovarian Cancer", "ASSOCIATED_WITH")
    kg.add_relationship("HER2", "Breast Cancer", "EXPRESSED_IN")
    kg.add_relationship("Breast Cancer", "Tamoxifen", "TREATED_BY_DRUG", {"condition": "Hormone Receptor Positive"})
    kg.add_relationship("Hormone Receptor Positive Breast Cancer", "Tamoxifen", "RESPONDS_TO")
    kg.add_relationship("HER2", "Herceptin", "TARGETED_BY_DRUG")
    kg.add_relationship("Herceptin", "CT_HER2_002", "BEING_TESTED_IN")
    kg.add_relationship("BRCA1", "CT_BRCA1_001", "ELIGIBLE_FOR_TRIAL")
    kg.add_relationship("Breast Cancer", "Chemotherapy", "TREATED_BY_CLASS")
    kg.add_relationship("Ovarian Cancer", "Chemotherapy", "TREATED_BY_CLASS")

    # 2. Initialize LLM Client (mocked)
    llm_client = LLMClient()

    # 3. Initialize Recommender System
    recommender = PrecisionMedicineRecommender(kg, llm_client)

    # 4. Define a patient case
    patient_case = "Patient presents with newly diagnosed ER/PR-positive, HER2-negative breast cancer. Genetic testing reveals a BRCA1 germline mutation. Patient is 55 years old with no significant comorbidities."

    # 5. Get recommendations
    recommendation_output = recommender.recommend_treatment(patient_case)

    print("\n----- Final Recommendation Output -----")
    print(f"Recommendation: {recommendation_output['recommendation']}")
    print(f"\nReasoning Explanation: {recommendation_output['reasoning_explanation']}")
    print(f"\nExtracted Entities: {recommendation_output['extracted_entities']}")
    print(f"\nKG Paths Found (first 3): {recommendation_output['kg_paths_found'][:3]}")
    print(f"\nKG Facts Used (first 3): {recommender._format_kg_facts_for_cot(recommendation_output['kg_facts_used'][:3])}")

    # 6. Demonstrate Iterative Prompting / Refinement
    print("\n--- Demonstrating Refinement ---")
    refinement_queries = [
        "Are there any new trials for BRCA1-positive breast cancer with better outcomes?",
        "What are the potential side effects of Tamoxifen based on recent studies?"
    ]
    refined_recommendation_output = recommender.recommend_treatment(patient_case, refinement_queries)
    print("\n----- Refined Recommendation Output -----")
    print(f"Recommendation: {refined_recommendation_output['recommendation']}")
    print(f"\nReasoning Explanation: {refined_recommendation_output['reasoning_explanation']}")
    print(f"\nConversation History Length: {len(refined_recommendation_output['conversation_history'])}")
