import networkx as nx
import spacy

# --- 1. Medical Knowledge Graph (KG) Module ---
class MedicalKnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_entity(self, entity_id, entity_type, attributes=None):
        if not self.graph.has_node(entity_id):
            self.graph.add_node(entity_id, type=entity_type, **(attributes if attributes else {}))

    def add_relationship(self, source_id, target_id, rel_type, attributes=None):
        if self.graph.has_node(source_id) and self.graph.has_node(target_id):
            self.graph.add_edge(source_id, target_id, type=rel_type, **(attributes if attributes else {}))
        else:
            print(f"Warning: Could not add relationship. One or both entities ({source_id}, {target_id}) do not exist.")

    def query_facts(self, entity_id=None, rel_type=None, target_entity_type=None):
        results = []
        if entity_id and self.graph.has_node(entity_id):
            for neighbor in self.graph.neighbors(entity_id):
                edge_data = self.graph.get_edge_data(entity_id, neighbor)
                if rel_type is None or edge_data.get("type") == rel_type:
                    if target_entity_type is None or self.graph.nodes[neighbor].get("type") == target_entity_type:
                        results.append((entity_id, edge_data.get("type"), neighbor))
            for source, _, edge_data in self.graph.in_edges(entity_id, data=True):
                if rel_type is None or edge_data.get("type") == rel_type:
                    if target_entity_type is None or self.graph.nodes[source].get("type") == target_entity_type:
                        results.append((source, edge_data.get("type"), entity_id))
        elif rel_type:
            for u, v, data in self.graph.edges(data=True):
                if data.get("type") == rel_type:
                    results.append((u, data.get("type"), v))
        return results

    def explore_paths(self, start_entity, end_entity=None, max_depth=2):
        paths = []
        if not self.graph.has_node(start_entity):
            return paths

        for path in nx.bfs_edges(self.graph, start_entity, depth_limit=max_depth):
            # This is a simplified path exploration. For a real beam search, LLM guidance would be crucial.
            # We'll just return direct edges for this demo.
            u, v = path
            edge_data = self.graph.get_edge_data(u, v)
            paths.append((u, edge_data.get("type"), v))
            if v == end_entity:
                break
        return paths

# --- Mock LLM for demonstration ---
def _mock_llm_response(prompt, tool_output=None):
    # This function simulates an LLM call. In a real system, this would be an API call.
    # It's highly simplified to demonstrate the flow.
    if "What diseases cause" in prompt and "fever" in prompt and "cough" in prompt:
        return {
            "thought": "The user is asking for diseases causing specific symptoms. I should query the KG for diseases associated with fever and cough.",
            "tool_call": {"name": "query_kg", "args": {"symptom1": "fever", "symptom2": "cough"}},
            "final_answer": None
        }
    elif "query_kg" in prompt and "hypertension" in prompt:
        return {
            "thought": "The user is asking about hypertension. I need to retrieve facts related to hypertension from the KG.",
            "tool_call": {"name": "query_kg_facts", "args": {"entity_id": "Hypertension"}},
            "final_answer": None
        }
    elif "explore_paths" in prompt and "hypertension" in prompt:
        return {
            "thought": "Exploring paths related to hypertension to find associated conditions or treatments.",
            "tool_call": {"name": "explore_kg_paths", "args": {"start_entity": "Hypertension"}},
            "final_answer": None
        }
    elif "interpret this information" in prompt and tool_output:
        if "Disease A - causes - Fever" in tool_output and "Disease A - causes - Cough" in tool_output:
            return {
                "thought": "Based on the KG, Disease A causes both fever and cough. I should suggest Disease A as a potential diagnosis.",
                "final_answer": "Potential Diagnosis: Disease A. Explanation: The knowledge graph indicates that Disease A is associated with both fever and cough." + f" KG Evidence: {tool_output}"
            }
        elif "Hypertension - treated_by - Drug X" in tool_output:
             return {
                "thought": "The KG shows that Drug X treats Hypertension. I will recommend it.",
                "final_answer": "Recommendation: Consider Drug X for Hypertension. Explanation: The knowledge graph indicates that Drug X is a treatment for Hypertension." + f" KG Evidence: {tool_output}"
            }
    return {"thought": "I need more information or a different approach.", "tool_call": None, "final_answer": None}


# --- 2. Natural Language Understanding (NLU) & Semantic Parsing Module ---
class SemanticParser:
    def __init__(self, kg):
        # Load a small English model for NER
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading spaCy model 'en_core_web_sm'. This may take a moment.")
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        self.kg = kg

    def extract_entities(self, text):
        doc = self.nlp(text)
        entities = [ent.text for ent in doc.ents]
        # Also look for terms that might be in our KG but not caught by default NER
        kg_entities = []
        for node in self.kg.graph.nodes:
            if node.lower() in text.lower():
                kg_entities.append(node)
        return list(set(entities + kg_entities))

    def parse_query_to_kg_format(self, natural_language_query, extracted_entities):
        # This is a highly simplified semantic parser using mock LLM for demonstration
        # In a real system, a more sophisticated LLM prompt and parsing logic would be used.

        # Mock LLM call for semantic parsing logic
        llm_prompt = f"Given the query: '{natural_language_query}' and extracted entities: {extracted_entities}. " \
                     "Convert this into a structured KG query format (e.g., {{'entity': 'Hypertension', 'rel_type': 'treated_by', 'target_type': 'Drug'}}). " \
                     "Focus on identifying the main subject and desired relationship/target."

        mock_llm_output = _mock_llm_response(llm_prompt) # Use generic mock response

        if "What diseases cause fever and cough?" in natural_language_query:
            return {"query_type": "facts_by_symptoms", "symptoms": ["fever", "cough"]}
        elif "treats hypertension" in natural_language_query.lower() or "hypertension treatment" in natural_language_query.lower():
             return {"query_type": "facts", "entity_id": "Hypertension", "rel_type": "treated_by", "target_entity_type": "Drug"}
        elif "what is hypertension" in natural_language_query.lower():
            return {"query_type": "facts", "entity_id": "Hypertension"}
        elif extracted_entities:
            return {"query_type": "facts", "entity_id": extracted_entities[0]}
        return None


# --- 3. LLM-KG Reasoning & Agent Module ---
class KGProxyAgent:
    def __init__(self, llm_mock_function, kg):
        self.llm = llm_mock_function
        self.kg = kg
        self.tools = {
            "query_kg_facts": self._query_kg_facts_tool,
            "explore_kg_paths": self._explore_kg_paths_tool,
        }

    def _query_kg_facts_tool(self, entity_id=None, rel_type=None, target_entity_type=None, symptom1=None, symptom2=None):
        if symptom1 and symptom2:
            # Special handling for symptom-based queries (simplified)
            results_s1 = set(self.kg.query_facts(entity_id=symptom1, rel_type="causes", target_entity_type="Disease"))
            results_s2 = set(self.kg.query_facts(entity_id=symptom2, rel_type="causes", target_entity_type="Disease"))
            # Find common diseases for both symptoms
            common_diseases = results_s1.intersection(results_s2)
            return list(common_diseases)
        else:
            return self.kg.query_facts(entity_id, rel_type, target_entity_type)

    def _explore_kg_paths_tool(self, start_entity, end_entity=None, max_depth=2):
        return self.kg.explore_paths(start_entity, end_entity, max_depth)

    def _format_kg_triples(self, triples):
        return [f"{s} - {p} - {o}" for s, p, o in triples]

    def reason(self, structured_query):
        reasoning_steps = []
        final_answer = None
        current_kg_context = []

        initial_prompt = f"The user has a medical query. Here is the structured query: {structured_query}. " \
                         "Think step-by-step. What KG tools should I use to answer this?"

        # Simulate iterative prompting (ThinkonGraph / KDCoT)
        for _ in range(3): # Max 3 reasoning steps for demo simplicity
            llm_output = self.llm(initial_prompt, tool_output="\n".join(self._format_kg_triples(current_kg_context)))
            thought = llm_output.get("thought")
            tool_call = llm_output.get("tool_call")
            final_answer = llm_output.get("final_answer")

            reasoning_steps.append(f"Thought: {thought}")

            if final_answer:
                break

            if tool_call and tool_call["name"] in self.tools:
                tool_args = tool_call["args"]
                print(f"Executing tool: {tool_call['name']} with args: {tool_args}")
                tool_result = self.tools[tool_call["name"]](**tool_args)
                if tool_result:
                    current_kg_context.extend(tool_result)
                    reasoning_steps.append(f"KG Tool Result: {self._format_kg_triples(tool_result)}")
                    initial_prompt = f"I have executed a KG tool and got these results: {self._format_kg_triples(tool_result)}. " \
                                     f"My current reasoning steps are: {reasoning_steps}. " \
                                     "Given the original query: {structured_query}, how should I proceed or form a final answer? " \
                                     "Interpret this information and provide a medical conclusion or recommendation. Focus on generating a faithful explanation using the provided KG facts."
                else:
                    reasoning_steps.append("KG Tool Result: No relevant information found.")
                    initial_prompt = f"No new information from KG tool. My current reasoning steps are: {reasoning_steps}. " \
                                     "Can I form a final answer or do I need a different approach?"
            else:
                reasoning_steps.append("No suitable KG tool call detected or tool not found.")
                initial_prompt = f"I could not use a KG tool. My current reasoning steps are: {reasoning_steps}. " \
                                 "Can I form a final answer or do I need a different approach?"

        # Final attempt to get an answer if not already found (RAG & Explanation)
        if not final_answer and current_kg_context:
            rag_prompt = f"Given the original structured query: {structured_query}. " \
                         f"And the following relevant facts from the Knowledge Graph: {self._format_kg_triples(current_kg_context)}. " \
                         "Provide a comprehensive medical conclusion or recommendation with a faithful explanation, citing the KG facts."
            final_llm_output = self.llm(rag_prompt, tool_output="\n".join(self._format_kg_triples(current_kg_context)))
            final_answer = final_llm_output.get("final_answer", "Could not determine a definitive answer based on available information.")
            reasoning_steps.append(f"Final RAG-driven generation.")
        elif not final_answer:
            final_answer = "Could not determine a definitive answer based on available information and KG exploration."

        return {"answer": final_answer, "explanation": "\n".join(reasoning_steps)}


# --- Main Execution Flow ---
if __name__ == "__main__":
    print("Initializing ICDSS-XAI System...")

    # 1. Initialize Medical Knowledge Graph
    kg = MedicalKnowledgeGraph()
    kg.add_entity("Hypertension", "Disease")
    kg.add_entity("Fever", "Symptom")
    kg.add_entity("Cough", "Symptom")
    kg.add_entity("Diabetes", "Disease")
    kg.add_entity("Influenza", "Disease")
    kg.add_entity("Pneumonia", "Disease")
    kg.add_entity("Drug X", "Drug")
    kg.add_entity("Drug Y", "Drug")
    kg.add_entity("Headache", "Symptom")

    kg.add_relationship("Fever", "Influenza", "causes")
    kg.add_relationship("Cough", "Influenza", "causes")
    kg.add_relationship("Fever", "Pneumonia", "causes")
    kg.add_relationship("Cough", "Pneumonia", "causes")
    kg.add_relationship("Hypertension", "Drug X", "treated_by")
    kg.add_relationship("Diabetes", "Drug Y", "treated_by")
    kg.add_relationship("Hypertension", "Headache", "associated_with")

    print("Medical Knowledge Graph loaded.")

    # 2. Initialize NLU & Semantic Parsing Module
    semantic_parser = SemanticParser(kg)
    print("NLU and Semantic Parsing Module initialized.")

    # 3. Initialize LLM-KG Reasoning & Agent Module
    agent = KGProxyAgent(_mock_llm_response, kg)
    print("LLM-KG Reasoning Agent initialized.")

    print("\n--- Scenario 1: Differential Diagnosis (Fever and Cough) ---")
    patient_query_1 = "Patient presents with fever and cough. What could be the potential diseases?"
    print(f"Clinician Query: {patient_query_1}")

    extracted_entities_1 = semantic_parser.extract_entities(patient_query_1)
    print(f"Extracted Entities: {extracted_entities_1}")

    structured_query_1 = semantic_parser.parse_query_to_kg_format(patient_query_1, extracted_entities_1)
    print(f"Structured KG Query: {structured_query_1}")

    if structured_query_1:
        result_1 = agent.reason(structured_query_1)
        print("\n--- Result (Scenario 1) ---")
        print(f"Diagnosis/Recommendation: {result_1['answer']}")
        print(f"\nExplanation:\n{result_1['explanation']}")

    print("\n--- Scenario 2: Treatment Recommendation (Hypertension) ---")
    patient_query_2 = "What drugs treat hypertension?"
    print(f"Clinician Query: {patient_query_2}")

    extracted_entities_2 = semantic_parser.extract_entities(patient_query_2)
    print(f"Extracted Entities: {extracted_entities_2}")

    structured_query_2 = semantic_parser.parse_query_to_kg_format(patient_query_2, extracted_entities_2)
    print(f"Structured KG Query: {structured_query_2}")

    if structured_query_2:
        result_2 = agent.reason(structured_query_2)
        print("\n--- Result (Scenario 2) ---")
        print(f"Diagnosis/Recommendation: {result_2['answer']}")
        print(f"\nExplanation:\n{result_2['explanation']}")

    print("\n--- Scenario 3: General Information Query (Hypertension) ---")
    patient_query_3 = "Tell me about hypertension."
    print(f"Clinician Query: {patient_query_3}")

    extracted_entities_3 = semantic_parser.extract_entities(patient_query_3)
    print(f"Extracted Entities: {extracted_entities_3}")

    structured_query_3 = semantic_parser.parse_query_to_kg_format(patient_query_3, extracted_entities_3)
    print(f"Structured KG Query: {structured_query_3}")

    if structured_query_3:
        result_3 = agent.reason(structured_query_3)
        print("\n--- Result (Scenario 3) ---")
        print(f"Information: {result_3['answer']}")
        print(f"\nExplanation:\n{result_3['explanation']}")

    print("\nICDSS-XAI System demonstration complete.")
