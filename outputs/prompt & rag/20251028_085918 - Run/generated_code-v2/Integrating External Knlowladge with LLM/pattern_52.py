import networkx as nx
import json

class MedicalKnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_entity(self, entity_type, entity_id, properties=None):
        if not properties:
            properties = {}
        properties["type"] = entity_type
        self.graph.add_node(entity_id, **properties)

    def add_relationship(self, source_id, target_id, rel_type, properties=None):
        if not properties:
            properties = {}
        properties["type"] = rel_type
        self.graph.add_edge(source_id, target_id, **properties)

    def get_entity_details(self, entity_id):
        if entity_id in self.graph:
            return self.graph.nodes[entity_id]
        return None

    def get_related_entities(self, entity_id, relationship_type=None):
        related = []
        if entity_id in self.graph:
            for neighbor in self.graph.neighbors(entity_id):
                edge_data = self.graph.get_edge_data(entity_id, neighbor)
                if relationship_type is None or edge_data.get("type") == relationship_type:
                    related.append({
                        "entity_id": neighbor,
                        "relationship": edge_data.get("type"),
                        "entity_details": self.graph.nodes[neighbor]
                    })
            for source, _, edge_data in self.graph.in_edges(entity_id, data=True):
                 if relationship_type is None or edge_data.get("type") == relationship_type:
                    related.append({
                        "entity_id": source,
                        "relationship": edge_data.get("type"),
                        "entity_details": self.graph.nodes[source]
                    })
        return related

    def search_entities(self, query, entity_type=None):
        found_entities = []
        query_lower = query.lower()
        for node_id, attributes in self.graph.nodes(data=True):
            matches_type = True
            if entity_type and attributes.get("type") != entity_type:
                matches_type = False

            if matches_type:
                # Check ID, name, or other common properties for the query
                if query_lower in str(node_id).lower() or \
                   any(query_lower in str(v).lower() for k, v in attributes.items() if k != "type"):
                    found_entities.append({"entity_id": node_id, "details": attributes})
        return found_entities

class MockLLMInterface:
    def __init__(self):
        self.step = 0
        self.mock_responses = [
            # Initial thought: search for diseases related to symptoms
            json.dumps({"action": "call_tool", "tool_name": "search_entities", "tool_args": {"query": "fever", "entity_type": "symptom"}}),
            # Found symptom, now search for diseases associated with that symptom
            json.dumps({"action": "call_tool", "tool_name": "get_related_entities", "tool_args": {"entity_id": "fever", "relationship_type": "has_symptom"}}),
            # Get details for one of the potential diseases (e.g., "influenza")
            json.dumps({"action": "call_tool", "tool_name": "get_entity_details", "tool_args": {"entity_id": "influenza"}}),
            # Final diagnosis based on accumulated info
            json.dumps({"action": "final_answer", "diagnosis": "Influenza", "explanation": "Based on the patient presenting with fever and the association of fever with influenza in the knowledge graph, and further details of influenza."})
        ]

    def generate_response(self, prompt):
        if self.step < len(self.mock_responses):
            response = self.mock_responses[self.step]
            self.step += 1
            return response
        return json.dumps({"action": "final_answer", "diagnosis": "Unknown", "explanation": "No further reasoning path."})

class DiagnosticAgent:
    def __init__(self, kg, llm_interface):
        self.kg = kg
        self.llm_interface = llm_interface
        self.reasoning_trace = []
        self.tools = {
            "search_entities": self.kg.search_entities,
            "get_entity_details": self.kg.get_entity_details,
            "get_related_entities": self.kg.get_related_entities,
        }

    def _call_llm(self, prompt_context):
        full_prompt = f"Current patient context: {prompt_context}\n\nWhat is the next step? (Return a JSON object with 'action', 'tool_name', 'tool_args' or 'final_answer')"
        llm_response_str = self.llm_interface.generate_response(full_prompt)
        self.reasoning_trace.append(f"LLM Input: {full_prompt}")
        self.reasoning_trace.append(f"LLM Output: {llm_response_str}")
        try:
            return json.loads(llm_response_str)
        except json.JSONDecodeError:
            return {"action": "error", "message": "Invalid JSON from LLM."}

    def _execute_tool(self, tool_name, **kwargs):
        tool_func = self.tools.get(tool_name)
        if tool_func:
            result = tool_func(**kwargs)
            self.reasoning_trace.append(f"Tool Call: {tool_name} with args {kwargs} -> Result: {result}")
            return result
        return f"Error: Tool {tool_name} not found."

    def diagnose(self, patient_symptoms, patient_history):
        context = f"Patient symptoms: {', '.join(patient_symptoms)}. Patient history: {patient_history}."
        diagnosis = ""
        explanation = ""
        
        while True:
            llm_decision = self._call_llm(context)
            action = llm_decision.get("action")

            if action == "call_tool":
                tool_name = llm_decision.get("tool_name")
                tool_args = llm_decision.get("tool_args", {})
                tool_result = self._execute_tool(tool_name, **tool_args)
                context += f"\n\nObservation from {tool_name}: {tool_result}"
            elif action == "final_answer":
                diagnosis = llm_decision.get("diagnosis", "")
                explanation = llm_decision.get("explanation", "")
                break
            else:
                diagnosis = "Undetermined (Error or no clear action)"
                explanation = f"LLM returned unexpected action: {action}"
                break

            if len(self.reasoning_trace) > 10: # Prevent infinite loops in a real LLM scenario
                diagnosis = "Undetermined (Too many reasoning steps)"
                explanation = "The agent exceeded the maximum reasoning steps."
                break

        return diagnosis, explanation, self.reasoning_trace


# --- Main Application Logic ----
if __name__ == "__main__":
    # 1. Initialize Medical Knowledge Graph and populate with sample data
    kg = MedicalKnowledgeGraph()

    kg.add_entity("symptom", "fever", {"description": "Elevated body temperature"})
    kg.add_entity("symptom", "cough", {"description": "Sudden expulsion of air from lungs"})
    kg.add_entity("symptom", "sore_throat", {"description": "Pain or irritation of the throat"})
    
    kg.add_entity("disease", "influenza", {"name": "Influenza (Flu)", "causes": "Influenza virus", "treatment": "Antivirals, rest"})
    kg.add_entity("disease", "common_cold", {"name": "Common Cold", "causes": "Rhinovirus", "treatment": "Symptomatic relief"})
    kg.add_entity("disease", "strep_throat", {"name": "Streptococcal Pharyngitis", "causes": "Streptococcus pyogenes", "treatment": "Antibiotics"})

    kg.add_relationship("fever", "influenza", "has_symptom", {"severity_factor": 0.8})
    kg.add_relationship("cough", "influenza", "has_symptom", {"severity_factor": 0.7})
    kg.add_relationship("sore_throat", "influenza", "has_symptom", {"severity_factor": 0.5})

    kg.add_relationship("fever", "common_cold", "has_symptom", {"severity_factor": 0.4})
    kg.add_relationship("cough", "common_cold", "has_symptom", {"severity_factor": 0.9})
    kg.add_relationship("sore_throat", "common_cold", "has_symptom", {"severity_factor": 0.7})

    kg.add_relationship("sore_throat", "strep_throat", "has_symptom", {"severity_factor": 0.9})
    kg.add_relationship("fever", "strep_throat", "has_symptom", {"severity_factor": 0.6})

    kg.add_entity("drug", "oseltamivir", {"name": "Oseltamivir", "class": "Antiviral"})
    kg.add_relationship("oseltamivir", "influenza", "treats_disease")

    # 2. Initialize LLM Interface (Mocked for this example)
    llm_interface = MockLLMInterface()

    # 3. Initialize Diagnostic Agent
    agent = DiagnosticAgent(kg, llm_interface)

    # 4. Simulate a diagnostic session
    patient_symptoms = ["fever", "cough"]
    patient_history = "No significant medical history, non-smoker."

    print("\n--- Starting Diagnostic Session ---")
    diagnosis, explanation, trace = agent.diagnose(patient_symptoms, patient_history)

    print("\n--- Final Diagnosis ---")
    print(f"Diagnosis: {diagnosis}")
    print(f"Explanation: {explanation}")

    print("\n--- Reasoning Trace ---")
    for step in trace:
        print(step)
