
class LLMModule:
    """Simulates the reasoning capabilities of a Large Language Model."""
    def generate_response(self, prompt: str) -> str:
        # In a real application, this would call an actual LLM API (e.g., OpenAI, Hugging Face transformers)
        # For demonstration, we simulate a simple response based on keywords.
        print(f"[LLM] Processing prompt: {prompt[:50]}...")
        if "symptoms" in prompt.lower() or "diagnosis" in prompt.lower():
            return f"Based on your input, the LLM suggests investigating further. I will consult other modules. Initial thought on '{prompt}'."
        elif "plan" in prompt.lower() or "action" in prompt.lower():
            return f"The LLM is guiding the diagnostic plan. I will proceed with the suggested steps for '{prompt}'."
        elif "data" in prompt.lower() or "history" in prompt.lower():
            return f"The LLM requires patient data from memory for '{prompt}'."
        elif "tool" in prompt.lower() or "database" in prompt.lower() or "lab" in prompt.lower():
            return f"The LLM recommends using external tools for '{prompt}'."
        return f"LLM response to: '{prompt}'."

class MemoryModule:
    """Stores and retrieves patient-specific data."""
    def __init__(self):
        self._patient_data = {}

    def add_patient_data(self, patient_id: str, key: str, value: any):
        if patient_id not in self._patient_data:
            self._patient_data[patient_id] = {}
        self._patient_data[patient_id][key] = value
        print(f"[Memory] Added data for patient {patient_id}: {key} = {value}")

    def get_patient_data(self, patient_id: str, key: str = None) -> dict or any:
        data = self._patient_data.get(patient_id, {})
        if key:
            print(f"[Memory] Retrieved data for patient {patient_id}, key {key}: {data.get(key)}")
            return data.get(key)
        print(f"[Memory] Retrieved all data for patient {patient_id}: {data}")
        return data

class PlanningModule:
    """Generates and executes diagnostic plans."""
    def __init__(self):
        self._workflows = {
            "initial assessment": [
                "Gather patient demographics",
                "Collect chief complaints and history of present illness",
                "Review past medical history and medications",
            ],
            "differential diagnosis": [
                "Generate a list of possible conditions",
                "Prioritize conditions based on likelihood",
                "Identify key discriminating factors",
            ],
            "order tests": [
                "Recommend relevant laboratory tests",
                "Suggest imaging studies if indicated",
                "Consider specialist consultations",
            ],
        }

    def generate_plan(self, patient_data: dict, query: str) -> list:
        plan = []
        query_lower = query.lower()
        print(f"[Planning] Generating plan for query: {query[:50]}...")

        if "symptoms" in query_lower and "history" in query_lower:
            plan.extend(self._workflows["initial assessment"])
        if "possible causes" in query_lower or "what could it be" in query_lower:
            plan.extend(self._workflows["differential diagnosis"])
        if "tests needed" in query_lower or "next steps" in query_lower:
            plan.extend(self._workflows["order tests"])

        if not plan and patient_data:
            # Default plan if specific keywords aren't found but patient data exists
            plan.append("Review existing patient data for clues.")
            if "chief_complaint" in patient_data:
                plan.append(f"Focus on the chief complaint: {patient_data['chief_complaint']}")

        if not plan: # Fallback if no specific plan is generated
            plan.append("Consult general diagnostic guidelines.")

        print(f"[Planning] Generated plan: {plan}")
        return plan

    def execute_plan(self, plan: list) -> str:
        print(f"[Planning] Executing plan: {plan}")
        results = []
        for step in plan:
            results.append(f"Executed: {step}")
            # In a real scenario, this would involve calling other modules or performing actual actions
        return "\n".join(results)

class ToolInterfaceModule:
    """Provides interfaces to external systems."""
    def query_medical_database(self, query: str) -> str:
        # Simulate querying a medical knowledge base
        print(f"[Tool] Querying medical database with: {query[:50]}...")
        if "fever" in query.lower() and "rash" in query.lower():
            return "Medical database suggests: Possible viral exanthem, consider measles or rubella. Refer to CDC guidelines."
        elif "hypertension" in query.lower():
            return "Medical database suggests: Review JNC 8 guidelines for management of hypertension."
        return f"Medical database found information related to: '{query}'."

    def order_lab_tests(self, patient_id: str, tests: list) -> dict:
        # Simulate ordering lab tests and retrieving results
        print(f"[Tool] Ordering lab tests for patient {patient_id}: {tests}")
        results = {}
        for test in tests:
            if test == "CBC":
                results["CBC"] = {"WBC": "8.5 K/uL", "RBC": "4.8 M/uL", "Hgb": "14.2 g/dL"}
            elif test == "CMP":
                results["CMP"] = {"Glucose": "95 mg/dL", "Creatinine": "0.9 mg/dL"}
            else:
                results[test] = "Pending/Normal"
        print(f"[Tool] Lab results for patient {patient_id}: {results}")
        return results

class ContextManagerModule:
    """Maintains the conversation context and state for each patient interaction."""
    def __init__(self):
        self._contexts = {}

    def add_to_context(self, patient_id: str, message_type: str, content: str):
        if patient_id not in self._contexts:
            self._contexts[patient_id] = []
        self._contexts[patient_id].append({"type": message_type, "content": content})
        print(f"[Context] Added to context for patient {patient_id}: {message_type} - {content[:30]}...")

    def get_context(self, patient_id: str) -> list:
        context = self._contexts.get(patient_id, [])
        print(f"[Context] Retrieved context for patient {patient_id}: {len(context)} entries")
        return context

    def clear_context(self, patient_id: str):
        if patient_id in self._contexts:
            del self._contexts[patient_id]
            print(f"[Context] Cleared context for patient {patient_id}")

class MedicalDiagnosticAgent:
    """The main orchestrator class for the Medical Diagnostic Assistant."""
    def __init__(self):
        self.llm = LLMModule()
        self.memory = MemoryModule()
        self.planning = PlanningModule()
        self.tools = ToolInterfaceModule()
        self.context_manager = ContextManagerModule()

    def process_patient_query(self, patient_id: str, query: str) -> str:
        self.context_manager.add_to_context(patient_id, "user_query", query)
        full_response = []

        # 1. LLM gets an initial understanding/response
        llm_initial_thought = self.llm.generate_response(query)
        full_response.append(f"LLM initial thought: {llm_initial_thought}")
        self.context_manager.add_to_context(patient_id, "llm_thought", llm_initial_thought)

        # 2. Consult Memory (if needed, based on LLM thought or explicit keywords)
        if "history" in query.lower() or "patient data" in query.lower():
            patient_history = self.memory.get_patient_data(patient_id)
            if patient_history:
                full_response.append(f"Memory consulted: Patient history found: {patient_history}")
                self.context_manager.add_to_context(patient_id, "memory_retrieval", str(patient_history))
            else:
                full_response.append("Memory consulted: No specific patient history found.")

        # 3. Generate and Execute Plan
        current_patient_data = self.memory.get_patient_data(patient_id)
        plan = self.planning.generate_plan(current_patient_data, query)
        if plan:
            plan_execution_result = self.planning.execute_plan(plan)
            full_response.append(f"Plan executed:\n{plan_execution_result}")
            self.context_manager.add_to_context(patient_id, "plan_execution", plan_execution_result)

        # 4. Interact with Tools (if needed, based on LLM thought or explicit keywords)
        if "database" in query.lower() or "medical info" in query.lower():
            db_result = self.tools.query_medical_database(query)
            full_response.append(f"External Tool (Medical DB): {db_result}")
            self.context_manager.add_to_context(patient_id, "tool_db_query", db_result)

        if "lab tests" in query.lower() or "order tests" in query.lower():
            # Simulate ordering specific tests
            tests_to_order = []
            if "cbc" in query.lower(): tests_to_order.append("CBC")
            if "cmp" in query.lower(): tests_to_order.append("CMP")
            if not tests_to_order: tests_to_order.append("General Blood Panel")

            lab_results = self.tools.order_lab_tests(patient_id, tests_to_order)
            # Add lab results to memory for future reference
            self.memory.add_patient_data(patient_id, "latest_lab_results", lab_results)
            full_response.append(f"External Tool (Lab System): Ordered {tests_to_order}. Results: {lab_results}")
            self.context_manager.add_to_context(patient_id, "tool_lab_results", str(lab_results))

        # 5. Final LLM synthesis (conceptual - in a real system, the LLM would synthesize all information)
        # For this demo, we'll combine the responses generated by the modules.
        final_llm_synthesis_prompt = f"Synthesize the following information for patient {patient_id} regarding query '{query}':\n" \
                                     + "\n".join(full_response)
        final_llm_output = self.llm.generate_response(final_llm_synthesis_prompt)
        full_response.append(f"\nFinal LLM Synthesis: {final_llm_output}")
        self.context_manager.add_to_context(patient_id, "final_synthesis", final_llm_output)

        return "\n---\n".join(full_response)

# --- Demonstration of the Medical Diagnostic Agent ---
if __name__ == "__main__":
    agent = MedicalDiagnosticAgent()

    # Scenario 1: Initial patient query and data input
    print("\n===== SCENARIO 1: Initial Patient Assessment =====")
    patient_id_1 = "P001"
    agent.memory.add_patient_data(patient_id_1, "name", "Alice Smith")
    agent.memory.add_patient_data(patient_id_1, "age", 45)
    agent.memory.add_patient_data(patient_id_1, "chief_complaint", "Persistent cough and fatigue for 2 weeks")
    agent.memory.add_patient_data(patient_id_1, "medical_history", "Hypertension, well-controlled")

    query_1 = "Patient P001 has persistent cough and fatigue. What are the initial steps for assessment?"
    response_1 = agent.process_patient_query(patient_id_1, query_1)
    print(f"\nAgent Response for P001 (Query 1):\n{response_1}")
    print("\n--- Current Context for P001 ---")
    for entry in agent.context_manager.get_context(patient_id_1):
        print(f"  [{entry['type']}] {entry['content'][:70]}...")

    # Scenario 2: Differential diagnosis and medical database query
    print("\n===== SCENARIO 2: Differential Diagnosis and Database Query =====")
    query_2 = "Considering P001's cough and fatigue, what are possible differential diagnoses and what does the medical database say about similar symptoms?"
    response_2 = agent.process_patient_query(patient_id_1, query_2)
    print(f"\nAgent Response for P001 (Query 2):\n{response_2}")
    print("\n--- Current Context for P001 ---")
    for entry in agent.context_manager.get_context(patient_id_1):
        print(f"  [{entry['type']}] {entry['content'][:70]}...")

    # Scenario 3: Ordering lab tests and reviewing results
    print("\n===== SCENARIO 3: Ordering Lab Tests =====")
    query_3 = "For patient P001, I want to order CBC and CMP to investigate further. Provide lab tests results."
    response_3 = agent.process_patient_query(patient_id_1, query_3)
    print(f"\nAgent Response for P001 (Query 3):\n{response_3}")
    print("\n--- Current Context for P001 ---")
    for entry in agent.context_manager.get_context(patient_id_1):
        print(f"  [{entry['type']}] {entry['content'][:70]}...")
    print("\n--- Latest Lab Results in Memory for P001 ---")
    print(agent.memory.get_patient_data(patient_id_1, "latest_lab_results"))

    # Scenario 4: New patient interaction
    print("\n===== SCENARIO 4: New Patient Interaction =====")
    patient_id_2 = "P002"
    agent.memory.add_patient_data(patient_id_2, "name", "Bob Johnson")
    agent.memory.add_patient_data(patient_id_2, "age", 60)
    agent.memory.add_patient_data(patient_id_2, "chief_complaint", "Chest pain and shortness of breath")

    query_4 = "Patient P002 reports sudden chest pain and shortness of breath. What initial actions should be taken?"
    response_4 = agent.process_patient_query(patient_id_2, query_4)
    print(f"\nAgent Response for P002 (Query 4):\n{response_4}")
    print("\n--- Current Context for P002 ---")
    for entry in agent.context_manager.get_context(patient_id_2):
        print(f"  [{entry['type']}] {entry['content'][:70]}...")

    print("\n===== End of Demonstration =====")
