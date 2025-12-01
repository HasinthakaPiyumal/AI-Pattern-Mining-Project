import json

class AIDiagnosticAssistant:
    def __init__(self):
        self.pre_existing_tools = {
            "get_lab_results": self._get_lab_results,
            "check_drug_interactions": self._check_drug_interactions
        }
        self.ai_generated_tools = {}
        self.medical_research_db = {
            "new_biomarker_X": "early detection for disease Y via blood test",
            "drug_Z_side_effects": "severe liver damage in elderly patients"
        }

    def _get_lab_results(self, patient_id):
        return {"patient_id": patient_id, "glucose": 120, "white_blood_cells": 7500}

    def _check_drug_interactions(self, drug1, drug2):
        if drug1 == "drug_A" and drug2 == "drug_B":
            return "Severe interaction: avoid combination"
        return "No known interaction"

    def _simulate_llm_response(self, prompt):
        if "new diagnostic tool for biomarker_Z" in prompt:
            return {"action": "create_tool", "tool_name": "analyze_biomarker_Z", "description": "Function to analyze biomarker Z for disease A.", "code": "def analyze_biomarker_Z(data):\n    if 'biomarker_Z_level' in data and data['biomarker_Z_level'] > 0.8:\n        return 'High risk for Disease A'\n    return 'Low risk for Disease A'"}
        elif "average temperature" in prompt:
            return {"action": "create_tool", "tool_name": "compute_avg_temp", "description": "Function to compute average temperature from a list of readings.", "code": "def compute_avg_temp(temperatures):\n    if not temperatures: return 0.0\n    return sum(temperatures) / len(temperatures)"}
        elif "diagnose patient with symptoms" in prompt:
            return {"action": "use_tool", "tool_name": "get_lab_results", "args": {"patient_id": "P001"}}
        elif "check drug interactions for drug_A and drug_B" in prompt:
            return {"action": "use_tool", "tool_name": "check_drug_interactions", "args": {"drug1": "drug_A", "drug2": "drug_B"}}
        return {"action": "none", "response": "I don't have a specific tool for that or need more information."}

    def _code_generation_engine(self, tool_description, code_template):
        return code_template

    def _tool_validation_safety(self, tool_name, tool_code):
        return True

    def _execute_tool(self, tool_name, args):
        if tool_name in self.pre_existing_tools:
            return self.pre_existing_tools[tool_name](**args)
        elif tool_name in self.ai_generated_tools:
            return self.ai_generated_tools[tool_name](**args)
        else:
            return f"Tool '{tool_name}' not found."

    def _integrate_new_tool(self, tool_name, tool_code):
        local_vars = {}
        exec(tool_code, globals(), local_vars)
        new_function = local_vars.get(tool_name)
        if new_function and callable(new_function):
            if self._tool_validation_safety(tool_name, tool_code):
                self.ai_generated_tools[tool_name] = new_function
                return f"Successfully integrated new tool: {tool_name}"
        return f"Failed to integrate tool: {tool_name}"

    def inquire(self, query):
        print(f"User query: {query}")
        llm_decision = self._simulate_llm_response(query)

        if llm_decision["action"] == "create_tool":
            tool_name = llm_decision["tool_name"]
            tool_code = llm_decision["code"]
            print(f"LLM decided to create tool: {tool_name}")
            integration_status = self._integrate_new_tool(tool_name, tool_code)
            print(integration_status)
            return integration_status
        elif llm_decision["action"] == "use_tool":
            tool_name = llm_decision["tool_name"]
            args = llm_decision["args"]
            print(f"LLM decided to use tool: {tool_name} with args: {args}")
            result = self._execute_tool(tool_name, args)
            print(f"Tool result: {result}")
            return result
        else:
            print(f"LLM response: {llm_decision['response']}")
            return llm_decision['response']


if __name__ == "__main__":
    assistant = AIDiagnosticAssistant()

    print("--- Scenario 1: Using a pre-existing tool ---")
    assistant.inquire("diagnose patient with symptoms for P001")
    print("\n--- Scenario 2: Checking drug interactions ---")
    assistant.inquire("check drug interactions for drug_A and drug_B")

    print("\n--- Scenario 3: AI autonomously creates a new tool based on research ---")
    assistant.inquire("Develop a new diagnostic tool for biomarker_Z in relation to disease A, based on recent research.")
    print(f"\nAI Generated Tools after creation: {list(assistant.ai_generated_tools.keys())}")
    print("Now using the newly created tool:")
    if "analyze_biomarker_Z" in assistant.ai_generated_tools:
        result = assistant.ai_generated_tools["analyze_biomarker_Z"]({"biomarker_Z_level": 0.9})
        print(f"Result from analyze_biomarker_Z with high level: {result}")
        result = assistant.ai_generated_tools["analyze_biomarker_Z"]({"biomarker_Z_level": 0.4})
        print(f"Result from analyze_biomarker_Z with low level: {result}")

    print("\n--- Scenario 4: AI creating another general-purpose tool ---")
    assistant.inquire("Can you create a tool to compute the average temperature from a list of readings?")
    print(f"\nAI Generated Tools after creation: {list(assistant.ai_generated_tools.keys())}")
    if "compute_avg_temp" in assistant.ai_generated_tools:
        result = assistant.ai_generated_tools["compute_avg_temp"]([20.5, 21.0, 19.8, 22.1])
        print(f"Result from compute_avg_temp: {result}")
        result = assistant.ai_generated_tools["compute_avg_temp"]([])
        print(f"Result from compute_avg_temp with empty list: {result}")

    print("\n--- Scenario 5: Query without specific tool match ---")
    assistant.inquire("Tell me about the history of medicine.")