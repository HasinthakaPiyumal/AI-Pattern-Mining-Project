tool_registry = {}
tool_definitions = {}

def simulate_tool_generation(tool_name, requirement):
    if tool_name == "calculate_bmi":
        tool_code = f"""
def {tool_name}(weight_kg, height_m):
    if height_m == 0:
        return \"Error: Height cannot be zero.\"
    bmi = weight_kg / (height_m ** 2)
    return f\"Your BMI is {{bmi:.2f}}\"
"""
        tool_def = {
            "name": tool_name,
            "description": "Calculates Body Mass Index (BMI) given weight in kg and height in meters.",
            "parameters": {
                "weight_kg": {"type": "float", "description": "Weight in kilograms"},
                "height_m": {"type": "float", "description": "Height in meters"}
            }
        }
    elif tool_name == "check_drug_interaction":
        tool_code = f"""
def {tool_name}(drug1, drug2):
    interactions = {
        ("aspirin", "warfarin"): "Increased bleeding risk.",
        ("paracetamol", "alcohol"): "Increased liver toxicity risk.",
    }
    d1 = drug1.lower()
    d2 = drug2.lower()
    if (d1, d2) in interactions:
        return f\"Interaction between {{drug1}} and {{drug2}}: {{interactions[(d1, d2)]}}\"
    elif (d2, d1) in interactions:
        return f\"Interaction between {{drug2}} and {{drug1}}: {{interactions[(d2, d1)]}}\"
    else:
        return f\"No common interaction found between {{drug1}} and {{drug2}} in our database.\"
"""
        tool_def = {
            "name": tool_name,
            "description": "Checks for potential drug interactions between two specified drugs.",
            "parameters": {
                "drug1": {"type": "str", "description": "First drug name"},
                "drug2": {"type": "str", "description": "Second drug name"}
            }
        }
    else:
        tool_code = f"""
def {tool_name}(*args, **kwargs):
    return f\"This is a newly generated tool for '{{tool_name}}' based on the requirement: '{{requirement}}'. Args: {{args}}, Kwargs: {{kwargs}}\"
"""
        tool_def = {
            "name": tool_name,
            "description": f"A generic tool created for the requirement: '{requirement}'.",
            "parameters": {}
        }

    return tool_code, tool_def

class SimulatedLLMAgent:
    def __init__(self):
        self.tool_registry = tool_registry
        self.tool_definitions = tool_definitions
        self._add_predefined_tools()

    def _add_predefined_tools(self):
        bmi_code, bmi_def = simulate_tool_generation("calculate_bmi", "calculate BMI")
        self._add_tool_to_registry("calculate_bmi", bmi_code, bmi_def)

    def _add_tool_to_registry(self, name, code, definition):
        try:
            local_scope = {}
            exec(code, globals(), local_scope)
            self.tool_registry[name] = local_scope[name]
            self.tool_definitions[name] = definition
            print(f"Tool '{name}' added to registry.")
        except Exception as e:
            print(f"Error adding tool '{name}': {e}")

    def process_query(self, query):
        print(f"\nLLM Agent processing query: '{query}'")

        if "bmi" in query.lower() and "calculate_bmi" in self.tool_registry:
            print("Detected 'BMI' calculation. Attempting to use existing tool.")
            try:
                weight = float(input("Enter weight in kg: "))
                height = float(input("Enter height in meters: "))
                result = self.tool_registry["calculate_bmi"](weight, height)
                print(f"Result from calculate_bmi: {result}")
                return
            except ValueError:
                print("Invalid input for BMI calculation.")
                return
            except Exception as e:
                print(f"Error executing calculate_bmi: {e}")
                return

        if "drug interaction" in query.lower() and "check_drug_interaction" in self.tool_registry:
            print("Detected 'drug interaction' query. Attempting to use existing tool.")
            try:
                drug1 = input("Enter first drug name: ")
                drug2 = input("Enter second drug name: ")
                result = self.tool_registry["check_drug_interaction"](drug1, drug2)
                print(f"Result from check_drug_interaction: {result}")
                return
            except Exception as e:
                print(f"Error executing check_drug_interaction: {e}")
                return

        if "create tool for" in query.lower():
            requirement = query.lower().split("create tool for", 1)[1].strip()
            tool_name_suggestion = requirement.replace(" ", "_").replace(".", "")
            print(f"LLM Agent identifies need for a new tool: '{requirement}'")
            print(f"Suggesting tool name: '{tool_name_suggestion}'")

            tool_code, tool_def = simulate_tool_generation(tool_name_suggestion, requirement)
            self._add_tool_to_registry(tool_name_suggestion, tool_code, tool_def)
            print(f"New tool '{tool_name_suggestion}' created and added.")
            print(f"Tool definition: {self.tool_definitions.get(tool_name_suggestion, 'N/A')}")
            if not tool_def["parameters"]:
                 print(f"Attempting to call newly created tool '{tool_name_suggestion}'...")
                 result = self.tool_registry[tool_name_suggestion]()
                 print(f"Result from newly created tool: {result}")
            return

        print(f"LLM Agent response: I cannot fulfill this request directly or generate a specific tool for it yet: '{query}'")
        print("Existing tools:")
        for name, definition in self.tool_definitions.items():
            print(f"- {name}: {definition['description']}")


if __name__ == "__main__":
    agent = SimulatedLLMAgent()

    print("Welcome to the Dynamic Medical Assistant Tool Creator (Simulated)!")
    print("Type 'exit' to quit.")
    print("Initial tools available:")
    for name, definition in tool_definitions.items():
        print(f"- {name}: {definition['description']}")

    drug_code, drug_def = simulate_tool_generation("check_drug_interaction", "check drug interaction")
    agent._add_tool_to_registry("check_drug_interaction", drug_code, drug_def)


    while True:
        user_query = input("\nEnter your medical query: ")
        if user_query.lower() == 'exit':
            break
        agent.process_query(user_query)