import math

def medical_database_tool(query):
    query = query.lower()
    if "ibuprofen dosing" in query:
        return "Ibuprofen dosing for adults: 200-400mg every 4-6 hours as needed, maximum 1200mg/day."
    elif "hypertension treatment" in query:
        return "Treatment for hypertension often involves lifestyle changes (diet, exercise) and medications like ACE inhibitors, ARBs, diuretics, or beta-blockers."
    elif "diabetes management" in query:
        return "Diabetes management includes blood sugar monitoring, diet control, regular exercise, and potentially medication or insulin therapy."
    else:
        return "Information for '{}' not found in medical database. Please refine your query.".format(query)

def calculator_tool(expression):
    try:
        return str(eval(expression))
    except Exception as e:
        return "Error in calculation: {}".format(str(e))

def symptom_checker_tool(symptoms):
    symptoms_list = [s.strip().lower() for s in symptoms.split(',')]
    if "fever" in symptoms_list and "cough" in symptoms_list and "fatigue" in symptoms_list:
        return "Possible conditions: Common cold, Flu, Bronchitis, early stages of Pneumonia."
    elif "headache" in symptoms_list and "nausea" in symptoms_list:
        return "Possible conditions: Migraine, Tension headache, Dehydration."
    elif "chest pain" in symptoms_list and "shortness of breath" in symptoms_list:
        return "Seek immediate medical attention. Possible conditions: Heart attack, Pulmonary embolism, Pneumonia, Anxiety."
    else:
        return "Based on the provided symptoms, further investigation or a doctor's consultation is recommended."

def literature_search_tool(topic):
    topic_lower = topic.lower()
    if "new diabetes treatments" in topic_lower:
        return "Recent research highlights SGLT2 inhibitors and GLP-1 receptor agonists as effective new treatments for type 2 diabetes, showing cardiovascular and renal benefits. (Simulated research summary)"
    elif "covid-19 vaccine efficacy" in topic_lower:
        return "Studies consistently demonstrate high efficacy of approved COVID-19 vaccines in preventing severe illness, hospitalization, and death, though effectiveness may wane over time requiring boosters. (Simulated research summary)"
    else:
        return "No specific recent literature found for '{}'. Try a broader or more specific topic. (Simulated research summary)".format(topic)

def llm_router(query):
    query_lower = query.lower()
    if "calculate" in query_lower:
        expression = query_lower.split("calculate ", 1)[1].strip() if "calculate " in query_lower else query_lower.replace("what is ", "").replace("compute ", "")
        return "calculator_tool", expression
    elif "dosing" in query_lower or "treatment for" in query_lower or "management of" in query_lower or "drug info" in query_lower:
        return "medical_database_tool", query
    elif "symptoms" in query_lower or "i have" in query_lower or "my symptoms are" in query_lower:
        symptoms_part = query_lower.split("symptoms are ", 1)[1].strip() if "symptoms are " in query_lower else \
                        query_lower.split("i have ", 1)[1].strip() if "i have " in query_lower else query
        return "symptom_checker_tool", symptoms_part
    elif "research on" in query_lower or "latest studies on" in query_lower or "scientific literature about" in query_lower:
        topic = query_lower.split("research on ", 1)[1].strip() if "research on " in query_lower else \
                query_lower.split("latest studies on ", 1)[1].strip() if "latest studies on " in query_lower else \
                query_lower.split("scientific literature about ", 1)[1].strip() if "scientific literature about " in query_lower else query
        return "literature_search_tool", topic
    else:
        return "default", query

def response_synthesizer(original_query, tool_output, tool_name):
    if tool_name == "calculator_tool":
        return f"For your query \"{original_query}\", the calculation result is: {tool_output}"
    elif tool_name == "medical_database_tool":
        return f"Regarding \"{original_query}\", here is the medical information: {tool_output}"
    elif tool_name == "symptom_checker_tool":
        return f"Based on your symptoms query \"{original_query}\", here are the potential insights: {tool_output}"
    elif tool_name == "literature_search_tool":
        return f"For your research on \"{original_query}\", here is a summary from scientific literature: {tool_output}"
    else:
        return f"I'm sorry, I couldn't find a specific tool for \"{original_query}\". However, I can offer this general response: {tool_output}"

class MedicalAssistant:
    def __init__(self):
        self.tools = {
            "medical_database_tool": medical_database_tool,
            "calculator_tool": calculator_tool,
            "symptom_checker_tool": symptom_checker_tool,
            "literature_search_tool": literature_search_tool,
        }

    def process_query(self, user_query):
        tool_name, tool_input = llm_router(user_query)
        
        if tool_name in self.tools:
            tool_function = self.tools[tool_name]
            tool_output = tool_function(tool_input)
            final_response = response_synthesizer(user_query, tool_output, tool_name)
        else:
            # Default response if no specific tool is matched
            tool_output = "Please ask about medical information, calculations, symptoms, or scientific research."
            final_response = response_synthesizer(user_query, tool_output, "default")
        
        return final_response

if __name__ == "__main__":
    assistant = MedicalAssistant()

    print("\n--- Test Case 1: Medical Database --- ")
    query1 = "What is the dosing for ibuprofen for adults?"
    print(f"User: {query1}")
    print(f"Assistant: {assistant.process_query(query1)}")

    print("\n--- Test Case 2: Calculator --- ")
    query2 = "Calculate 150 * 0.7 + 25."
    print(f"User: {query2}")
    print(f"Assistant: {assistant.process_query(query2)}")

    print("\n--- Test Case 3: Symptom Checker --- ")
    query3 = "I have symptoms like fever, cough, fatigue."
    print(f"User: {query3}")
    print(f"Assistant: {assistant.process_query(query3)}")

    print("\n--- Test Case 4: Scientific Literature --- ")
    query4 = "Latest studies on new diabetes treatments."
    print(f"User: {query4}")
    print(f"Assistant: {assistant.process_query(query4)}")

    print("\n--- Test Case 5: Default/Unmatched --- ")
    query5 = "Tell me a joke."
    print(f"User: {query5}")
    print(f"Assistant: {assistant.process_query(query5)}")

    print("\n--- Test Case 6: Medical Database - Another Query --- ")
    query6 = "What are the common treatments for hypertension?"
    print(f"User: {query6}")
    print(f"Assistant: {assistant.process_query(query6)}")

    print("\n--- Test Case 7: Calculator - Complex Expression --- ")
    query7 = "Compute (100 + 20) / 3."
    print(f"User: {query7}")
    print(f"Assistant: {assistant.process_query(query7)}")

    print("\n--- Test Case 8: Symptom Checker - Different Symptoms --- ")
    query8 = "My symptoms are headache, nausea."
    print(f"User: {query8}")
    print(f"Assistant: {assistant.process_query(query8)}")

    print("\n--- Test Case 9: Scientific Literature - Unfound Topic --- ")
    query9 = "Research on quantum entanglement in biology."
    print(f"User: {query9}")
    print(f"Assistant: {assistant.process_query(query9)}")
