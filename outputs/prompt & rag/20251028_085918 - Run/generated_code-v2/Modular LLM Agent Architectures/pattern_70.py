def mock_symptom_checker(symptoms):
    """Simulates an external API call for symptom checking."""
    symptoms_lower = [s.lower() for s in symptoms]
    if "fever" in symptoms_lower and "cough" in symptoms_lower:
        return {"tool": "symptom_checker", "result": "Potential conditions: Common Cold, Flu, Bronchitis."}
    elif "headache" in symptoms_lower and "nausea" in symptoms_lower:
        return {"tool": "symptom_checker", "result": "Potential conditions: Migraine, Food Poisoning, Dehydration."}
    elif "chest pain" in symptoms_lower:
        return {"tool": "symptom_checker", "result": "Seek immediate medical attention for chest pain. Potential conditions: Angina, Heart Attack, Anxiety."}
    else:
        return {"tool": "symptom_checker", "result": "No specific conditions found for these symptoms in our mock database."}

def mock_drug_interaction_checker(drugs):
    """Simulates an external API call for drug interaction checking."""
    drugs_lower = [d.lower() for d in drugs]
    if "ibuprofen" in drugs_lower and "aspirin" in drugs_lower:
        return {"tool": "drug_interaction_checker", "result": "Warning: Increased risk of gastrointestinal bleeding when taking Ibuprofen and Aspirin together."}
    elif "warfarin" in drugs_lower and "grapefruit juice" in drugs_lower:
        return {"tool": "drug_interaction_checker", "result": "Warning: Grapefruit juice can increase the effect of Warfarin, leading to a higher risk of bleeding."}
    else:
        return {"tool": "drug_interaction_checker", "result": "No significant interactions found for the given drugs in our mock database."}

def mock_medical_knowledge_base(query):
    """Simulates an external API call to a medical knowledge base."""
    query_lower = query.lower()
    if "diabetes" in query_lower:
        return {"tool": "knowledge_base", "result": "Diabetes is a chronic condition that affects how your body turns food into energy. It is characterized by high blood sugar levels. There are several types, including Type 1, Type 2, and Gestational Diabetes."}
    elif "hypertension" in query_lower:
        return {"tool": "knowledge_base", "result": "Hypertension, or high blood pressure, is a common condition in which the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease."}
    else:
        return {"tool": "knowledge_base", "result": f"No detailed information found for '{query}' in our mock knowledge base."}

def mock_medical_calculator(expression):
    """Simulates a simple medical calculation tool (e.g., BMI)."""
    try:
        # For simplicity, we'll implement a very basic BMI calculation example
        if "bmi" in expression.lower():
            parts = expression.lower().replace("bmi", "").strip().split(" ")
            if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                weight_kg = float(parts[0])
                height_m = float(parts[1])
                if height_m == 0: return {"tool": "calculator", "result": "Error: Height cannot be zero."}
                bmi = weight_kg / (height_m ** 2)
                return {"tool": "calculator", "result": f"Calculated BMI: {bmi:.2f}."}
        return {"tool": "calculator", "result": f"Could not process calculation: '{expression}'. Try 'BMI weight_kg height_m'."}
    except Exception as e:
        return {"tool": "calculator", "result": f"Error during calculation: {e}"}

def llm_router(query):
    """Simulates an LLM router that directs queries to appropriate tools.
    In a real system, this would involve an actual LLM making a decision.
    """
    query_lower = query.lower()
    tool_calls = []

    if "symptoms" in query_lower or "feel" in query_lower or "experiencing" in query_lower:
        # Extract symptoms - highly simplified for this mock
        if "fever" in query_lower and "cough" in query_lower:
            tool_calls.append(("symptom_checker", ["fever", "cough"]))
        elif "headache" in query_lower and "nausea" in query_lower:
            tool_calls.append(("symptom_checker", ["headache", "nausea"]))
        elif "chest pain" in query_lower:
            tool_calls.append(("symptom_checker", ["chest pain"]))
        else:
            # Default if specific symptoms not caught
            tool_calls.append(("symptom_checker", [query_lower])) # Pass whole query for general check

    if "interact" in query_lower or "take together" in query_lower or "drug" in query_lower:
        # Extract drugs - highly simplified
        if "ibuprofen" in query_lower and "aspirin" in query_lower:
            tool_calls.append(("drug_interaction_checker", ["ibuprofen", "aspirin"]))
        elif "warfarin" in query_lower and "grapefruit juice" in query_lower:
            tool_calls.append(("drug_interaction_checker", ["warfarin", "grapefruit juice"]))
        else:
            pass # More complex parsing needed for a real system

    if "what is" in query_lower or "tell me about" in query_lower or "info on" in query_lower:
        # Extract knowledge base query
        if "diabetes" in query_lower:
            tool_calls.append(("knowledge_base", "diabetes"))
        elif "hypertension" in query_lower:
            tool_calls.append(("knowledge_base", "hypertension"))
        else:
            pass # More complex parsing

    if "calculate" in query_lower or "bmi" in query_lower:
        # Extract calculation expression
        if "bmi" in query_lower:
            import re
            match = re.search(r"bmi (\d+\.?\d*)\s*kg\s*(\d+\.?\d*)\s*m", query_lower) # e.g., 'bmi 70 kg 1.75 m'
            if match:
                weight = match.group(1)
                height = match.group(2)
                tool_calls.append(("calculator", f"BMI {weight} {height}"))
            else:
                tool_calls.append(("calculator", "BMI calculation requested, but input format unclear. Please specify weight in kg and height in m (e.g., 'BMI 70 1.75')."))
        else:
            tool_calls.append(("calculator", query_lower)) # Pass whole query for general calculation attempt

    # If no specific tool is identified, assume general knowledge query
    if not tool_calls and query_lower:
        tool_calls.append(("knowledge_base", query_lower))

    return tool_calls

def synthesize_response(tool_results):
    """Combines results from multiple tools into a coherent response."""
    responses = []
    for res in tool_results:
        tool_name = res.get("tool", "Unknown Tool").replace("_", " ").title()
        result_text = res.get("result", "No information available.")
        responses.append(f"**{tool_name}**: {result_text}")

    if not responses:
        return "I couldn't find relevant information using my tools. Please try rephrasing your query."
    return "\n".join(responses)

def medical_diagnostic_assistant(user_query):
    """Main function for the Medical Diagnostic Assistant.
    It orchestrates the LLM router, tool calls, and response synthesis.
    """
    print(f"\nUser Query: {user_query}")
    print("--------------------------------------------------")

    # Step 1: LLM Router decides which tools to use
    calls_to_make = llm_router(user_query)
    if not calls_to_make:
        return "I couldn't determine which medical tools to use for your query. Can you please be more specific?"

    print(f"Router decided to call: {calls_to_make}")

    # Step 2: Execute tool calls
    tool_outputs = []
    for tool_name, args in calls_to_make:
        if tool_name == "symptom_checker":
            tool_outputs.append(mock_symptom_checker(args))
        elif tool_name == "drug_interaction_checker":
            tool_outputs.append(mock_drug_interaction_checker(args))
        elif tool_name == "knowledge_base":
            tool_outputs.append(mock_medical_knowledge_base(args))
        elif tool_name == "calculator":
            tool_outputs.append(mock_medical_calculator(args))
        else:
            tool_outputs.append({"tool": tool_name, "result": "Error: Unrecognized tool."})

    print(f"Tool outputs: {tool_outputs}")

    # Step 3: Synthesize response from tool outputs
    final_response = synthesize_response(tool_outputs)

    return final_response

# --- Example Usage ---
if __name__ == "__main__":
    # Example 1: Symptom check
    print(medical_diagnostic_assistant("I am experiencing a fever and a cough. What could it be?"))

    # Example 2: Drug interaction
    print(medical_diagnostic_assistant("Can I take ibuprofen and aspirin together?"))

    # Example 3: Medical knowledge lookup
    print(medical_diagnostic_assistant("Tell me about diabetes."))

    # Example 4: Medical calculation (BMI)
    print(medical_diagnostic_assistant("Calculate my BMI if I weigh 70 kg and am 1.75 m tall."))

    # Example 5: Multiple tools (simplified scenario - router prioritizes for mock)
    print(medical_diagnostic_assistant("I have a headache and want to know about hypertension."))

    # Example 6: Query not easily matched to a specific tool (defaults to knowledge base)
    print(medical_diagnostic_assistant("What are the benefits of exercise?"))

    # Example 7: Unmatched calculation format
    print(medical_diagnostic_assistant("Calculate BMI 70 kilograms and 1.75 meters."))

    # Example 8: Complex symptom query (mock simplification)
    print(medical_diagnostic_assistant("I feel very tired and have some muscle aches."))
