import gradio as gr
import json

# --- 1. Mock External Tools ---

def mock_api_search(query: str) -> dict:
    """Mocks an API search for medical APIs."""
    print(f"Searching for APIs related to: {query}")
    if "EHR" in query.upper():
        return {"EHR_API": {"name": "Epic FHIR API", "docs_url": "https://fhir.epic.com/", "description": "API for Electronic Health Records."}}
    elif "LAB" in query.upper() or "RESULTS" in query.upper():
        return {"LAB_API": {"name": "Quest Diagnostics API", "docs_url": "https://developer.questdiagnostics.com/", "description": "API for lab test results."}}
    else:
        return {"GENERIC_MEDICAL_API": {"name": "Generic Medical Data API", "docs_url": "https://example.com/medical-data-api", "description": "A generic API for medical data."}}

def mock_documentation_parser(docs_url: str) -> dict:
    """Mocks parsing API documentation."""
    print(f"Parsing documentation from: {docs_url}")
    if "epic.com" in docs_url:
        return {
            "endpoints": ["/Patient", "/Practitioner", "/Observation"],
            "auth": "OAuth2",
            "schemas": {"Patient": {"id": "string", "name": "string"}}
        }
    elif "questdiagnostics.com" in docs_url:
        return {
            "endpoints": ["/LabOrder", "/LabResult"],
            "auth": "API Key",
            "schemas": {"LabResult": {"id": "string", "patient_id": "string", "test_name": "string", "value": "string"}}
        }
    else:
        return {"endpoints": [], "auth": "None", "schemas": {}}

def mock_execute_code(code_snippet: str) -> str:
    """Mocks executing a code snippet in a sandbox."""
    print("Executing code snippet (mocking):")
    print(code_snippet)
    # In a real scenario, this would use a secure sandbox like `exec()` with caution or a dedicated service.
    if "import requests" in code_snippet:
        return "Mock execution successful. Assumed API call made."
    elif "def test_" in code_snippet:
        return "Mock test execution successful. Assumed tests passed."
    else:
        return "Mock execution completed. Output: Hello from mock sandbox!"

def mock_test_code(code_snippet: str) -> str:
    """Mocks testing a code snippet."""
    print("Testing code snippet (mocking):")
    print(code_snippet)
    if "raise Exception" in code_snippet:
        return "Mock test failed: Error detected in code."
    else:
        return "Mock test passed: Basic syntax and structure seem okay."

# --- 2. Mock Knowledge Base/Vector Store ---

medical_kb = {
    "ehr_integration_best_practices": "Always use secure authentication (OAuth2). Handle patient data with care. Implement robust error handling.",
    "lab_results_schema_example": "{\"patient_id\": \"123\", \"test_name\": \"CBC\", \"value\": \"Normal\"}"
}

# --- 3. Core AI Model (LLM Placeholder) ---

class LLM:
    def generate_code(self, prompt: str) -> str:
        print(f"LLM generating code for prompt: {prompt}")
        # This is a highly simplified placeholder for an actual LLM call.
        # In a real application, this would call a model like OpenAI GPT, Llama, etc.
        if "integrate EHR" in prompt:
            return """import requests\n\ndef get_patient_data(patient_id, access_token):\n    headers = {'Authorization': f'Bearer {access_token}'}\n    response = requests.get(f'https://fhir.epic.com/Patient/{patient_id}', headers=headers)\n    response.raise_for_status()\n    return response.json()\n"""
        elif "integrate lab results" in prompt:
            return """import requests\n\ndef post_lab_result(lab_data, api_key):\n    headers = {'X-API-Key': api_key, 'Content-Type': 'application/json'}\n    response = requests.post('https://developer.questdiagnostics.com/LabResult', headers=headers, json=lab_data)\n    response.raise_for_status()\n    return response.json()\n"""
        elif "test code" in prompt:
            return """def test_get_patient_data():\n    # Mock test implementation\n    assert True # Placeholder for actual test logic\n"""
        else:
            return f"# Generated code for: {prompt}\nprint('Hello from the medical API integration assistant!')"

llm_model = LLM()

# --- 4. Tool Orchestration (Simplified Langchain-like agent) ---

def medical_api_integration_agent(user_query: str) -> str:
    """Orchestrates tools to generate and validate API integration code."""
    response_log = []

    # Step 1: Understand the query and search for relevant APIs
    response_log.append(f"User query: {user_query}")
    api_search_results = mock_api_search(user_query)
    response_log.append(f"API Search Results: {json.dumps(api_search_results, indent=2)}")

    if not api_search_results:
        response_log.append("No relevant APIs found.")
        return "\n".join(response_log)

    # Pick the first found API for demonstration
    first_api_key = next(iter(api_search_results))
    selected_api = api_search_results[first_api_key]
    response_log.append(f"Selected API: {selected_api['name']} ({selected_api['docs_url']})")

    # Step 2: Parse documentation
    docs_info = mock_documentation_parser(selected_api['docs_url'])
    response_log.append(f"Parsed Documentation Info: {json.dumps(docs_info, indent=2)}")

    # Step 3: Retrieve relevant knowledge base information
    kb_info = []
    if "EHR" in user_query.upper():
        kb_info.append(medical_kb.get("ehr_integration_best_practices", ""))
    if "LAB" in user_query.upper() or "RESULTS" in user_query.upper():
        kb_info.append(medical_kb.get("lab_results_schema_example", ""))
    if kb_info:
        response_log.append(f"Relevant KB Info: {'; '.join(kb_info)}")

    # Step 4: Generate code using the LLM
    prompt = f"Generate Python code to {user_query} using the {selected_api['name']} (documentation: {docs_info}). Consider best practices: {'; '.join(kb_info)}. Include basic error handling."
    generated_code = llm_model.generate_code(prompt)
    response_log.append("\n--- Generated Code ---\n" + generated_code)

    # Step 5: Execute and Test the generated code (mocking)
    execution_result = mock_execute_code(generated_code)
    response_log.append("\n--- Code Execution Result ---\n" + execution_result)

    test_prompt = f"Generate a simple test for the following code:\n{generated_code}"
    generated_test_code = llm_model.generate_code(test_prompt)
    response_log.append("\n--- Generated Test Code ---\n" + generated_test_code)

    test_result = mock_test_code(generated_test_code + "\n" + generated_code) # Combine for context
    response_log.append("\n--- Code Testing Result ---\n" + test_result)

    return "\n".join(response_log)

# --- 5. User Interface (Gradio) ---

if __name__ == "__main__":
    gr.Interface(
        fn=medical_api_integration_agent,
        inputs=gr.Textbox(lines=2, placeholder="e.g., 'Integrate with an EHR system to fetch patient data' or 'Send lab results to a diagnostic API'"),
        outputs=gr.Textbox(lines=20),
        title="Medical API Integration Assistant",
        description="An AI assistant to help healthcare providers integrate various medical APIs."
    ).launch()
