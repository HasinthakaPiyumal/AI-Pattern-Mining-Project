import json

def simulate_llm_entity_extraction(query: str) -> str:
    """
    Simulates an LLM call to extract medical entities from a query.
    In a real application, this would involve an actual LLM API call.
    """
    # A simple mapping for demonstration purposes
    if "diabetes" in query.lower() and "symptoms" in query.lower():
        return json.dumps({"entities": ["diabetes", "symptoms"]})
    elif "insulin" in query.lower() and "side effects" in query.lower():
        return json.dumps({"entities": ["insulin", "side effects"]})
    elif "heart disease" in query.lower():
        return json.dumps({"entities": ["heart disease", "treatment"]})
    elif "ibuprofen" in query.lower():
        return json.dumps({"entities": ["ibuprofen", "uses"]})
    else:
        return json.dumps({"entities": ["general medical term"]})

def extract_entities_from_llm_response(llm_response_json: str) -> list:
    """
    Parses the JSON response from the simulated LLM to extract entities.
    """
    try:
        response_data = json.loads(llm_response_json)
        return response_data.get("entities", [])
    except json.JSONDecodeError:
        print("Error: Could not parse LLM response.")
        return []

medical_knowledge_graph = {
    "diabetes": {
        "type": "disease",
        "symptoms": ["frequent urination", "increased thirst", "fatigue", "blurred vision"],
        "treatments": ["insulin therapy", "lifestyle changes", "metformin"],
        "related_conditions": ["obesity", "heart disease", "kidney disease"]
    },
    "insulin": {
        "type": "drug",
        "uses": ["treats diabetes", "lowers blood sugar"],
        "side_effects": ["hypoglycemia", "weight gain", "allergic reactions"]
    },
    "heart disease": {
        "type": "disease",
        "symptoms": ["chest pain", "shortness of breath", "fatigue"],
        "treatments": ["medication", "surgery", "lifestyle changes"],
        "related_conditions": ["diabetes", "high blood pressure"]
    },
    "ibuprofen": {
        "type": "drug",
        "uses": ["pain relief", "fever reduction", "inflammation"],
        "side_effects": ["stomach upset", "headache", "dizziness"]
    },
    "frequent urination": {
        "type": "symptom",
        "associated_diseases": ["diabetes", "urinary tract infection"]
    },
    "metformin": {
        "type": "drug",
        "uses": ["treats type 2 diabetes"],
        "side_effects": ["nausea", "diarrhea"]
    }
}

def query_knowledge_graph(entities: list) -> dict:
    """
    Simulates querying a medical knowledge graph based on extracted entities.
    """
    results = {}
    if not entities:
        return {"message": "No specific entities found to query the KG."}

    for entity in entities:
        entity_lower = entity.lower()
        found_data = False
        for kg_entity, data in medical_knowledge_graph.items():
            if entity_lower in kg_entity.lower():
                results[entity] = data
                found_data = True
                break
        if not found_data:
            results[entity] = {"message": f"No direct information found for '{entity}' in KG."}
    return results

def main():
    print("Welcome to the LLM-based Medical Research Query Assistant!")
    print("Type 'exit' to quit.")

    while True:
        user_query = input("\nEnter your medical research query: ")
        if user_query.lower() == 'exit':
            break

        print(f"\nProcessing query: '{user_query}'...")

        # 1. LLM-based Entity Extraction
        llm_raw_response = simulate_llm_entity_extraction(user_query)
        extracted_entities = extract_entities_from_llm_response(llm_raw_response)

        print("--- Extracted Entities ---")
        if extracted_entities:
            print(f"Identified entities: {', '.join(extracted_entities)}")
        else:
            print("No specific entities could be extracted.")

        # 2. Medical Knowledge Graph Query
        kg_results = query_knowledge_graph(extracted_entities)

        # 3. Result Aggregation and Presentation
        print("\n--- Knowledge Graph Results ---")
        if kg_results:
            for entity, data in kg_results.items():
                print(f"\nInformation for '{entity}':")
                if "message" in data:
                    print(f"  {data['message']}")
                else:
                    for key, value in data.items():
                        if isinstance(value, list):
                            print(f"  {key.replace('_', ' ').title()}: {', '.join(value)}")
                        else:
                            print(f"  {key.replace('_', ' ').title()}: {value}")
        else:
            print("No relevant information found in the knowledge graph.")

    print("Thank you for using the Medical Research Query Assistant!")

if __name__ == "__main__":
    main()