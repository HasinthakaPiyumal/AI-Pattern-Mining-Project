
medical_knowledge_graph = {
    "Metformin": {
        "treats": "Type 2 Diabetes",
        "side_effects": ["Lactic Acidosis", "Diarrhea", "Nausea", "Vomiting"],
        "contraindications": ["Kidney Insufficiency", "Liver Disease"],
    },
    "Type 2 Diabetes": {
        "symptoms": ["Frequent urination", "Increased thirst", "Weight loss"],
        "treatments": ["Metformin", "Insulin"],
    },
    "Kidney Insufficiency": {
        "associated_with": ["Hypertension", "Diabetes"],
        "impact_on_drugs": {"Metformin": "increased risk of Lactic Acidosis"},
    },
}

def llm_to_sparql(natural_language_question: str) -> str:
    """
    Simulates an LLM converting a natural language question into a SPARQL query.
    In a real application, this would involve an actual LLM call with a sophisticated prompt.
    """
    natural_language_question = natural_language_question.lower()

    if "side effects of metformin" in natural_language_question and "kidney insufficiency" in natural_language_question:
        return """
        SELECT ?sideEffect WHERE {
          dbr:Metformin dbo:side_effect ?sideEffect .
          dbr:Metformin dbo:contraindication dbr:Kidney_Insufficiency .
        }
        """
    elif "treatments for type 2 diabetes" in natural_language_question:
        return """
        SELECT ?treatment WHERE {
          dbr:Type_2_Diabetes dbo:treatment ?treatment .
        }
        """
    elif "contraindications of metformin" in natural_language_question:
        return """
        SELECT ?contraindication WHERE {
          dbr:Metformin dbo:contraindication ?contraindication .
        }
        """
    else:
        return f"Could not generate SPARQL for: '{natural_language_question}'. Please try a different query."

def execute_sparql_query(sparql_query: str, kg: dict) -> list:
    """
    Simulates executing a SPARQL query against a simple Python dictionary-based KG.
    This is a highly simplified executor and doesn't fully parse SPARQL.
    It looks for keywords and patterns to infer the intended query.
    """
    results = []
    if "SELECT ?sideEffect WHERE { dbr:Metformin dbo:side_effect ?sideEffect" in sparql_query:
        if "dbr:Kidney_Insufficiency" in sparql_query and "contraindication" in sparql_query:
            metformin_info = kg.get("Metformin", {})
            side_effects = metformin_info.get("side_effects", [])
            kidney_impact = kg.get("Kidney Insufficiency", {}).get("impact_on_drugs", {}).get("Metformin")
            if kidney_impact:
                results.append({"sideEffect": "Increased risk of Lactic Acidosis due to Kidney Insufficiency"})
            for se in side_effects:
                results.append({"sideEffect": se})
        else:
            metformin_info = kg.get("Metformin", {})
            for se in metformin_info.get("side_effects", []):
                results.append({"sideEffect": se})

    elif "SELECT ?treatment WHERE { dbr:Type_2_Diabetes dbo:treatment ?treatment" in sparql_query:
        diabetes_info = kg.get("Type 2 Diabetes", {})
        for t in diabetes_info.get("treatments", []):
            results.append({"treatment": t})

    elif "SELECT ?contraindication WHERE { dbr:Metformin dbo:contraindication ?contraindication" in sparql_query:
        metformin_info = kg.get("Metformin", {})
        for c in metformin_info.get("contraindications", []):
            results.append({"contraindication": c})
    else:
        results.append({"error": "Unsupported SPARQL query for this mock KG."})

    return results

def run_clinical_knowledge_navigator(question: str) -> dict:
    """
    Runs the Clinical Knowledge Navigator workflow for a given natural language question.
    """
    print(f"User Question: \"{question}\"")

    sparql_query = llm_to_sparql(question)
    print(f"\nGenerated SPARQL Query:\n```sparql\n{sparql_query}\n```")

    if "Could not generate SPARQL" in sparql_query:
        return {"answer": sparql_query}

    results = execute_sparql_query(sparql_query, medical_knowledge_graph)
    print(f"\nQuery Execution Results: {results}")

    if not results or "error" in results[0]:
        return {"answer": "I could not find a definitive answer based on the knowledge graph for your query."}

    formatted_answer = "Based on the medical knowledge graph:\n"
    if "sideEffect" in results[0]:
        side_effects = [r["sideEffect"] for r in results if "sideEffect" in r]
        formatted_answer += f"Common side effects of Metformin, especially considering kidney insufficiency: {", ".join(side_effects)}."
    elif "treatment" in results[0]:
        treatments = [r["treatment"] for r in results if "treatment" in r]
        formatted_answer += f"Treatments for Type 2 Diabetes include: {", ".join(treatments)}."
    elif "contraindication" in results[0]:
        contraindications = [r["contraindication"] for r in results if "contraindication" in r]
        formatted_answer += f"Contraindications for Metformin include: {", ".join(contraindications)}."
    else:
        formatted_answer += "The query returned unknown information."

    return {"answer": formatted_answer}

if __name__ == "__main__":
    test_questions = [
        "What are the common side effects of Metformin for type 2 diabetes patients with kidney insufficiency?",
        "What are treatments for type 2 diabetes?",
        "What are the contraindications of Metformin?",
        "Tell me about ibuprofen.",
    ]

    for q in test_questions:
        print("\n" + "="*80)
        response = run_clinical_knowledge_navigator(q)
        print("\nFinal Answer:")
        print(response["answer"])
        print("="*80 + "\n")
