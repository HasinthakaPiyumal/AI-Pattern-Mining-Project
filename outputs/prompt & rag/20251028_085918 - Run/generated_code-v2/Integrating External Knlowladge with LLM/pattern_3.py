import streamlit as st
from rdflib import Graph, Literal, URIRef, Namespace
from rdflib.namespace import RDF, RDFS

# 1. Medical Knowledge Graph Setup
def create_medical_kg():
    g = Graph()

    # Define Namespaces
    ex = Namespace("http://example.org/medical/")
    schema = Namespace("http://schema.org/")
    
    g.bind("ex", ex)
    g.bind("schema", schema)

    # Add medical facts
    # Metformin
    g.add((ex.Metformin, RDF.type, ex.Drug))
    g.add((ex.Metformin, RDFS.label, Literal("Metformin")))
    g.add((ex.Metformin, ex.treats, ex.Type2Diabetes))
    g.add((ex.Metformin, ex.hasSideEffect, ex.Nausea))
    g.add((ex.Metformin, ex.hasSideEffect, ex.Diarrhea))
    g.add((ex.Metformin, ex.hasSideEffect, ex.AbdominalPain))
    g.add((ex.Metformin, ex.hasSideEffect, ex.LacticAcidosis))
    
    # Type 2 Diabetes
    g.add((ex.Type2Diabetes, RDF.type, ex.Condition))
    g.add((ex.Type2Diabetes, RDFS.label, Literal("Type 2 Diabetes")))

    # Side Effects
    g.add((ex.Nausea, RDF.type, ex.SideEffect))
    g.add((ex.Nausea, RDFS.label, Literal("Nausea")))
    g.add((ex.Diarrhea, RDF.type, ex.SideEffect))
    g.add((ex.Diarrhea, RDFS.label, Literal("Diarrhea")))
    g.add((ex.AbdominalPain, RDF.type, ex.SideEffect))
    g.add((ex.AbdominalPain, RDFS.label, Literal("Abdominal Pain")))
    g.add((ex.LacticAcidosis, RDF.type, ex.SideEffect))
    g.add((ex.LacticAcidosis, RDFS.label, Literal("Lactic Acidosis")))

    # Other Drug (for testing general queries)
    g.add((ex.Aspirin, RDF.type, ex.Drug))
    g.add((ex.Aspirin, RDFS.label, Literal("Aspirin")))
    g.add((ex.Aspirin, ex.treats, ex.Headache))
    g.add((ex.Aspirin, ex.hasSideEffect, ex.StomachUpset))

    g.add((ex.Headache, RDF.type, ex.Condition))
    g.add((ex.Headache, RDFS.label, Literal("Headache")))
    g.add((ex.StomachUpset, RDF.type, ex.SideEffect))
    g.add((ex.StomachUpset, RDFS.label, Literal("Stomach Upset")))

    return g

medical_kg = create_medical_kg()

# 2. Semantic Parsing Layer (Mock LLM)
def nl_to_sparql(question: str) -> str:
    question = question.lower()
    sparql_query = ""

    if "side effects of metformin" in question and "type 2 diabetes" in question:
        sparql_query = """
PREFIX ex: <http://example.org/medical/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT DISTINCT ?sideEffectLabel
WHERE {
    ex:Metformin ex:hasSideEffect ?sideEffect .
    ?sideEffect rdfs:label ?sideEffectLabel .
}
"""
    elif "side effects of" in question:
        drug_name = question.split("side effects of")[-1].split("for")[-1].strip().replace("?", "").capitalize()
        # Simple mapping for demo purposes
        if drug_name == "Metformin":
            drug_uri = "ex:Metformin"
        elif drug_name == "Aspirin":
            drug_uri = "ex:Aspirin"
        else:
            drug_uri = None

        if drug_uri:
            sparql_query = f"""
PREFIX ex: <http://example.org/medical/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT DISTINCT ?sideEffectLabel
WHERE {{
    {drug_uri} ex:hasSideEffect ?sideEffect .
    ?sideEffect rdfs:label ?sideEffectLabel .
}}
"""
    elif "treats" in question and "what drug" in question:
        condition_name = question.split("treats")[-1].replace("?", "").strip().replace("condition", "").replace("drug", "").strip().capitalize()
        if condition_name == "Type 2 Diabetes":
            condition_uri = "ex:Type2Diabetes"
        elif condition_name == "Headache":
            condition_uri = "ex:Headache"
        else:
            condition_uri = None
        
        if condition_uri:
            sparql_query = f"""
PREFIX ex: <http://example.org/medical/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT DISTINCT ?drugLabel
WHERE {{
    ?drug ex:treats {condition_uri} .
    ?drug rdfs:label ?drugLabel .
}}
"""
    else:
        sparql_query = "# No specific SPARQL query generated for this question based on current rules."

    return sparql_query

# 3. SPARQL Query Engine
def execute_sparql_query(graph: Graph, sparql_query: str):
    try:
        results = graph.query(sparql_query)
        return results
    except Exception as e:
        st.error(f"Error executing SPARQL query: {e}")
        return None

# 4. Result Formatter
def format_results(query_results):
    if not query_results:
        return "No results found or an error occurred."

    formatted_output = []
    for row in query_results:
        # Assuming single variable queries for simplicity (e.g., sideEffectLabel, drugLabel)
        if row:
            formatted_output.append(str(row[0])) 
            
    if formatted_output:
        return "; ".join(formatted_output)
    else:
        return "No specific information found for your query."

# 5. User Interface (Streamlit Application)
st.title("⚕️ Medical Diagnostic Assistant (KGQA Demo)")
st.markdown("Ask natural language questions about drugs, conditions, and their relationships.")

user_question = st.text_input(
    "Your Question:", 
    "What are the common side effects of Metformin for patients with Type 2 Diabetes?"
)

if st.button("Get Answer"):
    if user_question:
        st.subheader("Processing...")

        # Semantic Parsing
        sparql_query = nl_to_sparql(user_question)
        st.code(sparql_query, language="sparql")

        if "# No specific SPARQL query" not in sparql_query:
            # Execute Query
            results = execute_sparql_query(medical_kg, sparql_query)
            
            # Format Results
            formatted_answer = format_results(results)
            
            st.subheader("Answer:")
            st.success(formatted_answer)
        else:
            st.warning("Could not generate a specific SPARQL query for your question. Please try rephrasing or a different question relevant to the available medical facts (e.g., side effects of Metformin/Aspirin, drugs that treat Type 2 Diabetes/Headache).")
    else:
        st.warning("Please enter a question.")