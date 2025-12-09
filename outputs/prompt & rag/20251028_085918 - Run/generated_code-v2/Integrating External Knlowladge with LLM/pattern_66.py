import streamlit as st
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF, RDFS

# kg_utils.py content
def create_medical_knowledge_graph():
    g = Graph()

    # Define namespaces/URIs
    EX = URIRef("http://example.org/medical/")
    DRUG = URIRef(EX + "Drug/")
    CONDITION = URIRef(EX + "Condition/")
    INTERACTION = URIRef(EX + "Interaction/")
    CONTRAINDICATION = URIRef(EX + "Contraindication/")

    # Add some example data
    # Drugs
    aspirin = DRUG + "Aspirin"
    warfarin = DRUG + "Warfarin"
    ibuprofen = DRUG + "Ibuprofen"
    amoxicillin = DRUG + "Amoxicillin"

    g.add((aspirin, RDF.type, DRUG))
    g.add((aspirin, RDFS.label, Literal("Aspirin")))
    g.add((warfarin, RDF.type, DRUG))
    g.add((warfarin, RDFS.label, Literal("Warfarin")))
    g.add((ibuprofen, RDF.type, DRUG))
    g.add((ibuprofen, RDFS.label, Literal("Ibuprofen")))
    g.add((amoxicillin, RDF.type, DRUG))
    g.add((amoxicillin, RDFS.label, Literal("Amoxicillin")))

    # Conditions
    heart_disease = CONDITION + "HeartDisease"
    asthma = CONDITION + "Asthma"
    bleeding_disorder = CONDITION + "BleedingDisorder"
    bacterial_infection = CONDITION + "BacterialInfection"
    stomach_ulcer = CONDITION + "StomachUlcer"

    g.add((heart_disease, RDF.type, CONDITION))
    g.add((heart_disease, RDFS.label, Literal("Heart Disease")))
    g.add((asthma, RDF.type, CONDITION))
    g.add((asthma, RDFS.label, Literal("Asthma")))
    g.add((bleeding_disorder, RDF.type, CONDITION))
    g.add((bleeding_disorder, RDFS.label, Literal("Bleeding Disorder")))
    g.add((bacterial_infection, RDF.type, CONDITION))
    g.add((bacterial_infection, RDFS.label, Literal("Bacterial Infection")))
    g.add((stomach_ulcer, RDF.type, CONDITION))
    g.add((stomach_ulcer, RDFS.label, Literal("Stomach Ulcer")))

    # Contraindications
    g.add((aspirin, CONTRAINDICATION + "for", bleeding_disorder))
    g.add((warfarin, CONTRAINDICATION + "for", bleeding_disorder))
    g.add((ibuprofen, CONTRAINDICATION + "for", stomach_ulcer))
    g.add((ibuprofen, CONTRAINDICATION + "with", warfarin))
    g.add((aspirin, CONTRAINDICATION + "with", warfarin))

    g.add((amoxicillin, EX + "treats", bacterial_infection))

    return g

def execute_sparql_query(graph, query):
    results = graph.query(query)
    return results

# llm_parser.py content (Mock LLM)
def natural_language_to_sparql(question):
    question_lower = question.lower()
    
    if "contraindications for aspirin" in question_lower:
        return """PREFIX ex: <http://example.org/medical/>
SELECT ?conditionLabel
WHERE {
    ex:Drug/Aspirin ex:Contraindication/for ?condition .
    ?condition rdfs:label ?conditionLabel .
}"""
    elif "contraindications for ibuprofen with warfarin" in question_lower or "ibuprofen and warfarin interaction" in question_lower:
        return """PREFIX ex: <http://example.org/medical/>
SELECT ?contraindicationType ?target
WHERE {
    ex:Drug/Ibuprofen ex:Contraindication/with ex:Drug/Warfarin .
    ex:Drug/Ibuprofen ex:Contraindication/with ?target .
    BIND(IF(?target = ex:Drug/Warfarin, "contraindicated with", "unknown") AS ?contraindicationType)
}"""
    elif "what treats bacterial infection" in question_lower:
        return """PREFIX ex: <http://example.org/medical/>
SELECT ?drugLabel
WHERE {
    ?drug ex:treats ex:Condition/BacterialInfection .
    ?drug rdfs:label ?drugLabel .
}"""
    elif "what conditions is warfarin contraindicated for" in question_lower:
        return """PREFIX ex: <http://example.org/medical/>
SELECT ?conditionLabel
WHERE {
    ex:Drug/Warfarin ex:Contraindication/for ?condition .
    ?condition rdfs:label ?conditionLabel .
}"""
    elif "all drugs" in question_lower:
        return """PREFIX ex: <http://example.org/medical/>
SELECT ?drugLabel
WHERE {
    ?drug rdf:type ex:Drug/ .
    ?drug rdfs:label ?drugLabel .
}"""
    elif "all conditions" in question_lower:
        return """PREFIX ex: <http://example.org/medical/>
SELECT ?conditionLabel
WHERE {
    ?condition rdf:type ex:Condition/ .
    ?condition rdfs:label ?conditionLabel .
}"""
    else:
        return None

# app.py content

st.set_page_config(layout="wide")
st.title("💊 Medical Knowledge Navigator")

# Initialize KG (only once)
if "medical_kg" not in st.session_state:
    st.session_state.medical_kg = create_medical_knowledge_graph()
    st.success("Medical Knowledge Graph loaded!")

question_input = st.text_area("Ask a medical question:", "What are the contraindications for Aspirin?")

if st.button("Get Answer"):
    if question_input:
        sparql_query = natural_language_to_sparql(question_input)

        if sparql_query:
            st.subheader("Generated SPARQL Query:")
            st.code(sparql_query, language="sparql")

            results = execute_sparql_query(st.session_state.medical_kg, sparql_query)

            if results:
                st.subheader("Answer:")
                result_list = []
                for row in results:
                    row_str = []
                    for key, value in row.asdict().items():
                        # Extract string value from Literal or URIRef
                        val_str = str(value.toPython()) if isinstance(value, Literal) else str(value.split('/')[-1]) if isinstance(value, URIRef) else str(value)
                        row_str.append(f"{key}: {val_str}")
                    result_list.append(", ".join(row_str))
                
                if result_list:
                    for res in result_list:
                        st.write(f"- {res}")
                else:
                    st.info("No results found for this query.")
            else:
                st.info("No results found for this query.")
        else:
            st.warning("Could not convert your question to a SPARQL query. Please try rephrasing.")
    else:
        st.warning("Please enter a question.")

st.markdown("""---
**Example Questions:**
*   What are the contraindications for Aspirin?
*   What conditions is Warfarin contraindicated for?
*   What treats bacterial infection?
*   What are all drugs?
*   What are all conditions?
*   Tell me about Ibuprofen and Warfarin interaction.
""")
