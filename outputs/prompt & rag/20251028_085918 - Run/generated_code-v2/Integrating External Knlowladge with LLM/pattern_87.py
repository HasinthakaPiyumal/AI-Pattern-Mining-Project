import streamlit as st
import json
from langchain.prompts import PromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Mock LLM for demonstration purposes
# In a real application, you would replace this with an actual LLM client, e.g., ChatOpenAI
class MockLLM:
    def invoke(self, prompt: str) -> str:
        if "drug interactions for" in prompt.lower():
            drug = prompt.split("for")[-1].strip().replace("?", "").replace(".", "")
            return f"SELECT ?interaction WHERE {{ <http://example.org/drug/{drug.replace(' ', '_')}> <http://example.org/ontology/interactsWith> ?interaction . }}"
        elif "treatment for" in prompt.lower():
            condition = prompt.split("for")[-1].strip().replace("?", "").replace(".", "")
            return f"SELECT ?treatment WHERE {{ <http://example.org/condition/{condition.replace(' ', '_')}> <http://example.org/ontology/hasTreatment> ?treatment . }}"
        elif "symptoms of" in prompt.lower():
            condition = prompt.split("of")[-1].strip().replace("?", "").replace(".", "")
            return f"SELECT ?symptom WHERE {{ <http://example.org/condition/{condition.replace(' ', '_')}> <http://example.org/ontology/hasSymptom> ?symptom . }}"
        else:
            return "SELECT ?result WHERE { ?s ?p ?o . }"

# Mock SPARQL endpoint execution
def execute_sparql_query_mock(sparql_query: str) -> str:
    # This function simulates querying a knowledge graph.
    # In a real application, you would use a library like sparqlwrapper
    # to connect to a Virtuoso, Blazegraph, or other SPARQL endpoint.

    if "drug/Aspirin" in sparql_query and "interactsWith" in sparql_query:
        return json.dumps([
            {"interaction": {"value": "Warfarin"}},
            {"interaction": {"value": "Ibuprofen"}}
        ])
    elif "condition/Diabetes" in sparql_query and "hasTreatment" in sparql_query:
        return json.dumps([
            {"treatment": {"value": "Metformin"}},
            {"treatment": {"value": "Insulin Therapy"}},
            {"treatment": {"value": "Dietary Changes"}}
        ])
    elif "condition/Flu" in sparql_query and "hasSymptom" in sparql_query:
        return json.dumps([
            {"symptom": {"value": "Fever"}},
            {"symptom": {"value": "Cough"}},
            {"symptom": {"value": "Body aches"}}
        ])
    else:
        return json.dumps([
            {"result": {"value": "No specific information found for this query based on mock data."}}
        ])

# Streamlit UI
st.title("Medical Information Query System")
st.write("Ask natural language questions about medical conditions, drug interactions, and treatment protocols.")
st.info("Note: This system uses mocked LLM and SPARQL execution for demonstration purposes. Replace `MockLLM` with an actual LLM (e.g., `ChatOpenAI`) and `execute_sparql_query_mock` with a real SPARQL client for production use.")

user_question = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if user_question:
        st.subheader("Processing your question...")

        # Step 1: LLM generates SPARQL query
        llm = MockLLM()  # Replace with actual LLM in production (e.g., ChatOpenAI(temperature=0, model="gpt-4"))
        
        sparql_prompt_template = PromptTemplate(
            input_variables=["question"],
            template="You are a helpful assistant that converts natural language medical questions into SPARQL queries. Given the question: '{question}', generate a SPARQL query that can be executed against a medical knowledge graph. Focus on common medical entities like conditions, drugs, and treatments. If specific entities are mentioned, try to map them to a URI structure like http://example.org/drug/DrugName or http://example.org/condition/ConditionName. Be concise and provide only the SPARQL query."
        )

        sparql_generator_chain = (
            {"question": RunnablePassthrough()}
            | sparql_prompt_template
            | llm  # This is where your actual LLM would be invoked
            | StrOutputParser()
        )

        generated_sparql_query = sparql_generator_chain.invoke(user_question)
        
        st.subheader("Generated SPARQL Query:")
        st.code(generated_sparql_query, language="sparql")

        # Step 2: Execute SPARQL query against the Knowledge Graph
        st.subheader("Executing SPARQL Query (Mocked):")
        query_results_json = execute_sparql_query_mock(generated_sparql_query)
        
        st.subheader("Results:")
        try:
            parsed_results = json.loads(query_results_json)
            if parsed_results:
                for item in parsed_results:
                    for key, value_dict in item.items():
                        st.write(f"- **{key.capitalize()}**: {value_dict['value']}")
            else:
                st.write("No specific results found for this query in the mock data.")
        except json.JSONDecodeError:
            st.error("Error parsing SPARQL results. Raw output:")
            st.code(query_results_json)
    else:
        st.warning("Please enter a question.")