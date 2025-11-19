import streamlit as st
from langchain.llms import FakeListLLM
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser, RunnablePassthrough
from langchain.chains import LLMChain

# 1. Mock Knowledge Graph (Simplified for demonstration)
medical_knowledge_graph = [
    ("drug X", "contraindicated with", "condition Y", "due to severe allergic reaction"),
    ("drug X", "interacts with", "medication Z", "leading to reduced efficacy"),
    ("condition Y", "symptom", "fever"),
    ("medication Z", "class", "anticoagulant"),
    ("drug A", "treatment for", "condition B"),
    ("drug A", "side effect", "nausea"),
    ("condition Y", "treatment", "rest and hydration")
]

# 2. Natural Language Understanding (NLU) Module - Simplified Entity Extraction
def extract_entities(query):
    entities = {"drugs": [], "conditions": [], "medications": []}
    # Simple keyword matching for demonstration
    # In a real application, this would use a fine-tuned NER model (e.g., from transformers/spaCy)
    all_drugs = [item[0] for item in medical_knowledge_graph if item[0].startswith('drug')]
    all_conditions = [item[0] for item in medical_knowledge_graph if item[0].startswith('condition')]
    all_medications = [item[0] for item in medical_knowledge_graph if item[0].startswith('medication')]

    for drug in set(all_drugs):
        if drug.lower() in query.lower():
            entities["drugs"].append(drug)
    for condition in set(all_conditions):
        if condition.lower() in query.lower():
            entities["conditions"].append(condition)
    for medication in set(all_medications):
        if medication.lower() in query.lower():
            entities["medications"].append(medication)
            
    return entities

# 3. Knowledge Graph (KG) Module - Retrieval
def retrieve_kg_data(entities):
    relevant_facts = []
    search_terms = []
    for entity_type in entities:
        search_terms.extend(entities[entity_type])

    for fact in medical_knowledge_graph:
        if any(term.lower() in str(fact).lower() for term in search_terms):
            relevant_facts.append(" - ".join(fact))
    return "\n".join(relevant_facts) if relevant_facts else "No specific medical facts found for the given entities."

# 4. Large Language Model (LLM) Integration Module - Simulated LLM
# Using FakeListLLM to simulate LLM responses without actual API calls
llm = FakeListLLM(responses=[
    "Based on the medical knowledge, drug X is contraindicated with condition Y due to severe allergic reaction. Additionally, drug X interacts with medication Z, leading to reduced efficacy. For condition Y, rest and hydration are common treatments.",
    "Drug A is a treatment for condition B, but a common side effect is nausea.",
    "No specific contraindications or interactions found for this query in our current knowledge base, but always consult a medical professional.",
    "The query involves medical entities and requires careful consideration of interactions and contraindications. Please provide more context."
])

# 5. RAG Orchestration
# Define prompt template
rag_prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful healthcare AI assistant. Answer the medical question accurately based on the provided context. If the context does not contain enough information, state that you cannot answer fully."),
    ("user", "Medical Context:\n{context}\n\nQuestion: {question}")
])

# Build the RAG chain
def create_rag_chain():
    return (
        RunnablePassthrough.assign(
            context=lambda x: retrieve_kg_data(extract_entities(x["question"]))
        )
        | rag_prompt_template
        | llm
        | StrOutputParser()
    )

rag_chain = create_rag_chain()

# 6. User Interface (UI) Module - Streamlit App
st.title("Healthcare AI Assistant")
st.write("Ask complex medical questions to get AI-powered insights.")

user_question = st.text_area("Enter your medical question here:", height=100)

if st.button("Get Answer"):
    if user_question:
        with st.spinner("Retrieving and reasoning..."):
            response = rag_chain.invoke({"question": user_question})
            st.subheader("AI Assistant's Answer:")
            st.write(response)
    else:
        st.warning("Please enter a medical question.")
