
define_project_architecture_response = {
    "output": "Define the architecture for the AI application idea. Use suitable framworks and libraries to implement this AI App.\n Use frameworks libraries if need such as tensorflow, pytorch, jax, scikit-learn, lightgbm, xgboost, catboost, fastai, rapids-cuml, transformers, sentence-transformers, tokenizers, spacy, nltk, gensim, trl, accelerate, vllm, langchain, llama-index, chroma, faiss, weaviate, pinecone, milvus, qdrant, elasticsearch, haystack, pandas, numpy, dask, polars, datasets, pyarrow, langgraph, autogen, crewai, opendevin, dspy, semantic-kernel, langsmith, promptlayer, wandb, trulens, evals, guardrails-ai, pydantic, gradio, streamlit, openai, instructor-embedding, cohere, text2vec, clip, openclip, blip, blip2, lavis, diffusers, torchvision, opencv-python, fastapi, ray, bentoml, onnxruntime, tensorrt, tqdm, rich, loguru, python-dotenv, joblib, networkx, phidata, openaimultiswarm, lcel, memgpt, vectorhub, llmdatahub... Here is project idea:\n\nA Medical Diagnosis Assistant that uses Retrieval-Augmented Generation (RAG) with a medical Knowledge Graph to provide accurate, evidence-based responses to physician queries, reducing diagnostic errors and improving treatment recommendations. When a physician queries the assistant (e.g., \"What are common treatments for type 2 diabetes with kidney complications?\" ), the system first retrieves relevant facts, triples, and pathways from the medical KG related to type 2 diabetes, kidney complications, and their treatments. These retrieved, factual medical details serve as context and are then provided to a large language model. The LLM, grounded by the retrieved KG facts, generates a precise, evidence-based answer, including potential treatment options, contraindications, and relevant clinical guidelines."
}

# --- 1. Simulate a Medical Knowledge Graph (KG) --- 
# In a real-world scenario, this would be a sophisticated graph database (e.g., Neo4j)
# or a large set of structured medical data. For this example, we use a simple
# list of facts represented as dictionaries.
medical_knowledge_graph = [
    {"subject": "Type 2 Diabetes", "predicate": "has_treatment", "object": "Metformin"},
    {"subject": "Type 2 Diabetes", "predicate": "has_treatment", "object": "Insulin Therapy"},
    {"subject": "Type 2 Diabetes", "predicate": "can_cause", "object": "Kidney Complications"},
    {"subject": "Type 2 Diabetes", "predicate": "can_cause", "object": "Neuropathy"},
    {"subject": "Kidney Complications", "predicate": "related_to_condition", "object": "Type 2 Diabetes"},
    {"subject": "Kidney Complications", "predicate": "treatment_considerations_include", "object": "Renal function monitoring"},
    {"subject": "Kidney Complications", "predicate": "treatment_considerations_include", "object": "SGLT2 inhibitors"},
    {"subject": "SGLT2 inhibitors", "predicate": "is_a_type_of", "object": "Antidiabetic medication"},
    {"subject": "SGLT2 inhibitors", "predicate": "benefit_for", "object": "Kidney Complications"},
    {"subject": "Metformin", "predicate": "contraindicated_in", "object": "Severe renal impairment"},
    {"subject": "Insulin Therapy", "predicate": "may_be_required_for", "object": "Poorly controlled diabetes"},
    {"subject": "Diabetic Nephropathy", "predicate": "is_a_type_of", "object": "Kidney Complications"},
    {"subject": "Diabetic Nephropathy", "predicate": "management_involves", "object": "Blood pressure control"},
    {"subject": "Diabetic Nephropathy", "predicate": "management_involves", "object": "Glycemic control"},
    {"subject": "Blood pressure control", "predicate": "related_treatment", "object": "ACE inhibitors"},
    {"subject": "Blood pressure control", "predicate": "related_treatment", "object": "ARBs"},
]

# --- 2. Information Retrieval from KG --- 
def retrieve_from_medical_kg(query_keywords, knowledge_graph):
    """
    Retrieves relevant facts from the medical knowledge graph based on query keywords.
    In a more advanced system, this would involve graph traversal, embedding matching,
    or more sophisticated semantic search.
    """
    relevant_facts = []
    query_lower = [kw.lower() for kw in query_keywords]

    for fact in knowledge_graph:
        subject_lower = fact["subject"].lower()
        predicate_lower = fact["predicate"].lower()
        object_lower = fact["object"].lower()
        
        # Check if any part of the fact contains any of the query keywords
        if any(keyword in subject_lower for keyword in query_lower) or \
           any(keyword in predicate_lower for keyword in query_lower) or \
           any(keyword in object_lower for keyword in query_lower):
            relevant_facts.append(fact)
            
    # Also, retrieve facts related to the objects found in the initial search
    # This simulates a shallow graph traversal
    secondary_facts = []
    for fact in relevant_facts:
        object_entity = fact["object"]
        for secondary_fact in knowledge_graph:
            if secondary_fact["subject"] == object_entity and secondary_fact not in relevant_facts:
                secondary_facts.append(secondary_fact)
                
    return list(relevant_facts + secondary_facts)

# --- 3. Large Language Model (LLM) Placeholder --- 
# In a real application, this would be an API call to a powerful LLM like GPT-4, Gemini, etc.
def generate_response_with_llm(prompt, context_facts):
    """
    Simulates an LLM generating a response based on a prompt and retrieved context.
    """
    context_str = "\n".join([
        f"- {fact['subject']} {fact['predicate'].replace('_', ' ')} {fact['object']}."
        for fact in context_facts
    ])
    
    # Basic prompt engineering to integrate context
    if context_str:
        full_prompt = f"""Based on the following medical facts from a knowledge graph, answer the question accurately and concisely. Prioritize information from the facts provided, and mention any contraindications or specific considerations:

Medical Facts:
{context_str}

Question: {prompt}

Answer:"""
    else:
        full_prompt = f"""Answer the following medical question. If specific factual knowledge is missing, indicate that.

Question: {prompt}

Answer:"""

    # Simulate LLM response logic based on context and prompt
    response_lines = []
    response_lines.append("### Medical Diagnosis Assistant Response ###")

    if "type 2 diabetes" in prompt.lower() and "kidney complications" in prompt.lower():
        if any("Metformin" in fact["object"] and "Severe renal impairment" in fact["object"] for fact in context_facts):
            response_lines.append("For Type 2 Diabetes with kidney complications, Metformin is a common treatment, but it is contraindicated in severe renal impairment.")
        elif any("Metformin" in fact["object"] for fact in context_facts):
             response_lines.append("Common treatments for Type 2 Diabetes include Metformin.")
        if any("SGLT2 inhibitors" in fact["object"] and "Kidney Complications" in fact["subject"] for fact in context_facts):
            response_lines.append("SGLT2 inhibitors are often beneficial for patients with Type 2 Diabetes and kidney complications, as they are a type of antidiabetic medication that also benefits kidney function.")
        if any("Diabetic Nephropathy" in fact["object"] and "management_involves" in fact["predicate"] for fact in context_facts):
            response_lines.append("Management may involve blood pressure control with ACE inhibitors or ARBs, and glycemic control to address Diabetic Nephropathy.")

    if not response_lines or len(response_lines) == 1: # Only header is present
        response_lines.append("I can provide information based on the medical facts available. If the specific details for your exact query are not directly in my knowledge base, I will state general relevant information.")
        if context_str:
            response_lines.append(f"Relevant facts found: {context_str}")
        else:
            response_lines.append("No highly relevant facts were retrieved for this specific query. Please try rephrasing.")
            
    return "\n".join(response_lines)

# --- 4. Main Medical Diagnosis Assistant Function (RAG Orchestration) --- 
def medical_diagnosis_assistant(query, knowledge_graph):
    """
    Orchestrates the RAG process: retrieves facts from KG and then uses an LLM to generate a response.
    """
    # 1. Identify key terms for retrieval (simple keyword extraction for this example)
    # In a real system, this would involve NLP techniques like named entity recognition (NER).
    query_keywords = [
        word for word in query.lower().replace("?", "").replace(".", "").split()
        if len(word) > 2 and word not in ["what", "are", "common", "for", "with", "and", "the"]
    ]

    print(f"\n--- User Query: {query} ---")
    print(f"Identified Keywords for Retrieval: {query_keywords}")

    # 2. Retrieve relevant medical facts from the KG
    retrieved_facts = retrieve_from_medical_kg(query_keywords, knowledge_graph)
    print("\n--- Retrieved Facts from KG ---")
    if retrieved_facts:
        for fact in retrieved_facts:
            print(f"  - {fact['subject']} {fact['predicate'].replace('_', ' ')} {fact['object']}")
    else:
        print("  No directly relevant facts found in the knowledge graph.")

    # 3. Generate response using the LLM (placeholder) with retrieved context
    llm_response = generate_response_with_llm(query, retrieved_facts)
    print("\n--- LLM Generated Response (Augmented) ---")
    print(llm_response)
    return llm_response

# --- Example Usage --- 
if __name__ == "__main__":
    # Example Query 1: Specific treatment with complications
    query1 = "What are common treatments for type 2 diabetes with kidney complications?"
    medical_diagnosis_assistant(query1, medical_knowledge_graph)

    # Example Query 2: General query about a condition
    query2 = "What are treatments for Type 2 Diabetes?"
    medical_diagnosis_assistant(query2, medical_knowledge_graph)

    # Example Query 3: Specific contraindication
    query3 = "When is Metformin contraindicated for kidney issues?"
    medical_diagnosis_assistant(query3, medical_knowledge_graph)
    
    # Example Query 4: Query with less direct match
    query4 = "Tell me about managing Diabetic Nephropathy."
    medical_diagnosis_assistant(query4, medical_knowledge_graph)

    # Example Query 5: Query with no direct match in KG (to show fallback)
    query5 = "What is the best exercise for heart health?"
    medical_diagnosis_assistant(query5, medical_knowledge_graph)
