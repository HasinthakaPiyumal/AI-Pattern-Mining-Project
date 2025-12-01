import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import HuggingFacePipeline

# 2. Knowledge Base Management
medical_documents = [
    "Aspirin is commonly used for pain relief, fever reduction, and anti-inflammatory purposes. It's also used to prevent blood clots.",
    "Diabetes mellitus is a chronic metabolic disease characterized by high blood sugar levels. Type 1 diabetes is an autoimmune condition, while Type 2 is often linked to lifestyle.",
    "Hypertension, or high blood pressure, is a common condition where the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease.",
    "Common cold symptoms include a runny nose, sore throat, cough, and congestion. It is caused by viruses and usually resolves within a week.",
    "The human heart has four chambers: two atria and two ventricles. It pumps blood throughout the body, supplying oxygen and nutrients to the tissues.",
    "Cancer is a disease caused by an uncontrolled division of abnormal cells in a part of the body. Treatments include chemotherapy, radiation, and surgery.",
    "Antibiotics are medications that fight bacterial infections. They do not work against viral infections like the common cold or flu.",
    "Vaccines stimulate the body's immune system to protect against infection or disease. They are crucial for public health.",
    "Asthma is a chronic respiratory condition characterized by inflammation and narrowing of the airways, leading to difficulty breathing.",
    "Osteoporosis is a condition that causes bones to become weak and brittle, making them more susceptible to fractures. Calcium and Vitamin D are important for bone health."
]

# Load SentenceTransformer model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings
doc_embeddings = embedding_model.encode(medical_documents, convert_to_tensor=True).cpu().numpy()

# Create FAISS index
dimension = doc_embeddings.shape[1]
faiss_index = faiss.IndexFlatIP(dimension) # Using Inner Product for similarity
faiss_index.add(doc_embeddings)

# 3. Document Retrieval Module
def retrieve_documents(question: str, top_k: int = 3) -> list[str]:
    question_embedding = embedding_model.encode(question, convert_to_tensor=True).cpu().numpy().reshape(1, -1)
    distances, indices = faiss_index.search(question_embedding, top_k)
    retrieved_docs = [medical_documents[i] for i in indices[0]]
    return retrieved_docs

# 4. Prompt Construction Module
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a medical information assistant. Answer the question based ONLY on the provided context. If the answer is not in the context, state that you don't have enough information."),
        ("user", "Context: {context}\n\nQuestion: {question}"),
    ]
)

# 5. Large Language Model (LLM) Interaction Module
# Using a small, local transformers model for demonstration
llm_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    max_new_tokens=100, # Increased for potentially longer answers
    device=-1, # -1 for CPU, 0 for GPU
)

llm = HuggingFacePipeline(pipeline=llm_pipeline)

# Main Application Script
if __name__ == "__main__":
    print("Medical Information Assistant initialized. Ask a medical question or type 'exit' to quit.")
    while True:
        user_question = input("\nYour question: ")
        if user_question.lower() == 'exit':
            break

        # 2. Document Retrieval
        relevant_docs = retrieve_documents(user_question, top_k=3)
        context = "\n".join(relevant_docs)
        # print(f"\nRetrieved Context:\n{context}\n") # For debugging purposes

        # 3. Prompt Construction
        formatted_prompt = prompt_template.format(context=context, question=user_question)
        
        # 4. LLM Interaction
        try:
            response = llm.invoke(formatted_prompt)
            print(f"Assistant: {response}")
        except Exception as e:
            print(f"An error occurred with the LLM: {e}")
            print("Please ensure 'google/flan-t5-small' model is downloaded or accessible.")

