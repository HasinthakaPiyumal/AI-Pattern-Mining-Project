import gradio as gr
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage

# --- 1. Data Ingestion and Vector Store ---

# Placeholder for actual medical journals, clinical guidelines, and reputable medical databases.
# For demonstration, we'll use a small set of example text documents.
medical_texts = [
    Document(page_content="""
        Cough is a common symptom of many respiratory diseases, including the common cold, influenza, bronchitis, and pneumonia.
        Persistent cough lasting more than 8 weeks is considered chronic and may warrant further investigation.
        Smoking is a significant risk factor for chronic cough.
        Treatment often involves identifying and addressing the underlying cause.
        For acute cough, symptomatic relief with antitussives may be prescribed.
        """, metadata={"source": "American Thoracic Society Guidelines", "url": "http://atsjournals.org/cough"}),
    Document(page_content="""
        Fever, defined as a body temperature above 100.4°F (38°C), is a frequent sign of infection or inflammation.
        It can also be caused by drug reactions, autoimmune disorders, and certain cancers.
        In children, fever should be monitored carefully.
        Antipyretics like acetaminophen or ibuprofen are commonly used to reduce fever.
        However, the primary goal is to treat the underlying cause of the fever.
        """, metadata={"source": "Mayo Clinic", "url": "https://www.mayoclinic.org/fever"}),
    Document(page_content="""
        Headache is a very common condition, with tension headaches, migraines, and cluster headaches being the most prevalent types.
        Tension headaches are often characterized by a band-like pressure sensation.
        Migraines typically involve throbbing pain, often on one side of the head, accompanied by nausea, vomiting, and sensitivity to light and sound.
        Red flag symptoms for headaches that require urgent medical attention include sudden onset of severe headache, headache with fever and stiff neck, or headache following head trauma.
        """, metadata={"source": "World Health Organization", "url": "https://www.who.int/headache"}),
    Document(page_content="""
        Shortness of breath, also known as dyspnea, can be a symptom of heart or lung problems. It can occur suddenly or gradually.
        Conditions like asthma, COPD, heart failure, and anxiety can cause shortness of breath. Emergency care is needed if it is severe or accompanied by chest pain.
        Pulmonary embolism is a serious cause of sudden shortness of breath. Diagnosis often involves imaging tests like CT pulmonary angiography.
        """, metadata={"source": "National Heart, Lung, and Blood Institute", "url": "https://www.nhlbi.nih.gov/health-topics/shortness-breath"})
]

# Text Splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
all_splits = text_splitter.split_documents(medical_texts)

# Embedding Model
# Using a common sentence-transformers model. Ensure it's downloaded/available.
embeddings_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Vector Database
vectorstore = Chroma.from_documents(documents=all_splits, embedding=embeddings_model)
retriever = vectorstore.as_retriever()

# --- 2. Retrieval Augmented Generation (RAG) Pipeline ---

# Placeholder for a Language Model (LLM)
# For a real application, replace this with a powerful medical LLM (e.g., fine-tuned Llama 2, Med-PaLM)
# For demonstration, we'll use a mock LLM that formats the output with references.
def mock_llm_with_references(input_dict):
    question = input_dict["question"]
    context_docs = input_dict["context"] # This will be a list of Document objects

    # Simulate diagnosis and recommendations based on the question and context
    diagnosis_parts = []
    recommendation_parts = []
    references_str = []

    # Simple logic to simulate using context for diagnosis/recommendation
    if "cough" in question.lower():
        diagnosis_parts.append("Possible respiratory infection (e.g., bronchitis, pneumonia).")
        recommendation_parts.append("Investigate duration and accompanying symptoms. Consider antitussives for symptomatic relief if acute.")
    if "fever" in question.lower():
        diagnosis_parts.append("Likely infection or inflammatory process.")
        recommendation_parts.append("Identify underlying cause of fever. Use antipyretics if necessary.")
    if "headache" in question.lower():
        diagnosis_parts.append("Consider tension headache or migraine. Rule out red flag symptoms.")
        recommendation_parts.append("Assess headache characteristics and patient history. Consider neuroimaging if red flags present.")
    if "shortness of breath" in question.lower():
        diagnosis_parts.append("Potential cardiovascular or pulmonary issue.")
        recommendation_parts.append("Immediate assessment for severity and accompanying symptoms. Consider imaging for pulmonary embolism.")

    if not diagnosis_parts:
        diagnosis_parts.append("No specific diagnosis could be inferred from the provided context or general symptoms.")
    if not recommendation_parts:
        recommendation_parts.append("General medical evaluation recommended.")

    for i, doc in enumerate(context_docs):
        content_snippet = doc.page_content.strip()
        source_info = doc.metadata.get("source", f"Unknown Source {i+1}")
        url_info = doc.metadata.get("url", "No URL provided")
        references_str.append(f'- "{content_snippet}" (Source: [{source_info}]({url_info}))')

    final_diagnosis = "\n- ".join(diagnosis_parts)
    final_recommendation = "\n- ".join(recommendation_parts)
    final_references = "\n".join(references_str)

    response_content = (
        f"Differential Diagnoses:\n- {final_diagnosis}\n\n"
        f"Treatment Recommendations:\n- {final_recommendation}\n\n"
        f"References:\n{final_references}"
    )
    return AIMessage(content=response_content)

mock_llm_runnable = RunnableLambda(mock_llm_with_references)

# Prompt Engineering (used for general guidance, but mock_llm_with_references directly handles output)
# For a real LLM, this prompt would be crucial.
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a medical diagnostic AI assistant. Provide differential diagnoses and treatment recommendations based on the patient's symptoms and medical history. Crucially, include direct quotes from the provided medical context as references for each claim."),
    ("human", "Medical Context:\n{context}\n\nPatient Symptoms and History: {question}\n\n")
])

# RAG Chain
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()} # Retrieves context based on question
    | mock_llm_runnable # Passes context and question to the mock LLM
    | StrOutputParser() # Parses the AIMessage to string
)

# --- 3. User Interface (Gradio) ---

def medical_assistant_interface(patient_info: str):
    """
    Processes patient information and returns AI-generated diagnosis, recommendations, and references.
    """
    response = rag_chain.invoke(patient_info)
    return response

# Gradio Interface
if __name__ == "__main__":
    demo = gr.Interface(
        fn=medical_assistant_interface,
        inputs=gr.Textbox(lines=5, label="Enter Patient Symptoms and Medical History"),
        outputs=gr.Markdown(label="AI Diagnosis, Recommendations, and References"),
        title="Medical Diagnostic AI Assistant",
        description="Provide patient symptoms and history to get differential diagnoses, treatment recommendations, and supporting medical references."
    )
    demo.launch()
