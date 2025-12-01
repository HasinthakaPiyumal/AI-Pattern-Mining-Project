import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import torch
from langchain_text_splitters import RecursiveCharacterTextSplitter
import requests

# 1. Data Ingestion and API Documentation Management (Simplified Mock)
# In a real scenario, this would involve scraping and constant updates
medical_api_docs = [
    {"id": "doc_001", "content": "Drug A dosage: 10mg daily for adults. Side effects include nausea. Source: DrugDB v2.1 (2023-01-15)", "source": "DrugDB", "version": "2.1", "date": "2023-01-15", "valid": True},
    {"id": "doc_002", "content": "Treatment for Condition X: First-line therapy is Drug B. Second-line is Drug C. Source: ClinicalGuidelines v1.5 (2023-03-20)", "source": "ClinicalGuidelines", "version": "1.5", "date": "2023-03-20", "valid": True},
    {"id": "doc_003", "content": "Diagnostic Imaging API: MRI brain scan requires 'sequence' and 'contrast' parameters. Source: ImagingAPI v3.0 (2024-02-01)", "source": "ImagingAPI", "version": "3.0", "date": "2024-02-01", "valid": True},
    {"id": "doc_004", "content": "Outdated: Drug A dosage: 5mg daily for adults. Side effects include headache. Source: DrugDB v1.0 (2022-06-01)", "source": "DrugDB", "version": "1.0", "date": "2022-06-01", "valid": False},
    {"id": "doc_005", "content": "Drug D dosage: 20mg twice daily. Use with caution in renal impairment. Source: DrugDB v2.2 (2024-03-10)", "source": "DrugDB", "version": "2.2", "date": "2024-03-10", "valid": True},
    {"id": "doc_006", "content": "Treatment for Condition Y: Initial approach involves lifestyle changes. Medication (Drug E) if severe. Source: ClinicalGuidelines v2.0 (2024-01-05)", "source": "ClinicalGuidelines", "version": "2.0", "date": "2024-01-05", "valid": True}
]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)

# Mock Vector Database and Embedding Model
class MockVectorDB:
    def __init__(self, embedding_model):
        self.documents = []
        self.embeddings = []
        self.embedding_model = embedding_model

    def add_documents(self, docs):
        for doc in docs:
            chunks = text_splitter.split_text(doc["content"])
            for i, chunk in enumerate(chunks):
                self.documents.append({"id": f"{doc['id']}_chunk{i}", "content": chunk, **{k: v for k, v in doc.items() if k != 'content'}})
                self.embeddings.append(self.embedding_model.encode(chunk, convert_to_tensor=True).cpu())

    def search(self, query, k=3):
        if not self.embeddings:
            return []
        query_embedding = self.embedding_model.encode(query, convert_to_tensor=True).cpu()
        similarities = torch.nn.functional.cosine_similarity(query_embedding, torch.stack(self.embeddings))
        top_k_indices = torch.topk(similarities, min(k, len(self.embeddings))).indices.tolist()
        return [self.documents[i] for i in top_k_indices]

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_embedding_model()
vector_db = MockVectorDB(embedding_model)
vector_db.add_documents(medical_api_docs)

# 2. Retriever Module
def retrieve_documentation(query, num_docs=3):
    return vector_db.search(query, k=num_docs)

# 3. LLM Finetuning Module (Mock - real finetuning is complex and resource-intensive)
# We'll use a pre-trained model and simulate the 'retriever-aware' aspect in its prompt and internal logic
@st.cache_resource
def load_llm():
    # Using a smaller model for demonstration, replace with a more capable one for production
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device=0 if torch.cuda.is_available() else -1,
    )
    return pipe

llm_pipeline = load_llm()

def generate_llm_response(prompt):
    try:
        response = llm_pipeline(prompt, max_new_tokens=500, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        return response[0]["generated_text"]
    except Exception as e:
        st.error(f"LLM generation error: {e}")
        return "An error occurred while generating a response from the LLM."

# 4. Inference Module
def construct_augmented_prompt(user_query, retrieved_docs):
    system_instruction = (
        "You are a Medical Diagnosis and Treatment Assistant. "
        "Carefully evaluate the provided medical documentation. "
        "Prioritize the most recent and relevant information. "
        "If you find conflicting or outdated information in the provided documentation, identify it "
        "and explain why you are using different information or ignoring it. "
        "If you decide to make an API call, state the API name and parameters you would use."
    )
    doc_context = ""
    if retrieved_docs:
        doc_context = "Retrieved Medical Documentation:\n"
        for doc in retrieved_docs:
            valid_status = "(VALID)" if doc.get("valid", True) else "(OUTDATED/POTENTIALLY INCORRECT)"
            doc_context += f"- Source: {doc.get('source', 'N/A')} v{doc.get('version', 'N/A')} ({doc.get('date', 'N/A')}) {valid_status}\n  Content: {doc['content']}\n"
    else:
        doc_context = "No relevant documentation found.\n"

    full_prompt = f"{system_instruction}\n\n{doc_context}\nUser Query: {user_query}\nAssistant:"
    return full_prompt

# 5. API Integration Layer (Mock)
class MockMedicalAPI:
    def get_drug_dosage(self, drug_name):
        if drug_name == "Drug A":
            return {"drug": "Drug A", "dosage": "10mg daily", "source": "DrugDB v2.2"}
        elif drug_name == "Drug B":
            return {"drug": "Drug B", "dosage": "250mg twice daily", "source": "DrugDB v1.8"}
        return {"error": f"Dosage for {drug_name} not found."}

    def get_treatment_protocols(self, condition):
        if condition == "Condition X":
            return {"condition": "Condition X", "protocol": "First-line: Drug B. Second-line: Drug C.", "source": "ClinicalGuidelines v1.5"}
        return {"error": f"Treatment protocols for {condition} not found."}

    def perform_imaging_scan(self, scan_type, params):
        if scan_type == "MRI brain":
            required_params = ["sequence", "contrast"]
            if all(p in params for p in required_params):
                return {"scan_type": scan_type, "status": "Scan initiated with parameters.", "params": params}
            else:
                return {"error": f"Missing required parameters for MRI brain scan: {', '.join([p for p in required_params if p not in params])}"}
        return {"error": f"Scan type {scan_type} not supported."}

mock_apis = {
    "drug_db": MockMedicalAPI(),
    "clinical_guidelines": MockMedicalAPI(),
    "imaging_api": MockMedicalAPI()
}

def execute_api_call(api_name, method_name, **kwargs):
    api = mock_apis.get(api_name)
    if not api:
        return {"error": f"API {api_name} not found."}
    method = getattr(api, method_name, None)
    if not method:
        return {"error": f"Method {method_name} not found in {api_name}."}
    try:
        return method(**kwargs)
    except TypeError as e:
        return {"error": f"Invalid parameters for {api_name}.{method_name}: {e}"}

# 6. User Interface (Streamlit)
st.set_page_config(layout="wide", page_title="Medical Assistant (RAT Demo)")

st.title("🩺 Medical Diagnosis and Treatment Assistant (RAT Demo)")
st.markdown("This application demonstrates Retriever-Aware Training (RAT) for an LLM assistant interacting with dynamic medical API documentation.")

user_query = st.text_area("Enter your medical query here:", 
"What is the recommended dosage for Drug A and what are the latest treatment protocols for Condition X? Also, how would I initiate an MRI brain scan API call?", height=150)

if st.button("Get Assistant Response"):
    if user_query:
        st.subheader("1. Retrieving Documentation...")
        retrieved_docs = retrieve_documentation(user_query, num_docs=5)
        
        st.json([{"content": doc["content"], "source": doc["source"], "version": doc["version"], "date": doc["date"], "valid": doc["valid"]} for doc in retrieved_docs])
        
        st.subheader("2. Augmenting Prompt and Generating LLM Response...")
        augmented_prompt = construct_augmented_prompt(user_query, retrieved_docs)
        
        st.text_area("Augmented Prompt sent to LLM:", augmented_prompt, height=400)

        llm_response_full = generate_llm_response(augmented_prompt)
        
        # Extract only the assistant's response part
        assistant_prefix = "Assistant:"
        if assistant_prefix in llm_response_full:
            llm_response = llm_response_full.split(assistant_prefix, 1)[1].strip()
        else:
            llm_response = llm_response_full.strip()

        st.subheader("3. Assistant's Response:")
        st.write(llm_response)

        # Simple check for API call suggestion and execution
        st.subheader("4. Checking for API Call Suggestions...")
        if "call drug_db.get_dosage" in llm_response.lower() and "drug a" in llm_response.lower():
            st.info("Detected suggestion to call 'drug_db.get_dosage' for Drug A. Executing mock API call...")
            api_result = execute_api_call("drug_db", "get_drug_dosage", drug_name="Drug A")
            st.json(api_result)
        
        if "call imaging_api.perform_imaging_scan" in llm_response.lower() and "mri brain" in llm_response.lower():
            st.info("Detected suggestion to call 'imaging_api.perform_imaging_scan' for MRI brain. Executing mock API call...")
            # Example parameters, LLM would ideally suggest these
            api_result = execute_api_call("imaging_api", "perform_imaging_scan", scan_type="MRI brain", params={"sequence": "T1W", "contrast": True})
            st.json(api_result)

    else:
        st.warning("Please enter a query.")