import gradio as gr
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.schema import Document
import requests # For simulating external API calls

# --- 1. Simulated External Data Connectors/Tools ---

def _simulate_pubmed_api(query: str) -> str:
    """Simulates fetching medical literature from PubMed."""
    if "diabetes treatment" in query.lower():
        return "Recent study (2023) on GLP-1 receptor agonists for type 2 diabetes shows significant A1c reduction and cardiovascular benefits. (Source: NEJM)."
    elif "hypertension guidelines" in query.lower():
        return "The ACC/AHA 2023 guidelines recommend lifestyle modifications and pharmacotherapy for blood pressure control, with target BP <130/80 mmHg for most adults. (Source: JACC)."
    return f"Simulated PubMed search for \"{query}\" found general medical articles."

def _simulate_rxnorm_api(drug_name: str) -> str:
    """Simulates fetching drug information from RxNorm."""
    if "metformin" in drug_name.lower():
        return "Metformin: Oral biguanide, first-line for type 2 diabetes. Reduces hepatic glucose production. Common side effects: GI upset. (Source: RxNorm)."
    elif "lisinopril" in drug_name.lower():
        return "Lisinopril: ACE inhibitor, used for hypertension and heart failure. Side effects: cough, hyperkalemia. Contraindicated in pregnancy. (Source: RxNorm)."
    return f"Simulated RxNorm data for \"{drug_name}\" indicates it's a common medication with known uses and side effects."

def _simulate_clinical_guidelines_api(disease: str) -> str:
    """Simulates fetching clinical guidelines for a specific disease."""
    if "asthma" in disease.lower():
        return "GINA 2023 guidelines emphasize personalized asthma management based on symptom control and risk reduction, including inhaled corticosteroids as foundational therapy. (Source: GINA)."
    elif "heart failure" in disease.lower():
        return "ESC 2022 guidelines for heart failure recommend a quadruple therapy regimen for HFrEF, including SGLT2 inhibitors, beta-blockers, MRA, and ARNI/ACEi. (Source: ESC)."
    return f"Simulated clinical guidelines for \"{disease}\" recommend standard diagnostic and therapeutic approaches."

def _simulate_web_search(topic: str) -> str:
    """Simulates a controlled web search for real-time news/alerts."""
    if "new COVID-19 variant" in topic.lower():
        return "Emerging reports suggest a new highly transmissible Omicron sub-variant, EG.5, is gaining prevalence globally. Public health agencies are monitoring its impact. (Source: WHO-like simulated alert)."
    elif "drug recall" in topic.lower():
        return "FDA-like simulated alert: Voluntary recall issued for certain batches of XYZ drug due to potential contamination. Patients advised to consult their physician. (Source: FDA-like simulated alert)."
    return f"Simulated web search for \"{topic}\" found recent general medical news."

# --- 2. RAG System Setup ---

# Initialize embedding model
# Using a small, fast model for demonstration. For production, consider larger models.
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Dummy medical documents for the vector store
dummy_medical_texts = [
    "Type 2 diabetes is a chronic condition characterized by high blood sugar levels resulting from insulin resistance or insufficient insulin production.",
    "Hypertension, or high blood pressure, significantly increases the risk of heart disease, stroke, and kidney disease.",
    "Asthma is a chronic respiratory condition where airways narrow and swell, producing extra mucus, making breathing difficult.",
    "Heart failure is a condition in which the heart can't pump enough blood to meet the body's needs.",
    "Common medications for type 2 diabetes include metformin, sulfonylureas, and SGLT2 inhibitors.",
    "Beta-blockers are a class of drugs that block the effects of epinephrine (adrenaline) and are used to treat conditions like high blood pressure, angina, and heart rhythm problems.",
    "Anaphylaxis is a severe, potentially life-threatening allergic reaction. It can occur within seconds or minutes of exposure to something you're allergic to.",
    "Pneumonia is an infection that inflames air sacs in one or both lungs, which may fill with fluid or pus. Symptoms include cough with phlegm or pus, fever, chills, and difficulty breathing.",
    "Diagnosis of appendicitis typically involves a physical exam, blood tests, urine tests, and imaging tests such as ultrasound or CT scan. Emergency surgery is often required.",
    "Migraine is a severe headache often accompanied by symptoms such as throbbing in the head, sensitivity to light and sound, nausea, and vomiting. Triggers can include stress, certain foods, and hormonal changes."
]

# Create LangChain Document objects
docs = [Document(page_content=text) for text in dummy_medical_texts]

# Initialize Chroma vector store with dummy data
# In a real application, data would be loaded and updated regularly.
vectorstore = Chroma.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()

# --- 3. Information Consolidation and Refinement Pipeline ---

def _consolidate_and_refine(rag_docs: list[Document], external_data_results: dict) -> str:
    """Consolidates and refines information from RAG and external tools."""
    context_parts = []

    if rag_docs:
        context_parts.append("\n--- Retrieved Medical Knowledge ---\n")
        for i, doc in enumerate(rag_docs):
            context_parts.append(f"Document {i+1}: {doc.page_content}")

    if external_data_results:
        context_parts.append("\n--- Real-time External Data ---\n")
        for source, data in external_data_results.items():
            if data:
                context_parts.append(f"{source}: {data}")

    if not context_parts:
        return "No relevant information found from available sources."

    # Simple concatenation for demonstration. Advanced pipelines would include summarization, deduplication, etc.
    return "\n".join(context_parts)

# --- 4. LLM Core Simulation (Placeholder) ---

def _simulate_llm_response(prompt: str) -> str:
    """Simulates an LLM generating a response based on a prompt."""
    # In a real application, this would call an actual LLM (e.g., OpenAI, HuggingFace model)
    # For this simulation, we'll try to extract the 'answer' part or return a generic response.
    if "Based on the following context" in prompt and "Answer:" in prompt:
        # Simple attempt to extract an 'answer' from a structured prompt
        try:
            context_end = prompt.find("Answer:")
            context = prompt[prompt.find("Context:") + len("Context:"):context_end].strip()
            question = prompt[prompt.find("Question:") + len("Question:"):prompt.find("Context:")].strip()
            
            # A very simple rule-based 'answer' simulation
            if "metformin" in context.lower() and "type 2 diabetes" in context.lower() and "uses" in question.lower():
                return "Metformin is a first-line medication for type 2 diabetes that reduces hepatic glucose production."
            if "hypertension" in context.lower() and "risk factors" in question.lower():
                 return "Hypertension increases the risk of heart disease, stroke, and kidney disease."
            if "anaphylaxis" in context.lower() and "severe allergic reaction" in context.lower() and "symptoms" in question.lower():
                return "Anaphylaxis is a severe, life-threatening allergic reaction that can occur rapidly."
            
            return f"Simulated LLM generated response based on the provided context. (Original question: {question})"
        except Exception:
            pass # Fallback to generic if extraction fails

    return f"Simulated LLM response: I processed your request using the available information. (Prompt: {prompt[:100]}...)"

# --- 5. Main Clinical Assistant Logic ---

class ClinicalAssistant:
    def __init__(self):
        self.retriever = retriever

    def _call_external_tool(self, tool_name: str, query: str) -> str:
        """Dispatches calls to simulated external APIs."""
        if tool_name == "pubmed":
            return _simulate_pubmed_api(query)
        elif tool_name == "rxnorm":
            return _simulate_rxnorm_api(query)
        elif tool_name == "clinical_guidelines":
            return _simulate_clinical_guidelines_api(query)
        elif tool_name == "web_search":
            return _simulate_web_search(query)
        return ""

    def process_query(self, user_query: str) -> str:
        # 1. RAG Retrieval
        rag_docs = self.retriever.get_relevant_documents(user_query)

        # 2. Determine and call external tools (simple keyword-based logic for simulation)
        external_data_results = {}
        if "recent study" in user_query.lower() or "latest research" in user_query.lower():
            external_data_results["PubMed"] = self._call_external_tool("pubmed", user_query)
        if "drug info" in user_query.lower() or "medication for" in user_query.lower() or "side effects of" in user_query.lower():
            # Simple extraction of drug name for simulation
            drug_keywords = ["metformin", "lisinopril"]
            found_drug = next((d for d in drug_keywords if d in user_query.lower()), None)
            if found_drug:
                external_data_results["RxNorm"] = self._call_external_tool("rxnorm", found_drug)
            else:
                external_data_results["RxNorm"] = self._call_external_tool("rxnorm", user_query) # Fallback
        if "guidelines for" in user_query.lower() or "management of" in user_query.lower():
            disease_keywords = ["asthma", "heart failure", "diabetes", "hypertension"]
            found_disease = next((d for d in disease_keywords if d in user_query.lower()), None)
            if found_disease:
                external_data_results["Clinical Guidelines"] = self._call_external_tool("clinical_guidelines", found_disease)
            else:
                external_data_results["Clinical Guidelines"] = self._call_external_tool("clinical_guidelines", user_query)
        if "new variant" in user_query.lower() or "drug recall" in user_query.lower() or "breaking medical news" in user_query.lower():
            external_data_results["Web Search (News)"] = self._call_external_tool("web_search", user_query)

        # 3. Information Consolidation and Refinement
        context = _consolidate_and_refine(rag_docs, external_data_results)

        # 4. Prepare prompt for LLM and get simulated response
        llm_prompt = f"Question: {user_query}\n\nContext: {context}\n\nBased on the following context, please provide a comprehensive and accurate answer to the medical question. If the context is insufficient, state that.\nAnswer:"

        llm_response = _simulate_llm_response(llm_prompt)

        return llm_response

# --- 6. Gradio UI Setup ---

clinical_assistant = ClinicalAssistant()

def chat_interface(message, history):
    response = clinical_assistant.process_query(message)
    return response

# Create a Gradio ChatInterface
demo = gr.ChatInterface(
    fn=chat_interface,
    title="Real-time Clinical Assistant LLM (Simulated)",
    description="Ask medical questions to get dynamically augmented answers from simulated RAG and external medical databases.",
    examples=[
        "What are the latest treatments for type 2 diabetes?",
        "Tell me about Metformin, including its side effects.",
        "What are the current guidelines for managing asthma?",
        "Are there any new COVID-19 variants being monitored?",
        "What is anaphylaxis?"
    ]
)

if __name__ == "__main__":
    demo.launch()