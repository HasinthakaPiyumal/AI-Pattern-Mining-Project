import streamlit as st
import spacy
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
import json
from typing import List, Dict, Any

# --- 1. Pydantic Models for Query and Data --- 
class MedicalQuery(BaseModel):
    disease: str = None
    drug: str = None
    gene_marker: str = None
    demographics: str = None
    keywords: List[str] = []
    query_type: str = "summary"

# --- 2. Mock External API Tools --- 
class PubMedTool:
    def search(self, keywords: List[str]) -> Dict[str, Any]:
        st.sidebar.write(f"[MOCK] Searching PubMed for: {', '.join(keywords)}")
        # Simulate API call and return structured data
        if "glioblastoma" in [k.lower() for k in keywords]:
            return {
                "source": "PubMed",
                "results": [
                    {
                        "title": "Recent advances in glioblastoma treatment strategies",
                        "abstract": "This paper reviews the latest therapeutic approaches for glioblastoma, including novel chemotherapy regimens and immunotherapies. Key findings suggest improved progression-free survival with combination therapies.",
                        "url": "https://pubmed.example.com/glioblastoma_advances"
                    },
                    {
                        "title": "Genetic mutations in glioblastoma and drug resistance",
                        "abstract": "Analysis of IDH1 mutations and their correlation with resistance to temozolomide. Highlights the importance of personalized medicine approaches.",
                        "url": "https://pubmed.example.com/glioblastoma_genetics"
                    }
                ]
            }
        return {"source": "PubMed", "results": []}

class ClinicalTrialsGovTool:
    def search(self, keywords: List[str]) -> Dict[str, Any]:
        st.sidebar.write(f"[MOCK] Searching ClinicalTrials.gov for: {', '.join(keywords)}")
        # Simulate API call
        if "glioblastoma" in [k.lower() for k in keywords] and "phase 3" in [k.lower() for k in keywords]:
            return {
                "source": "ClinicalTrials.gov",
                "results": [
                    {
                        "id": "NCT01234567",
                        "title": "Phase 3 Trial of Drug X in Newly Diagnosed Glioblastoma",
                        "status": "Completed",
                        "summary": "A randomized, double-blind, placebo-controlled Phase 3 study evaluating the efficacy and safety of Drug X in combination with standard radiation and temozolomide for newly diagnosed glioblastoma. Primary endpoint met, showing statistically significant improvement in overall survival.",
                        "url": "https://clinicaltrials.gov/ct2/show/NCT01234567"
                    }
                ]
            }
        return {"source": "ClinicalTrials.gov", "results": []}

# --- 3. Query Processing Module --- 
# Load a small English model for spaCy
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.error("SpaCy model 'en_core_web_sm' not found. Please run: python -m spacy download en_core_web_sm")
    st.stop()

def process_user_query(user_input: str) -> MedicalQuery:
    doc = nlp(user_input.lower())
    query_params = {"keywords": []}
    
    # Simple keyword extraction (can be improved with NER for medical terms)
    if "glioblastoma" in user_input.lower():
        query_params["disease"] = "glioblastoma"
        query_params["keywords"].append("glioblastoma")
    if "treatment" in user_input.lower():
        query_params["keywords"].append("treatment")
    if "drug" in user_input.lower() or "medication" in user_input.lower():
        query_params["drug"] = "" # Placeholder for actual drug extraction
        query_params["keywords"].append("drug")
    if "phase 3" in user_input.lower():
        query_params["keywords"].append("phase 3")
    if "summarize" in user_input.lower() or "summary" in user_input.lower():
        query_params["query_type"] = "summary"
    if "explain" in user_input.lower() or "implications" in user_input.lower():
        query_params["query_type"] = "explanation"

    # Add remaining tokens as general keywords
    for token in doc:
        if not token.is_stop and not token.is_punct and token.text not in [v.lower() for v in query_params.values() if isinstance(v, str)]:
            query_params["keywords"].append(token.text)
            
    return MedicalQuery(**query_params)

# --- 4. Raw Output Processing Module (Simplified) --- 
def process_raw_tool_output(tool_output: Dict[str, Any]) -> str:
    processed_data = []
    if tool_output.get("source") == "PubMed":
        for result in tool_output.get("results", []):
            processed_data.append(f"Title: {result['title']}\nAbstract: {result['abstract']}\nURL: {result['url']}\n")
    elif tool_output.get("source") == "ClinicalTrials.gov":
        for result in tool_output.get("results", []):
            processed_data.append(f"Trial ID: {result['id']}\nTitle: {result['title']}\nStatus: {result['status']}\nSummary: {result['summary']}\nURL: {result['url']}\n")
    return "\n---\n".join(processed_data)

# --- 5. LLM Integration & Synthesis Module (Mock) --- 
def synthesize_response_with_llm(processed_data: List[str], query: MedicalQuery) -> str:
    st.sidebar.write("[MOCK] Sending processed data to LLM for synthesis...")
    
    if not processed_data:
        return "I couldn't find relevant information for your query from the available tools."

    # Simulate LLM's synthesis logic based on query type
    combined_text = "\n".join(processed_data)

    if query.query_type == "summary":
        if query.disease and "glioblastoma" in query.disease.lower():
            return f"**Summary of Glioblastoma Research:**\n\nBased on the retrieved information, recent advances in glioblastoma treatment include novel chemotherapy and immunotherapies, showing improved progression-free survival. Genetic mutations like IDH1 are critical for understanding drug resistance, emphasizing personalized medicine. A completed Phase 3 trial for Drug X combined with standard therapy showed significant improvement in overall survival. \n\n---\nRaw Data Snippets:\n{combined_text}"
        else:
            return f"**General Summary:**\n\nI've gathered some information based on your query. Here's a synthesized summary:\n\n{combined_text}\n\nThis is a general overview; for specific insights, further analysis is often required."
    elif query.query_type == "explanation":
        if query.disease and "glioblastoma" in query.disease.lower() and query.drug and "drug x" in combined_text.lower():
            return f"**Explanation of Drug X Phase 3 Trial Implications for Glioblastoma:**\n\nThe Phase 3 trial of Drug X in newly diagnosed glioblastoma patients demonstrated a statistically significant improvement in overall survival when combined with standard radiation and temozolomide. This suggests that Drug X could be a valuable addition to the current treatment paradigm, potentially extending patient lives. Further research will likely focus on long-term outcomes and specific patient subgroups benefiting most.\n\n---\nRaw Data Snippets:\n{combined_text}"
        else:
            return f"**General Explanation:**\n\nI've gathered information, and here's a general explanation based on the findings:\n\n{combined_text}\n\nThis explanation integrates available data, but a human expert should review for clinical decision-making."
    
    return f"**Synthesized Response:**\n\n{combined_text}"

# --- Main Streamlit Application --- 
st.set_page_config(layout="wide", page_title="Medical Research Summarizer")
st.title("🧠 Medical Research Summarizer & Explainer AI")
st.subheader("Leveraging AI to synthesize complex medical research for healthcare professionals.")

st.sidebar.header("Configuration")

user_query = st.text_area("Enter your medical research query (e.g., 'Summarize latest glioblastoma treatment research' or 'Explain implications of Drug X Phase 3 trial for glioblastoma'):",
                          "Summarize latest glioblastoma treatment research, considering IDH1 mutations.",
                          height=100)

if st.button("Get Insights"):
    if not user_query:
        st.warning("Please enter a query.")
    else:
        st.info("Processing your query...")
        
        with st.spinner("Analyzing query and invoking tools..."):
            # Query Processing
            medical_query = process_user_query(user_query)
            st.sidebar.markdown(f"**Processed Query:**\n`{medical_query.model_dump_json(indent=2)}`")

            # Tool Orchestration and Invocation
            all_tool_outputs = []
            
            pubmed_tool = PubMedTool()
            pubmed_results = pubmed_tool.search(medical_query.keywords)
            if pubmed_results["results"]:
                all_tool_outputs.append(pubmed_results)

            clinical_trials_tool = ClinicalTrialsGovTool()
            clinical_trials_results = clinical_trials_tool.search(medical_query.keywords)
            if clinical_trials_results["results"]:
                all_tool_outputs.append(clinical_trials_results)
            
            if not all_tool_outputs:
                st.warning("No relevant data found from external tools for your query.")
                st.stop()

        with st.spinner("Processing raw outputs and synthesizing response..."):
            # Raw Output Processing
            processed_data_for_llm = []
            for output in all_tool_outputs:
                processed_data_for_llm.append(process_raw_tool_output(output))
            
            st.sidebar.markdown("**Processed Raw Data (for LLM):**")
            for data_snippet in processed_data_for_llm:
                st.sidebar.text(data_snippet[:200] + "...") # Show snippet

            # LLM Integration & Synthesis
            final_response = synthesize_response_with_llm(processed_data_for_llm, medical_query)
            
            st.success("Response Generated!")
            st.markdown(final_response)

st.sidebar.markdown("---")
st.sidebar.markdown("This is a demonstration of the \"Tool-Augmented Response Synthesis\" pattern.")
