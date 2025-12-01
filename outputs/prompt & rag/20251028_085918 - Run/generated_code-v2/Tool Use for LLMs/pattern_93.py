import gradio as gr
import time
import random
import json
from collections import defaultdict


class MockLLM:
    def __init__(self, name="MockLLM"):
        self.name = name

    def generate(self, prompt, stop=None):
        response_map = {
            "analyze symptoms": "Extracted key symptoms: fever, fatigue, rash. Potential categories: autoimmune, infectious.",
            "search literature for rare diseases related to fever, fatigue, rash": "Found articles on 'Still's disease' and 'Systemic Lupus Erythematosus' (SLE). Both involve fever, fatigue, and rash. Still's disease often has salmon-pink rash and arthralgia; SLE has butterfly rash and multi-organ involvement.",
            "interpret genetic report for variants related to Still's disease, Systemic Lupus Erythematosus": "No direct pathogenic variants found for Still's disease or SLE. However, some common SNPs associated with inflammatory responses were noted.",
            "generate differential diagnosis based on all gathered info": "Considering patient data and literature, top differential diagnoses are: 1. Still's Disease (moderate confidence) due to general inflammatory symptoms, 2. Systemic Lupus Erythematosus (moderate confidence) given widespread symptoms. Further tests needed to differentiate.",
            "check drug interactions for ibuprofen, prednisone": "Ibuprofen and prednisone can both cause gastrointestinal irritation. Concurrent use increases risk of ulcers. Monitor for GI side effects.",
            "match clinical trials for Still's Disease, Systemic Lupus Erythematosus": "Clinical trials found for novel biologics targeting inflammatory pathways in Still's disease and SLE. Locations: Major academic medical centers in [City A], [City B]."
        }

        for key, value in response_map.items():
            if key in prompt:
                return value
        return "No specific response for this prompt, but I am reasoning..."


class Tool:
    def __init__(self, name, description, func):
        self.name = name
        self.description = description
        self.func = func

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


def symptom_and_medical_history_analyzer(patient_data):
    symptoms = patient_data.get("symptoms", "")
    history = patient_data.get("medical_history", "")
    key_symptoms = ", ".join(list(set([s.strip().lower() for s in symptoms.split(',') if s.strip()]))) if symptoms else "no specific symptoms"
    return {"summary": f"Analyzed symptoms and history. Key identified symptoms: {key_symptoms}.", "key_symptoms": key_symptoms}

def medical_literature_search_and_summarizer(query):
    mock_results = {
        "fever, fatigue, rash": "Still's disease: Characterized by high fever, evanescent salmon-pink rash, and arthralgia. Systemic Lupus Erythematosus (SLE): Autoimmune disease with varied symptoms including fatigue, fever, rash (often malar), arthritis, and potential organ involvement.",
        "Still's disease": "Still's disease is a rare inflammatory disorder. Diagnosis involves exclusion and specific criteria.",
        "Systemic Lupus Erythematosus": "SLE is a chronic autoimmune condition affecting multiple organ systems."
    }
    summary = mock_results.get(query.lower(), f"Searched for '{query}'. Found general information on inflammatory conditions.")
    return {"summary": summary, "relevant_diseases": [d for d in mock_results.keys() if query.lower() in d.lower() or d.lower() in query.lower()]}

def genomic_data_interpreter(genetic_report_data):
    if "mutation_X" in genetic_report_data:
        return {"summary": "Detected mutation_X, associated with a rare metabolic disorder.", "findings": ["mutation_X"]}
    if "common_snp_inflammatory" in genetic_report_data:
        return {"summary": "Identified common SNPs associated with inflammatory response, not directly pathogenic for rare diseases but indicates predisposition.", "findings": ["common_snp_inflammatory"]}
    return {"summary": "No significant pathogenic variants for known rare diseases found.", "findings": []}

def drug_interaction_and_contraindication_checker(medications, conditions):
    if "ibuprofen" in medications and "prednisone" in medications:
        return {"summary": "WARNING: Increased risk of gastrointestinal ulcers with concurrent ibuprofen and prednisone use.", "interactions": ["Ibuprofen-Prednisone GI risk"]}
    return {"summary": "No significant drug interactions or contraindications found for the given medications and conditions.", "interactions": []}

def differential_diagnosis_generator(all_patient_info_summary):
    if "Still's disease" in all_patient_info_summary and "fever" in all_patient_info_summary:
        return {"diagnoses": [{"name": "Still's Disease", "confidence": "moderate", "evidence": "Matches key symptoms and literature findings."}, {"name": "Systemic Lupus Erythematosus", "confidence": "low-moderate", "evidence": "Some overlapping symptoms, but less specific."}] }
    return {"diagnoses": [{"name": "Undetermined Rare Disease", "confidence": "low", "evidence": "Insufficient specific data for a clear rare diagnosis."}]}

def clinical_trial_matcher(diseases, genetic_markers):
    trials = []
    if "Still's Disease" in diseases:
        trials.append("Trial for IL-1 inhibitors in Still's Disease (NCT012345)")
    if "Systemic Lupus Erythematosus" in diseases:
        trials.append("Study on new immunosuppressants for SLE (NCT067890)")
    if genetic_markers and "mutation_X" in genetic_markers:
        trials.append("Gene therapy trial for rare metabolic disorder (NCT112233)")
    return {"summary": f"Found {len(trials)} relevant clinical trials.", "trials": trials}


class AgenticLLMCore:
    def __init__(self, llm):
        self.llm = llm
        self.tools = {
            "symptom_analyzer": Tool("Symptom and Medical History Analyzer", "Analyzes patient symptoms and history.", symptom_and_medical_history_analyzer),
            "literature_searcher": Tool("Medical Literature Search and Summarizer", "Searches and summarizes medical literature.", medical_literature_search_and_summarizer),
            "genomic_interpreter": Tool("Genomic Data Interpreter", "Interprets genetic sequencing data.", genomic_data_interpreter),
            "drug_checker": Tool("Drug Interaction and Contraindication Checker", "Checks drug interactions.", drug_interaction_and_contraindication_checker),
            "diagnosis_generator": Tool("Differential Diagnosis Generator", "Generates a list of possible diagnoses.", differential_diagnosis_generator),
            "trial_matcher": Tool("Clinical Trial Matcher", "Finds relevant clinical trials.", clinical_trial_matcher)
        }
        self.memory = []

    def _log_interaction(self, role, content):
        self.memory.append({"role": role, "content": content})

    def process_patient_case(self, patient_data):
        self._log_interaction("User Input", json.dumps(patient_data))
        output_stream = []

        output_stream.append("\n--- LLM Reasoning Process ---")
        output_stream.append("Thinking: Initial assessment of patient symptoms...")

        # Step 1: Analyze symptoms and history
        symptom_analysis_input = {
            "symptoms": patient_data.get("symptoms", ""),
            "medical_history": patient_data.get("medical_history", "")
        }
        symptoms_analysis = self.tools["symptom_analyzer"](symptom_analysis_input)
        output_stream.append(f"Tool Call: symptom_analyzer(patient_data={{'symptoms': '{patient_data.get('symptoms', '')}', 'medical_history': '{patient_data.get('medical_history', '')}'}})")
        output_stream.append(f"Tool Output (Symptom Analyzer): {symptoms_analysis['summary']}")
        self._log_interaction("Tool Output", f"Symptom Analyzer: {symptoms_analysis['summary']}")

        # Simulate LLM deciding to search literature based on key symptoms
        output_stream.append("Thinking: Symptoms analyzed. Now, searching medical literature for related rare diseases...")
        literature_query = symptoms_analysis.get("key_symptoms", "inflammatory conditions")
        literature_summary = self.tools["literature_searcher"](literature_query)
        output_stream.append(f"Tool Call: literature_searcher(query='{literature_query}')")
        output_stream.append(f"Tool Output (Medical Literature Searcher): {literature_summary['summary']}")
        self._log_interaction("Tool Output", f"Medical Literature Searcher: {literature_summary['summary']}")

        # Simulate LLM deciding to interpret genomic data if available
        genomic_findings = []
        if patient_data.get("genetic_report"):
            output_stream.append("Thinking: Genetic report provided. Interpreting genomic data for relevant variants...")
            genomic_interpretation = self.tools["genomic_interpreter"](patient_data["genetic_report"])
            output_stream.append(f"Tool Call: genomic_interpreter(genetic_report_data='{patient_data['genetic_report']}')")
            output_stream.append(f"Tool Output (Genomic Data Interpreter): {genomic_interpretation['summary']}")
            self._log_interaction("Tool Output", f"Genomic Data Interpreter: {genomic_interpretation['summary']}")
            genomic_findings = genomic_interpretation.get("findings", [])

        # Simulate LLM synthesizing information for differential diagnosis
        output_stream.append("Thinking: Synthesizing all gathered information to generate differential diagnoses...")
        all_info_summary = f"Symptoms: {symptoms_analysis['summary']}, Literature Findings: {literature_summary['summary']}"
        if genomic_findings:
            all_info_summary += f", Genomic Findings: {', '.join(genomic_findings)}"

        differential_diagnosis_result = self.tools["diagnosis_generator"](all_info_summary)
        output_stream.append(f"Tool Call: differential_diagnosis_generator(all_patient_info_summary='{all_info_summary[:100]}...')")
        output_stream.append(f"Tool Output (Differential Diagnosis Generator): {json.dumps(differential_diagnosis_result['diagnoses'])}")
        self._log_interaction("Tool Output", f"Differential Diagnosis Generator: {json.dumps(differential_diagnosis_result['diagnoses'])}")

        # Simulate LLM checking drug interactions for potential treatments related to suggested diagnoses
        output_stream.append("Thinking: Considering potential treatments for suggested diagnoses and checking for drug interactions...")
        mock_medications = patient_data.get("medications", [])
        for diag in differential_diagnosis_result["diagnoses"]:
            if "Still's Disease" in diag["name"]:
                mock_medications.extend(["ibuprofen", "prednisone"])
            elif "Systemic Lupus Erythematosus" in diag["name"]:
                mock_medications.extend(["hydroxychloroquine"])
        mock_medications = list(set(mock_medications))
        drug_interaction_result = self.tools["drug_checker"](mock_medications, all_info_summary)
        output_stream.append(f"Tool Call: drug_checker(medications={mock_medications}, conditions='{all_info_summary[:50]}...')")
        output_stream.append(f"Tool Output (Drug Interaction Checker): {drug_interaction_result['summary']}")
        self._log_interaction("Tool Output", f"Drug Interaction Checker: {drug_interaction_result['summary']}")

        # Simulate LLM matching clinical trials
        output_stream.append("Thinking: Finally, searching for relevant clinical trials...")
        diseases_for_trials = [d["name"] for d in differential_diagnosis_result["diagnoses"]]
        trial_matching_result = self.tools["trial_matcher"](diseases_for_trials, genomic_findings)
        output_stream.append(f"Tool Call: trial_matcher(diseases={diseases_for_trials}, genetic_markers={genomic_findings})")
        output_stream.append(f"Tool Output (Clinical Trial Matcher): {trial_matching_result['summary']}")
        self._log_interaction("Tool Output", f"Clinical Trial Matcher: {trial_matching_result['summary']}")

        output_stream.append("\n--- Final Summary ---")
        final_diagnosis_str = "Proposed Differential Diagnoses:\n"
        for diag in differential_diagnosis_result["diagnoses"]:
            final_diagnosis_str += f"- {diag['name']} (Confidence: {diag['confidence']}) - Evidence: {diag['evidence']}\n"

        final_drug_interactions_str = "Drug Interaction Notes:\n"
        if drug_interaction_result["interactions"]:
            final_drug_interactions_str += f"  {drug_interaction_result['summary']}\n"
        else:
            final_drug_interactions_str += "  No significant interactions found.\n"

        final_trials_str = "Relevant Clinical Trials:\n"
        if trial_matching_result["trials"]:
            for trial in trial_matching_result["trials"]:
                final_trials_str += f"- {trial}\n"
        else:
            final_trials_str += "  No specific trials matched at this time.\n"

        final_output = "\n".join(output_stream) + "\n\n" + final_diagnosis_str + "\n" + final_drug_interactions_str + "\n" + final_trials_str
        return final_output


def run_diagnostic_assistant(symptoms, medical_history, genetic_report, medications):
    patient_data = {
        "symptoms": symptoms,
        "medical_history": medical_history,
        "genetic_report": genetic_report,
        "medications": [m.strip() for m in medications.split(',') if m.strip()]
    }
    llm_mock = MockLLM()
    agent = AgenticLLMCore(llm_mock)
    result = agent.process_patient_case(patient_data)
    return result


if __name__ == "__main__":
    interface = gr.Interface(
        fn=run_diagnostic_assistant,
        inputs=[
            gr.Textbox(label="Patient Symptoms (comma-separated)", placeholder="e.g., fever, fatigue, rash, joint pain"),
            gr.Textbox(label="Medical History", placeholder="e.g., childhood allergies, recent infections"),
            gr.Textbox(label="Genetic Report Data (e.g., mutation_X, common_snp_inflammatory)", placeholder="e.g., 'mutation_X' or leave blank"),
            gr.Textbox(label="Current Medications (comma-separated)", placeholder="e.g., ibuprofen, paracetamol"),
        ],
        outputs=gr.Textbox(label="Diagnostic Assistant Output"),
        title="AI-Powered Rare Disease Diagnostic Assistant (Emergent Tool Composition)",
        description="This assistant simulates an LLM agent dynamically composing tools to aid in rare disease diagnosis. The LLM's 'reasoning' and tool calls are logged below."
    )
    interface.launch(share=False)
