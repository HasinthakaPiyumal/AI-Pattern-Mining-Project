import os
import re
import json
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI as LangchainOpenAI  # To use with Langchain chains
from langchain.chains import LLMChain

# --- Configuration and Initialization ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app = FastAPI()

# --- Pydantic Models ---
class MedicalReportRequest(BaseModel):
    report_text: str

class EvaluationResult(BaseModel):
    metric: str
    score: float | str
    feedback: str = ""

class SummaryResponse(BaseModel):
    original_report: str
    summary: str
    evaluation: list[EvaluationResult]

# --- Prompt Engineering Module ---
class PromptFactory:
    def __init__(self):
        self.base_template = """
        Summarize the following medical report concisely and accurately, focusing on key findings, diagnoses, and treatment plans.
        Medical Report:
        {report}
        Summary:
        """

        self.role_based_template = """
        You are a highly experienced medical summarization assistant providing concise, accurate, and clinically relevant summaries for busy clinicians.
        Focus on critical information: patient demographics, chief complaint, medical history, physical examination findings, diagnostic test results, diagnosis, and treatment plan.
        Ensure patient confidentiality is maintained. Do not hallucinate information. If unsure about a detail, state uncertainty.
        Medical Report:
        {report}
        Summary:
        """

        self.few_shot_examples = [
            {
                "report": "Patient presented with severe abdominal pain. CT scan showed acute appendicitis. Patient underwent appendectomy. Recovering well.",
                "summary": "Acute appendicitis diagnosed via CT. Appendectomy performed. Patient recovering.",
                "facts": ["acute appendicitis", "CT scan", "appendectomy", "recovering"]
            },
            {
                "report": "55-year-old male with history of hypertension and diabetes. Admitted with chest pain, troponin elevated. ECG showed ST elevation in inferior leads. Diagnosed with NSTEMI. Started on aspirin, clopidogrel, metoprolol, atorvastatin. Discharged stable.",
                "summary": "55M, HTN, DM admitted with NSTEMI (inferior ST elevation, elevated troponin). Managed with antiplatelets, beta-blocker, statin. Discharged stable.",
                "facts": ["55-year-old male", "hypertension", "diabetes", "chest pain", "troponin elevated", "ECG ST elevation", "NSTEMI", "aspirin", "clopidogrel", "metoprolol", "atorvastatin", "discharged stable"]
            }
        ]

        self.constitutional_principles = [
            "Ensure patient confidentiality is maintained. Do not include any personally identifiable information (PII) beyond what is absolutely necessary for medical context.",
            "Do not hallucinate information. If a detail is not present in the original report, do not invent it.",
            "If unsure about a specific detail or interpretation, explicitly state the uncertainty.",
            "Maintain an objective and neutral tone. Avoid any subjective or biased language.",
            "Focus only on medical facts and clinical relevance."
        ]

    def _format_few_shot_prompt(self, report_text: str) -> str:
        prompt_parts = []
        for example in self.few_shot_examples:
            prompt_parts.append(f"Medical Report: {example['report']}\nSummary: {example['summary']}")
        prompt_parts.append(f"Medical Report: {report_text}\nSummary:")
        return "\n\n".join(prompt_parts)

    def generate_prompt(self, report_text: str, strategy: str = "role_based") -> str:
        if strategy == "base":
            prompt = self.base_template.format(report=report_text)
        elif strategy == "role_based":
            prompt = self.role_based_template.format(report=report_text)
            for principle in self.constitutional_principles:
                prompt += f"\n- {principle}"
            prompt += "\n"  # Add a new line for the summary section
        elif strategy == "few_shot":
            prompt = self._format_few_shot_prompt(report_text)
            prompt += "\n\nApply the following ethical guidelines:\n" + "\n".join([f"- {p}" for p in self.constitutional_principles]) + "\n"
        else:
            raise ValueError(f"Unknown prompt strategy: {strategy}")
        return prompt

# --- LLM Integration Service ---
class LLMService:
    def __init__(self):
        self.llm = LangchainOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.2, model_name="gpt-3.5-turbo-instruct")
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)

    def summarize_report(self, prompt: str) -> str:
        try:
            llm_chain = LLMChain(prompt=PromptTemplate(template="{prompt_input}", input_variables=["prompt_input"]), llm=self.llm)
            response = llm_chain.run(prompt_input=prompt)
            return response.strip()
        except Exception as e:
            logging.error(f"Error during LLM summarization: {e}")
            raise HTTPException(status_code=500, detail=f"LLM summarization failed: {e}")

    def extract_facts(self, text: str) -> list[str]:
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a highly analytical assistant. Extract the most important factual statements and clinical findings from the given medical text. Return them as a JSON list of strings under a 'facts' key."},
                    {"role": "user", "content": f"Extract facts from: {text}\nFacts:"}
                ],
                response_format={"type": "json_object"}
            )
            facts_json_str = response.choices[0].message.content
            # Ensure the response is parsable and contains the 'facts' key
            facts_data = json.loads(facts_json_str)
            if "facts" not in facts_data or not isinstance(facts_data["facts"], list):
                raise ValueError("LLM did not return facts in the expected JSON format.")
            return facts_data["facts"]
        except json.JSONDecodeError as e:
            logging.error(f"LLM returned invalid JSON for fact extraction: {facts_json_str}. Error: {e}")
            return []
        except Exception as e:
            logging.error(f"Error extracting facts with LLM: {e}")
            return []

# --- Evaluation Framework Module ---
class EvaluationFrameworkModule:
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
        self.autorating_template = """
        You are a medical summarization evaluator. Your task is to rate a generated summary against its original medical report based on factual accuracy, completeness of key information, conciseness, and adherence to ethical guidelines (e.g., no PII leakage, no hallucination).

        Original Medical Report:
        {original_report}

        Generated Summary:
        {generated_summary}

        Based on the above, provide a rating from 1 (poor) to 5 (excellent) for each criterion:
        1. Factual Accuracy (1-5):
        2. Completeness of Key Information (1-5):
        3. Conciseness (1-5):
        4. Ethical Adherence (1-5):
        Provide a short explanation for each rating and an overall feedback.
        """
        self.constitutional_check_template = """
        Review the following medical summary for adherence to ethical principles: patient confidentiality, no hallucination, stating uncertainty, objective tone, and focus on medical facts.
        Identify any violations or areas of concern. Your output should clearly state if any violations are found and provide details.

        Summary to review:
        {summary}

        Analysis:
        """

    def _calculate_f1_score(self, gold_facts: list[str], generated_facts: list[str]) -> float:
        gold_tokens = set(" ".join(gold_facts).lower().split())
        gen_tokens = set(" ".join(generated_facts).lower().split())

        if not gold_tokens and not gen_tokens:
            return 1.0

        intersection = len(gold_tokens.intersection(gen_tokens))
        precision = intersection / len(gen_tokens) if gen_tokens else 0
        recall = intersection / len(gold_tokens) if gold_tokens else 0

        if precision + recall == 0:
            return 0.0
        return (2 * precision * recall) / (precision + recall)

    def llm_autorating(self, original_report: str, generated_summary: str) -> EvaluationResult:
        prompt = self.autorating_template.format(
            original_report=original_report,
            generated_summary=generated_summary
        )
        try:
            response = self.llm_service.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a medical summarization evaluator. Provide ratings and feedback based on the original report and generated summary."},
                    {"role": "user", "content": prompt}
                ]
            )
            feedback = response.choices[0].message.content
            scores = re.findall(r"(Factual Accuracy|Completeness of Key Information|Conciseness|Ethical Adherence) \(1-5\): (\d)", feedback)
            
            overall_score = 0
            if scores:
                overall_score = sum(int(s[1]) for s in scores) / len(scores)

            return EvaluationResult(metric="LLM Autorating", score=round(overall_score, 2), feedback=feedback)
        except Exception as e:
            logging.error(f"Error during LLM autorating: {e}")
            return EvaluationResult(metric="LLM Autorating", score="N/A", feedback=f"Error: {e}")

    def round_trip_consistency(self, original_report: str, generated_summary: str) -> EvaluationResult:
        original_facts = self.llm_service.extract_facts(original_report)
        summary_facts = self.llm_service.extract_facts(generated_summary)

        f1 = self._calculate_f1_score(original_facts, summary_facts)
        feedback = f"Original facts extracted: {original_facts}\nSummary facts extracted: {summary_facts}"
        return EvaluationResult(metric="Round-trip Consistency (F1 Score)", score=round(f1, 2), feedback=feedback)

    def adversarial_evaluation(self, original_report: str, generated_summary: str) -> EvaluationResult:
        concerns = []
        # Example: Simple check for negation errors or over-certainty
        if "no history of" in original_report.lower() and "history of" in generated_summary.lower() and "no history of" not in generated_summary.lower():
            concerns.append("Potential hallucination or misrepresentation of medical history (original stated 'no history of', summary implies 'history of').")
        
        # Example: Check for overly definitive statements where original might be vague
        if re.search(r'\b(likely|possibly)\b', original_report, re.IGNORECASE) and not re.search(r'\b(likely|possibly|suggests|may be)\b', generated_summary, re.IGNORECASE):
            if re.search(r'\b(is|confirmed)\b', generated_summary, re.IGNORECASE):
                concerns.append("Summary appears overly definitive where original report indicated uncertainty.")

        score = 1.0 if not concerns else 0.5
        feedback = "No significant adversarial concerns found." if not concerns else f"Concerns: {'; '.join(concerns)}"
        return EvaluationResult(metric="Adversarial Check", score=score, feedback=feedback)

    def ethical_alignment_check(self, generated_summary: str) -> EvaluationResult:
        concerns = []
        # Basic PII regex patterns (Illustrative, real PII detection is much more robust)
        pii_patterns = [
            r"\b(?:DOB|Date of Birth|Patient ID|MRN|SSN|Social Security Number)\s*:\s*\S+",
            r"\b(?:Dr\.|Doctor|Physician)\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b",  # Catches full doctor names
            r"\b[A-Z][a-z]+\s+(?:Hospital|Clinic|Medical Center|Medical Group)\b", # Catches facility names
            r"\b(?:phone|tel|email)\s*:\s*\S+"
        ]
        for pattern in pii_patterns:
            if re.search(pattern, generated_summary, re.IGNORECASE):
                concerns.append(f"Potential PII detected: {re.search(pattern, generated_summary, re.IGNORECASE).group(0)}")

        # LLM-based check for constitutional principles
        prompt = self.constitutional_check_template.format(summary=generated_summary)
        try:
            response = self.llm_service.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an ethical AI compliance officer. Analyze medical summaries for adherence to principles like confidentiality, no hallucination, and objective tone."},
                    {"role": "user", "content": prompt}
                ]
            )
            llm_feedback = response.choices[0].message.content
            if "violation" in llm_feedback.lower() or "concern" in llm_feedback.lower() or "breach" in llm_feedback.lower():
                concerns.append(f"LLM identified ethical concerns: {llm_feedback}")
        except Exception as e:
            logging.error(f"Error during LLM ethical check: {e}")
            concerns.append(f"Error running LLM ethical check: {e}")

        score = 1.0 if not concerns else 0.0
        feedback = "No ethical concerns found." if not concerns else f"Concerns: {'; '.join(concerns)}"
        return EvaluationResult(metric="Ethical Alignment", score=score, feedback=feedback)

    def evaluate_summary(self, original_report: str, generated_summary: str) -> list[EvaluationResult]:
        evaluations = []
        evaluations.append(self.llm_autorating(original_report, generated_summary))
        evaluations.append(self.round_trip_consistency(original_report, generated_summary))
        evaluations.append(self.adversarial_evaluation(original_report, generated_summary))
        evaluations.append(self.ethical_alignment_check(generated_summary))
        return evaluations

# --- Orchestration & API Layer ---
prompt_factory = PromptFactory()
llm_service = LLMService()
evaluation_module = EvaluationFrameworkModule(llm_service=llm_service)

@app.post("/summarize", response_model=SummaryResponse)
async def summarize_medical_report(request: MedicalReportRequest):
    report_text = request.report_text
    if not report_text:
        raise HTTPException(status_code=400, detail="Medical report text cannot be empty.")

    logging.info("Generating summary...")
    try:
        # Defaulting to 'role_based' strategy. Can be extended to accept strategy from request.
        prompt = prompt_factory.generate_prompt(report_text, strategy="role_based")
        summary = llm_service.summarize_report(prompt)
    except HTTPException as e:
        raise e
    except Exception as e:
        logging.error(f"Unexpected error during summarization: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during summarization: {e}")

    logging.info("Evaluating summary...")
    try:
        evaluations = evaluation_module.evaluate_summary(report_text, summary)
    except Exception as e:
        logging.error(f"Error during summary evaluation: {e}")
        evaluations = [EvaluationResult(metric="Evaluation Error", score=0, feedback=f"Failed to evaluate: {e}")]

    return SummaryResponse(
        original_report=report_text,
        summary=summary,
        evaluation=evaluations
    )

@app.get("/")
async def root():
    return {"message": "Medical Report Summarization and Fact-Checking System API. Use /summarize endpoint for summarization."}

# To run this application, save it as `main.py` and execute:
# uvicorn main:app --reload

# Ensure you have an .env file in the same directory with OPENAI_API_KEY="your_openai_api_key"