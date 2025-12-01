import json
from pydantic import BaseModel, Field
from transformers import pipeline
import gradio as gr

class MedicalSummary(BaseModel):
    patient_name: str = Field(..., description="Full name of the patient")
    date_of_report: str = Field(..., description="Date when the medical report was issued (YYYY-MM-DD)")
    diagnosis: str = Field(..., description="Primary diagnosis of the patient")
    key_findings: list[str] = Field(..., description="List of the most important findings from the report")
    treatment_plan: str = Field(..., description="Overview of the proposed treatment plan")
    medications: list[str] = Field(..., description="List of prescribed medications")
    follow_up: str = Field(..., description="Recommended follow-up actions or appointments")

class MedicalReportSummarizer:
    def __init__(self):
        self.summarizer_pipeline = pipeline("summarization", model="t5-small")

    def _generate_prompt(self, medical_notes: str, output_schema: str) -> str:
        prompt = f"""Summarize the following medical notes into a structured JSON format. Adhere strictly to the provided JSON schema. Ensure all fields are present and valid according to their descriptions.

Medical Notes:
{medical_notes}

JSON Schema:
{output_schema}

Output ONLY the JSON summary without any additional text or formatting. Make sure the JSON is valid and complete."""
        return prompt

    def _call_llm(self, prompt: str) -> str:
        result = self.summarizer_pipeline(prompt, max_length=512, min_length=30, do_sample=False)
        return result[0]['summary_text']

    def summarize(self, medical_notes: str) -> dict:
        output_schema_str = MedicalSummary.schema_json(indent=2)
        prompt = self._generate_prompt(medical_notes, output_schema_str)
        raw_llm_output = self._call_llm(prompt)

        try:
            json_output = json.loads(raw_llm_output)
            validated_summary = MedicalSummary(**json_output)
            return validated_summary.dict()
        except json.JSONDecodeError as e:
            return {"error": "LLM output was not valid JSON", "details": str(e), "raw_output": raw_llm_output}
        except Exception as e:
            return {"error": "Validation or parsing error", "details": str(e), "raw_output": raw_llm_output}

def gradio_interface(notes: str):
    summarizer = MedicalReportSummarizer()
    summary = summarizer.summarize(notes)
    return summary

if __name__ == "__main__":
    iface = gr.Interface(
        fn=gradio_interface,
        inputs=gr.Textbox(lines=10, label="Medical Notes", placeholder="Enter unstructured medical notes here..."),
        outputs=gr.JSON(label="Structured Medical Summary"),
        title="Medical Report Summarization AI",
        description="Enter unstructured medical notes, and the AI will summarize them into a structured JSON format."
    )
    iface.launch()