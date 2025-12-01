from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
import os

# Ensure you have your OpenAI API key set as an environment variable
# os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"

class MedicalSummary(BaseModel):
    patient_name: str = Field(description="The full name of the patient.")
    diagnosis: str = Field(description="The primary diagnosis mentioned in the report.")
    prescribed_medications: list[str] = Field(description="A list of all prescribed medications.")
    follow_up_date: str = Field(description="The recommended follow-up date, if specified.")

parser = JsonOutputParser(pydantic_object=MedicalSummary)

# Define the prompt template with explicit output format instructions
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert medical assistant tasked with extracting key information from medical reports."
            "Always output the extracted information as a JSON object adhering to the following schema:\n{format_instructions}\n"
            "If a piece of information is not available, provide an empty string or an empty list as appropriate.",
        ),
        ("human", "Extract information from the following medical report:\n\n{medical_report}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

# Initialize the LLM (ensure OPENAI_API_KEY is set in your environment)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Create the LangChain chain
chain = prompt | llm | parser

def summarize_medical_report(report_text: str) -> MedicalSummary:
    """Summarizes a medical report by extracting key entities into a structured JSON format."""
    try:
        summary = chain.invoke({"medical_report": report_text})
        return summary
    except Exception as e:
        print(f"An error occurred during summarization: {e}")
        return None

if __name__ == "__main__":
    print("\n--- Medical Report Summarizer ---")
    print("Please paste the medical report text. Type 'END' on a new line to finish input.")

    report_lines = []
    while True:
        line = input()
        if line.strip().upper() == 'END':
            break
        report_lines.append(line)
    
    medical_report_text = "\n".join(report_lines)

    if not medical_report_text.strip():
        print("No report text provided. Exiting.")
    else:
        print("\nProcessing report...")
        summary = summarize_medical_report(medical_report_text)

        if summary:
            print("\n--- Extracted Information (JSON) ---")
            print(summary.model_dump_json(indent=2))
            print("\n--- Validation Status ---")
            print("Output successfully validated against the Pydantic schema.")
        else:
            print("\nFailed to extract information or validate the output.")
