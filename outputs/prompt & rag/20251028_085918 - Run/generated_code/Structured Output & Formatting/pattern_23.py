import os
import json
from enum import Enum
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError
from loguru import logger
import openai

# Load environment variables
load_dotenv()

# Configure OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    logger.error("OPENAI_API_KEY environment variable not set.")
    raise ValueError("OPENAI_API_KEY environment variable not set.")

# Initialize OpenAI client
client = openai.OpenAI(api_key=openai_api_key)

# --- Pydantic Models for Structured Output ---

class TicketUrgency(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class TicketCategory(str, Enum):
    TECHNICAL_SUPPORT = "Technical Support"
    BILLING_INQUIRY = "Billing Inquiry"
    FEATURE_REQUEST = "Feature Request"
    ACCOUNT_MANAGEMENT = "Account Management"
    PRODUCT_INQUIRY = "Product Inquiry"
    OTHER = "Other"

class ExtractedTicketInfo(BaseModel):
    ticket_id: str = Field(..., description="Unique identifier for the support ticket.")
    customer_name: str = Field(..., description="Name of the customer.")
    email: str = Field(..., description="Email address of the customer.")
    problem_summary: str = Field(..., description="A concise summary of the customer's problem.")
    product_affected: str = Field(..., description="The product or service affected, if known.")
    urgency: TicketUrgency = Field(..., description="The urgency level of the ticket.")
    category: TicketCategory = Field(..., description="The primary category of the support ticket.")
    recommended_action: str = Field(..., description="A brief recommended next step or action.")

class TicketInquiry(BaseModel):
    inquiry_text: str = Field(..., description="The raw, unstructured customer support inquiry text.")

class ProcessedTicketResponse(ExtractedTicketInfo):
    assigned_department: str = Field(..., description="The department assigned to handle this ticket.")

# --- LLM Processing Module ---

def extract_and_categorize_ticket_info(inquiry_text: str) -> ExtractedTicketInfo:
    """Extracts structured information from unstructured ticket text using an LLM."""
    logger.info(f"Processing inquiry: {inquiry_text[:100]}...")

    # Generate JSON schema for the LLM to follow
    json_schema = json.dumps(ExtractedTicketInfo.model_json_schema(), indent=2)

    prompt = f"""You are an AI assistant designed to extract key information from customer support inquiries and format it as a JSON object. The JSON object must strictly adhere to the following Pydantic schema. Provide only the JSON object in your response, no additional text or formatting outside the JSON.

JSON Schema:
{json_schema}

Customer Inquiry:
{inquiry_text}

Extracted Information (JSON only):
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # Using a capable model for structured output
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts information in JSON format."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}, # Explicitly request JSON
            temperature=0.0 # Keep temperature low for structured output
        )

        llm_output_str = response.choices[0].message.content
        logger.debug(f"Raw LLM output: {llm_output_str}")

        # Parse the JSON string from the LLM
        llm_data = json.loads(llm_output_str)

        # Validate against Pydantic model
        extracted_info = ExtractedTicketInfo.model_validate(llm_data)
        logger.info("Successfully extracted and validated ticket information.")
        return extracted_info

    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON from LLM: {e} - Raw output: {llm_output_str}")
        raise HTTPException(status_code=500, detail=f"LLM output not valid JSON: {e}")
    except ValidationError as e:
        logger.error(f"Pydantic validation failed for LLM output: {e}")
        raise HTTPException(status_code=500, detail=f"LLM output schema mismatch: {e}")
    except openai.APIError as e:
        logger.error(f"OpenAI API error: {e}")
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during LLM processing: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

# --- Categorization and Assignment Logic Module ---

def assign_department(category: TicketCategory) -> str:
    """Assigns a department based on the ticket category."""
    department_map = {
        TicketCategory.TECHNICAL_SUPPORT: "Technical Support Department",
        TicketCategory.BILLING_INQUIRY: "Billing Department",
        TicketCategory.FEATURE_REQUEST: "Product Development Department",
        TicketCategory.ACCOUNT_MANAGEMENT: "Account Management Department",
        TicketCategory.PRODUCT_INQUIRY: "Sales Department",
        TicketCategory.OTHER: "General Support Department",
    }
    assigned_dept = department_map.get(category, "Unassigned")
    logger.info(f"Assigned department: {assigned_dept} for category: {category}")
    return assigned_dept

# --- FastAPI Application ---

app = FastAPI(
    title="Automated Customer Support Ticket Processor",
    description="API for extracting structured information and categorizing customer support inquiries using LLMs."
)

@app.post("/process-ticket", response_model=ProcessedTicketResponse)
async def process_customer_ticket(inquiry: TicketInquiry):
    """Processes a raw customer support inquiry to extract structured data and categorize it."""
    logger.info("Received new customer inquiry.")
    
    # 1. LLM Processing and Information Extraction
    extracted_info = extract_and_categorize_ticket_info(inquiry.inquiry_text)
    
    # 2. Categorization and Department Assignment
    assigned_dept = assign_department(extracted_info.category)
    
    # 3. CRM Simulation/Integration (return combined data)
    response_data = extracted_info.model_dump()
    response_data["assigned_department"] = assigned_dept
    
    logger.info("Ticket successfully processed and categorized.")
    return ProcessedTicketResponse(**response_data)

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI application...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
