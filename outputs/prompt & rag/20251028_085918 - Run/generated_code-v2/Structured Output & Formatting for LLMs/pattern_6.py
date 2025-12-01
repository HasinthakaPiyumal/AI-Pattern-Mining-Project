from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationError
import json

app = FastAPI()

class TicketInquiry(BaseModel):
    inquiry_text: str

class TicketAnalysisResult(BaseModel):
    customer_name: str
    product: str
    issue: str
    urgency: str
    next_steps: str

def mock_llm_response(prompt: str) -> str:
    # This function simulates an LLM's response formatted as JSON.
    # In a real application, this would be an actual API call to an LLM provider.
    # For demonstration, we're returning a consistent structured JSON string.
    
    # Example of how the LLM would "understand" and extract based on the prompt.
    # For simplicity, this mock doesn't actually parse the prompt content to generate the response.
    # It just returns a valid structured JSON.
    
    # A more sophisticated mock could try to parse keywords from the prompt_text
    # but for demonstrating output formatting, a fixed valid JSON is sufficient.
    
    if "customer_name" in prompt and "product" in prompt and "issue" in prompt:
        return json.dumps({
            "customer_name": "John Doe",
            "product": "Premium Widget",
            "issue": "Device not powering on after update",
            "urgency": "High",
            "next_steps": "Schedule a callback with a senior technical support agent to troubleshoot software and hardware."
        })
    else:
        # Return an invalid or incomplete JSON if the prompt instructions were not met (simulated)
        return "{\"error\": \"Prompt instructions for structured output not clear.\"}"

@app.post("/analyze_ticket", response_model=TicketAnalysisResult)
async def analyze_ticket(ticket: TicketInquiry):
    prompt_template = """Extract the customer's name, product, issue, urgency, and proposed next steps from the following customer inquiry and provide the output as a JSON object with keys 'customer_name', 'product', 'issue', 'urgency', and 'next_steps'.

Customer Inquiry: {inquiry_text}

JSON Output:"""

    full_prompt = prompt_template.format(inquiry_text=ticket.inquiry_text)
    
    # Simulate LLM call
    llm_raw_response = mock_llm_response(full_prompt)
    
    try:
        # Attempt to parse the LLM's raw string response as JSON
        parsed_json = json.loads(llm_raw_response)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail=f"LLM returned invalid JSON: {llm_raw_response}")

    try:
        # Validate the parsed JSON against our Pydantic model
        analysis_result = TicketAnalysisResult(**parsed_json)
        return analysis_result
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=f"LLM output did not conform to schema: {e.errors()}")

