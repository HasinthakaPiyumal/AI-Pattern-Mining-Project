import os
import json
from enum import Enum
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

# 1. Environment Variable Loading
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY environment variable not set. Using mock client.")
    print("To use the real OpenAI API, please set the OPENAI_API_KEY in a .env file or as an environment variable.")

# Mock OpenAI Client for demonstration purposes if API key is not set
class MockOpenAIClient:
    def chat(self):
        return self

    def completions(self):
        return self

    def create(self, model, messages, response_model=None, **kwargs):
        # Simulate LLM response based on prompt. This is a simplification.
        prompt = messages[0]["content"] if messages else ""
        
        # Simple heuristic to mock responses
        if "refund" in prompt.lower() or "charge" in prompt.lower() or "billing" in prompt.lower():
            mock_category = "Billing"
            mock_sentiment = "Negative"
            product_match = next((word for word in ['Gizmo 2000', 'Quantum Leaper', 'Deluxe Widget', 'Standard Sprocket'] if word.lower() in prompt.lower()), None)
            order_match = next((s for s in prompt.split() if s.startswith(('ORD', 'REF', 'WID')) and len(s) > 3), None)
            mock_summary = "Customer is unhappy with a billing issue and requests a refund or clarification."
        elif "defect" in prompt.lower() or "power on" in prompt.lower() or "broken" in prompt.lower() or "technical" in prompt.lower():
            mock_category = "Technical Support"
            mock_sentiment = "Negative"
            product_match = next((word for word in ['Gizmo 2000', 'Quantum Leaper', 'Deluxe Widget', 'Standard Sprocket'] if word.lower() in prompt.lower()), None)
            order_match = next((s for s in prompt.split() if s.startswith(('ORD', 'REF', 'WID')) and len(s) > 3), None)
            mock_summary = "Customer reports a technical issue with a product, requiring troubleshooting or replacement."
        elif "release date" in prompt.lower() or "features" in prompt.lower() or "inquiry" in prompt.lower():
            mock_category = "Product Inquiry"
            mock_sentiment = "Neutral"
            product_match = next((word for word in ['Gizmo 2000', 'Quantum Leaper', 'Deluxe Widget', 'Standard Sprocket'] if word.lower() in prompt.lower()), None)
            order_match = None
            mock_summary = "Customer is asking about product information, such as release dates or features."
        elif "package" in prompt.lower() or "shipping" in prompt.lower() or "wrong product" in prompt.lower():
            mock_category = "Shipping"
            mock_sentiment = "Negative"
            product_match = next((word for word in ['Gizmo 2000', 'Quantum Leaper', 'Deluxe Widget', 'Standard Sprocket'] if word.lower() in prompt.lower()), None)
            order_match = next((s for s in prompt.split() if s.startswith(('ORD', 'REF', 'WID')) and len(s) > 3), None)
            mock_summary = "Customer received an incorrect product or has an issue with shipping."
        elif "upgrade" in prompt.lower() or "subscription" in prompt.lower():
            mock_category = "Billing"
            mock_sentiment = "Positive"
            product_match = None
            order_match = None
            mock_summary = "Customer inquiring about upgrading their subscription plan."
        elif "login" in prompt.lower() or "account" in prompt.lower() or "password" in prompt.lower():
            mock_category = "Technical Support"
            mock_sentiment = "Negative"
            product_match = None
            order_match = None
            mock_summary = "Customer is unable to log in or access their account."
        else:
            mock_category = "Other"
            mock_sentiment = "Neutral"
            product_match = None
            order_match = None
            mock_summary = "General inquiry not fitting other categories."

        response_dict = {
            "category": mock_category,
            "sentiment": mock_sentiment,
            "key_information": {
                "product_name": product_match,
                "order_id": order_match
            },
            "summary": mock_summary
        }
        return {"choices": [{"message": {"content": json.dumps(response_dict)}}]}

client = None
if OPENAI_API_KEY:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        print("OpenAI client initialized with API key.")
    except ImportError:
        print("Warning: 'openai' library not found. Please install it to use the real OpenAI API (pip install openai).")
        print("Falling back to mock client.")
        client = MockOpenAIClient()
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}. Falling back to mock client.")
        client = MockOpenAIClient()
else:
    client = MockOpenAIClient()
    print("Using mock OpenAI client.")


# 2. Pydantic Models for Structured Output
class TicketCategory(str, Enum):
    BILLING = "Billing"
    TECHNICAL_SUPPORT = "Technical Support"
    PRODUCT_INQUIRY = "Product Inquiry"
    SHIPPING = "Shipping"
    RETURNS = "Returns"
    OTHER = "Other"

class Sentiment(str, Enum):
    POSITIVE = "Positive"
    NEUTRAL = "Neutral"
    NEGATIVE = "Negative"

class KeyInformation(BaseModel):
    product_name: Optional[str] = Field(None, description="The name of the product mentioned in the ticket.")
    order_id: Optional[str] = Field(None, description="The order ID mentioned in the ticket.")

class TicketAnalysisOutput(BaseModel):
    category: TicketCategory = Field(..., description="The categorized type of the customer support ticket.")
    sentiment: Sentiment = Field(..., description="The overall sentiment expressed in the ticket.")
    key_information: KeyInformation = Field(..., description="Extracted key entities like product name and order ID.")
    summary: str = Field(..., description="A concise summary of the customer's issue.")

# 3. Prompt Engineering Module
def generate_ticket_analysis_prompt(ticket_text: str) -> str:
    """
    Generates the prompt for the LLM to categorize and summarize a customer support ticket
    into a structured JSON format.
    """
    # Use .model_json_schema() for Pydantic v2+
    schema = json.dumps(TicketAnalysisOutput.model_json_schema(), indent=2)

    prompt = f"""
You are an AI assistant designed to categorize and summarize customer support tickets.
Your goal is to extract key information and present it in a structured JSON format.

Here is the customer support ticket:
---
{ticket_text}
---

Please analyze the ticket and provide your response in the following JSON format.
Ensure your output is *only* the JSON object, with no additional text or formatting outside the JSON.

JSON Schema:
{schema}

Example of expected JSON output:
```json
{{
  "category": "Billing",
  "sentiment": "Negative",
  "key_information": {{
    "product_name": null,
    "order_id": "ORD123456"
  }},
  "summary": "Customer is disputing a charge of $25 and requests clarification."
}}
```
"""
    return prompt

# 4. LLM Interaction Module
def get_llm_structured_response(prompt: str, llm_client) -> dict:
    """
    Sends the engineered prompt to the LLM and attempts to get a JSON response.
    """
    llm_output = ""
    try:
        if isinstance(llm_client, MockOpenAIClient):
            response = llm_client.chat().completions().create(
                model="mock-model",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that outputs JSON."},
                    {"role": "user", "content": prompt}
                ]
            )
            llm_output = response["choices"][0]["message"]["content"]
        else: # Real OpenAI client
            response = llm_client.chat.completions.create(
                model="gpt-3.5-turbo", # Consider using a more capable model like 'gpt-4' or 'gpt-4o' for production
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that outputs JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={ "type": "json_object" } # Instruct OpenAI API for JSON output
            )
            llm_output = response.choices[0].message.content
        
        # For debugging purposes, you can uncomment this to see the raw LLM output
        # print(f"\nRaw LLM Output:\n{llm_output}")

        return json.loads(llm_output)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from LLM response: {e}")
        print(f"LLM raw output was: {llm_output}")
        raise ValueError("LLM did not return valid JSON. Check LLM output and prompt engineering.")
    except Exception as e:
        print(f"An error occurred during LLM interaction: {e}")
        raise

# 5. Main Processing Function
def process_customer_ticket(ticket_text: str) -> Optional[TicketAnalysisOutput]:
    """
    Processes a raw customer support ticket to generate a structured analysis.
    """
    print(f"\n--- Processing New Ticket ---")
    print(f"Ticket Text: \"{ticket_text[:100]}...\"")

    try:
        # 1. Generate prompt
        prompt = generate_ticket_analysis_prompt(ticket_text)
        
        # 2. Interact with LLM (using the globally configured client)
        llm_raw_output_dict = get_llm_structured_response(prompt, client)

        # 3. Validate and parse output using Pydantic
        structured_output = TicketAnalysisOutput.model_validate(llm_raw_output_dict)
        print("Ticket processed successfully.")
        return structured_output
    except ValidationError as e:
        print(f"Validation Error: LLM output did not match schema: {e}")
        return None
    except ValueError as e:
        print(f"Processing Error: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

# 6. Example Usage
if __name__ == "__main__":
    # Test cases
    ticket_1 = "I am very upset with the recent charge of $50 on my credit card. I never authorized this! I demand a refund for order ID REF456789 immediately."
    ticket_2 = "My new Gizmo 2000 arrived today (Order: ORD789012) but it won't power on. I've tried everything. It seems defective. Please help me fix it or replace it!"
    ticket_3 = "When will the new 'Quantum Leaper' be released? I'm so excited about its new features, especially the improved battery life. Is there a pre-order link yet?"
    ticket_4 = "I just received my package, but the product inside is completely wrong. I ordered the 'Deluxe Widget' (Order: WID112233) but got a 'Standard Sprocket' instead. This needs to be corrected quickly."
    ticket_5 = "My subscription renews on the 15th, but I want to upgrade to the premium plan. Will the upgrade be immediate, or will it wait until the next billing cycle?"
    ticket_6 = "I am having trouble logging into my account. My username is 'johndoe' and I tried resetting my password but didn't receive an email. Can you assist?"
    ticket_7 = "I need to return a product I bought last week. It's the 'Super Speaker' from order number XYZ98765. How do I initiate a return?"


    tickets = [ticket_1, ticket_2, ticket_3, ticket_4, ticket_5, ticket_6, ticket_7]

    for i, ticket in enumerate(tickets):
        result = process_customer_ticket(ticket)
        if result:
            print(f"\n--- Structured Output for Ticket {i+1} ---")
            print(json.dumps(result.model_dump(), indent=2))
            print("-" * 40)
        else:
            print(f"\n--- Failed to process Ticket {i+1} ---")
            print("-" * 40)

    print("\nDemonstration complete.")
