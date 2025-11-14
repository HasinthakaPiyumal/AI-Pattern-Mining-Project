import json
from typing import List, Literal, Optional
from pydantic import BaseModel, ValidationError

# 1. Structured Output Pydantic Model
class StructuredTicketOutput(BaseModel):
    ticket_id: str
    customer_name: str
    issue_summary: str
    product_or_service: str
    priority: Literal["Low", "Medium", "High", "Urgent"]
    sentiment: Literal["Positive", "Neutral", "Negative"]
    action_items: List[str]

# 2. LLM Integration Layer (Simulated)
# In a real application, this would interact with an actual LLM API (e.g., OpenAI, Gemini).
def simulated_llm_response(prompt: str, ticket_text: str) -> str:
    """
    Simulates an LLM generating a structured JSON response based on the prompt
    and ticket text. For demonstration, it returns a fixed JSON string.
    """
    print(f"\n--- Simulated LLM Input ---")
    print(f"Prompt: {prompt[:200]}...") # Print a truncated prompt
    print(f"Ticket Text: {ticket_text[:100]}...") # Print a truncated ticket
    print(f"-------------------------\n")

    # This is a hardcoded example of what a real LLM would generate
    # based on the prompt, formatted as JSON.
    if "Refund for delayed delivery" in ticket_text:
        return json.dumps({
            "ticket_id": "CS-78901",
            "customer_name": "Jane Doe",
            "issue_summary": "Customer requesting a refund for a delayed delivery of product X. Delivery was 3 days late.",
            "product_or_service": "Product X",
            "priority": "High",
            "sentiment": "Negative",
            "action_items": [
                "Process refund for Product X",
                "Apologize for delay",
                "Offer discount on next purchase"
            ]
        })
    elif "Difficulty configuring router" in ticket_text:
         return json.dumps({
            "ticket_id": "CS-78902",
            "customer_name": "John Smith",
            "issue_summary": "Customer is having trouble setting up their new wireless router Model Y and needs technical assistance.",
            "product_or_service": "Wireless Router Model Y",
            "priority": "Medium",
            "sentiment": "Negative",
            "action_items": [
                "Schedule technical support call",
                "Send link to setup guide",
                "Troubleshoot common configuration issues"
            ]
        })
    else:
        return json.dumps({
            "ticket_id": "CS-00000",
            "customer_name": "Test User",
            "issue_summary": "Generic issue summary for a test ticket.",
            "product_or_service": "Unknown Product/Service",
            "priority": "Low",
            "sentiment": "Neutral",
            "action_items": [
                "Investigate further"
            ]
        })

# 3. Ticket Processing Logic
class TicketProcessor:
    def __init__(self, llm_inference_func):
        self.llm_inference_func = llm_inference_func

    def _create_prompt(self, ticket_text: str) -> str:
        """
        Generates a detailed prompt for the LLM to extract structured information.
        """
        prompt = f"""
        Analyze the following customer support ticket and extract the key information.
        Format your response as a JSON object strictly adhering to the following schema:
        {{
            "ticket_id": "<unique_ticket_identifier>",
            "customer_name": "<customer_full_name>",
            "issue_summary": "<concise_summary_of_the_issue>",
            "product_or_service": "<name_of_product_or_service_involved>",
            "priority": "<Low|Medium|High|Urgent>",
            "sentiment": "<Positive|Neutral|Negative>",
            "action_items": [
                "<action_item_1>",
                "<action_item_2>",
                "..."
            ]
        }}

        Ensure that 'action_items' is always a list, even if empty. The 'priority'
        and 'sentiment' fields must strictly be one of the specified literal values.

        Customer Support Ticket:
        """
        prompt += f"""
        {ticket_text}
        """
        return prompt

    def process_ticket(self, raw_ticket_text: str) -> Optional[StructuredTicketOutput]:
        """
        Processes a raw customer ticket to extract structured information.
        """
        prompt = self._create_prompt(raw_ticket_text)
        llm_json_string = self.llm_inference_func(prompt, raw_ticket_text)

        try:
            # Parse the JSON string from the LLM
            parsed_data = json.loads(llm_json_string)
            # Validate the parsed data against our Pydantic model
            structured_data = StructuredTicketOutput(**parsed_data)
            print(f"\nSuccessfully extracted structured data for ticket.")
            return structured_data
        except json.JSONDecodeError as e:
            print(f"\nError: LLM did not return valid JSON. Details: {e}")
            print(f"LLM Raw Response: {llm_json_string}")
            return None
        except ValidationError as e:
            print(f"\nError: LLM output did not conform to the Pydantic schema. Details: {e}")
            print(f"LLM Raw Response: {llm_json_string}")
            return None
        except Exception as e:
            print(f"\nAn unexpected error occurred during processing: {e}")
            return None

# 4. Main Application Logic
if __name__ == "__main__":
    # Simulate incoming raw customer support tickets
    ticket1_text = (
        "Hello, I need help. My order #12345 for Product X was supposed to arrive on Monday, "
        "but it's now Thursday and it's still not here. This is very frustrating, I want a refund!"
    )
    ticket2_text = (
        "I just bought your new wireless router, Model Y, and I cannot for the life of me "
        "get it to connect to my internet. The instructions are very confusing. Please help!"
    )
    ticket3_text = (
        "I really love your service! Everything works perfectly. Just wanted to say thanks."
    )

    # Initialize the TicketProcessor with the simulated LLM
    processor = TicketProcessor(simulated_llm_response)

    print("\n--- Processing Ticket 1 ---")
    structured_output1 = processor.process_ticket(ticket1_text)
    if structured_output1:
        print("\nStructured Output 1 (Pydantic Object):")
        print(structured_output1.model_dump_json(indent=2))

    print("\n--- Processing Ticket 2 ---")
    structured_output2 = processor.process_ticket(ticket2_text)
    if structured_output2:
        print("\nStructured Output 2 (Pydantic Object):")
        print(structured_output2.model_dump_json(indent=2))

    print("\n--- Processing Ticket 3 ---")
    structured_output3 = processor.process_ticket(ticket3_text)
    if structured_output3:
        print("\nStructured Output 3 (Pydantic Object):")
        print(structured_output3.model_dump_json(indent=2))

    # Example of a malformed LLM response (simulated error)
    class MalformedLLM:
        def __call__(self, prompt, ticket_text):
            return "this is not valid json"

    print("\n--- Processing Ticket with Malformed LLM Response ---")
    malformed_processor = TicketProcessor(MalformedLLM())
    malformed_output = malformed_processor.process_ticket(ticket1_text)
    if malformed_output is None:
        print("\nSuccessfully handled malformed JSON response.")

    class SchemaViolatingLLM:
        def __call__(self, prompt, ticket_text):
            return json.dumps({
                "ticket_id": "CS-ERROR",
                "customer_name": "Invalid Priority User",
                "issue_summary": "LLM tried to set an invalid priority.",
                "product_or_service": "N/A",
                "priority": "CRITICAL", # Invalid priority
                "sentiment": "Neutral",
                "action_items": []
            })

    print("\n--- Processing Ticket with Schema Violating LLM Response ---")
    schema_violating_processor = TicketProcessor(SchemaViolatingLLM())
    schema_violating_output = schema_violating_processor.process_ticket(ticket1_text)
    if schema_violating_output is None:
        print("\nSuccessfully handled schema violating JSON response.")
