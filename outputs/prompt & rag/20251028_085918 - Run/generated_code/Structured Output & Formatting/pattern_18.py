import json
import re

def process_support_ticket(ticket_text: str) -> dict:
    """
    Processes a free-form customer support ticket using a simulated LLM
    to categorize it and extract key information into a structured JSON format.

    Args:
        ticket_text: The raw, free-form text of the customer support ticket.

    Returns:
        A dictionary containing the structured information (category, priority,
        extracted entities, summary) parsed from the LLM's output.
    """
    # In a real application, this would be an API call to an LLM (e.g., OpenAI, Gemini).
    # The prompt would instruct the LLM to generate JSON.
    
    # --- Simulated LLM Prompt Engineering ---
    # The actual prompt sent to the LLM would look something like this:
    """
    Analyze the following customer support ticket. Categorize it, assign a priority,
    extract key entities (like product names, order IDs, customer IDs), and provide a concise summary.
    Output the result strictly in JSON format with the following keys:
    'category': (e.g., 'Billing', 'Technical Issue', 'Product Inquiry', 'Shipping')
    'priority': (e.g., 'High', 'Medium', 'Low')
    'extracted_entities': A list of objects, each with 'type' and 'value' (e.g., [{"type": "order_id", "value": "ORD12345"}] )
    'summary': A brief summary of the issue.

    Customer Support Ticket:
    "{ticket_text}"
    """

    # --- Simulated LLM Response ---
    # For demonstration, we'll simulate the LLM's JSON output based on keywords.
    # This part would be replaced by an actual LLM API call and parsing of its response.
    if "billing" in ticket_text.lower() or "invoice" in ticket_text.lower():
        simulated_llm_output = {
            "category": "Billing",
            "priority": "High",
            "extracted_entities": [{"type": "customer_id", "value": "CUST7890"}], # Placeholder
            "summary": "Customer has an issue with their latest invoice."
        }
    elif "not working" in ticket_text.lower() or "bug" in ticket_text.lower() or "error" in ticket_text.lower():
        simulated_llm_output = {
            "category": "Technical Issue",
            "priority": "High",
            "extracted_entities": [{"type": "product", "value": "Software X"}], # Placeholder
            "summary": "User reports a critical bug in Software X functionality."
        }
    elif "delivery" in ticket_text.lower() or "shipment" in ticket_text.lower() or "order #" in ticket_text.lower():
        order_id = "N/A"
        match = re.search(r"order #([a-zA-Z0-9]+)", ticket_text, re.IGNORECASE)
        if match:
            order_id = match.group(1)
        simulated_llm_output = {
            "category": "Shipping",
            "priority": "Medium",
            "extracted_entities": [{"type": "order_id", "value": order_id}],
            "summary": "Customer inquiring about the status of their recent order."
        }
    else:
        simulated_llm_output = {
            "category": "General Inquiry",
            "priority": "Low",
            "extracted_entities": [],
            "summary": "General question from the customer."
        }
    
    # In a real scenario, you'd parse the actual LLM text response into JSON
    # For this simulation, we already have a dict.
    # structured_output = json.loads(llm_raw_text_response) # Example if LLM returns a JSON string

    return simulated_llm_output

if __name__ == "__main__":
    print("--- Automated Customer Support Ticket Processor ---")

    ticket1 = "My internet service is not working. I can't connect to anything since yesterday. My account ID is 12345."
    print(f"\nProcessing Ticket 1:\n'{ticket1}'")
    structured_data1 = process_support_ticket(ticket1)
    print("Structured Output 1:")
    print(json.dumps(structured_data1, indent=2))

    ticket2 = "I have a question about my latest invoice. It seems too high. My customer ID is CUST-001."
    print(f"\nProcessing Ticket 2:\n'{ticket2}'")
    structured_data2 = process_support_ticket(ticket2)
    print("Structured Output 2:")
    print(json.dumps(structured_data2, indent=2))

    ticket3 = "Where is my order #XYZ789? It was supposed to arrive last week."
    print(f"\nProcessing Ticket 3:\n'{ticket3}'")
    structured_data3 = process_support_ticket(ticket3)
    print("Structured Output 3:")
    print(json.dumps(structured_data3, indent=2))

    ticket4 = "Just a general question about your new product features."
    print(f"\nProcessing Ticket 4:\n'{ticket4}'")
    structured_data4 = process_support_ticket(ticket4)
    print("Structured Output 4:")
    print(json.dumps(structured_data4, indent=2))