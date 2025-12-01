import json

def mock_llm_evaluate_ticket(ticket_text: str) -> str:
    # This function simulates an LLM call that returns a JSON string.
    # In a real application, you would replace this with actual LLM API calls
    # (e.g., using openai.ChatCompletion.create or a Hugging Face pipeline).

    # The prompt engineering would involve telling the LLM to output in JSON.
    # For this mock, we'll hardcode some logic based on keywords.
    sentiment = "neutral"
    category = "general inquiry"
    summary = f"Customer inquiry about: {ticket_text[:50]}..."

    if "issue" in ticket_text.lower() or "problem" in ticket_text.lower() or "broken" in ticket_text.lower():
        sentiment = "negative"
        category = "technical issue"
        summary = f"Customer reports an issue: {ticket_text[:50]}..."
    elif "thank you" in ticket_text.lower() or "appreciate" in ticket_text.lower() or "happy" in ticket_text.lower():
        sentiment = "positive"
        category = "feedback"
        summary = f"Positive feedback received: {ticket_text[:50]}..."
    elif "question" in ticket_text.lower() or "?" in ticket_text:
        category = "information request"
        summary = f"Customer asks a question: {ticket_text[:50]}..."

    response_data = {
        "sentiment": sentiment,
        "category": category,
        "summary": summary,
        "original_ticket": ticket_text
    }
    return json.dumps(response_data, indent=2)

def analyze_support_ticket(ticket_text: str) -> dict:
    """
    Analyzes a customer support ticket using a mock LLM and returns structured JSON output.
    """
    print(f"Analyzing ticket:\n---START---\n{ticket_text}\n---END---")

    # Simulate LLM call to get structured JSON output
    llm_raw_output = mock_llm_evaluate_ticket(ticket_text)
    print("\nLLM Raw Output (simulated):")
    print(llm_raw_output)

    try:
        # Parse the JSON output from the LLM
        parsed_data = json.loads(llm_raw_output)
        print("\nSuccessfully parsed LLM output.")
        return parsed_data
    except json.JSONDecodeError as e:
        print(f"\nError parsing JSON from LLM output: {e}")
        print("Raw output that caused error:", llm_raw_output)
        return {"error": "JSON parsing failed", "raw_output": llm_raw_output}

if __name__ == "__main__":
    sample_ticket_1 = "I am having a major issue with my internet connection. It keeps dropping every hour. Please help!"
    sample_ticket_2 = "Just wanted to say thank you for the quick resolution to my previous ticket. Great support!"
    sample_ticket_3 = "What are your operating hours during the holidays? I need to speak with someone about billing."
    sample_ticket_4 = "I received my order, everything looks good."

    print("\n--- Processing Sample Ticket 1 ---")
    analysis_1 = analyze_support_ticket(sample_ticket_1)
    print("\nAnalysis Result 1:")
    print(json.dumps(analysis_1, indent=2))

    print("\n--- Processing Sample Ticket 2 ---")
    analysis_2 = analyze_support_ticket(sample_ticket_2)
    print("\nAnalysis Result 2:")
    print(json.dumps(analysis_2, indent=2))

    print("\n--- Processing Sample Ticket 3 ---")
    analysis_3 = analyze_support_ticket(sample_ticket_3)
    print("\nAnalysis Result 3:")
    print(json.dumps(analysis_3, indent=2))

    print("\n--- Processing Sample Ticket 4 ---")
    analysis_4 = analyze_support_ticket(sample_ticket_4)
    print("\nAnalysis Result 4:")
    print(json.dumps(analysis_4, indent=2))