def get_llm_binary_score(email_content: str) -> str:
    urgent_keywords = ["urgent", "critical", "immediately", "down", "bug", "issue", "emergency", "failed", "escalate"]
    email_content_lower = email_content.lower()
    for keyword in urgent_keywords:
        if keyword in email_content_lower:
            return "True"
    return "False"

def triage_email(llm_score: str) -> str:
    if llm_score == "True":
        return "Urgent"
    else:
        return "Non-Urgent"

if __name__ == "__main__":
    simulated_emails = [
        "My website is down and customers cannot access it! This is urgent.",
        "I have a question about my last order, when will it ship?",
        "There's a critical bug in the latest software update, please fix it immediately.",
        "I'd like to update my contact information.",
        "My payment failed unexpectedly. Please investigate.",
        "Can you provide more details about your new product line?"
    ]

    print("--- Automated Email Triage System ---")
    for i, email in enumerate(simulated_emails):
        print(f"\nEmail {i + 1}:")
        print(f"Content: {email}")
        
        llm_binary_result = get_llm_binary_score(email)
        urgency_level = triage_email(llm_binary_result)
        
        print(f"LLM Binary Score: {llm_binary_result}")
        print(f"Triage Urgency: {urgency_level}")