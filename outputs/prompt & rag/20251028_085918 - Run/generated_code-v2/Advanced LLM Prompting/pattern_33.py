def simulate_llm_response(prompt: str) -> str:
    if prompt.startswith("Read the question again: "):
        original_query = prompt.replace("Read the question again: ", "", 1)
        return f"LLM processed query with RE2 pattern. Original intent: '{original_query}'. Elaborating on the re-read query..."
    else:
        return f"LLM processed query without RE2 pattern. Query received: '{prompt}'. Generating standard response..."

def customer_support_agent(query: str, apply_re2: bool = True) -> str:
    processed_query = query
    if apply_re2:
        processed_query = f"Read the question again: {query}"
    
    llm_response = simulate_llm_response(processed_query)
    return llm_response

if __name__ == "__main__":
    print("--- Testing with RE2 enabled ---")
    complex_query_re2 = "I need to understand the implications of the new privacy policy changes, specifically how they affect data sharing with third-party advertising partners and what steps users can take to opt-out."
    response_re2 = customer_support_agent(complex_query_re2, apply_re2=True)
    print(f"Customer Query: {complex_query_re2}")
    print(f"Agent Response: {response_re2}")
    print("\n--- Testing without RE2 disabled ---")
    simple_query_no_re2 = "What are your operating hours?"
    response_no_re2 = customer_support_agent(simple_query_no_re2, apply_re2=False)
    print(f"Customer Query: {simple_query_no_re2}")
    print(f"Agent Response: {response_no_re2}")
    print("\n--- Testing with RE2 enabled for a simple query ---")
    simple_query_re2 = "How do I reset my password?"
    response_simple_re2 = customer_support_agent(simple_query_re2, apply_re2=True)
    print(f"Customer Query: {simple_query_re2}")
    print(f"Agent Response: {response_simple_re2}")