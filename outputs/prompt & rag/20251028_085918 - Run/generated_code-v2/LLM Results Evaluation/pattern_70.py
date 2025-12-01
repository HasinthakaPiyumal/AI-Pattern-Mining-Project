
import collections

def tokenize(text):
    """Simple tokenizer to split text into lowercase words."""
    return set(text.lower().split())

def jaccard_similarity(set1, set2):
    """Calculates Jaccard Similarity between two sets."""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    if union == 0:
        return 0.0
    return intersection / union

def simulate_llm_response(prompt_text, query):
    """Simulates an LLM's response based on the prompt and query.
    In a real application, this would involve calling an actual LLM API.
    """
    # A very simple simulation: if prompt contains 'greeting', add a greeting.
    # If prompt contains 'issue', incorporate the query.
    if "greeting" in prompt_text.lower():
        base_response = "Hello! How can I assist you today? "
    else:
        base_response = ""

    if "{{query}}" in prompt_text:
        return f"{base_response}Regarding your query: {query}. We are looking into this."
    elif "explain" in prompt_text.lower():
        return f"{base_response}Let me explain: The solution to '{query}' usually involves several steps."
    else:
        return f"{base_response}Thank you for contacting us about '{query}'. We will provide a solution shortly."

def calculate_mutual_information_proxy(prompt_text, llm_output):
    """Calculates a proxy for mutual information using Jaccard similarity.
    A higher similarity indicates more shared information between the prompt and output.
    """
    prompt_tokens = tokenize(prompt_text)
    output_tokens = tokenize(llm_output)
    return jaccard_similarity(prompt_tokens, output_tokens)

def select_optimal_prompt(query, prompt_templates):
    """Selects the prompt template that maximizes the proxy mutual information
    with the simulated LLM output for a given query.
    """
    best_prompt = None
    max_mi = -1
    best_llm_output = ""

    for template in prompt_templates:
        # Format the template with the current query
        formatted_prompt = template.replace("{{query}}", query)

        # Simulate LLM response
        llm_output = simulate_llm_response(formatted_prompt, query)

        # Calculate proxy mutual information
        current_mi = calculate_mutual_information_proxy(formatted_prompt, llm_output)

        if current_mi > max_mi:
            max_mi = current_mi
            best_prompt = template
            best_llm_output = llm_output
            
    return best_prompt, best_llm_output, max_mi

def generate_chatbot_response(query, prompt_templates):
    """Generates a chatbot response by first selecting the optimal prompt
    and then using the simulated LLM response from that prompt.
    """
    print(f"\nCustomer Query: '{query}'")
    optimal_prompt_template, llm_response, mi_score = select_optimal_prompt(query, prompt_templates)
    print(f"Selected Optimal Prompt Template: '{optimal_prompt_template}' (MI Proxy: {mi_score:.3f})")
    print(f"Chatbot Response: {llm_response}")
    return llm_response

if __name__ == "__main__":
    # Define multiple prompt templates
    prompt_templates = [
        "Please provide a brief and direct answer to the customer's question: {{query}}",
        "Hello, I need you to explain in detail the solution for the following issue: {{query}}",
        "Generate a polite customer support response addressing: {{query}}",
        "What is the best way to handle this customer inquiry about: {{query}}"
    ]

    # Example usage with different customer queries
    generate_chatbot_response("My internet is not working", prompt_templates)
    generate_chatbot_response("How do I reset my password?", prompt_templates)
    generate_chatbot_response("I want to know my account balance", prompt_templates)
    generate_chatbot_response("Explain the benefits of the premium plan", prompt_templates)
