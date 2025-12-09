from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

# 2. Knowledge Base (KB) - Primarily in English
knowledge_base = {
    "return policy": "Our return policy allows returns within 30 days of purchase, provided the item is in its original condition. Please visit our website for more details.",
    "shipping costs": "Shipping costs vary based on your location and the shipping method chosen. Standard shipping is free for orders over $50.",
    "product warranty": "All our products come with a one-year manufacturer's warranty. For claims, please contact our support team with your proof of purchase.",
    "payment methods": "We accept major credit cards (Visa, MasterCard, Amex), PayPal, and bank transfers."
}

# 3. Cross-lingual In-Context Learning (InCLT) Example Store
inclt_examples = [
    {
        "source_lang": "de",
        "source_query": "Wie kann ich ein Produkt zurücksenden?",
        "target_equivalent": "How can I return a product?",
        "target_answer_topic": "return policy"
    },
    {
        "source_lang": "de",
        "source_query": "Ist die Rücksendung kostenlos?",
        "target_equivalent": "Are returns free?",
        "target_answer_topic": "shipping costs" # Simulating a slightly indirect mapping to show flexibility
    },
    {
        "source_lang": "es",
        "source_query": "Cuál es la política de devoluciones?",
        "target_equivalent": "What is your return policy?",
        "target_answer_topic": "return policy"
    },
    {
        "source_lang": "es",
        "source_query": "Cuánto cuesta el envío?",
        "target_equivalent": "How much is shipping?",
        "target_answer_topic": "shipping costs"
    }
]

# 2. Language Detection Module
def detect_language(text):
    try:
        return detect(text)
    except:
        return "en" # Default to English if detection fails

# 4. Cross-lingual Example Retrieval Module (Simple keyword matching for demo)
def retrieve_inclt_examples(query, detected_lang, inclt_examples, num_examples=2):
    relevant_examples = []
    query_lower = query.lower()
    for example in inclt_examples:
        if example["source_lang"] == detected_lang and any(keyword in example["source_query"].lower() for keyword in query_lower.split()):
            relevant_examples.append(example)
        if len(relevant_examples) >= num_examples:
            break
    return relevant_examples

# 5. Prompt Construction Module
def construct_prompt(customer_query, detected_lang, relevant_examples):
    prompt_parts = ["You are a helpful multilingual customer support assistant. Your goal is to understand the customer's query, possibly translate it to English, and provide an answer in English based on the provided knowledge."]

    if relevant_examples:
        prompt_parts.append("\\nHere are some examples of queries and their English equivalents/answers to help you:")
        for ex in relevant_examples:
            prompt_parts.append(f"Customer ({ex['source_lang']}): {ex['source_query']}")
            prompt_parts.append(f"English Equivalent: {ex['target_equivalent']}")
            if ex['target_answer_topic'] and ex['target_answer_topic'] in knowledge_base:
                prompt_parts.append(f"English Answer: {knowledge_base[ex['target_answer_topic']]}")
            prompt_parts.append("")

    prompt_parts.append(f"\\nCustomer ({detected_lang}): {customer_query}")
    prompt_parts.append("English Equivalent:")
    prompt_parts.append("English Answer:")

    return "\n".join(prompt_parts)

# 6. Large Language Model (LLM) Interaction Module (Simulated)
def simulate_llm_response(prompt):
    # In a real application, this would call an actual LLM (e.g., via transformers library, OpenAI API, etc.)
    # For demonstration, we'll try to extract the English equivalent and find an answer from KB.
    
    # Simple simulation logic
    lines = prompt.split('\n')
    customer_query_line = [line for line in lines if line.startswith(f"Customer (") and line.endswith(":")]
    if customer_query_line:
        original_query = customer_query_line[-1].split(': ', 1)[1]
    else:
        original_query = ""

    # Try to find an English equivalent from the prompt's examples or guess based on keywords
    english_equivalent = ""
    for line in lines:
        if line.startswith("English Equivalent:") and line != "English Equivalent:":
            english_equivalent = line.split(': ', 1)[1]
            break
    
    if not english_equivalent:
        # Very basic keyword-based translation for simulation if no example match
        if "Rücksendung" in original_query or "return" in original_query.lower():
            english_equivalent = "What is the return policy?"
        elif "Versandkosten" in original_query or "shipping" in original_query.lower():
            english_equivalent = "What are the shipping costs?"
        elif "garantia" in original_query or "warranty" in original_query.lower():
            english_equivalent = "What is the product warranty?"
        elif "pago" in original_query or "payment" in original_query.lower():
            english_equivalent = "What payment methods are available?"
        else:
            english_equivalent = f"query about {original_query}"
            
    response_answer = "I'm sorry, I couldn't find a definitive answer in the knowledge base for that specific query." # Default generic response
    for topic, answer in knowledge_base.items():
        if topic in english_equivalent.lower():
            response_answer = answer
            break

    return f"English Equivalent: {english_equivalent}\nEnglish Answer: {response_answer}"

# Main Chatbot Logic
def chat_with_bot(customer_query):
    print(f"Customer: {customer_query}")

    # 1. Language Detection
    detected_lang = detect_language(customer_query)
    print(f"Detected Language: {detected_lang}")

    # 2. Cross-lingual Example Retrieval
    relevant_examples = retrieve_inclt_examples(customer_query, detected_lang, inclt_examples)
    print(f"Relevant InCLT Examples Found: {len(relevant_examples)}")

    # 3. Prompt Construction
    prompt = construct_prompt(customer_query, detected_lang, relevant_examples)
    # print(f"\\n--- Constructed Prompt ---\\n{prompt}\\n--------------------------")

    # 4. LLM Interaction (Simulated)
    llm_output = simulate_llm_response(prompt)
    # print(f"\\n--- LLM Raw Output ---\\n{llm_output}\\n----------------------")

    # 7. Response Formatting (extract answer from simulated LLM output)
    response_lines = llm_output.split('\n')
    final_answer = ""
    for line in response_lines:
        if line.startswith("English Answer:"):
            final_answer = line.split(': ', 1)[1]
            break
    
    if not final_answer: # Fallback if for some reason 'English Answer:' wasn't found
        final_answer = llm_output

    print(f"Chatbot (English): {final_answer}")
    print("\n" + "="*50 + "\n")
    return final_answer

if __name__ == "__main__":
    print("Multilingual Customer Support Chatbot (InCLT Crosslingual Transfer Prompting Demo)")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        chat_with_bot(user_input)
