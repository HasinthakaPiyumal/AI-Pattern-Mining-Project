import random

def prepare_finetuning_data(raw_data, abstention_percentage=0.2, abstention_phrase="I don't know, please provide more details or contact human support."):
    """
    Prepares a finetuning dataset by replacing a subset of answers with an abstention phrase.

    Args:
        raw_data (list): A list of dictionaries, each with 'query' and 'answer' keys.
        abstention_percentage (float): The percentage of examples to modify for abstention.
        abstention_phrase (str): The phrase to use for abstention.

    Returns:
        list: The modified dataset ready for finetuning.
    """
    modified_data = [item.copy() for item in raw_data]
    num_to_abstain = int(len(modified_data) * abstention_percentage)

    # Randomly select indices to replace answers with abstention phrase
    abstention_indices = random.sample(range(len(modified_data)), num_to_abstain)

    print(f"\n--- Dataset Preparation ---")
    print(f"Original dataset size: {len(raw_data)}")
    print(f"Number of samples to modify for abstention: {num_to_abstain}")

    for i in abstention_indices:
        modified_data[i]["answer"] = abstention_phrase

    print(f"First 5 modified samples (showing abstention if applied):\n")
    for i in range(min(5, len(modified_data))):
        print(f"Query: {modified_data[i]['query']}\nAnswer: {modified_data[i]['answer']}\n")

    return modified_data

def simulate_llm_response(finetuned_model_placeholder, query):
    """
    Simulates a response from a finetuned LLM, demonstrating abstention behavior.
    In a real application, 'finetuned_model_placeholder' would be an actual loaded LLM
    and its prediction would be used.

    Args:
        finetuned_model_placeholder: A placeholder for the finetuned LLM (not used directly).
        query (str): The customer query.

    Returns:
        str: The simulated response.
    """
    # This is a simplified simulation. A real LLM would process the query
    # and its confidence/uncertainty would determine abstention.
    known_answers = {
        "what is the status of my order 12345": "Your order 12345 is currently being processed and is expected to ship within 2 business days.",
        "how do i return a product": "You can initiate a return by visiting our 'Returns & Exchanges' page on the website and following the instructions.",
        "what are your shipping options": "We offer standard, expedited, and express shipping options. Details are available on our shipping information page.",
        "reset my password": "You can reset your password by clicking on 'Forgot Password' on the login page and entering your email address."
    }

    abstention_triggers = [
        "tell me about quantum physics",
        "what is the capital of mars",
        "predict the stock market tomorrow",
        "how many stars are in the universe",
        "what is the meaning of life",
        "i need a deep philosophical answer"
    ]

    query_lower = query.lower()

    for trigger in abstention_triggers:
        if trigger in query_lower:
            return "I don't know, please provide more details or contact human support. This query is outside my current knowledge domain."

    for known_query, answer in known_answers.items():
        if known_query in query_lower:
            return answer

    # Default abstention for queries not explicitly handled, simulating uncertainty
    return "I'm not sure how to answer that question. Could you please rephrase it or provide more context?"


def main_conceptual_pipeline():
    print("Starting Conceptual LLM Finetuning for Controlled Abstention Pipeline")

    # 1. Simulate Raw Customer Interaction Data
    raw_customer_data = [
        {"query": "What is the return policy for electronics?", "answer": "You can return electronics within 30 days of purchase, provided they are in their original packaging and condition."},
        {"query": "How do I track my order?", "answer": "You can track your order by logging into your account and navigating to the 'My Orders' section."},
        {"query": "What are the payment methods available?", "answer": "We accept Visa, Mastercard, American Express, PayPal, and Apple Pay."},
        {"query": "Can I change my shipping address after placing an order?", "answer": "Shipping addresses cannot be changed once an order has been placed. Please contact customer support immediately for assistance."},
        {"query": "Do you ship internationally?", "answer": "Yes, we ship to over 100 countries worldwide. International shipping fees may apply."},
        {"query": "What's the warranty on product X?", "answer": "Product X comes with a 1-year manufacturer's warranty. Please refer to the product page for details."},
        {"query": "How do I apply a discount code?", "answer": "Discount codes can be applied at checkout in the 'Promo Code' field."},
        {"query": "Where is my order #XYZ123?", "answer": "Please provide your full order number for tracking. If you've logged in, you can find it in 'My Orders'."},
        {"query": "What's the meaning of life?", "answer": "This is a philosophical question that falls outside the scope of customer support. I cannot provide an answer."},
        {"query": "When was the internet invented?", "answer": "The internet was developed in the 1960s with ARPANET, but its public use expanded significantly in the 1990s."},
        {"query": "Who won the last Super Bowl?", "answer": "The Kansas City Chiefs won Super Bowl LVIII."}
    ]

    # 2. Prepare Finetuning Dataset with Abstention Examples
    finetuning_dataset = prepare_finetuning_data(raw_customer_data, abstention_percentage=0.25)

    # 3. Conceptual LoRA Finetuning Process (Requires external libraries like Hugging Face Transformers, PEFT, PyTorch)
    print("\n--- Conceptual LLM Finetuning (LoRA) ---")
    print("This step would involve:")
    print("  a. Loading a base LLM (e.g., Mistral 7B) using `transformers` library.")
    print("  b. Preparing the finetuning_dataset into a format suitable for the LLM's tokenizer.")
    print("  c. Configuring LoRA adapters using `peft` library (e.g., `LoraConfig`).")
    print("  d. Applying LoRA adapters to the base model.")
    print("  e. Training the model on the `finetuning_dataset` using `transformers.Trainer` or a custom training loop with PyTorch/TensorFlow.")
    print("  f. Saving the finetuned LoRA adapters.")
    print("  NOTE: Actual code for this step requires external libraries and significant computational resources, and cannot be directly provided as a simple built-in Python script.")
    
    # Placeholder for the finetuned model
    finetuned_model_placeholder = "<Finetuned_Mistral_7B_with_Abstention_LoRA_Adapters>"

    # 4. Simulate Chatbot Inference with the Finetuned LLM
    print("\n--- Simulating Chatbot Interaction ---")
    queries_to_test = [
        "How do I return a product?",
        "What is the status of my order 12345?",
        "What is the capital of France?", # Expected to abstain (not in e-commerce domain, unless finetuned on general knowledge)
        "Who is the current CEO of Apple?", # Expected to abstain (domain knowledge gap if not explicitly included)
        "I need a deep philosophical answer about existence.", # Designed to trigger explicit abstention
        "What are your shipping options?"
    ]

    for i, query in enumerate(queries_to_test):
        print(f"\nCustomer Query {i+1}: {query}")
        response = simulate_llm_response(finetuned_model_placeholder, query)
        print(f"Chatbot Response: {response}")

if __name__ == "__main__":
    main_conceptual_pipeline()
