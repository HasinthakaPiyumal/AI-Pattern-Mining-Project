"""
This script demonstrates a simplified Multilingual Customer Support Chatbot leveraging the InCLT Crosslingual Transfer Prompting pattern.
It includes a simulated LLM, a knowledge base for in-context examples, and a prompt construction module.
"""

# --- 1. Simulated Multilingual LLM ---
def _simulate_llm_response(prompt: str) -> str:
    """Simulates a response from a multilingual LLM based on the prompt."""
    # In a real application, this would involve calling a true LLM API (e.g., via transformers library)
    if "¿Cuál es el estado de mi pedido?" in prompt:
        return "Su pedido está en tránsito y se espera que llegue el 15 de marzo. (Your order is in transit and expected to arrive on March 15th.)"
    elif "I have a problem with my recent purchase" in prompt:
        return "Please provide your order number so I can assist you further. (Por favor, proporcione su número de pedido para que pueda ayudarle mejor.)"
    elif "problema con mi compra reciente" in prompt:
        return "Por favor, proporcione su número de pedido para que pueda ayudarle mejor. (Please provide your order number so I can assist you further.)"
    elif "how to return an item" in prompt:
        return "To return an item, please visit our returns page and follow the instructions. (Para devolver un artículo, visite nuestra página de devoluciones y siga las instrucciones.)"
    elif "cómo devolver un artículo" in prompt:
        return "Para devolver un artículo, visite nuestra página de devoluciones y siga las instrucciones. (To return an item, please visit our returns page and follow the instructions.)"
    return f"Entendido. Su consulta: \"{prompt[max(0, prompt.rfind('Customer Query:') + len('Customer Query:')):].strip()}\". Necesito más detalles. (Understood. Your query: \"{prompt[max(0, prompt.rfind('Customer Query:') + len('Customer Query:')):].strip()}\". I need more details.)"

# --- 2. ICL Example Management (Knowledge Base) ---
knowledge_base = [
    {
        "source_language_query": "¿Cuál es el estado de mi pedido?",
        "source_language_answer": "Su pedido número XYZ está en tránsito y se espera que llegue el 15 de marzo.",
        "target_language_query": "What is the status of my order?",
        "target_language_answer": "Your order number XYZ is in transit and expected to arrive on March 15th."
    },
    {
        "source_language_query": "Tengo un problema con mi compra reciente.",
        "source_language_answer": "Por favor, proporcione su número de pedido para que podamos investigar.",
        "target_language_query": "I have a problem with my recent purchase.",
        "target_language_answer": "Please provide your order number so we can investigate."
    },
    {
        "source_language_query": "¿Cómo puedo devolver un artículo?",
        "source_language_answer": "Para devolver un artículo, visite la sección de devoluciones en nuestro sitio web y siga las instrucciones.",
        "target_language_query": "How can I return an item?",
        "target_language_answer": "To return an item, please visit the returns section on our website and follow the instructions."
    },
    {
        "source_language_query": "¿Cuál es su política de reembolso?",
        "source_language_answer": "Nuestra política de reembolso permite devoluciones dentro de los 30 días posteriores a la compra, siempre que el artículo esté en su estado original.",
        "target_language_query": "What is your refund policy?",
        "target_language_answer": "Our refund policy allows returns within 30 days of purchase, provided the item is in its original condition."
    },
]

# --- 2. Example Retrieval Mechanism (Simulated) ---
def _retrieve_examples(query: str, k: int = 2) -> list:
    """Simulates retrieval of relevant in-context examples based on keywords in the query."""
    # In a real application, this would use embedding similarity (e.g., sentence-transformers + vector store)
    relevant_examples = []
    query_lower = query.lower()
    for example in knowledge_base:
        if any(keyword in example["source_language_query"].lower() for keyword in query_lower.split() if len(keyword) > 2) or \
           any(keyword in example["target_language_query"].lower() for keyword in query_lower.split() if len(keyword) > 2):
            relevant_examples.append(example)
            if len(relevant_examples) >= k:
                break
    return relevant_examples

# --- 3. Prompt Construction Module (InCLT Logic) ---
def _construct_in_clt_prompt(
    customer_query: str,
    source_lang: str,
    target_lang: str,
    examples: list
) -> str:
    """Constructs a prompt using the InCLT Crosslingual Transfer pattern.
    Combines source and target language examples in the prompt.
    """
    system_instruction = (
        f"You are a helpful multilingual customer support assistant for an e-commerce platform. "
        f"Your task is to answer customer queries accurately and politely in {source_lang}. "
        f"Use the following examples for in-context learning, which provide context in both the customer's language ({source_lang}) "
        f"and the primary operational language ({target_lang}).\n\n"
    )

    icl_examples_str = []
    for i, example in enumerate(examples):
        icl_examples_str.append(
            f"Example {i+1}:\n"
            f"Source Language Query ({source_lang}): {example['source_language_query']}\n"
            f"Source Language Answer ({source_lang}): {example['source_language_answer']}\n"
            f"Target Language Query ({target_lang}): {example['target_language_query']}\n"
            f"Target Language Answer ({target_lang}): {example['target_language_answer']}\n"
        )
    icl_examples_str = "\n".join(icl_examples_str)

    final_prompt = (
        f"{system_instruction}"
        f"{icl_examples_str}"
        f"---\n"
        f"Customer Query ({source_lang}): {customer_query}\n"
        f"Assistant Response ({source_lang}):"
    )
    return final_prompt

# --- Main Chatbot Logic ---
def multilingual_chatbot(
    customer_query: str,
    source_lang: str = "es",
    target_lang: str = "en"
) -> str:
    """Simulates the multilingual chatbot's interaction using InCLT prompting."""
    print(f"\nCustomer ({source_lang}): {customer_query}")

    # 1. Retrieve relevant examples
    retrieved_examples = _retrieve_examples(customer_query, k=2)
    print(f"[Debug] Retrieved {len(retrieved_examples)} examples for ICL.")

    # 2. Construct the InCLT prompt
    prompt = _construct_in_clt_prompt(
        customer_query, source_lang, target_lang, retrieved_examples
    )
    # print(f"[Debug] Constructed Prompt:\n{prompt}\n---") # Uncomment to see the full prompt

    # 3. Simulate LLM response
    llm_response = _simulate_llm_response(prompt)

    return llm_response.split(f"({target_lang}):")[-1].strip() if f"({target_lang}):" in llm_response else llm_response

# --- Demonstration ---
if __name__ == "__main__":
    print("\n--- Multilingual Customer Support Chatbot Demonstration (InCLT) ---")
    print("Using Spanish (es) as source language and English (en) as target/operational language.")

    # Example 1: Query in Spanish, relevant example exists
    response1 = multilingual_chatbot(
        customer_query="¿Cuál es el estado de mi pedido?",
        source_lang="es",
        target_lang="en"
    )
    print(f"Chatbot ({'es'}): {response1}")

    # Example 2: Query in Spanish about a problem, relevant example exists
    response2 = multilingual_chatbot(
        customer_query="Tengo un problema con mi compra reciente.",
        source_lang="es",
        target_lang="en"
    )
    print(f"Chatbot ({'es'}): {response2}")

    # Example 3: Query in Spanish about returns, relevant example exists
    response3 = multilingual_chatbot(
        customer_query="Cómo devolver un artículo?",
        source_lang="es",
        target_lang="en"
    )
    print(f"Chatbot ({'es'}): {response3}")

    # Example 4: Query in Spanish, less direct match (demonstrates fallback)
    response4 = multilingual_chatbot(
        customer_query="Necesito información sobre un producto.",
        source_lang="es",
        target_lang="en"
    )
    print(f"Chatbot ({'es'}): {response4}")

    print("\n--- Demonstration Complete ---")
