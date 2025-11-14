from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class InCLTProcessor:
    """Implements the InCLT Crosslingual Transfer Prompting pattern."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.examples = []

    def add_example(self, source_query: str, source_lang: str, target_response: str, target_lang: str):
        """Adds a cross-lingual in-context example."""
        self.examples.append({
            "source_query": source_query,
            "source_lang": source_lang,
            "target_response": target_response,
            "target_lang": target_lang
        })

    def _construct_prompt(self, current_query: str, current_query_lang: str, target_response_lang: str) -> str:
        """Constructs the prompt for the LLM using InCLT examples."""
        prompt_parts = []

        # Add in-context examples
        for ex in self.examples:
            prompt_parts.append(f"{ex['source_lang']} Query: {ex['source_query']}")
            prompt_parts.append(f"{ex['target_lang']} Response: {ex['target_response']}")
            prompt_parts.append("") # Add a newline for separation between examples

        # Add the current query, instructing the LLM to generate a response in the target language
        prompt_parts.append(f"{current_query_lang} Query: {current_query}")
        prompt_parts.append(f"{target_response_lang} Response:")

        return "\n".join(prompt_parts)

    def generate_response(self, customer_query: str, query_language: str, response_language: str, max_new_tokens: int = 50) -> str:
        """Generates a customer support response using the LLM and InCLT prompting."""
        prompt = self._construct_prompt(customer_query, query_language, response_language)
        print(f"\n--- PROMPT TO LLM ---\n{prompt}\n---------------------") # For debugging/demonstration

        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

        # Ensure inputs are on the correct device (CPU or GPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        self.model.to(device) # Move model to device as well

        # Generate response
        output_sequences = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id, # Or use bos_token_id if eos_token_id is not set
            num_return_sequences=1,
            do_sample=True, # Enable sampling for more varied responses
            temperature=0.7,
            top_k=50,
            top_p=0.95
        )

        generated_text = self.tokenizer.decode(output_sequences[0], skip_special_tokens=True)
        print(f"\n--- RAW LLM GENERATION ---\n{generated_text}\n------------------------") # For debugging/demonstration

        # Extract the relevant part of the response
        # We expect the LLM to continue after "[Target_Lang] Response:"
        response_tag = f"{response_language} Response:"
        if response_tag in generated_text:
            start_index = generated_text.find(response_tag) + len(response_tag)
            extracted_response = generated_text[start_index:].strip()
            # Basic heuristic to cut off at the first newline or period if it seems like a full stop
            cut_off_index_newline = extracted_response.find('\n')
            cut_off_index_period = extracted_response.find('.')

            if cut_off_index_newline != -1 and cut_off_index_period != -1:
                end_index = min(cut_off_index_newline, cut_off_index_period + 1) if cut_off_index_period + 1 > 0 else cut_off_index_newline
            elif cut_off_index_newline != -1:
                end_index = cut_off_index_newline
            elif cut_off_index_period != -1:
                end_index = cut_off_index_period + 1
            else:
                end_index = len(extracted_response)

            final_response = extracted_response[:end_index].strip()
            return final_response
        else:
            return "Sorry, I could not generate a clear response in the target language." # Fallback if parsing fails

# --- Main Application Logic (Customer Query Handler and Demo) ---

if __name__ == "__main__":
    print("Loading multilingual LLM and tokenizer...")
    # Using a smaller model like 'distilgpt2' for demonstration purposes.
    # For a truly multilingual and robust system, consider models like 'Helsinki-NLP/opus-mt-en-es', 'facebook/mbart-large-50', or commercial LLM APIs.
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Ensure the tokenizer has a pad_token if not already set (common for GPT-like models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    icl_processor = InCLTProcessor(model, tokenizer)
    print("LLM and tokenizer loaded. InCLTProcessor initialized.")

    print("Adding cross-lingual in-context examples...")
    # Example 1: English Query -> Spanish Response
    icl_processor.add_example(
        source_query="I have a problem with my order #12345.",
        source_lang="English",
        target_response="Tengo un problema con mi pedido #12345. ¿Puede darme más detalles?",
        target_lang="Spanish"
    )

    # Example 2: Spanish Query -> English Response
    icl_processor.add_example(
        source_query="¿Cuál es el estado de mi envío?",
        source_lang="Spanish",
        target_response="What is the status of your shipment? Please provide your order number.",
        target_lang="English"
    )

    # Example 3: English Query -> French Response
    icl_processor.add_example(
        source_query="I want to return an item.",
        source_lang="English",
        target_response="Je souhaite retourner un article. Quel est le numéro de commande et la raison du retour?",
        target_lang="French"
    )

    # Example 4: French Query -> English Response
    icl_processor.add_example(
        source_query="Mon colis n'est pas arrivé.",
        source_lang="French",
        target_response="My package has not arrived. Could you please provide your tracking number or order ID?",
        target_lang="English"
    )

    print("In-context examples added.")

    # --- Simulate Customer Queries ---
    print("\n--- Simulating Customer Interactions ---")

    # Query 1: New English query, expect Spanish response
    customer_query_1 = "Where is my package?"
    print(f"\nCustomer (English): {customer_query_1}")
    response_1 = icl_processor.generate_response(customer_query_1, "English", "Spanish")
    print(f"Chatbot (Spanish): {response_1}")

    # Query 2: New Spanish query, expect English response
    customer_query_2 = "Necesito ayuda con un producto defectuoso."
    print(f"\nCustomer (Spanish): {customer_query_2}")
    response_2 = icl_processor.generate_response(customer_query_2, "Spanish", "English")
    print(f"Chatbot (English): {response_2}")

    # Query 3: New French query, expect French response
    customer_query_3 = "J'ai une question sur ma facture."
    print(f"\nCustomer (French): {customer_query_3}")
    response_3 = icl_processor.generate_response(customer_query_3, "French", "French")
    print(f"Chatbot (French): {response_3}")

    # Query 4: New English query, expect French response
    customer_query_4 = "How do I change my shipping address?"
    print(f"\nCustomer (English): {customer_query_4}")
    response_4 = icl_processor.generate_response(customer_query_4, "English", "French")
    print(f"Chatbot (French): {response_4}")

    print("\n--- Simulation Complete ---")