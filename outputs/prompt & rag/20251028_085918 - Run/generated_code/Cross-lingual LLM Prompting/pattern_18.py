from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class InCLTChatbot:
    def __init__(self, model_name="google/flan-t5-base"):
        """
        Initializes the InCLTChatbot with a pre-trained multilingual LLM.

        Args:
            model_name (str): The name of the pre-trained model to use.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.in_context_examples = []
        print(f"Chatbot initialized with model: {model_name}")

    def add_example(self, source_query: str, target_response: str, 
                    source_lang: str = "en", target_lang: str = "en", 
                    is_cross_lingual: bool = False):
        """
        Adds an in-context example to the chatbot's memory.

        Args:
            source_query (str): The query in the source language.
            target_response (str): The expected response in the target language.
            source_lang (str): The language code for the source query (e.g., 'en', 'es', 'fr').
            target_lang (str): The language code for the target response (e.g., 'en', 'es', 'fr').
            is_cross_lingual (bool): True if the example involves different source and target languages.
        """
        self.in_context_examples.append({
            "source_query": source_query,
            "target_response": target_response,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "is_cross_lingual": is_cross_lingual
        })
        print(f"Added example: '{source_query}' -> '{target_response}' (Cross-lingual: {is_cross_lingual})")

    def _construct_prompt(self, user_query: str, target_lang: str) -> str:
        """
        Constructs the full prompt including in-context examples and the user's query.

        Args:
            user_query (str): The customer's query.
            target_lang (str): The desired language for the response.

        Returns:
            str: The fully constructed prompt.
        """
        prompt_parts = []
        for example in self.in_context_examples:
            # For simplicity, we'll include all examples. In a real scenario, you might select relevant ones.
            # The InCLT pattern suggests using both source and target language in examples.
            prompt_parts.append(f"Source ({example['source_lang']}) Query: {example['source_query']}")
            prompt_parts.append(f"Target ({example['target_lang']}) Response: {example['target_response']}")
            prompt_parts.append("") # Add a newline for separation
        
        # Add the user's actual query
        prompt_parts.append(f"User ({target_lang}) Query: {user_query}")
        prompt_parts.append(f"Target ({target_lang}) Response:")

        full_prompt = "\n".join(prompt_parts)
        print(f"\nConstructed Prompt:\n---\n{full_prompt}\n---")
        return full_prompt

    def get_response(self, user_query: str, target_lang: str = "en", max_new_tokens: int = 50) -> str:
        """
        Generates a response to the user's query using the LLM and in-context examples.

        Args:
            user_query (str): The customer's query.
            target_lang (str): The desired language for the response (e.g., 'en', 'es', 'fr').
            max_new_tokens (int): The maximum number of tokens to generate for the response.

        Returns:
            str: The generated response from the chatbot.
        """
        full_prompt = self._construct_prompt(user_query, target_lang)
        
        inputs = self.tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=512)
        
        # Generate response
        outputs = self.model.generate(
            inputs["input_ids"],
            max_new_tokens=max_new_tokens,
            num_beams=5, # Use beam search for better quality
            early_stopping=True
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nGenerated Response: {response}")
        return response

if __name__ == "__main__":
    print("Initializing Multilingual Customer Support Chatbot...")
    chatbot = InCLTChatbot()

    # Add Monolingual In-Context Examples (English)
    chatbot.add_example(
        source_query="What is the status of my order?",
        target_response="Please provide your order number and I can check its status for you.",
        source_lang="en", target_lang="en", is_cross_lingual=False
    )
    chatbot.add_example(
        source_query="How do I return an item?",
        target_response="You can initiate a return through your account's order history page.",
        source_lang="en", target_lang="en", is_cross_lingual=False
    )

    # Add Cross-lingual In-Context Examples (English-Spanish)
    chatbot.add_example(
        source_query="Mi pedido está tardando mucho.", # Spanish query
        target_response="I understand. Can you please provide your order number?", # English response
        source_lang="es", target_lang="en", is_cross_lingual=True
    )
    chatbot.add_example(
        source_query="How can I track my package?", # English query
        target_response="Puedes rastrear tu paquete utilizando el número de seguimiento proporcionado en tu correo electrónico de confirmación.", # Spanish response
        source_lang="en", target_lang="es", is_cross_lingual=True
    )
    chatbot.add_example(
        source_query="¿Dónde está mi paquete?", # Spanish query
        target_response="Para rastrear su paquete, por favor, ingrese el número de seguimiento.", # Spanish response
        source_lang="es", target_lang="es", is_cross_lingual=False # Spanish monolingual
    )

    print("\n--- Testing Chatbot Responses ---")

    # Test 1: Monolingual English query
    print("\n--- Test 1: English Monolingual Query ---")
    response1 = chatbot.get_response(user_query="My delivery is late.", target_lang="en")
    # Expected: Should leverage general customer support knowledge and possibly cross-lingual understanding

    # Test 2: Spanish query, expecting English response (cross-lingual transfer)
    print("\n--- Test 2: Spanish Query, English Response ---")
    response2 = chatbot.get_response(user_query="Necesito ayuda con una compra.", target_lang="en")
    # Expected: Should understand Spanish and respond in English, potentially using cross-lingual example knowledge

    # Test 3: English query, expecting Spanish response (cross-lingual transfer)
    print("\n--- Test 3: English Query, Spanish Response ---")
    response3 = chatbot.get_response(user_query="I want to change my shipping address.", target_lang="es")
    # Expected: Should understand English and respond in Spanish

    # Test 4: Spanish query, expecting Spanish response (monolingual but demonstrating multilingual capability)
    print("\n--- Test 4: Spanish Query, Spanish Response ---")
    response4 = chatbot.get_response(user_query="¿Cuál es su política de devoluciones?", target_lang="es")
    # Expected: Should understand Spanish and respond in Spanish

    print("\nChatbot demonstration complete.")
