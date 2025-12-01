import json

class InCLTCrosslingualPrompter:
    def __init__(self, examples: list[dict]):
        self.examples = examples

    def _format_example(self, example: dict, query_lang: str, target_lang: str) -> str:
        # Determine which query/response to use for the example based on query_lang and target_lang
        # For simplicity, assume source_lang in example is the query_lang and target_lang is the target_lang for response in the example
        # The core idea is to show both source and target in the example for cross-lingual transfer
        
        example_query_key = f"{query_lang}_query"
        example_target_query_key = f"{target_lang}_query"
        example_response_key = f"{target_lang}_response_template"

        formatted_example = (
            f"Query ({query_lang}): {example.get(example_query_key, example.get('source_query'))}\n"
            f"Query ({target_lang}): {example.get(example_target_query_key, example.get('target_query'))}\n"
            f"Intent: {example['intent']}\n"
            f"Sentiment: {example['sentiment']}\n"
            f"Response ({target_lang}): {example.get(example_response_key, example.get('target_response_template'))}"
        )
        return formatted_example

    def create_prompt(self, user_query: str, query_lang: str, target_lang: str) -> str:
        prompt_parts = [
            "You are a helpful multilingual customer support assistant. Analyze the user's query for intent and sentiment, then provide a concise response in the target language. Below are examples of how to respond."
        ]

        for example in self.examples:
            prompt_parts.append(self._format_example(example, query_lang, target_lang))
        
        prompt_parts.append(
            f"\n---\n"
            f"User Query ({query_lang}): {user_query}\n"
            f"Based on the above examples, what is the Intent, Sentiment, and your Response for the following query (in {target_lang} language)?\n"
            f"Intent:"
        )
        return "\n\n".join(prompt_parts)

class MultilingualChatbot:
    def __init__(self, prompter: InCLTCrosslingualPrompter):
        self.prompter = prompter
        self.llm = self._load_simulated_llm()

    def _load_simulated_llm(self):
        # In a real scenario, this would load a model like HuggingFace Transformers mBART, XLM-R, or OpenAI models.
        # For this example, we simulate a response based on keywords for demonstration.
        def simulated_llm_response(prompt: str) -> str:
            # Simple keyword-based simulation to extract intent, sentiment, and generate a response
            intent = "unknown"
            sentiment = "neutral"
            response = "I understand you have a question. How can I help you further?"

            if "return" in prompt.lower() or "product back" in prompt.lower():
                intent = "product_return"
                sentiment = "negative"
                response = "We can help you with your return. Please provide your order number."
            elif "damaged" in prompt.lower() or "broken" in prompt.lower():
                intent = "product_complaint"
                sentiment = "negative"
                response = "I'm sorry to hear about the damaged product. Please provide details and your order number."
            elif "hello" in prompt.lower() or "hi" in prompt.lower():
                intent = "greeting"
                sentiment = "positive"
                response = "Hello! How can I assist you today?"
            elif "order status" in prompt.lower() or "where is my order" in prompt.lower():
                intent = "order_status_inquiry"
                sentiment = "neutral"
                response = "To check your order status, please provide your order number."
            elif "شكرا" in prompt.lower() or "thank you" in prompt.lower():
                intent = "gratitude"
                sentiment = "positive"
                response = "You're welcome! Is there anything else I can help you with?"

            # Simulate LLM trying to follow the prompt's instruction to output structured data
            # This is a simplification; real LLMs would generate text that needs robust parsing.
            output_template = (
                '{ "intent": "{{intent}}", "sentiment": "{{sentiment}}", "response": "{{response}}" }'
            )
            # Attempt to extract intent/sentiment from the prompt itself if the simulation missed it
            if "intent: product_return" in prompt.lower():
                intent = "product_return"
            if "sentiment: negative" in prompt.lower():
                sentiment = "negative"
            
            # More advanced parsing for a real LLM output would be needed here.
            # For this simulation, we'll try to infer the target language for the response
            # from the prompt, although the LLM's own generation capability is key.
            # Let's assume the simulated response is always in the target language specified in the prompt for simplicity.
            # In a real scenario, the LLM would generate the 'response' in target_lang.

            # For a better simulation that respects target_lang in response:
            target_lang_placeholder = "in {target_lang} language)"
            if target_lang_placeholder in prompt:
                # This is a very rough way to get the target lang from the prompt itself
                # A real LLM call would explicitly specify target_lang or rely on its cross-lingual capabilities
                start_idx = prompt.find(target_lang_placeholder) + len(target_lang_placeholder) - 13 # Adjust to get the lang
                end_idx = start_idx + 2
                extracted_target_lang = prompt[start_idx:end_idx]
                if extracted_target_lang == "es": # Example translation
                    if intent == "product_return": response = "Podemos ayudarte con tu devolución. Por favor, proporciona tu número de pedido."
                    elif intent == "product_complaint": response = "Lamento lo del producto dañado. Por favor, danos los detalles y tu número de pedido."
                    elif intent == "greeting": response = "¡Hola! ¿Cómo puedo ayudarte hoy?"
                    elif intent == "order_status_inquiry": response = "Para verificar el estado de tu pedido, por favor, proporciona tu número de pedido."
                    elif intent == "gratitude": response = "¡De nada! ¿Hay algo más en lo que pueda ayudarte?"
                    else: response = "Entiendo que tienes una pregunta. ¿Cómo puedo ayudarte más?"
                elif extracted_target_lang == "ar": # Example translation
                    if intent == "product_return": response = "يمكننا مساعدتك في إرجاع منتجك. يرجى تقديم رقم طلبك."
                    elif intent == "product_complaint": response = "آسف لسماع عن المنتج التالف. يرجى تزويدنا بالتفاصيل ورقم طلبك."
                    elif intent == "greeting": response = "مرحباً! كيف يمكنني مساعدتك اليوم؟"
                    elif intent == "order_status_inquiry": response = "للتحقق من حالة طلبك، يرجى تقديم رقم طلبك."
                    elif intent == "gratitude": response = "عفواً! هل يمكنني مساعدتك بأي شيء آخر؟"
                    else: response = "أتفهم أن لديك سؤالاً. كيف يمكنني مساعدتك أكثر؟"


            return output_template.replace("{{intent}}", intent).replace("{{sentiment}}", sentiment).replace("{{response}}", response)

        return simulated_llm_response

    def process_query(self, user_query: str, query_lang: str, target_lang: str) -> dict:
        llm_prompt = self.prompter.create_prompt(user_query, query_lang, target_lang)
        print(f"\n--- LLM Prompt ---\n{llm_prompt}\n-------------------\n") # For debugging/visualization
        
        simulated_output_str = self.llm(llm_prompt)
        print(f"\n--- Simulated LLM Output ---\n{simulated_output_str}\n------------------------\n") # For debugging/visualization
        
        # Parse the simulated LLM's structured JSON output
        try:
            llm_response_parsed = json.loads(simulated_output_str)
            return llm_response_parsed
        except json.JSONDecodeError:
            # Fallback if the simulation doesn't perfectly return JSON
            print("Warning: Simulated LLM output was not perfectly JSON. Attempting fallback.")
            # A more robust parsing for real LLM output would be needed here.
            # For this simple simulation, we'll try to extract parts if JSON parsing fails.
            # This is a very basic fallback and might not work for all simulated outputs.
            intent = "unknown"
            sentiment = "neutral"
            response = "Error processing your request."
            if 'intent: ' in simulated_output_str.lower():
                intent_start = simulated_output_str.lower().find('intent: ') + len('intent: ')
                intent_end = simulated_output_str.lower().find('\n', intent_start)
                if intent_end == -1: intent_end = len(simulated_output_str)
                intent = simulated_output_str[intent_start:intent_end].strip()

            if 'sentiment: ' in simulated_output_str.lower():
                sentiment_start = simulated_output_str.lower().find('sentiment: ') + len('sentiment: ')
                sentiment_end = simulated_output_str.lower().find('\n', sentiment_start)
                if sentiment_end == -1: sentiment_end = len(simulated_output_str)
                sentiment = simulated_output_str[sentiment_start:sentiment_end].strip()
            
            # The response part in a real LLM can be complex. For simulation, just assume the default or extracted.
            if 'response: ' in simulated_output_str.lower():
                response_start = simulated_output_str.lower().find('response: ') + len('response: ')
                response = simulated_output_str[response_start:].strip()

            return {"intent": intent, "sentiment": sentiment, "response": response}


if __name__ == "__main__":
    # Example In-Context Learning (ICL) examples with cross-lingual pairs
    icl_examples = [
        {
            "source_lang": "en",
            "target_lang": "es",
            "source_query": "I want to return a broken product.",
            "target_query": "Quiero devolver un producto roto.",
            "intent": "product_return",
            "sentiment": "negative",
            "source_response_template": "We can help you with your return. Please provide your order number.",
            "target_response_template": "Podemos ayudarte con tu devolución. Por favor, proporciona tu número de pedido."
        },
        {
            "source_lang": "en",
            "target_lang": "ar",
            "source_query": "My order is damaged, what should I do?",
            "target_query": "طلبي تالف، ماذا علي أن أفعل؟",
            "intent": "product_complaint",
            "sentiment": "negative",
            "source_response_template": "I'm sorry to hear about the damaged product. Please provide details and your order number.",
            "target_response_template": "آسف لسماع عن المنتج التالف. يرجى تزويدنا بالتفاصيل ورقم طلبك."
        },
        {
            "source_lang": "en",
            "target_lang": "es",
            "source_query": "Hello, how are you?",
            "target_query": "Hola, ¿cómo estás?",
            "intent": "greeting",
            "sentiment": "positive",
            "source_response_template": "Hello! How can I assist you today?",
            "target_response_template": "¡Hola! ¿Cómo puedo ayudarte hoy?"
        },
        {
            "source_lang": "es",
            "target_lang": "en",
            "source_query": "¿Cuál es el estado de mi pedido?",
            "target_query": "What is the status of my order?",
            "intent": "order_status_inquiry",
            "sentiment": "neutral",
            "source_response_template": "Para verificar el estado de tu pedido, por favor, proporciona tu número de pedido.",
            "target_response_template": "To check your order status, please provide your order number."
        },
        {
            "source_lang": "ar",
            "target_lang": "en",
            "source_query": "شكرا جزيلا لمساعدتك.",
            "target_query": "Thank you very much for your help.",
            "intent": "gratitude",
            "sentiment": "positive",
            "source_response_template": "You're welcome! Is there anything else I can help you with?",
            "target_response_template": "You're welcome! Is there anything else I can help you with?"
        }
    ]

    # Initialize the prompter with ICL examples
    prompter = InCLTCrosslingualPrompter(icl_examples)

    # Initialize the chatbot
    chatbot = MultilingualChatbot(prompter)

    print("\n--- Test Case 1: English Query, Spanish Response ---")
    user_query_en = "I want to return my order, it arrived broken."
    query_lang_en = "en"
    target_lang_es = "es"
    response_es = chatbot.process_query(user_query_en, query_lang_en, target_lang_es)
    print(f"Chatbot Response (ES): {response_es}")

    print("\n--- Test Case 2: Spanish Query, English Response ---")
    user_query_es = "Mi pedido está dañado, ¿qué hago?"
    query_lang_es = "es"
    target_lang_en = "en"
    response_en = chatbot.process_query(user_query_es, query_lang_es, target_lang_en)
    print(f"Chatbot Response (EN): {response_en}")

    print("\n--- Test Case 3: Arabic Query, English Response ---")
    user_query_ar = "أين طلبي؟ أريد معرفة حالة الشحن."
    query_lang_ar = "ar"
    target_lang_en = "en"
    response_en_from_ar = chatbot.process_query(user_query_ar, query_lang_ar, target_lang_en)
    print(f"Chatbot Response (EN from AR): {response_en_from_ar}")

    print("\n--- Test Case 4: English Query, Arabic Response (New Intent Simulation) ---")
    user_query_en_new_intent = "Hello, can I check my order status?"
    query_lang_en_new_intent = "en"
    target_lang_ar_new_intent = "ar"
    response_ar_new_intent = chatbot.process_query(user_query_en_new_intent, query_lang_en_new_intent, target_lang_ar_new_intent)
    print(f"Chatbot Response (AR - New Intent): {response_ar_new_intent}")

    print("\n--- Test Case 5: Spanish Query, Spanish Response (General Query) ---")
    user_query_es_general = "Tengo una pregunta general."
    query_lang_es_general = "es"
    target_lang_es_general = "es"
    response_es_general = chatbot.process_query(user_query_es_general, query_lang_es_general, target_lang_es_general)
    print(f"Chatbot Response (ES - General): {response_es_general}")