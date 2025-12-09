"""This module implements a multimodal customer support chatbot with mocked AI services."""

def _mock_speech_to_text(audio_file_path):
    if audio_file_path:
        print(f"[MOCK] Processing audio from: {audio_file_path}")
        # Simulate speech-to-text conversion
        if "broken_product.wav" in audio_file_path:
            return "My product is broken and I want a refund."
        elif "delivery_status.wav" in audio_file_path:
            return "What is the status of my order number 12345?"
        else:
            return "User spoke about a general query."
    return ""

def _mock_image_analysis(image_file_path):
    if image_file_path:
        print(f"[MOCK] Analyzing image from: {image_file_path}")
        # Simulate image analysis
        if "broken_item.jpg" in image_file_path:
            return "Image shows a broken electronic device. Possible defect or damage during shipping."
        elif "wrong_item.png" in image_file_path:
            return "Image shows a shoe instead of a shirt. Incorrect item received."
        else:
            return "Image shows a generic product. No obvious issues detected."
    return ""

def _mock_translate(text, target_language):
    if not text or target_language == "en":
        return text
    print(f"[MOCK] Translating \"{text}\" to {target_language}")
    # Simulate machine translation
    translations = {
        "es": {
            "My product is broken and I want a refund.": "Mi producto está roto y quiero un reembolso.",
            "What is the status of my order number 12345?": "¿Cuál es el estado de mi pedido número 12345?",
            "Hello, I need help.": "Hola, necesito ayuda.",
            "I received the wrong item.": "Recibí el artículo equivocado."
        },
        "fr": {
            "My product is broken and I want a refund.": "Mon produit est cassé et je souhaite un remboursement.",
            "What is the status of my order number 12345?": "Quel est le statut de ma commande numéro 12345 ?",
            "Hello, I need help.": "Bonjour, j'ai besoin d'aide.",
            "I received the wrong item.": "J'ai reçu le mauvais article."
        }
    }
    return translations.get(target_language, {}).get(text, f"[Translated to {target_language}] {text}")

def _mock_llm_process(aggregated_text):
    print(f"[MOCK] LLM processing aggregated text: \"{aggregated_text}\"")
    # Simulate LLM's intent recognition and response generation
    if "broken" in aggregated_text.lower() or "defect" in aggregated_text.lower():
        return "I understand you have a broken item. Please provide your order number so we can initiate a return or replacement."
    elif "status" in aggregated_text.lower() and "order" in aggregated_text.lower():
        return "To check your order status, please visit our 'My Orders' section or provide your order number."
    elif "wrong item" in aggregated_text.lower():
        return "I see you received the wrong item. Can you please confirm the item you ordered and the item you received?"
    elif "refund" in aggregated_text.lower():
        return "For refunds, we typically require the item to be returned. Can I help you with the return process?"
    else:
        return "Thank you for contacting support. How else can I assist you today?"

def multimodal_chatbot_agent(text_input=None, audio_file_path=None, image_file_path=None, input_language="en"):
    processed_texts = []

    # 1. Input Layer and Multimodal Processing
    if audio_file_path:
        speech_text = _mock_speech_to_text(audio_file_path)
        if speech_text:
            processed_texts.append(speech_text)
    
    if image_file_path:
        image_description = _mock_image_analysis(image_file_path)
        if image_description:
            processed_texts.append(image_description)

    if text_input:
        processed_texts.append(text_input)

    # Translate any input to English if it's not already, before LLM processing
    # For simplicity, we assume the mock speech and image analysis return English directly.
    # Only translate explicit text_input if specified language is not English.
    final_texts = []
    for text in processed_texts:
        if input_language != "en":
            final_texts.append(_mock_translate(text, "en"))
        else:
            final_texts.append(text)

    aggregated_text = " ".join(final_texts).strip()
    
    if not aggregated_text:
        return "No valid input detected. Please provide text, audio, or an image."

    # 2. LLM Core
    llm_response = _mock_llm_process(aggregated_text)

    # 3. Output Layer (Translate response back to original input language if needed)
    if input_language != "en":
        final_response = _mock_translate(llm_response, input_language)
    else:
        final_response = llm_response

    return final_response

if __name__ == "__main__":
    print("\n--- Test Case 1: Text Input (English) ---")
    response = multimodal_chatbot_agent(text_input="Hello, I need help with my recent order.", input_language="en")
    print(f"Chatbot: {response}")

    print("\n--- Test Case 2: Audio Input (simulated broken product) ---")
    response = multimodal_chatbot_agent(audio_file_path="./mock_audio/broken_product.wav", input_language="en")
    print(f"Chatbot: {response}")

    print("\n--- Test Case 3: Image Input (simulated wrong item) ---")
    response = multimodal_chatbot_agent(image_file_path="./mock_images/wrong_item.png", input_language="en")
    print(f"Chatbot: {response}")

    print("\n--- Test Case 4: Multilingual Text Input (Spanish) ---")
    response = multimodal_chatbot_agent(text_input="Hola, recibí el artículo equivocado.", input_language="es")
    print(f"Chatbot: {response}")

    print("\n--- Test Case 5: Combined Audio and Text Input (English) ---")
    response = multimodal_chatbot_agent(text_input="and I'm very upset.", audio_file_path="./mock_audio/broken_product.wav", input_language="en")
    print(f"Chatbot: {response}")

    print("\n--- Test Case 6: Multilingual Audio Input (Spanish, assuming audio returns English then translates response) ---")
    # In a real scenario, speech-to-text would ideally support multiple languages directly.
    # Here, we simulate by having the mock speech-to-text return English, then translate the *final* response.
    response = multimodal_chatbot_agent(audio_file_path="./mock_audio/delivery_status.wav", input_language="es")
    print(f"Chatbot: {response}")

    print("\n--- Test Case 7: No Input ---")
    response = multimodal_chatbot_agent()
    print(f"Chatbot: {response}")
