import gradio as gr
from transformers import pipeline

# --- Global Components ---
# Load translation pipeline for German to English
translator_de_en = pipeline("translation", model="Helsinki-NLP/opus-mt-de-en")

# In-Context Learning Examples leveraging both source (German) and target (English)
# for cross-lingual transfer. These examples are crucial for the InCLT pattern.
# The LLM is expected to learn cross-lingual capabilities from these pairs.
INCLT_EXAMPLES = [
    {
        "source_lang_name": "German",
        "target_lang_name": "English", # Assuming English as a pivot or common intermediate for transfer
        "user_query_source": "Ich habe Probleme mit meiner Bestellung.",
        "user_query_target": "I have problems with my order.",
        "assistant_response_source": "Es tut mir leid zu hören, dass Sie Probleme mit Ihrer Bestellung haben. Könnten Sie mir bitte Ihre Bestellnummer mitteilen?",
        "assistant_response_target": "I'm sorry to hear you're having trouble with your order. Could you please provide your order number?",
    },
    {
        "source_lang_name": "German",
        "target_lang_name": "English",
        "user_query_source": "Wann wird meine Lieferung ankommen?",
        "user_query_target": "When will my delivery arrive?",
        "assistant_response_source": "Um den Status Ihrer Lieferung zu überprüfen, benötige ich Ihre Bestellnummer.",
        "assistant_response_target": "To check the status of your delivery, I need your order number.",
    },
    {
        "source_lang_name": "German",
        "target_lang_name": "English",
        "user_query_source": "Kann ich ein Produkt zurücksenden?",
        "user_query_target": "Can I return a product?",
        "assistant_response_source": "Ja, Sie können Produkte innerhalb von 30 Tagen nach Erhalt zurücksenden. Bitte stellen Sie sicher, dass das Produkt unbenutzt und in der Originalverpackung ist.",
        "assistant_response_target": "Yes, you can return products within 30 days of receipt. Please ensure the product is unused and in its original packaging.",
    },
]

# --- Functions ---

def construct_inclt_prompt(user_query: str, user_query_english_translation: str, chat_history: list) -> str:
    """
    Constructs the prompt for the LLM, integrating InCLT examples and the current query.
    The structure emphasizes the cross-lingual transfer aspect.
    """
    prompt_parts = [
        "You are a multilingual customer support assistant for an e-commerce company.",
        "Your goal is to provide helpful and accurate responses to customers in their native language (German in this case).",
        "Leverage both the original query language and its English translation for better cross-lingual understanding.",
        "Here are some in-context examples to guide your responses:",
        ""
    ]

    # Add InCLT examples to the prompt
    for example in INCLT_EXAMPLES:
        prompt_parts.append(f"Example ({example['source_lang_name']}):")
        prompt_parts.append(f"User Query ({example['source_lang_name']}): {example['user_query_source']}")
        prompt_parts.append(f"User Query ({example['target_lang_name']} Translation): {example['user_query_target']}")
        prompt_parts.append(f"Assistant Response ({example['source_lang_name']}): {example['assistant_response_source']}")
        prompt_parts.append("")

    # Add current chat history (simplified for this example, a real system would format it carefully)
    if chat_history:
        prompt_parts.append("--- Previous Conversation History ---")
        for human_msg, ai_msg in chat_history:
            prompt_parts.append(f"Customer: {human_msg}")
            prompt_parts.append(f"Assistant: {ai_msg}")
        prompt_parts.append("")

    # Add the current user query for the LLM to answer, providing both original and translated versions
    prompt_parts.append("--- Current Customer Query ---")
    prompt_parts.append(f"Customer Query (German): {user_query}")
    prompt_parts.append(f"Customer Query (English Translation): {user_query_english_translation}")
    prompt_parts.append(f"Assistant Response (German):") # LLM is expected to complete this in German

    return "\n".join(prompt_parts)

def mock_llm_response(prompt: str) -> str:
    """
    A placeholder function to simulate an LLM's response generation.
    In a real application, this would be an actual call to a powerful multilingual LLM
    (e.g., from OpenAI, HuggingFace, Google).
    This mock generates simple keyword-based responses, assuming the LLM's capability
    to respond in German, guided by the InCLT examples.
    """
    lower_prompt = prompt.lower()
    if "bestellnummer" in lower_prompt or "order number" in lower_prompt or "bestellung" in lower_prompt:
        return "Bitte geben Sie Ihre Bestellnummer an, damit ich Ihnen besser helfen kann."
    elif "lieferung" in lower_prompt or "delivery" in lower_prompt or "paket" in lower_prompt:
        return "Um den Status Ihrer Lieferung zu überprüfen, benötige ich Ihre Bestellnummer. Sie können diese in Ihrer Bestellbestätigung finden."
    elif "zurücksenden" in lower_prompt or "return a product" in lower_prompt or "rückgabe" in lower_prompt:
        return "Ja, Sie können unbenutzte Artikel innerhalb von 30 Tagen nach Erhalt zurücksenden. Bitte besuchen Sie unser Retourenportal auf der Webseite."
    elif "probleme" in lower_prompt or "problems" in lower_prompt:
        return "Es tut mir leid zu hören, dass Sie Probleme haben. Könnten Sie Ihr Anliegen genauer beschreiben und Ihre Bestellnummer angeben?"
    else:
        return "Ich verstehe Ihre Anfrage nicht ganz. Könnten Sie sie bitte präziser formulieren oder ein anderes Stichwort verwenden?"


def chatbot_response(message: str, history: list) -> str:
    """
    Handles the chatbot interaction flow:
    1. Translates the user's message to an intermediate language (English).
    2. Constructs the InCLT prompt using examples and the current query (in both languages).
    3. Obtains a response from a simulated LLM.
    4. Returns the response.
    """
    # 1. Translate user message from German to English
    translated_message_obj = translator_de_en(message, max_length=512)
    user_query_english_translation = translated_message_obj[0]['translation_text'] if translated_message_obj else ""

    # 2. Construct the InCLT prompt with examples and the current user query
    inclt_prompt = construct_inclt_prompt(message, user_query_english_translation, history)

    # 3. Get LLM response (using a mock for demonstration)
    # In a real scenario, this 'inclt_prompt' would be sent to a multilingual LLM.
    llm_response = mock_llm_response(inclt_prompt)

    return llm_response

# --- Gradio Interface ---
# A simple web interface for demonstrating the chatbot
iface = gr.ChatInterface(
    chatbot_response,
    chatbot=gr.Chatbot(height=500),
    textbox=gr.Textbox(placeholder="Stellen Sie Ihre Frage auf Deutsch... (e.g., Ich habe Probleme mit meiner Bestellung.)", container=False, scale=7),
    title="Multilingual Customer Support Chatbot with InCLT Crosslingual Transfer Prompting",
    description="This chatbot demonstrates the InCLT Crosslingual Transfer Prompting pattern. It processes your German query by internally translating it to English and then constructing a prompt that includes in-context examples in *both* German and English. This design aims to boost the cross-lingual cognitive capabilities of the underlying (mock) LLM, leading to more accurate and contextually appropriate responses in German.",
    examples=[
        ["Ich habe Probleme mit meiner Bestellung."],
        ["Wann kommt mein Paket an?"],
        ["Kann ich ein Produkt zurücksenden?"],
        ["Wie lange dauert die Lieferung?"],
    ],
    theme="soft",
)

if __name__ == "__main__":
    iface.launch()