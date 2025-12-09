import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langdetect import detect, DetectorFactory

# Ensure consistent language detection results
DetectorFactory.seed = 0

# 1. Multilingual Large Language Model (LLM)
# Using a smaller mT5 model for demonstration purposes.
# For production, consider larger models like google/mt5-base or similar.
model_name = "google/mt5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Create a text generation pipeline
generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# 3. In-Context Example Store
in_context_examples = [
    {
        "query_en": "Where is my order?",
        "response_en": "Your order #12345 is currently in transit and is expected to arrive by Friday.",
        "response_es": "Su pedido #12345 está actualmente en tránsito y se espera que llegue el viernes.",
        "query_es": "¿Dónde está mi pedido?"
    },
    {
        "query_en": "How can I return an item?",
        "response_en": "You can return items within 30 days of purchase. Please visit our returns page for more details.",
        "response_es": "Puede devolver artículos dentro de los 30 días posteriores a la compra. Visite nuestra página de devoluciones para más detalles.",
        "query_es": "¿Cómo puedo devolver un artículo?"
    },
    {
        "query_en": "What are your shipping options?",
        "response_en": "We offer standard and express shipping. Standard shipping takes 3-5 business days.",
        "response_es": "Ofrecemos envío estándar y express. El envío estándar tarda de 3 a 5 días hábiles.",
        "query_es": "¿Cuáles son sus opciones de envío?"
    }
]

# 2. InCLT Prompting Module
def create_inclt_prompt(user_query: str, source_lang: str, target_lang: str = "en") -> str:
    """
    Constructs a prompt incorporating InCLT examples.
    """
    prompt_parts = []
    
    for example in in_context_examples:
        if source_lang == "en":
            prompt_parts.append(f"Query: {example['query_en']}")
        elif source_lang == "es":
            prompt_parts.append(f"Query: {example['query_es']}")
        else:
            prompt_parts.append(f"Query: {example.get(f'query_{source_lang}', example['query_en'])}") # Fallback to EN
            
        prompt_parts.append(f"Response (EN): {example['response_en']}")
        prompt_parts.append(f"Response (ES): {example['response_es']}")
        prompt_parts.append("")

    prompt_parts.append(f"Query: {user_query}")
    prompt_parts.append(f"Response ({target_lang.upper()}):")

    return "\n".join(prompt_parts)

# 5. Language Detection
def detect_language(text: str) -> str:
    """
    Detects the language of the input text.
    """
    try:
        return detect(text)
    except Exception:
        return "en" # Default to English if detection fails

# Chatbot Logic
def chatbot_response(user_input: str) -> str:
    """
    Processes user input and returns a chatbot response using InCLT.
    """
    source_lang = detect_language(user_input)
    target_lang = "en" # Default target response language

    prompt = create_inclt_prompt(user_input, source_lang, target_lang)
    
    # print(f"Generated Prompt:\n{prompt}\n---") # Uncomment for debugging

    outputs = generator(prompt, max_new_tokens=100, do_sample=False, num_beams=1)
    
    generated_text = outputs[0]["generated_text"]
    
    response_prefix = f"Response ({target_lang.upper()}):"
    
    if response_prefix in generated_text:
        response_start_index = generated_text.rfind(response_prefix)
        final_response = generated_text[response_start_index + len(response_prefix):].strip()
        final_response = final_response.split("\n")[0].strip()
        return final_response if final_response else "Sorry, I couldn't generate a clear response. Please try rephrasing."
    else:
        return "Sorry, I couldn't generate a clear response. Please try rephrasing."

# 4. Chatbot Interface (Gradio)
iface = gr.Interface(
    fn=chatbot_response,
    inputs=gr.Textbox(lines=2, placeholder="Enter your query here..."),
    outputs="text",
    title="Multilingual E-commerce Support Chatbot (InCLT Prompting)",
    description="This chatbot uses InCLT (In-Context Learning Transfer) with a multilingual LLM to understand and respond to queries in multiple languages. It uses both source and target language examples to boost cross-lingual capabilities. Try asking in English or Spanish!",
    examples=[
        ["Where is my order?"],
        ["¿Cuál es el estado de mi envío?"],
        ["How do I track my package?"],
        ["Necesito ayuda con una devolución."],
    ]
)

if __name__ == "__main__":
    iface.launch()