import gradio as gr
from langdetect import detect, DetectorFactory
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Ensure reproducibility for langdetect
DetectorFactory.seed = 0

# --- 1. Knowledge Base/FAQ ---
# In a real application, this would be a database.
# We'll represent it as a list of dictionaries for simplicity.
knowledge_base = [
    {
        "id": 1,
        "en_question": "Where is my order?",
        "en_answer": "You can track your order status on our 'My Orders' page after logging in.",
        "es_question": "¿Dónde está mi pedido?",
        "es_answer": "Puede rastrear el estado de su pedido en nuestra página 'Mis Pedidos' después de iniciar sesión."
    },
    {
        "id": 2,
        "en_question": "How do I return an item?",
        "en_answer": "Please visit our Returns Policy page and follow the instructions to initiate a return.",
        "es_question": "¿Cómo devuelvo un artículo?",
        "es_answer": "Por favor, visite nuestra página de Política de Devoluciones y siga las instrucciones para iniciar una devolución."
    },
    {
        "id": 3,
        "en_question": "What are the payment methods accepted?",
        "en_answer": "We accept Visa, MasterCard, American Express, PayPal, and Google Pay.",
        "es_question": "¿Qué métodos de pago aceptan?",
        "es_answer": "Aceptamos Visa, MasterCard, American Express, PayPal y Google Pay."
    },
    {
        "id": 4,
        "en_question": "Can I change my shipping address after placing an order?",
        "en_answer": "Unfortunately, once an order is placed, we cannot change the shipping address. Please ensure your address is correct before confirming your purchase.",
        "es_question": "¿Puedo cambiar mi dirección de envío después de realizar un pedido?",
        "es_answer": "Lamentablemente, una vez que se realiza un pedido, no podemos cambiar la dirección de envío. Por favor, asegúrese de que su dirección sea correcta antes de confirmar su compra."
    },
    {
        "id": 5,
        "en_question": "Do you offer international shipping?",
        "en_answer": "Yes, we offer international shipping to most countries. Shipping costs and delivery times vary by destination.",
        "es_question": "¿Ofrecen envíos internacionales?",
        "es_answer": "Sí, ofrecemos envíos internacionales a la mayoría de los países. Los costos de envío y los tiempos de entrega varían según el destino."
    }
]

# --- 2. Embedding Model Initialization ---
# Using a multilingual sentence transformer model
model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        model_output = model(**inputs)
    # Mean pooling to get sentence embedding
    sentence_embedding = model_output.last_hidden_state.mean(dim=1).squeeze().numpy()
    return sentence_embedding

# Pre-compute embeddings for the knowledge base questions
kb_embeddings_en = np.array([get_embedding(item["en_question"]) for item in knowledge_base])
kb_embeddings_es = np.array([get_embedding(item["es_question"]) for item in knowledge_base])

# --- 3. Language Detection Module ---
def detect_language(text):
    try:
        return detect(text)
    except:
        return "en" # Default to English if detection fails

# --- 4. In-Context Example Retrieval Module ---
def retrieve_in_context_examples(query, detected_lang, num_examples=2):
    query_embedding = get_embedding(query)

    # Use embeddings for the detected language for similarity search
    if detected_lang == "es":
        similarities = cosine_similarity([query_embedding], kb_embeddings_es)[0]
    else: # Default to English or other languages using English KB if no specific KB
        similarities = cosine_similarity([query_embedding], kb_embeddings_en)[0]

    # Get top N indices
    top_indices = np.argsort(similarities)[::-1][:num_examples]

    retrieved_examples = []
    for idx in top_indices:
        item = knowledge_base[idx]
        retrieved_examples.append({
            "source_q": item["en_question"],
            "source_a": item["en_answer"],
            "target_q": item.get(f"{detected_lang}_question", item["en_question"]),
            "target_a": item.get(f"{detected_lang}_answer", item["en_answer"])
        })
    return retrieved_examples

# --- 5. Mock Multilingual Large Language Model (LLM) ---
def mock_llm_response(prompt):
    # This is a placeholder for a real LLM API call (e.g., OpenAI, Google Generative AI)
    # In a real scenario, you'd send the prompt to an LLM and get a generated response.
    print(f"\n--- Mock LLM Prompt ---\n{prompt}\n---\n")
    if "¿Dónde está mi pedido?" in prompt:
        return "Según la información proporcionada en los ejemplos, puedes verificar el estado de tu pedido en la página 'Mis Pedidos' después de iniciar sesión. Si tienes más preguntas, no dudes en preguntar."
    elif "Where is my order?" in prompt:
        return "Based on the provided examples, you can track your order status on your 'My Orders' page after logging in. Feel free to ask if you have more questions."
    elif "return an item" in prompt.lower() or "devuelvo un artículo" in prompt.lower():
        return "Para devolver un artículo, por favor consulta nuestra política de devoluciones y sigue los pasos indicados. Si necesitas ayuda adicional, estamos aquí para ayudarte."
    elif "payment methods" in prompt.lower() or "métodos de pago" in prompt.lower():
        return "Aceptamos varias formas de pago, incluyendo Visa, MasterCard, American Express, PayPal y Google Pay. ¿Hay algún método específico sobre el que tengas dudas?"
    else:
        return "Lo siento, no tengo suficiente información para responder a tu pregunta con precisión. ¿Podrías reformularla o proporcionar más detalles?"

# --- 6. Prompt Construction Module ---
def construct_prompt(user_query, detected_lang, examples):
    instructions = (
        "You are a helpful customer support assistant for an e-commerce platform. "
        "Use the provided examples to answer the customer's query accurately and in their language. "
        "If the examples are not sufficient, try to provide a general helpful response."
    )

    example_str = ""
    for i, ex in enumerate(examples):
        example_str += f"\n### Example {i+1} (English):\nQ: {ex['source_q']}\nA: {ex['source_a']}"
        if detected_lang != "en": # Only add target language example if different from source
             example_str += f"\n### Example {i+1} ({detected_lang.upper()}):\nQ: {ex['target_q']}\nA: {ex['target_a']}"

    prompt = f"""
{instructions}

{example_str}

### Customer Query ({detected_lang.upper()}):
Q: {user_query}
A:
"""
    return prompt

# --- Main Chatbot Logic ---
def chatbot_response(user_query):
    # 1. Detect Language
    detected_lang = detect_language(user_query)
    print(f"Detected Language: {detected_lang}")

    # 2. Retrieve In-Context Examples
    relevant_examples = retrieve_in_context_examples(user_query, detected_lang)

    # 3. Construct Prompt
    llm_prompt = construct_prompt(user_query, detected_lang, relevant_examples)

    # 4. Get LLM Response (Mocked)
    response = mock_llm_response(llm_prompt)

    return response

# --- Gradio UI ---
if __name__ == "__main__":
    interface = gr.Interface(
        fn=chatbot_response,
        inputs=gr.Textbox(lines=2, placeholder="Ask me anything about your order or our products..."),
        outputs="text",
        title="Multilingual E-commerce Support Chatbot (InCLT Crosslingual Transfer Prompting)",
        description="This chatbot uses InCLT Crosslingual Transfer Prompting to provide better cross-lingual support. "
                    "Ask questions in English or Spanish! (e.g., 'Where is my order?', '¿Dónde está mi pedido?')"
    )
    interface.launch()
