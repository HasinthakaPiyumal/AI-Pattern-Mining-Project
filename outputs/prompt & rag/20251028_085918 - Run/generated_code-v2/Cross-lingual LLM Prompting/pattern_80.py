import gradio as gr
from langdetect import detect
from sentence_transformers import SentenceTransformer, util
import json
import numpy as np

# 1. Knowledge Base (Simplified for demonstration)
# In a real application, this would be much larger and potentially in a vector database.
knowledge_base = [
    {
        "id": 1,
        "question_en": "How do I reset my password?",
        "answer_en": "You can reset your password by going to the 'Forgot Password' link on the login page.",
        "question_es": "¿Cómo restablezco mi contraseña?",
        "answer_es": "Puede restablecer su contraseña yendo al enlace 'Olvidé mi contraseña' en la página de inicio de sesión."
    },
    {
        "id": 2,
        "question_en": "What are your operating hours?",
        "answer_en": "Our customer support is available 24/7.",
        "question_es": "¿Cuáles son sus horarios de atención?",
        "answer_es": "Nuestro servicio de atención al cliente está disponible las 24 horas del día, los 7 días de la semana."
    },
    {
        "id": 3,
        "question_en": "How can I track my order?",
        "answer_en": "You can track your order using the tracking number provided in your shipping confirmation email.",
        "question_es": "¿Cómo puedo rastrear mi pedido?",
        "answer_es": "Puede rastrear su pedido utilizando el número de seguimiento proporcionado en su correo electrónico de confirmación de envío."
    },
    {
        "id": 4,
        "question_en": "What is your return policy?",
        "answer_en": "Our return policy allows returns within 30 days of purchase with a valid receipt.",
        "question_es": "¿Cuál es su política de devoluciones?",
        "answer_es": "Nuestra política de devoluciones permite devoluciones dentro de los 30 días posteriores a la compra con un recibo válido."
    },
]

# 2. Embedding Model
# Using a multilingual model
embedding_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

# Pre-compute embeddings for knowledge base questions
kb_questions_en = [item["question_en"] for item in knowledge_base]
kb_questions_es = [item["question_es"] for item in knowledge_base]

kb_embeddings_en = embedding_model.encode(kb_questions_en, convert_to_tensor=True)
kb_embeddings_es = embedding_model.encode(kb_questions_es, convert_to_tensor=True)

# 3. Simulated Multilingual LLM (Placeholder)
# In a real application, this would be an API call to a service like OpenAI, Cohere, or a self-hosted model.
def multilingual_llm(prompt: str, target_lang: str) -> str:
    print(f"--- LLM Input (Target Lang: {target_lang}) ---\n{prompt}\n---")
    # Simple rule-based simulation for demonstration
    if "reset password" in prompt.lower() or "restablecer contraseña" in prompt.lower():
        if target_lang == "es":
            return "Para restablecer su contraseña, visite el enlace 'Olvidé mi contraseña' en la página de inicio de sesión."
        else:
            return "To reset your password, please visit the 'Forgot Password' link on the login page."
    elif "operating hours" in prompt.lower() or "horarios de atención" in prompt.lower():
        if target_lang == "es":
            return "Nuestro soporte al cliente está disponible 24/7."
        else:
            return "Our customer support is available 24/7."
    elif "track my order" in prompt.lower() or "rastrear mi pedido" in prompt.lower():
        if target_lang == "es":
            return "Puede rastrear su pedido con el número de seguimiento de su correo electrónico de confirmación de envío."
        else:
            return "You can track your order using the tracking number in your shipping confirmation email."
    elif "return policy" in prompt.lower() or "política de devoluciones" in prompt.lower():
        if target_lang == "es":
            return "Nuestra política de devoluciones permite devoluciones dentro de los 30 días de la compra con un recibo válido."
        else:
            return "Our return policy allows returns within 30 days of purchase with a valid receipt."
    else:
        if target_lang == "es":
            return "Lo siento, no tengo suficiente información para responder a eso. ¿Puede reformular su pregunta?"
        else:
            return "I'm sorry, I don't have enough information to answer that. Can you rephrase your question?"

# 4. InCLT Prompt Construction
def get_icl_examples(user_query: str, query_lang: str, k: int = 2) -> list:
    query_embedding = embedding_model.encode(user_query, convert_to_tensor=True)

    # Use embeddings corresponding to the query language for initial retrieval
    if query_lang == 'es':
        cosine_scores = util.cos_sim(query_embedding, kb_embeddings_es)[0]
    else: # Default to English if detection fails or for 'en'
        cosine_scores = util.cos_sim(query_embedding, kb_embeddings_en)[0]
    
    top_k_indices = np.argsort(cosine_scores.cpu().numpy())[::-1][:k]
    
    icl_examples = []
    for idx in top_k_indices:
        example = knowledge_base[idx]
        # Constructing InCLT example: leveraging both source (e.g., English) and target (user's query language)
        # For simplicity, we assume English is the primary source language of our KB.
        # The 'target' language content is directly taken if available, or a placeholder for translation.
        
        # Example in English (source language context)
        icl_example_source = f"Question: {example['question_en']}\nAnswer: {example['answer_en']}"

        # Example in Target Language (user's query language context)
        if query_lang == 'es':
            icl_example_target = f"Pregunta: {example['question_es']}\nRespuesta: {example['answer_es']}"
        else:
            icl_example_target = f"Question: {example['question_en']}\nAnswer: {example['answer_en']}" # Fallback to English if target is not Spanish

        icl_examples.append({"source": icl_example_source, "target": icl_example_target})
    return icl_examples

def construct_prompt(user_query: str, query_lang: str, icl_examples: list) -> str:
    instruction = "You are a helpful customer support assistant. Answer the user's question concisely based on the provided examples. If the answer is not in the examples, indicate that you don't have enough information.\n\n"
    if query_lang == 'es':
        instruction = "Eres un asistente de atención al cliente útil. Responde a la pregunta del usuario de forma concisa basándote en los ejemplos proporcionados. Si la respuesta no está en los ejemplos, indica que no tienes suficiente información.\n\n"

    example_str = ""
    for ex in icl_examples:
        # Combine source and target language examples to stimulate cross-lingual transfer
        example_str += f"-- Example --\n{ex['source']}\n{ex['target']}\n\n"
    
    final_prompt = f"{instruction}{example_str}User Query ({query_lang}): {user_query}\nAnswer ({query_lang}):"
    return final_prompt

# 5. Main Chatbot Function
def chatbot_response(user_input: str, history: list) -> str:
    # 5.1 Language Detection
    try:
        detected_lang = detect(user_input)
    except:
        detected_lang = 'en' # Default to English if detection fails
    
    # 5.2 Get InCLT Examples
    icl_examples = get_icl_examples(user_input, detected_lang)
    
    # 5.3 Construct Prompt
    prompt = construct_prompt(user_input, detected_lang, icl_examples)
    
    # 5.4 Get LLM Response
    response = multilingual_llm(prompt, detected_lang)
    
    return response

# 6. Gradio Interface
if __name__ == "__main__":
    gr.ChatInterface(
        fn=chatbot_response,
        title="Multilingual Customer Support Chatbot",
        description="Ask questions in English or Spanish. The chatbot uses InCLT for better cross-lingual understanding."
    ).launch()