import streamlit as st
from langdetect import detect, LangDetectException
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
# from transformers import pipeline # Uncomment for actual LLM integration

# --- 1. Configuration and Model Loading ---

@st.cache_resource
def load_embedding_model():
    """Loads the multilingual sentence transformer model."""
    return SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

# @st.cache_resource
# def load_llm_pipeline():
#     """Loads a multilingual LLM pipeline. (Example using a smaller model)"""
#     # For a real application, consider a larger, more capable multilingual model
#     # and potentially running it on a more powerful backend or using an API.
#     return pipeline("text2text-generation", model="Helsinki-NLP/opus-mt-en-es")

# embedding_model = load_embedding_model()
# llm_pipeline = load_llm_pipeline() # Uncomment for actual LLM integration

# For demonstration, we'll use a placeholder for the LLM response generation
embedding_model = load_embedding_model()

def get_llm_response_placeholder(prompt, target_language):
    """Placeholder for LLM response generation."""
    st.warning("LLM integration is simulated. In a real application, a multilingual LLM would process the prompt.")
    
    # Simple logic to simulate LLM using parts of the prompt
    if "Customer Query:" in prompt and "FAQ Snippets:" in prompt:
        query_start = prompt.find("Customer Query:") + len("Customer Query:")
        query_end = prompt.find("FAQ Snippets:", query_start)
        customer_query = prompt[query_start:query_end].strip()
        
        faq_start = prompt.find("FAQ Snippets:") + len("FAQ Snippets:")
        faq_snippets = prompt[faq_start:].strip().split('\n- ')
        faq_snippets = [snippet.strip() for snippet in faq_snippets if snippet.strip()]

        response = f"Based on your query '{customer_query}' and our knowledge base, here's what we found: "
        if faq_snippets:
            response += ". ".join(faq_snippets[:2]) # Take top 2 snippets for simulation
        else:
            response += "We couldn't find a direct answer in our FAQs, but we will connect you to a human agent."
        
        # Simulate a very basic translation if the target language is different from English
        if target_language == 'es':
            return f"Respuesta simulada en español: {response}"
        elif target_language == 'fr':
            return f"Réponse simulée en français : {response}"
        else:
            return f"Simulated Response in {target_language.upper()}: {response}"
    return "Simulated LLM response: Please refine your query."


# --- 2. FAQ Data Management ---

# Example FAQ data (can be loaded from JSON/DB in a real app)
FAQ_DATA = [
    {"id": 1, "question": "What is your return policy?", "answer": "You can return items within 30 days of purchase with a valid receipt.", "lang": "en"},
    {"id": 2, "question": "How do I track my order?", "answer": "You can track your order using the tracking number provided in your shipping confirmation email.", "lang": "en"},
    {"id": 3, "question": "¿Cuál es su política de devolución?", "answer": "Puede devolver artículos dentro de los 30 días posteriores a la compra con un recibo válido.", "lang": "es"},
    {"id": 4, "question": "¿Cómo hago un seguimiento de mi pedido?", "answer": "Puede seguir su pedido utilizando el número de seguimiento proporcionado en su correo electrónico de confirmación de envío.", "lang": "es"},
    {"id": 5, "question": "Comment retourner un article?", "answer": "Vous pouvez retourner les articles dans les 30 jours suivant l'achat avec un reçu valide.", "lang": "fr"},
    {"id": 6, "question": "Livrez-vous à l'international?", "answer": "Oui, nous livrons dans la plupart des pays du monde. Les frais d'expédition varient.", "lang": "fr"}
]

@st.cache_resource
def create_faiss_index(faq_data, model):
    """Creates a FAISS index from FAQ questions."""
    questions = [entry["question"] for entry in faq_data]
    embeddings = model.encode(questions)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    return index, questions

faq_index, indexed_questions = create_faiss_index(FAQ_DATA, embedding_model)

# --- 3. In-Context Cross-lingual Examples (InCLT Pattern) ---

# These examples demonstrate cross-lingual transfer for the LLM
INCLT_EXAMPLES = [
    {
        "instruction": "Translate the following customer query from Spanish to English and then find a suitable answer.",
        "spanish_query": "¿Cuál es la política de reembolso?",
        "english_equivalent": "What is the refund policy?",
        "english_answer": "You can return items within 30 days of purchase for a full refund with a valid receipt."
    },
    {
        "instruction": "Given an English query, find the relevant French FAQ and respond in French.",
        "english_query": "How do I return a product?",
        "french_equivalent_faq": "Comment retourner un article?",
        "french_answer": "Vous pouvez retourner les articles dans les 30 jours suivant l'achat avec un reçu valide."
    },
    {
        "instruction": "Respond to the customer's query in their original language, even if the primary information is in another language.",
        "source_query": "¿Necesito una cuenta para comprar?", # Spanish
        "retrieved_english_faq_q": "Do I need an account to buy?",
        "retrieved_english_faq_a": "No, you can check out as a guest, but creating an account has benefits.",
        "target_spanish_response": "No, puede realizar la compra como invitado, pero crear una cuenta tiene beneficios."
    }
]

def format_inclt_examples(examples):
    """Formats the in-context examples for the LLM prompt."""
    formatted_examples = []
    for ex in examples:
        example_str = """
Example:
Instruction: {instruction}
""".format(instruction=ex["instruction"])
        if "spanish_query" in ex: example_str += f"Spanish Query: {ex['spanish_query']}\n"
        if "english_equivalent" in ex: example_str += f"English Equivalent: {ex['english_equivalent']}\n"
        if "english_answer" in ex: example_str += f"English Answer: {ex['english_answer']}\n"
        if "english_query" in ex: example_str += f"English Query: {ex['english_query']}\n"
        if "french_equivalent_faq" in ex: example_str += f"French Equivalent FAQ: {ex['french_equivalent_faq']}\n"
        if "french_answer" in ex: example_str += f"French Answer: {ex['french_answer']}\n"
        if "source_query" in ex: example_str += f"Source Query: {ex['source_query']}\n"
        if "retrieved_english_faq_q" in ex: example_str += f"Retrieved English FAQ Question: {ex['retrieved_english_faq_q']}\n"
        if "retrieved_english_faq_a" in ex: example_str += f"Retrieved English FAQ Answer: {ex['retrieved_english_faq_a']}\n"
        if "target_spanish_response" in ex: example_str += f"Target Spanish Response: {ex['target_spanish_response']}\n"
        formatted_examples.append(example_str)
    return "\n".join(formatted_examples)


# --- 4. Prompt Construction Module ---

def build_llm_prompt(customer_query, detected_lang, relevant_faqs, inclt_examples_str, top_k=3):
    """Constructs the prompt for the LLM, incorporating InCLT examples and FAQs."""
    prompt_parts = []

    prompt_parts.append("You are a helpful multilingual customer support assistant for an e-commerce platform.")
    prompt_parts.append(f"The customer's original query is in {detected_lang.upper()}. You must respond in {detected_lang.upper()}.")
    prompt_parts.append("Use the following examples to understand how to leverage information across different languages:")
    prompt_parts.append(inclt_examples_str)
    prompt_parts.append("\n---")

    prompt_parts.append(f"Customer Query: {customer_query}")

    if relevant_faqs:
        prompt_parts.append("FAQ Snippets (potentially in various languages): ")
        for i, faq in enumerate(relevant_faqs[:top_k]):
            prompt_parts.append(f"- Q ({faq['lang'].upper()}): {faq['question']}")
            prompt_parts.append(f"- A ({faq['lang'].upper()}): {faq['answer']}")
    else:
        prompt_parts.append("No direct FAQ snippets found, try to answer based on general knowledge or politely state lack of info.")

    prompt_parts.append(f"Please provide a comprehensive answer in {detected_lang.upper()} based on the query and the provided information.")

    return "\n".join(prompt_parts)


# --- 5. Streamlit Application ---

st.set_page_config(page_title="Multilingual E-commerce Chatbot", layout="centered")
st.title("🌍 Multilingual E-commerce Support Chatbot")
st.markdown("Ask a question in English, Spanish, or French about our e-commerce platform!")

user_query = st.text_input("Your Question:", "How do I return a product?")

if st.button("Get Answer"):
    if user_query:
        st.subheader("Chatbot Response:")
        
        # 1. Language Detection
        try:
            detected_lang = detect(user_query)
            st.info(f"Detected Language: {detected_lang.upper()}")
        except LangDetectException:
            detected_lang = "en" # Default to English if detection fails
            st.warning("Could not reliably detect language. Defaulting to English.")

        # 2. Embedding Generation for Query
        query_embedding = embedding_model.encode([user_query])

        # 3. FAQ Retrieval
        D, I = faq_index.search(np.array(query_embedding).astype('float32'), k=3) # Search top 3
        
        relevant_faqs = []
        for idx in I[0]:
            if idx != -1: # Ensure a valid index was found
                relevant_faqs.append(FAQ_DATA[idx])
        
        st.write("Relevant FAQs Found:")
        for faq in relevant_faqs:
            st.code(f"Q ({faq['lang'].upper()}): {faq['question']} | A ({faq['lang'].upper()}): {faq['answer']}")

        # 4. Prompt Construction
        formatted_inclt = format_inclt_examples(INCLT_EXAMPLES)
        llm_prompt = build_llm_prompt(user_query, detected_lang, relevant_faqs, formatted_inclt)
        
        st.subheader("\n--- LLM Prompt (for debugging/understanding) ---")
        st.text_area("Prompt sent to LLM:", llm_prompt, height=400)

        # 5. LLM Integration (Simulated)
        # For actual LLM:
        # response = llm_pipeline(llm_prompt, max_new_tokens=200)[0]['generated_text']
        response = get_llm_response_placeholder(llm_prompt, detected_lang)
        
        st.subheader("\n--- Final Answer ---")
        st.success(response)

    else:
        st.warning("Please enter a question.")
