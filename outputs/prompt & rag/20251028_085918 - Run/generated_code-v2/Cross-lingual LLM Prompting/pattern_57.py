import os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from langdetect import detect, DetectorFactory
import openai

DetectorFactory.seed = 0 # for consistent language detection results

# --- Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # Ensure you set this environment variable
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set.")
openai.api_key = OPENAI_API_KEY
LLM_MODEL = "gpt-3.5-turbo"
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"

# --- FAQ Knowledge Base ---
faq_data = [
    {"question": "What are your business hours?", "answer": "Our business hours are Monday to Friday, 9 AM to 5 PM EST."}, 
    {"question": "How can I contact customer support?", "answer": "You can contact customer support via email at support@example.com or call us at 1-800-123-4567."}, 
    {"question": "What is your return policy?", "answer": "We offer a 30-day return policy for most items. Please visit our website for more details."}, 
    {"question": "How do I track my order?", "answer": "You can track your order by logging into your account on our website and visiting the 'My Orders' section."}, 
    {"question": "Do you ship internationally?", "answer": "Yes, we offer international shipping to select countries. Shipping fees and times vary by destination."}, 
    {"question": "¿Cuál es el horario de atención?", "answer": "Nuestro horario de atención es de lunes a viernes, de 9 AM a 5 PM EST."},
    {"question": "Comment puis-je contacter le support client ?", "answer": "Vous pouvez contacter le support client par email à support@example.com ou nous appeler au 1-800-123-4567."},
    {"question": "Wie verfolge ich meine Bestellung?", "answer": "Sie können Ihre Bestellung verfolgen, indem Sie sich auf unserer Website in Ihr Konto einloggen und den Bereich 'Meine Bestellungen' besuchen."}
]

# --- Multilingual Embedding Model ---
print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
print("Embedding model loaded.")

# --- Vector Store for FAQ Retrieval ---
faq_questions = [item["question"] for item in faq_data]
print("Generating FAQ embeddings...")
faq_embeddings = embedding_model.encode(faq_questions)
faq_embeddings = np.array(faq_embeddings).astype("float32")

dimension = faq_embeddings.shape[1]
faiss_index = faiss.IndexFlatIP(dimension) # Using Inner Product for cosine similarity with normalized vectors
faiss_index.add(faq_embeddings)
print("FAISS index built with", faiss_index.ntotal, "FAQ entries.")

# --- Language Detection Module ---
def detect_language(text):
    try:
        return detect(text)
    except:
        return "en" # Default to English if detection fails

# --- FAQ Retrieval ---
def retrieve_faq(query_embedding, k=3):
    D, I = faiss_index.search(np.array([query_embedding]).astype("float32"), k)
    retrieved_items = [faq_data[i] for i in I[0]]
    return retrieved_items

# --- Prompt Engineering Module (InCLT Implementation) ---
SYSTEM_INSTRUCTION = (
    "You are a multilingual customer support chatbot. Your goal is to answer user queries accurately and concisely, "
    "leveraging the provided FAQ information. If the answer is not in the FAQ, state that you don't know but offer to connect them to a human."
    "Pay close attention to the examples provided to understand how to handle cross-lingual queries. "
    "Respond in the same language as the user's query unless explicitly asked otherwise."
)

# Cross-lingual in-context examples to stimulate transfer
CROSS_LINGUAL_EXAMPLES = [
    {
        "query": "How do I reset my password?",
        "faq_context": "Question: How do I reset my password? Answer: To reset your password, visit the login page and click 'Forgot Password'. Follow the instructions sent to your email.",
        "response": "To reset your password, visit the login page and click 'Forgot Password'. Follow the instructions sent to your email."
    },
    {
        "query": "¿Cómo restablezco mi contraseña?", # Spanish query
        "faq_context": "Question: How do I reset my password? Answer: To reset your password, visit the login page and click 'Forgot Password'. Follow the instructions sent to your email.",
        "response": "Para restablecer su contraseña, visite la página de inicio de sesión y haga clic en 'Olvidé mi contraseña'. Siga las instrucciones enviadas a su correo electrónico."
    },
    {
        "query": "Quel est le délai de livraison standard ?", # French query
        "faq_context": "Question: What is your return policy? Answer: Standard delivery time is 3-5 business days.", # Using a different FAQ for this example
        "response": "Le délai de livraison standard est de 3 à 5 jours ouvrables."
    }
]

def build_in_clt_prompt(user_query, retrieved_faqs, user_language):
    prompt_parts = []
    prompt_parts.append(f"System Instruction: {SYSTEM_INSTRUCTION}\n\n")

    prompt_parts.append("Here are some examples of how to answer user questions, even across different languages:")
    for example in CROSS_LINGUAL_EXAMPLES:
        prompt_parts.append(f"User Query (Example): {example['query']}")
        prompt_parts.append(f"Relevant FAQ Context (Example): {example['faq_context']}")
        prompt_parts.append(f"Chatbot Response (Example): {example['response']}\n")

    prompt_parts.append("\nBased on the following relevant FAQ entries, please answer the user's query:")
    for i, faq in enumerate(retrieved_faqs):
        prompt_parts.append(f"FAQ {i+1} - Question: {faq['question']}")
        prompt_parts.append(f"FAQ {i+1} - Answer: {faq['answer']}\n")

    prompt_parts.append(f"User's Query (Language: {user_language}): {user_query}\n")
    prompt_parts.append(f"Chatbot Response:")

    return "".join(prompt_parts)

# --- Large Language Model (LLM) Integration ---
def get_llm_response(prompt):
    try:
        response = openai.ChatCompletion.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTION}, # System instruction here for clarity
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=150
        )
        return response.choices[0].message['content'].strip()
    except openai.error.OpenAIError as e:
        print(f"An error occurred with the OpenAI API: {e}")
        return "I apologize, but I'm having trouble connecting to my knowledge base right now. Please try again later."
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return "I apologize, an unexpected error occurred. Please try again later."

# --- Chatbot Interface (Basic) ---
def main():
    print("\n--- Multilingual Customer Support Chatbot ---")
    print("Type 'exit' or 'quit' to end the conversation.")
    print("---------------------------------------------")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Chatbot: Goodbye!")
            break

        user_language = detect_language(user_input)
        print(f"(Detected language: {user_language})")

        query_embedding = embedding_model.encode([user_input])[0]
        retrieved_faqs = retrieve_faq(query_embedding, k=3)
        
        # Debug print for retrieved FAQs
        # print("\nRetrieved FAQs:")
        # for faq in retrieved_faqs:
        #     print(f"- Q: {faq['question']} | A: {faq['answer']}")

        prompt = build_in_clt_prompt(user_input, retrieved_faqs, user_language)
        
        # Debug print for the full prompt
        # print("\nFull Prompt sent to LLM:")
        # print(prompt)

        chatbot_response = get_llm_response(prompt)
        print(f"Chatbot: {chatbot_response}")

if __name__ == "__main__":
    main()
