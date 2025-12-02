import torch
import numpy as np
from transformers import pipeline, MarianMTModel, MarianTokenizer
from sentence_transformers import SentenceTransformer
from typing import List, Dict

# Global model loaders and caches
translation_pipelines: Dict[str, pipeline] = {}
embedding_model: SentenceTransformer = None
llm_pipeline: pipeline = None
knowledge_base_embeddings: np.ndarray = None
knowledge_base_articles: List[str] = []

def load_models():
    global embedding_model, llm_pipeline
    print("Loading embedding model...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Embedding model loaded.")

    print("Loading LLM (GPT2 for demonstration)...")
    # Check for GPU availability
    device = 0 if torch.cuda.is_available() else -1
    llm_pipeline = pipeline('text-generation', model='gpt2', device=device)
    print("LLM loaded.")

def get_translator(source_lang: str, target_lang: str):
    model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
    if model_name not in translation_pipelines:
        print(f"Loading translation model: {model_name}...")
        # Check for GPU availability
        device = 0 if torch.cuda.is_available() else -1
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        translator = pipeline("translation", model=model, tokenizer=tokenizer, device=device)
        translation_pipelines[model_name] = translator
        print(f"Translation model {model_name} loaded.")
    return translation_pipelines[model_name]

# Language Translation Module
def translate(text: str, source_lang: str, target_lang: str) -> str:
    translator = get_translator(source_lang, target_lang)
    translated_text = translator(text, max_length=512)[0]['translation_text'] # Added max_length to prevent warnings
    return translated_text

# Embedding Generation Module
def get_embedding(text: str) -> List[float]:
    if embedding_model is None:
        raise RuntimeError("Embedding model not loaded. Call load_models() first.")
    return embedding_model.encode(text).tolist()

# Knowledge Base & Retrieval Module
def load_knowledge_base(data: List[str]):
    global knowledge_base_articles, knowledge_base_embeddings
    if embedding_model is None:
        raise RuntimeError("Embedding model not loaded. Call load_models() first.")
    knowledge_base_articles = data
    print("Embedding knowledge base articles...")
    # convert_to_tensor=True for GPU if available, then move to cpu for numpy ops
    embeddings_tensor = embedding_model.encode(data, convert_to_tensor=True)
    knowledge_base_embeddings = embeddings_tensor.cpu().numpy()
    print("Knowledge base embedded.")

def retrieve_exemplars(query_embedding: List[float], top_k: int = 3) -> List[str]:
    global knowledge_base_embeddings, knowledge_base_articles

    if knowledge_base_embeddings is None or not knowledge_base_articles:
        return []

    query_embedding_np = np.array(query_embedding).reshape(1, -1)

    query_norm = np.linalg.norm(query_embedding_np, axis=1, keepdims=True)
    if np.any(query_norm == 0):
        return []

    kb_norms = np.linalg.norm(knowledge_base_embeddings, axis=1, keepdims=True)
    kb_norms[kb_norms == 0] = 1 # Avoid division by zero

    query_embedding_normalized = query_embedding_np / query_norm
    kb_embeddings_normalized = knowledge_base_embeddings / kb_norms

    similarities = np.dot(query_embedding_normalized, kb_embeddings_normalized.T).flatten()

    top_k_indices = np.argsort(similarities)[::-1][:top_k]
    
    return [knowledge_base_articles[i] for i in top_k_indices]

# Prompt Engineering & LLM Interaction Module
def build_prompt(original_query: str, exemplars: List[str], target_lang_name: str) -> str:
    prompt_parts = []
    prompt_parts.append(f"You are a helpful customer support assistant for an e-commerce platform. Respond to the customer's query in {target_lang_name}. Keep the response concise and helpful.")
    if exemplars:
        prompt_parts.append("\n\nHere are some relevant examples that might help you formulate a response:")
        for i, exemplar in enumerate(exemplars):
            prompt_parts.append(f"Example {i+1}: {exemplar}")
    prompt_parts.append(f"\n\nCustomer Query: {original_query}\n\nAgent Response in {target_lang_name}:")
    return "\n".join(prompt_parts)

def generate_response(prompt: str) -> str:
    global llm_pipeline
    if llm_pipeline is None:
        raise RuntimeError("LLM not loaded. Call load_models() first.")
    
    # Generate text, ensuring it returns only the new generation
    generation = llm_pipeline(
        prompt,
        max_new_tokens=150, 
        num_return_sequences=1,
        truncation=True,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.2,
        pad_token_id=llm_pipeline.tokenizer.eos_token_id,
        return_full_text=False # Crucial for getting only the generated part
    )
    
    if generation and 'generated_text' in generation[0]:
        response_text = generation[0]['generated_text'].strip()
        
        # Post-processing to get a clean response
        # Look for common sentence endings or clear breaks
        # Heuristic: stop at the first double newline, or a single newline if it seems like a complete thought.
        # Or if the response seems to end with a question mark or period.
        
        first_newline_idx = response_text.find('\n\n')
        if first_newline_idx != -1:
            response_text = response_text[:first_newline_idx].strip()
        else:
            # Try single newline as a softer break
            single_newline_idx = response_text.find('\n')
            if single_newline_idx != -1:
                response_text = response_text[:single_newline_idx].strip()

        # Further refine by looking for sentence endings if it's still long
        if len(response_text) > 70 and '.' in response_text:
            last_period_idx = response_text.rfind('.')
            if last_period_idx > len(response_text) * 0.5: # Only if a period is in the latter half
                response_text = response_text[:last_period_idx + 1].strip()
        elif len(response_text) > 70 and '?' in response_text: # Handle questions
            last_q_mark_idx = response_text.rfind('?')
            if last_q_mark_idx > len(response_text) * 0.5: 
                response_text = response_text[:last_q_mark_idx + 1].strip()

        return response_text
    return "Error: Could not generate response from LLM."

# Orchestration Module (Chatbot Pipeline)
def process_query(customer_query: str, low_resource_lang_code: str, low_resource_lang_name: str) -> str:
    # 1. Translate LRL query to English
    print(f"Translating customer query from {low_resource_lang_code} to en...")
    english_query = translate(customer_query, low_resource_lang_code, 'en')
    print(f"English Query: {english_query}")

    # 2. Get embedding for English query
    query_embedding = get_embedding(english_query)

    # 3. Retrieve relevant English exemplars
    print("Retrieving relevant English exemplars...")
    english_exemplars = retrieve_exemplars(query_embedding, top_k=3)
    print(f"Retrieved English exemplars: {english_exemplars}")

    # 4. Translate English exemplars back to LRL
    lrl_exemplars = []
    if english_exemplars:
        print(f"Translating English exemplars to {low_resource_lang_code}...")
        for ex in english_exemplars:
            lrl_exemplars.append(translate(ex, 'en', low_resource_lang_code))
    print(f"Translated LRL exemplars: {lrl_exemplars}")

    # 5. Build prompt for LLM
    prompt = build_prompt(customer_query, lrl_exemplars, low_resource_lang_name)
    print(f"\n--- LLM Prompt ---\n{prompt}\n------------------")

    # 6. Generate response in LRL
    print(f"Generating response in {low_resource_lang_code} using LLM...")
    lrl_response = generate_response(prompt)
    
    return lrl_response

# Main execution block
if __name__ == "__main__":
    # Load models once at startup
    load_models()

    # Sample English Knowledge Base
    sample_knowledge_base = [
        "How do I track my order? You can track your order using the tracking number provided in your shipping confirmation email on our website's 'Track Order' page.",
        "What is your return policy? We offer a 30-day return policy for unused items with original packaging. Please visit our returns page for more details and to initiate a return.",
        "How can I contact customer support? You can reach our customer support team via live chat on our website, email at support@example.com, or by calling us at 1-800-123-4567.",
        "My item arrived damaged. What should I do? Please take photos of the damaged item and packaging, and contact our customer support immediately. We will arrange for a replacement or refund.",
        "Do you offer international shipping? Yes, we offer international shipping to many countries. Shipping costs and delivery times vary by destination. Check our shipping information page for details."
    ]
    load_knowledge_base(sample_knowledge_base)

    print("\n--- Cross-Lingual Customer Support Chatbot ---")
    print("Enter 'exit' to quit.")

    # User can choose a low-resource language. For simplicity, let's use French as an example.
    # Ensure you have the corresponding MarianMT models available (e.g., opus-mt-fr-en and opus-mt-en-fr).
    low_resource_lang_code = "fr"  # Example: French
    low_resource_lang_name = "French"

    # Pre-load required translation models for the chosen LRL to avoid delays during interaction
    get_translator(low_resource_lang_code, 'en')
    get_translator('en', low_resource_lang_code)

    while True:
        try:
            user_query = input(f"\nEnter your query in {low_resource_lang_name} (e.g., 'Je veux suivre ma commande'): ")
            if user_query.lower() == 'exit':
                break

            response = process_query(user_query, low_resource_lang_code, low_resource_lang_name)
            print(f"\nChatbot ({low_resource_lang_name}): {response}")

        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please ensure you have internet access for model downloads and enough memory.")
            # For robust production systems, more specific error handling and logging would be needed.
            break
