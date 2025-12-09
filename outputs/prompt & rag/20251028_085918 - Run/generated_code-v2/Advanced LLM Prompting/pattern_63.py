import gradio as gr
from sentence_transformers import SentenceTransformer, util
import numpy as np

# 3. Multilingual Embedding Model
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

# 4. In-Context Example Store
# For simplicity, examples are in English and Spanish, and embeddings are pre-computed.
# In a real application, you'd have a more extensive and diverse set of examples.
examples = [
    {"text": "My internet is not working.", "label": "technical_support", "lang": "en"},
    {"text": "I cannot connect to the Wi-Fi.", "label": "technical_support", "lang": "en"},
    {"text": "How do I reset my password?", "label": "account_management", "lang": "en"},
    {"text": "I want to change my subscription plan.", "label": "billing_inquiry", "lang": "en"},
    {"text": "Mi internet no funciona.", "label": "technical_support", "lang": "es"},
    {"text": "No puedo conectar el Wi-Fi.", "label": "technical_support", "lang": "es"},
    {"text": "¿Cómo restablezco mi contraseña?", "label": "account_management", "lang": "es"},
    {"text": "Quiero cambiar mi plan de suscripción.", "label": "billing_inquiry", "lang": "es"},
    {"text": "I have a question about my last bill.", "label": "billing_inquiry", "lang": "en"},
    {"text": "Tengo una pregunta sobre mi última factura.", "label": "billing_inquiry", "lang": "es"},
]

# Pre-compute embeddings for examples
example_texts = [ex["text"] for ex in examples]
example_embeddings = model.encode(example_texts, convert_to_tensor=True)
for i, ex in enumerate(examples):
    ex["embedding"] = example_embeddings[i]

# 5. Example Alignment and Selection Module (XInSTA Prompting Core)
def get_aligned_examples(query: str, k_semantic: int = 2, k_task: int = 1):
    query_embedding = model.encode(query, convert_to_tensor=True)

    # Semantic Similarity Retrieval
    cosine_scores = util.cos_sim(query_embedding, example_embeddings)[0]
    top_semantic_indices = np.argsort(cosine_scores.cpu().numpy())[::-1][:k_semantic]
    semantic_examples = [examples[i] for i in top_semantic_indices]

    # For task-based alignment, we need a preliminary way to guess the intent.
    # For this example, we'll simply pick examples with the most frequent label among semantic examples
    # or if we have a simple keyword-based intent detection for illustration.
    # In a real system, a separate, perhaps simpler, classifier would provide this initial label.

    # Simplified Task-based Alignment: Infer a potential label from semantic examples
    potential_labels = [ex["label"] for ex in semantic_examples]
    if potential_labels:
        # Get the most frequent label if available, otherwise just pick one
        from collections import Counter
        most_common_label = Counter(potential_labels).most_common(1)[0][0]
    else:
        most_common_label = None
    
    task_examples = []
    if most_common_label:
        task_examples = [ex for ex in examples if ex["label"] == most_common_label and ex not in semantic_examples][:k_task]

    # Combined Strategy: Prioritize semantic, then add task-based
    selected_examples = list(semantic_examples)
    for ex in task_examples:
        if ex not in selected_examples:
            selected_examples.append(ex)

    return selected_examples

# 6. Prompt Construction Module
def construct_prompt(query: str, selected_examples: list):
    prompt_parts = ["You are a helpful customer support assistant. Classify the user's intent based on the query."]

    for ex in selected_examples:
        prompt_parts.append(f"Example query: {ex['text']}\nIntent: {ex['label']}")
    
    prompt_parts.append(f"User query: {query}\nIntent:")

    return "\n\n".join(prompt_parts)

# Placeholder for LLM Call (replace with actual LLM API integration)
def call_llm(prompt: str):
    # In a real application, you would integrate with an LLM API here (e.g., OpenAI, Google Gemini)
    # For this example, we'll simulate a response based on keywords or a very simple rule.
    if "internet" in prompt.lower() or "wi-fi" in prompt.lower():
        return "technical_support"
    elif "bill" in prompt.lower() or "factura" in prompt.lower() or "subscription" in prompt.lower() or "suscripción" in prompt.lower():
        return "billing_inquiry"
    elif "password" in prompt.lower() or "contraseña" in prompt.lower() or "account" in prompt.lower():
        return "account_management"
    else:
        return "unknown_intent"

# Main chatbot logic
def chatbot_response(user_query: str):
    if not user_query.strip():
        return "Please enter a query."

    # 5. Example Alignment and Selection
    aligned_examples = get_aligned_examples(user_query)

    # 6. Prompt Construction
    prompt = construct_prompt(user_query, aligned_examples)
    
    # 7. LLM Call (Simulated)
    predicted_intent = call_llm(prompt)

    return f"Detected Intent: {predicted_intent}\n\n--- Prompt Used ---\n{prompt}"

# 1. User Interface (UI) with Gradio
if __name__ == "__main__":
    interface = gr.Interface(
        fn=chatbot_response,
        inputs=gr.Textbox(lines=3, placeholder="Enter your customer support query here..."),
        outputs=gr.Textbox(),
        title="Multilingual XInSTA Customer Support Chatbot",
        description="This chatbot uses XInSTA Prompting to classify customer intents in multiple languages."
    )
    interface.launch()