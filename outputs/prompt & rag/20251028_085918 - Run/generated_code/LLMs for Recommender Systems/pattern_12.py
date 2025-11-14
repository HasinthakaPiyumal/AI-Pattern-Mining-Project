
import gradio as gr
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import random

# --- 1. Global Data and Model Loading ---

# Simulate a product catalog
product_catalog = [
    {"id": "P001", "name": "Smart Home Hub", "description": "Central control for all smart devices, voice-activated."},
    {"id": "P002", "name": "Wireless Noise-Cancelling Headphones", "description": "Immersive sound, comfortable fit, long battery life."},
    {"id": "P003", "name": "4K Ultra HD Smart TV", "description": "Stunning visuals, integrated streaming apps, voice control."},
    {"id": "P004", "name": "Ergonomic Office Chair", "description": "Full back support, adjustable armrests, breathable mesh."},
    {"id": "P005", "name": "Portable Bluetooth Speaker", "description": "Powerful sound, waterproof design, 12-hour playtime."},
    {"id": "P006", "name": "Robot Vacuum Cleaner", "description": "Automatic cleaning, smart mapping, app control."},
    {"id": "P007", "name": "Fitness Tracker Watch", "description": "Heart rate monitoring, sleep analysis, GPS tracking."},
    {"id": "P008", "name": "Professional DSLR Camera", "description": "High-resolution photos, 4K video, interchangeable lenses."},
    {"id": "P009", "name": "Electric Standing Desk", "description": "Adjustable height, memory presets, spacious desktop."},
    {"id": "P010", "name": "Gaming Laptop", "description": "High performance, powerful graphics card, backlit keyboard."},
    {"id": "P011", "name": "Air Fryer", "description": "Healthy cooking with less oil, multiple presets, easy to clean."},
    {"id": "P012", "name": "Smart Water Bottle", "description": "Tracks hydration, reminds you to drink, glows when thirsty."},
    {"id": "P013", "name": "Projector Mini", "description": "Portable, connect to phone, movie night anywhere."},
    {"id": "P014", "name": "Digital Drawing Tablet", "description": "Pressure sensitivity, precise strokes, for artists and designers."},
    {"id": "P015", "name": "Smart Garden Kit", "description": "Grow herbs indoors, self-watering, LED grow light."},
]

product_descriptions = [p["description"] for p in product_catalog]
product_ids = [p["id"] for p in product_catalog]

# Load Sentence-Transformer model for embeddings
# This model will be used to create rich semantic representations of products and user queries.
print("Loading SentenceTransformer model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("SentenceTransformer model loaded.")

# Generate embeddings for all products
product_embeddings = embedding_model.encode(product_descriptions, convert_to_tensor=False)

# Load a smaller conversational LLM from Hugging Face for explanations and interaction
# Using a conversational pipeline for a more natural human-AI engagement.
print("Loading conversational LLM...")
llm_explainer_pipeline = pipeline(
    "text-generation", 
    model="distilgpt2", 
    torch_dtype=random.choice([None])
) # Use default dtype for simplicity
print("Conversational LLM loaded.")

# --- 2. Recommendation Logic ---

def recommend_products(user_query: str, num_recommendations: int = 3) -> list:
    """
    Recommends products based on user query using semantic similarity.
    This function leverages the enriched data representation provided by the embedding model.
    """
    user_embedding = embedding_model.encode([user_query], convert_to_tensor=False)
    similarities = cosine_similarity(user_embedding, product_embeddings)[0]
    
    # Get top N product indices
    top_indices = np.argsort(similarities)[::-1][:num_recommendations]
    
    recommended_product_details = [
        product_catalog[i] for i in top_indices
    ]
    return recommended_product_details

# --- 3. LLM for Explanations and Dynamic Adaptability ---

def generate_llm_explanation_and_adaptation(
    recommended_products: list,
    user_query: str,
    conversation_history: list
) -> str:
    """
    Generates human-centric explanations for recommendations and simulates dynamic adaptation
    based on user query and conversation history using an LLM.
    """
    product_names = ", ".join([p["name"] for p in recommended_products])
    product_descriptions_summary = "\n".join([
        f"- {p['name']}: {p['description']}" for p in recommended_products
    ])

    # Craft a prompt for the LLM to generate explanations and adapt
    prompt = f"""
        You are an intelligent e-commerce recommender system. Your goal is to provide helpful product recommendations and explain why they are suitable, considering the user's needs and previous conversation.

        --- Conversation History ---
        {"".join([f"{item[0]}\n{item[1]}\n" for item in conversation_history])}

        --- Current User Query ---
        User: {user_query}

        --- Recommended Products ---
        {product_descriptions_summary}

        Based on the user's query and the recommended products ({product_names}), explain why these products are a good fit. Also, suggest how the user can refine their search or what other products might be relevant based on their implied interests. Be conversational and helpful.
        Explanation:
        """
    
    # Generate explanation using the LLM
    # The max_new_tokens and num_return_sequences are used to control the LLM output.
    llm_output = llm_explainer_pipeline(
        prompt,
        max_new_tokens=200,
        num_return_sequences=1,
        temperature=0.7, # Controls randomness
        do_sample=True, # Enables sampling for more creative output
        top_k=50, # Limits the vocabulary to top-k words
        eos_token_id=llm_explainer_pipeline.tokenizer.eos_token_id
    )
    
    # Extract the generated text, removing the prompt itself
    explanation = llm_output[0]["generated_text"].replace(prompt, "").strip()
    return explanation

# --- 4. Gradio Interface ---

def chat_interface(user_message, history):
    """
    Main function for the Gradio chatbot interface.
    Handles user input, calls the recommender, generates explanations, and manages conversation.
    """
    history = history or []

    # 1. Get initial recommendations based on the user's current message
    recommendations = recommend_products(user_message)
    recommended_names = [p["name"] for p in recommendations]
    initial_response = f"Based on your query, I recommend: {", ".join(recommended_names)}."

    # 2. Use LLM to generate a more detailed explanation and adaptable response
    llm_enhanced_explanation = generate_llm_explanation_and_adaptation(
        recommended_products=recommendations,
        user_query=user_message,
        conversation_history=history # Pass history for dynamic adaptation
    )

    # Combine initial response with LLM explanation
    full_response = f"{initial_response}\n\n{llm_enhanced_explanation}"

    history.append((user_message, full_response))
    return "", history


# Define the Gradio Blocks interface
with gr.Blocks() as demo:
    gr.Markdown(
        """
        # Intelligent E-commerce Recommender with LLM-powered Explanations
        Ask me for product recommendations! I'll provide explanations and adapt to your needs.
        Example queries: "headphones for travel", "smart devices for home automation", "gaming laptop for high performance"
        """
    )
    chatbot = gr.Chatbot(height=400)
    msg = gr.Textbox(label="Your Query")
    clear = gr.Button("Clear")

    msg.submit(chat_interface, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: (None, None), None, [msg, chatbot])

    gr.Markdown(
        """
        **How it works:**
        1.  **Enriched Data Representation:** Product descriptions are converted into semantic embeddings using `SentenceTransformer`.
        2.  **Recommendation Engine:** User queries are embedded, and similar products are found using cosine similarity.
        3.  **LLM for Explanations & Adaptability:** A `distilgpt2` model (via `transformers` pipeline) generates natural language explanations for recommendations and can adapt its suggestions based on the ongoing conversation, simulating dynamic responsiveness and human-centric interaction.
        """
    )

if __name__ == "__main__":
    demo.launch()
