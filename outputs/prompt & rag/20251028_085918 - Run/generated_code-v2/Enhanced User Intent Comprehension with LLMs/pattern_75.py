
import re
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import chromadb
import gradio as gr

# 1. Initialize models and components
# LLM for conversational responses
# Using 'gpt2' for demonstration purposes due to its balance of size and conversational ability.
# For production, consider fine-tuning a more capable instruction-tuned model.
llm_tokenizer = AutoTokenizer.from_pretrained("gpt2")
llm_model = AutoModelForCausalLM.from_pretrained("gpt2")
# Add pad token for generation if not present (common with GPT-2)
if llm_tokenizer.pad_token is None:
    llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    llm_model.resize_token_embeddings(len(llm_tokenizer))

llm_pipeline = pipeline(
    "text-generation",
    model=llm_model,
    tokenizer=llm_tokenizer,
    max_new_tokens=60, # Max tokens for AI's response
    num_return_sequences=1,
    do_sample=True, # Enable sampling for more varied responses
    top_k=50,
    top_p=0.95,
    pad_token_id=llm_tokenizer.pad_token_id, # Ensure pad_token_id is set
    device="cpu" # Use "cuda" if GPU is available
)

# Sentence Transformer for embeddings, used for semantic search in FAQs
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# ChromaDB for FAQ/Product information storage
# Using an in-memory client for simplicity in a single file demo
client = chromadb.Client()
# Get or create the collection for FAQs
faq_collection = client.get_or_create_collection(name="ecommerce_faqs")

# Populate FAQ/Product data
faqs_data = [
    {"question": "How can I track my order?", "answer": "You can track your order by logging into your account and visiting the 'My Orders' section, or by using the tracking link provided in your shipping confirmation email.", "id": "faq_track_order"},
    {"question": "What is your return policy?", "answer": "We offer a 30-day return policy for most items, provided they are unused and in their original packaging. Please see our full return policy for details.", "id": "faq_return_policy"},
    {"question": "How do I contact customer support?", "answer": "You can contact customer support via email at support@example.com or by calling 1-800-BUY-NOW.", "id": "faq_contact_support"},
    {"question": "Do you offer international shipping?", "answer": "Yes, we offer international shipping to many countries. Shipping costs and delivery times vary by destination.", "id": "faq_international_shipping"},
    {"question": "What payment methods do you accept?", "answer": "We accept major credit cards (Visa, Mastercard, Amex), PayPal, and Apple Pay.", "id": "faq_payment_methods"},
    {"question": "Where is my order?", "answer": "I can help you check your order status. Do you have your order number?", "id": "faq_where_is_my_order"},
    {"question": "How long does shipping take?", "answer": "Standard shipping usually takes 5-7 business days, while expedited shipping takes 2-3 business days.", "id": "faq_shipping_time"}
]

# Add FAQs to ChromaDB if the collection is empty
if faq_collection.count() == 0:
    faq_documents = [faq["question"] + " " + faq["answer"] for faq in faqs_data]
    faq_ids = [faq["id"] for faq in faqs_data]
    faq_metadata = [{"answer": faq["answer"]} for faq in faqs_data]

    faq_embeddings = embedding_model.encode(faq_documents).tolist()
    faq_collection.add(
        embeddings=faq_embeddings,
        documents=faq_documents,
        metadatas=faq_metadata,
        ids=faq_ids
    )

# 2. Intent Classification (Simplified rule-based for core intents)
def classify_intent(query):
    query_lower = query.lower()
    if any(keyword in query_lower for keyword in ["track order", "where is my order", "order status"]):
        return "order_status"
    if any(keyword in query_lower for keyword in ["return", "refund", "return policy"]):
        return "return_policy"
    if any(keyword in query_lower for keyword in ["contact", "help", "support", "customer service"]):
        return "contact_support"
    if any(keyword in query_lower for keyword in ["shipping", "deliver", "delivery time", "international shipping"]):
        return "shipping_inquiry"
    if any(keyword in query_lower for keyword in ["payment", "pay with", "methods", "accepted payments"]):
        return "payment_methods"
    if any(keyword in query_lower for keyword in ["product", "item", "stock", "availability", "details about"]):
        return "product_inquiry"
    return "general_inquiry"

# 3. Tool/Action Executor (Simulated functionality)
def get_order_status(order_number):
    # This function simulates an API call to an external order management system.
    # In a real application, this would involve actual API integrations.
    if order_number == "12345":
        return f"Order {order_number} is currently 'Shipped' and expected to arrive by December 25th."
    elif order_number == "67890":
        return f"Order {order_number} is 'Processing'. Estimated delivery in 3-5 business days."
    else:
        return f"Could not find details for order number {order_number}. Please double-check it."

def retrieve_faq_answer(query):
    # Performs a semantic search against the ChromaDB FAQ collection
    query_embedding = embedding_model.encode([query]).tolist()
    results = faq_collection.query(
        query_embeddings=query_embedding,
        n_results=1,
        include=['metadatas', 'distances']
    )
    # A similarity threshold is used to ensure only highly relevant answers are returned
    if results and results["distances"] and results["distances"][0][0] < 0.6: # Lower distance means higher similarity
        return results["metadatas"][0][0]["answer"]
    return None

# 4. Dialogue Manager
def generate_bot_response(user_message, history):
    intent = classify_intent(user_message)
    response = ""

    if intent == "order_status":
        # Attempt to extract an order number from the user's message
        match = re.search(r'\b\d{5,}\b', user_message) # Matches 5 or more digits
        if match:
            order_number = match.group(0)
            response = get_order_status(order_number)
        else:
            response = "I can help with your order status. What is your order number?"
    elif intent == "return_policy":
        response = retrieve_faq_answer("return policy") or "Our return policy allows returns within 30 days for most items. Do you have a specific item in mind?"
    elif intent == "contact_support":
        response = retrieve_faq_answer("contact customer support") or "You can reach us at support@example.com or call 1-800-BUY-NOW for assistance."
    elif intent == "shipping_inquiry":
        response = retrieve_faq_answer(user_message) or "We offer various shipping options. Are you asking about domestic, international shipping, or delivery times?"
    elif intent == "payment_methods":
        response = retrieve_faq_answer("payment methods") or "We accept major credit cards (Visa, Mastercard, Amex), PayPal, and Apple Pay. Is there a specific payment method you're curious about?"
    elif intent == "product_inquiry":
        response = "What product are you interested in? I can check its details, stock, or availability for you."
    else: # General inquiry or no specific intent matched, try FAQ or LLM fallback
        faq_answer = retrieve_faq_answer(user_message)
        if faq_answer:
            response = faq_answer
        else:
            # Fallback to LLM for open-ended conversation using conversational prompt
            conversation_history_str = "\n".join([f"Human: {h[0]}\nAI: {h[1]}" for h in history])
            prompt = f"The following is a friendly conversation between a human and an AI assistant named E-Chat. E-Chat is helpful, creative, clever, and very friendly.\n\n{conversation_history_str}\nHuman: {user_message}\nAI:"
            
            llm_output = llm_pipeline(prompt, num_return_sequences=1)[0]['generated_text']
            
            # Extract only the AI's response part from the LLM's output
            response_start_idx = llm_output.rfind("AI:")
            if response_start_idx != -1:
                response = llm_output[response_start_idx + len("AI:"):].strip()
                # Clean up any trailing partial sentences or unintended human prompts generated by LLM
                response = re.split(r'\nHuman:', response, 1)[0].strip()
            else:
                response = "I'm not sure how to respond to that. Can you rephrase or ask something else?"

    return response

# 5. Gradio Interface for user interaction
def chatbot_interface(user_message, history):
    # The 'history' parameter from Gradio is a list of [user_message, bot_message] pairs.
    # It's passed to generate_bot_response to maintain context for the LLM.
    
    bot_response = generate_bot_response(user_message, history)
    
    # Update history for the Gradio chatbot display
    history.append([user_message, bot_response])
    # Return empty string for the message input box and updated history for the chatbot component
    return "", history


# Gradio application launch
with gr.Blocks() as demo:
    gr.Markdown("# E-commerce Customer Support Chatbot")
    gr.Markdown("I can help you with your orders, returns, shipping, and general inquiries!")
    
    chatbot = gr.Chatbot(label="E-Chat Support")
    msg = gr.Textbox(label="Your Message", placeholder="Ask me anything about your order or our products...")
    
    with gr.Row():
        submit_btn = gr.Button("Send")
        clear_btn = gr.Button("Clear Chat")

    # gr.State is used to maintain the chat history across turns
    chat_history = gr.State([])

    # Event handlers for sending messages (button click or Enter key)
    submit_btn.click(chatbot_interface, [msg, chat_history], [msg, chatbot, chat_history])
    msg.submit(chatbot_interface, [msg, chat_history], [msg, chatbot, chat_history]) # Allows pressing Enter

    # Event handlers for clearing the chat
    clear_btn.click(lambda: None, None, chatbot, queue=False)
    clear_btn.click(lambda: [], None, chat_history, queue=False)
    clear_btn.click(lambda: "", None, msg, queue=False) # Also clear the message input box

demo.launch(debug=True)
