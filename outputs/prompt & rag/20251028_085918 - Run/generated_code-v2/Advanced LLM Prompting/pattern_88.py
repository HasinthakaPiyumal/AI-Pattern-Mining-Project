import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# 1. Load Translation Models
@st.cache_resource
def load_translator_sw_en():
    return pipeline("translation", model="Helsinki-NLP/opus-mt-sw-en")

@st.cache_resource
def load_translator_en_sw():
    return pipeline("translation", model="Helsinki-NLP/opus-mt-en-sw")

translator_sw_en = load_translator_sw_en()
translator_en_sw = load_translator_en_sw()

# 2. Load Sentence Transformer for Embeddings
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

# 3. Define English Knowledge Base and build FAISS index
english_knowledge_base = [
    "What is your return policy? Our return policy allows returns within 30 days of purchase with a valid receipt.",
    "How can I track my order? You can track your order using the tracking number provided in your shipping confirmation email.",
    "Do you offer international shipping? Yes, we offer international shipping to most countries. Shipping fees apply.",
    "What payment methods do you accept? We accept Visa, Mastercard, American Express, PayPal, and Google Pay.",
    "How do I contact customer support? You can reach customer support via email at support@ecommerce.com or by calling us at 1-800-123-4567.",
    "Can I change my shipping address after placing an order? Unfortunately, we cannot change the shipping address once an order has been placed. Please ensure your address is correct before confirming.",
    "What are your operating hours? Our customer support operates from Monday to Friday, 9 AM to 5 PM EST.",
    "Do you have a loyalty program? Yes, our loyalty program offers discounts and exclusive benefits to our members. Sign up on our website.",
    "How do I reset my password? You can reset your password by clicking on the 'Forgot Password' link on the login page.",
    "Are your products ethically sourced? We are committed to ethical sourcing and work with suppliers who adhere to fair labor practices."
]

@st.cache_resource
def build_faiss_index(knowledge_base, model):
    embeddings = model.encode(knowledge_base, convert_to_tensor=False)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype('float32'))
    return index, embeddings

faiss_index, kb_embeddings = build_faiss_index(english_knowledge_base, embedding_model)

# 4. Load LLM (GPT-2)
@st.cache_resource
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    return tokenizer, model

llm_tokenizer, llm_model = load_llm()
llm_tokenizer.pad_token = llm_tokenizer.eos_token # Set pad_token

# Streamlit UI
st.title("🛒 Cross-lingual E-commerce Chatbot (Swahili-English)")
st.write("Ask your questions in Swahili, and get answers based on our English knowledge base!")

user_query_sw = st.text_input("Your question in Swahili:")

if user_query_sw:
    st.subheader("Processing...")

    # 1. Translate Swahili query to English
    translated_query_en = translator_sw_en(user_query_sw)[0]["translation_text"]
    st.write(f"Translated to English: {translated_query_en}")

    # 2. Retrieve relevant English exemplars
    query_embedding = embedding_model.encode([translated_query_en], convert_to_tensor=False)
    D, I = faiss_index.search(np.array(query_embedding).astype('float32'), k=2) # Retrieve top 2
    retrieved_exemplars = [english_knowledge_base[i] for i in I[0]]
    
    st.write(f"Retrieved exemplars: {retrieved_exemplars}")

    # 3. Prompt Augmentation
    augmented_prompt = f"Context:\n{'\n'.join(retrieved_exemplars)}\n\nQuestion: {translated_query_en}\nAnswer:"
    st.write(f"Augmented Prompt: {augmented_prompt}")

    # 4. Generate response with LLM
    inputs = llm_tokenizer(augmented_prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
    output_sequences = llm_model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=inputs["input_ids"].shape[-1] + 100, # Generate up to 100 new tokens
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
    )
    generated_text = llm_tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    
    # Extract only the answer part if the prompt is included in generation
    response_en = generated_text[len(augmented_prompt):].strip()
    st.write(f"Generated English Response: {response_en}")

    # 5. Translate English response to Swahili
    final_response_sw = translator_en_sw(response_en)[0]["translation_text"]
    st.subheader("Chatbot's Answer in Swahili:")
    st.success(final_response_sw)
else:
    st.info("Please enter your question in Swahili.")