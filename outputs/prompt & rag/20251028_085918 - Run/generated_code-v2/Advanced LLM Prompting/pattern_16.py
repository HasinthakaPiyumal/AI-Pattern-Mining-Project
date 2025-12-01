import streamlit as st
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import chromadb
from langchain.prompts import PromptTemplate

translator_en_lr = pipeline("translation", model="Helsinki-NLP/opus-mt-en-hi")
translator_lr_en = pipeline("translation", model="Helsinki-NLP/opus-mt-hi-en")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.Client()
collection_name = "customer_support_knowledge"
try:
    kb_collection = client.get_collection(name=collection_name)
except:
    kb_collection = client.create_collection(name=collection_name)
    documents = [
        "How to reset your password? Go to settings and click 'Forgot Password'.",
        "Our shipping policy states that orders are delivered within 5-7 business days.",
        "To contact customer support, please email support@example.com.",
        "Returns are accepted within 30 days of purchase with original receipt.",
        "Your account can be updated in the 'My Profile' section.",
    ]
    metadatas = [{"source": "FAQ"}] * len(documents)
    ids = [f"doc_{i}" for i in range(len(documents))]
    embeddings = embedding_model.encode(documents).tolist()
    kb_collection.add(
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

llm_pipeline = pipeline("text-generation", model="distilgpt2", max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95, temperature=0.7)

def detect_language(text):
    if "नमस्ते" in text or "क्या हाल है" in text:
        return "hi"
    return "en"

def translate_text(text, source_lang, target_lang):
    if source_lang == "en" and target_lang == "hi":
        return translator_en_lr(text, clean_up_tokenization_spaces=True)[0]['translation_text']
    elif source_lang == "hi" and target_lang == "en":
        return translator_lr_en(text, clean_up_tokenization_spaces=True)[0]['translation_text']
    return text

template = """
You are a helpful customer support assistant. Answer the user's question based on the provided context.
If you don't know the answer, politely state that you cannot provide it.

Context:
{context}

Question: {question}
Answer:
"""
prompt = PromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc for doc in docs)

def get_parc_enhanced_response(user_query_english_text):
    query_embedding_np = embedding_model.encode(user_query_english_text)
    retrieved_docs_english = kb_collection.query(
        query_embeddings=[query_embedding_np.tolist()],
        n_results=2,
        include=['documents']
    )['documents'][0]
    
    formatted_context = format_docs(retrieved_docs_english)
    
    full_prompt_text = prompt.format(context=formatted_context, question=user_query_english_text)
    
    llm_output = llm_pipeline(full_prompt_text)[0]['generated_text']
    
    answer_prefix = "\nAnswer:"
    if answer_prefix in llm_output:
        extracted_answer = llm_output.split(answer_prefix, 1)[1].strip()
        if "\n" in extracted_answer:
            extracted_answer = extracted_answer.split("\n")[0].strip()
    else:
        extracted_answer = llm_output.replace(full_prompt_text, "").strip()
        if "\n" in extracted_answer:
            extracted_answer = extracted_answer.split("\n")[0].strip()
            
    if extracted_answer.startswith(user_query_english_text):
        extracted_answer = "I can help with that."
    if len(extracted_answer) > 200:
        extracted_answer = extracted_answer[:200] + "..."
            
    return extracted_answer

st.title("Global Customer Support Chatbot (PARC-enhanced)")
st.write("Ask your question. For demonstration, we simulate Hindi as a low-resource language.")

user_input = st.text_input("Your question:")

if user_input:
    detected_lang = detect_language(user_input)
    st.write(f"Detected language: {detected_lang.upper()}")

    user_query_english = user_input
    if detected_lang == "hi":
        user_query_english = translate_text(user_input, "hi", "en")
        st.write(f"Translated to English: {user_query_english}")
    
    with st.spinner("Generating response..."):
        llm_answer_english = get_parc_enhanced_response(user_query_english)
        st.write(f"LLM's English response: {llm_answer_english}")

    final_response = llm_answer_english
    if detected_lang == "hi":
        final_response = translate_text(llm_answer_english, "en", "hi")
        st.success(f"Chatbot Response (Hindi): {final_response}")
    else:
        st.success(f"Chatbot Response (English): {final_response}")