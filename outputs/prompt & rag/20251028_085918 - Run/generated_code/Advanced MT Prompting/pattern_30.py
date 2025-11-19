import nltk
from nltk.tokenize import sent_tokenize
from transformers import pipeline, MarianMTModel, MarianTokenizer
from langchain.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer

# Download NLTK data (if not already downloaded)
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

# Mock Medical Knowledge Base
medical_knowledge_base = {
    "hypertension": "Hypertension, or high blood pressure, is a common condition in which the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease. Normal blood pressure is typically around 120/80 mmHg.",
    "diabetes": "Diabetes mellitus is a chronic metabolic disease characterized by high blood sugar levels (hyperglycemia), which over time leads to serious damage to the heart, blood vessels, eyes, kidneys and nerves. Type 1 diabetes is an autoimmune disease, while Type 2 diabetes is often lifestyle-related.",
    "cardiology": "Cardiology is a branch of medicine dealing with disorders of the heart and the cardiovascular system.",
    "neurology": "Neurology is a branch of medicine dealing with disorders of the nervous system.",
    "medication": "A substance used for medical treatment, especially a medicine or drug.",
    "diagnosis": "The identification of the nature of an illness or other problem by examination of the symptoms.",
}

# Initialize Models
# Translation Models
marian_model_name_en_es = "Helsinki-NLP/opus-mt-en-es"
marian_tokenizer_en_es = MarianTokenizer.from_pretrained(marian_model_name_en_es)
marian_model_en_es = MarianMTModel.from_pretrained(marian_model_name_en_es)
marian_translator_en_es = pipeline("translation", model=marian_model_en_es, tokenizer=marian_tokenizer_en_es)

marian_model_name_es_en = "Helsinki-NLP/opus-mt-es-en"
marian_tokenizer_es_en = MarianTokenizer.from_pretrained(marian_model_name_es_en)
marian_model_es_en = MarianMTModel.from_pretrained(marian_model_name_es_en)
marian_translator_es_en = pipeline("translation", model=marian_model_es_en, tokenizer=marian_tokenizer_es_en)

marian_model_name_mul_en = "Helsinki-NLP/opus-mt-mul-en" # Multilingual to English
marian_tokenizer_mul_en = MarianTokenizer.from_pretrained(marian_model_name_mul_en)
marian_model_mul_en = MarianMTModel.from_pretrained(marian_model_name_mul_en)
marian_translator_mul_en = pipeline("translation", model=marian_model_mul_en, tokenizer=marian_tokenizer_mul_en)

# Summarization Model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Sentence Transformer for context retrieval (mocked if embeddings are not used directly for search)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# --- Services ---

def detect_language_mock(text):
    # This is a mock function. In a real system, fasttext or similar would be used.
    if any(char in text for char in 'áéíóúñ'):
        return "es" # Assume Spanish
    return "en" # Assume English

def chunk_text(text, max_chunk_size=500):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0
    for sentence in sentences:
        if current_length + len(sentence) + 1 <= max_chunk_size:
            current_chunk.append(sentence)
            current_length += len(sentence) + 1
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = len(sentence) + 1
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def retrieve_medical_context(query):
    # Simple keyword-based retrieval from the mock knowledge base
    found_context = []
    query_lower = query.lower()
    for term, definition in medical_knowledge_base.items():
        if term in query_lower or any(word in query_lower for word in definition.lower().split()):
            found_context.append(f"Term: {term}. Definition: {definition}")
    return "\n".join(found_context)

def augment_prompt(original_text, context, task="translate"):
    if task == "translate":
        prompt_template = PromptTemplate(
            input_variables=["context", "text"],
            template="Given the following medical context:\n{context}\n\nTranslate the following medical text, ensuring medical accuracy and consistency with the provided context:\n{text}\nTranslation:"
        )
    elif task == "summarize":
        prompt_template = PromptTemplate(
            input_variables=["context", "text"],
            template="Given the following medical context:\n{context}\n\nSummarize the following medical text, highlighting key medical information and using terminology consistent with the context:\n{text}\nSummary:"
        )
    else:
        raise ValueError("Invalid task specified. Choose 'translate' or 'summarize'.")

    return prompt_template.format(context=context, text=original_text)

def translate_text(text, source_lang, target_lang):
    if source_lang == target_lang:
        return text

    if source_lang == "en" and target_lang == "es":
        return marian_translator_en_es(text)[0]['translation_text']
    elif source_lang == "es" and target_lang == "en":
        return marian_translator_es_en(text)[0]['translation_text']
    elif source_lang != "en" and target_lang == "en":
        # Use multilingual to English for low-resource/other non-English to English
        return marian_translator_mul_en(text)[0]['translation_text']
    elif source_lang != "en" and target_lang != "en":
        # Multi-step translation: source -> en -> target
        print(f"Performing multi-step translation: {source_lang} -> en -> {target_lang}")
        intermediate_en = marian_translator_mul_en(text)[0]['translation_text']
        if target_lang == "es":
             return marian_translator_en_es(intermediate_en)[0]['translation_text']
        else:
             # Fallback or specific model for en -> target if available
             print(f"Warning: Direct translation from English to {target_lang} not explicitly defined. Using general purpose model if available or returning English.")
             return f"[Translated via English to {target_lang} (unspecified model)]: {intermediate_en}"
    else:
        return f"Translation from {source_lang} to {target_lang} not supported yet for this combination."

def summarize_text(text, max_length=150, min_length=30):
    return summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']

# --- Main Orchestration --- 

def process_medical_document(
    document_text: str,
    target_language: str,
    summarize: bool = False,
):
    print(f"\nProcessing document for target language: {target_language}, Summarization: {summarize}")
    source_language = detect_language_mock(document_text)
    print(f"Detected source language: {source_language}")

    chunks = chunk_text(document_text)
    processed_chunks = []

    for i, chunk in enumerate(chunks):
        print(f"\n--- Processing Chunk {i+1}/{len(chunks)} ---")
        print(f"Original Chunk: {chunk[:100]}...")

        # Step 1: Pre-process non-English inputs to high-resource language (English)
        current_chunk_text = chunk
        current_source_lang = source_language
        if source_language != "en":
            print(f"Translating chunk from {source_language} to English for better processing.")
            current_chunk_text = translate_text(chunk, source_language, "en")
            current_source_lang = "en"
            print(f"Translated to English: {current_chunk_text[:100]}...")

        # Step 2: Context Retrieval and Prompt Augmentation
        context = retrieve_medical_context(current_chunk_text)
        print(f"Retrieved Context: {context if context else 'None'}")

        augmented_translation_prompt = augment_prompt(current_chunk_text, context, task="translate")
        augmented_summarization_prompt = augment_prompt(current_chunk_text, context, task="summarize")

        # Step 3: Translation Service
        final_translated_chunk = ""
        # If the original source was not English, the current_chunk_text is already in English
        # We then translate this English text to the final target_language
        if current_source_lang == "en": # This means original was English or already translated to English
            if target_language == "en":
                final_translated_chunk = current_chunk_text
            else:
                print(f"Translating augmented text from English to {target_language}.")
                # For actual translation, we'd ideally use the augmented_translation_prompt to guide the NMT model
                # However, current MarianMT pipeline doesn't directly take a 'prompt' for generation.
                # This is a simplification. A more advanced setup would involve LLM-based translation using the prompt.
                final_translated_chunk = translate_text(current_chunk_text, "en", target_language)
        else: # This path should ideally not be taken if we preprocess to English first
             print(f"Warning: Unexpected path in translation logic. Attempting direct translation from {source_language} to {target_language}.")
             final_translated_chunk = translate_text(chunk, source_language, target_language)

        print(f"Final Translated Chunk ({target_language}): {final_translated_chunk[:100]}...")

        # Step 4: Summarization Service (on the English version for consistency if preprocessed)
        chunk_summary = ""
        if summarize:
            print(f"Summarizing current chunk (using English version if preprocessed).")
            # Summarize the English version for consistency with the summarizer model's training
            chunk_summary = summarize_text(current_chunk_text)
            print(f"Chunk Summary (English): {chunk_summary[:100]}...")
            # If target language is not English, translate the summary
            if target_language != "en":
                print(f"Translating summary to {target_language}.")
                chunk_summary = translate_text(chunk_summary, "en", target_language)
                print(f"Chunk Summary ({target_language}): {chunk_summary[:100]}...")

        processed_chunks.append({"original_chunk": chunk, "translated_chunk": final_translated_chunk, "summary": chunk_summary})

    # Aggregate results
    full_translated_document = " ".join([pc["translated_chunk"] for pc in processed_chunks])
    full_summary = " ".join([pc["summary"] for pc in processed_chunks if pc["summary"]])

    return {
        "source_language": source_language,
        "target_language": target_language,
        "full_translated_document": full_translated_document,
        "full_summary": full_summary,
        "processed_chunks": processed_chunks,
    }

# --- Example Usage (Simplified UI Interaction) ---
if __name__ == "__main__":
    # Example 1: English Medical Document, Translate to Spanish and Summarize
    medical_doc_en = "The patient presented with classic symptoms of hypertension, including a consistent blood pressure reading of 160/100 mmHg. Treatment will involve medication and lifestyle changes to reduce the risk of cardiovascular events. Regular monitoring by a cardiologist is advised. Diabetes was ruled out after initial tests."
    print("\n=== Example 1: English to Spanish Translation & Summarization ===")
    result1 = process_medical_document(medical_doc_en, target_language="es", summarize=True)
    print("\n--- Final Results (Example 1) ---")
    print(f"Original Document: {medical_doc_en}")
    print(f"Translated Document (ES): {result1['full_translated_document']}")
    print(f"Summary (ES): {result1['full_summary']}")

    print("\n" + "="*80 + "\n")

    # Example 2: Spanish Medical Document, Translate to English (No Summarization)
    medical_doc_es = "El paciente fue admitido con un diagnóstico de diabetes tipo 2. Se observaron niveles altos de glucosa en sangre. Se iniciará un régimen de medicación y una dieta estricta. La evaluación por un neurólogo es necesaria debido a complicaciones."
    print("=== Example 2: Spanish to English Translation ===")
    result2 = process_medical_document(medical_doc_es, target_language="en", summarize=False)
    print("\n--- Final Results (Example 2) ---")
    print(f"Original Document: {medical_doc_es}")
    print(f"Translated Document (EN): {result2['full_translated_document']}")
    print(f"Summary (EN): {result2['full_summary'] if result2['full_summary'] else 'Not requested'}")

    print("\n" + "="*80 + "\n")

    # Example 3: Low-resource language (mocked as 'unknown non-en') to English, with context-aware translation and summarization
    # For demonstration, we'll use a slightly modified English text but assume it came from a low-resource language
    medical_doc_low_resource_mock = "The patient is experiencing severe headache and dizziness. A full neurological examination is required. Suspecting an issue related to the nervous system. Urgent attention is needed for this neurology case."
    # To simulate low-resource, we'll manually set source_language for demonstration purposes if needed
    # In a real app, detect_language_mock would handle this.
    print("=== Example 3: Low-Resource (Mocked) to English Translation & Summarization ===")
    result3 = process_medical_document(medical_doc_low_resource_mock, target_language="en", summarize=True)
    print("\n--- Final Results (Example 3) ---")
    print(f"Original Document: {medical_doc_low_resource_mock}")
    print(f"Translated Document (EN): {result3['full_translated_document']}")
    print(f"Summary (EN): {result3['full_summary']}")

    print("\n" + "="*80 + "\n")

    # Example 4: English to English, Summarize only
    medical_doc_en_sum_only = "A review of the patient's records indicates a history of controlled hypertension. Recent blood tests show stable glucose levels, ruling out diabetes at this time. The patient has been compliant with medication and regular follow-ups. No new cardiovascular events have been reported. Continual monitoring of blood pressure is recommended."
    print("=== Example 4: English to English (Summarize Only) ===")
    result4 = process_medical_document(medical_doc_en_sum_only, target_language="en", summarize=True)
    print("\n--- Final Results (Example 4) ---")
    print(f"Original Document: {medical_doc_en_sum_only}")
    print(f"Translated Document (EN): {result4['full_translated_document']}")
    print(f"Summary (EN): {result4['full_summary']}")
