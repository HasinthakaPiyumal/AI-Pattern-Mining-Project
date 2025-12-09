from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import gradio as gr

# 1. Multilingual LLM (Placeholder for a real LLM)
class MultilingualLLM:
    def __init__(self, model_name="t5-small"): # Using a small model for demonstration purposes
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.generator = pipeline("text2text-generation", model=self.model, tokenizer=self.tokenizer)

    def generate_response(self, prompt: str) -> str:
        # In a real scenario, this would be a more sophisticated LLM call
        # For this example, we'll just simulate a response or use a simple T5 model
        if "Please provide a general answer if you cannot find a specific one." in prompt:
            return "I am a multilingual customer support chatbot, and I'm here to assist you with your queries."
        response = self.generator(prompt, max_length=150, num_return_sequences=1)[0]['generated_text']
        return response

# 2. Translation Service
class TranslationService:
    def __init__(self):
        self.translators = {}

    def _get_translator(self, src_lang, tgt_lang):
        model_key = f"opus-mt-{src_lang}-{tgt_lang}"
        if model_key not in self.translators:
            try:
                model_name = f"Helsinki-NLP/{model_key}"
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                self.translators[model_key] = pipeline("translation", model=model, tokenizer=tokenizer)
            except Exception as e:
                print(f"Could not load translator for {src_lang} to {tgt_lang}: {e}")
                self.translators[model_key] = None # Mark as failed to load
        return self.translators[model_key]

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        if src_lang == tgt_lang:
            return text
        
        translator = self._get_translator(src_lang, tgt_lang)
        if translator:
            try:
                return translator(text, max_length=512)[0]['translation_text']
            except Exception as e:
                print(f"Translation error from {src_lang} to {tgt_lang}: {e}")
                return f"[Translation failed for '{text}']"
        else:
            return f"[No translator available for {src_lang} to {tgt_lang} for '{text}']"

# 3. Knowledge Base
KNOWLEDGE_BASE = {
    "en": [
        {"q": "What are your shipping options?", "a": "We offer standard and express shipping worldwide. Standard shipping takes 5-7 business days, and express takes 2-3 business days."},
        {"q": "How can I track my order?", "a": "You can track your order using the tracking number provided in your shipping confirmation email on our website's 'Track Order' page."},
        {"q": "What is your return policy?", "a": "Items can be returned within 30 days of purchase, provided they are unused and in their original packaging. Please see our full return policy online."},
        {"q": "Do you ship internationally?", "a": "Yes, we ship to most countries worldwide. International shipping fees may apply."},
        {"q": "How do I contact customer support?", "a": "You can reach our customer support team via email at support@example.com or by phone at +1-800-123-4567 during business hours."}
    ]
}

# 4. Prompt Engineering Module
class PromptEngineer:
    def __init__(self, knowledge_base: dict, translation_service: TranslationService):
        self.knowledge_base = knowledge_base
        self.translation_service = translation_service

    def _retrieve_relevant_examples(self, query: str, num_examples: int = 2) -> list:
        # Simple keyword-based matching for demonstration. 
        # In a real system, this would use semantic search (e.g., Sentence Transformers + Faiss/Chroma).
        relevant_examples = []
        query_lower = query.lower()
        
        for qa_pair in self.knowledge_base['en']:
            if any(keyword in qa_pair['q'].lower() for keyword in query_lower.split()):
                relevant_examples.append(qa_pair)
            if len(relevant_examples) >= num_examples:
                break
        if not relevant_examples and self.knowledge_base['en']:
            # If no specific match, pick a couple of general examples
            relevant_examples.extend(self.knowledge_base['en'][:num_examples])
        return relevant_examples[:num_examples]

    def _construct_in_context_prompt(self, query: str, target_lang: str, examples: list) -> str:
        prompt_parts = []
        
        if target_lang != "en":
            # Introduce the ICL with target language context
            prompt_parts.append(f"You are a helpful multilingual assistant. Here are some examples of questions and answers in English and {target_lang} to help you understand the context and respond accurately:")
        else:
             prompt_parts.append(f"You are a helpful multilingual assistant. Here are some examples of questions and answers to help you understand the context and respond accurately:")

        for example in examples:
            en_q = example['q']
            en_a = example['a']
            
            if target_lang != "en":
                tgt_q = self.translation_service.translate(en_q, "en", target_lang)
                tgt_a = self.translation_service.translate(en_a, "en", target_lang)
                
                if "[Translation failed" not in tgt_q and "[Translation failed" not in tgt_a:
                    prompt_parts.append(f"\nEnglish Question: {en_q}")
                    prompt_parts.append(f"English Answer: {en_a}")
                    prompt_parts.append(f"{target_lang.capitalize()} Question: {tgt_q}")
                    prompt_parts.append(f"{target_lang.capitalize()} Answer: {tgt_a}")
                else:
                    # Fallback to English only if translation fails for an example
                    prompt_parts.append(f"\nEnglish Question: {en_q}")
                    prompt_parts.append(f"English Answer: {en_a}")
            else:
                prompt_parts.append(f"\nQuestion: {en_q}")
                prompt_parts.append(f"Answer: {en_a}")

        prompt_parts.append(f"\nNow, answer the following question in {target_lang}: {query}")
        prompt_parts.append("Please provide a general answer if you cannot find a specific one.")

        return "\n".join(prompt_parts)

    def generate_prompt(self, query: str, target_lang: str) -> str:
        relevant_examples = self._retrieve_relevant_examples(query)
        prompt = self._construct_in_context_prompt(query, target_lang, relevant_examples)
        return prompt

# 5. Chatbot Interface (Orchestrator)
class MultilingualChatbot:
    def __init__(self):
        self.translation_service = TranslationService()
        self.llm = MultilingualLLM()
        self.prompt_engineer = PromptEngineer(KNOWLEDGE_BASE, self.translation_service)

    def get_response(self, user_query: str, target_language: str) -> str:
        # Step 1: Generate the InCLT prompt
        prompt = self.prompt_engineer.generate_prompt(user_query, target_language)
        
        # Step 2: Get response from LLM
        llm_raw_response = self.llm.generate_response(prompt)
        
        # Step 3: (Optional) Post-process LLM response if needed (e.g., translate back to target_language if LLM responds in English)
        # For this setup, we expect the LLM to respond in target_language due to the prompt instruction
        return llm_raw_response

# Gradio Interface
chatbot_instance = MultilingualChatbot()

def chatbot_interface(query, target_lang):
    # Map display names to language codes for Helsinki-NLP models
    lang_map = {
        "English": "en", 
        "Spanish": "es", 
        "French": "fr", 
        "German": "de", 
        "Vietnamese": "vi", 
        "Russian": "ru", 
        "Chinese": "zh", 
        "Japanese": "ja"
    }
    
    actual_target_lang = lang_map.get(target_lang, "en")
    
    response = chatbot_instance.get_response(query, actual_target_lang)
    return response

if __name__ == "__main__":
    # Example usage (for testing without Gradio if desired)
    # query_en = "What is your return policy?"
    # response_en = chatbot_instance.get_response(query_en, "en")
    # print(f"English Response: {response_en}")

    # query_vi = "Chính sách hoàn trả của bạn là gì?" # What is your return policy?
    # response_vi = chatbot_instance.get_response(query_vi, "vi")
    # print(f"Vietnamese Response: {response_vi}")

    iface = gr.Interface(
        fn=chatbot_interface,
        inputs=[
            gr.Textbox(lines=2, placeholder="Enter your question here..."),
            gr.Dropdown(
                ["English", "Spanish", "French", "German", "Vietnamese", "Russian", "Chinese", "Japanese"],
                label="Target Language",
                value="English"
            )
        ],
        outputs="text",
        title="Multilingual Customer Support Chatbot (InCLT)",
        description="Ask a question and receive an answer, leveraging In-Context Learning Transfer for cross-lingual understanding."
    )

    print("Loading models... This may take a moment.")
    try:
        # Pre-load some common translators to avoid delay on first query
        _ = chatbot_instance.translation_service.translate("hello", "en", "es")
        _ = chatbot_instance.translation_service.translate("hello", "en", "vi")
        _ = chatbot_instance.translation_service.translate("hello", "es", "en")
        _ = chatbot_instance.translation_service.translate("hello", "vi", "en")
        print("Initial models loaded.")
    except Exception as e:
        print(f"Error pre-loading translation models: {e}. Some translations might be slower initially.")

    iface.launch()
