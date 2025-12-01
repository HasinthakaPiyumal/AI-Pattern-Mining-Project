import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

class KnowledgeBase:
    def __init__(self, documents):
        self.documents = documents

class VectorStore:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.embeddings = []
        self.texts = []

    def add_documents(self, documents):
        if not documents:
            return
        new_embeddings = self.embedding_model.encode(documents, convert_to_tensor=False)
        self.embeddings.extend(new_embeddings)
        self.texts.extend(documents)

    def search(self, query_embedding, top_k=3):
        if not self.embeddings:
            return []
        similarities = [1 - cosine(query_embedding, doc_emb) for doc_emb in self.embeddings]
        top_k_indices = np.argsort(similarities)[::-1][:top_k]
        return [(self.texts[i], similarities[i]) for i in top_k_indices]

class RetrieverModule:
    def __init__(self, embedding_model, vector_store):
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def retrieve(self, query, top_k=3):
        query_embedding = self.embedding_model.encode(query, convert_to_tensor=False)
        results = self.vector_store.search(query_embedding, top_k=top_k)
        return [doc_text for doc_text, _ in results]

class DynamicFAQChatbot:
    def __init__(self, kb_documents, retrieval_stride=4, top_k_retrieval=3, llm_model_name="distilgpt2", embedding_model_name="all-MiniLM-L6-v2"):
        self.knowledge_base = KnowledgeBase(kb_documents)
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        self.vector_store = VectorStore(self.embedding_model)
        self.vector_store.add_documents(self.knowledge_base.documents)
        
        self.retriever = RetrieverModule(self.embedding_model, self.vector_store)
        
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self.llm = AutoModelForCausalLM.from_pretrained(llm_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token # For distilgpt2

        self.retrieval_stride = retrieval_stride
        self.top_k_retrieval = top_k_retrieval
        self.conversation_history = []
        self.generated_tokens_since_last_retrieval = 0
        self.last_retrieved_context = []
        self.first_turn = True

    def _build_prompt(self, user_query, context):
        system_instruction = "You are an e-commerce customer support chatbot. Answer questions based on the provided context. If the answer is not in the context, state that you don't have enough information."
        
        context_str = "\n".join(context) if context else "No additional context provided."

        history_str = "\n".join([f"Customer: {q}\nBot: {a}" for q, a in self.conversation_history])
        
        prompt = f"{system_instruction}\n\nContext:\n{context_str}\n\nConversation History:\n{history_str}\n\nCustomer: {user_query}\nBot:"
        return prompt

    def _get_llm_response(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        
        # Generate response, limiting length to prevent infinite generation
        output = self.llm.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=50, 
            num_return_sequences=1,
            do_sample=True, 
            top_k=50, 
            top_p=0.95,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        # Decode only the newly generated tokens
        response_tokens = output[0, input_ids.shape[-1]:]
        response_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
        
        return response_text, len(response_tokens)

    def chat(self, user_query):
        current_context = []

        if self.first_turn or self.generated_tokens_since_last_retrieval >= self.retrieval_stride:
            current_context = self.retriever.retrieve(user_query, top_k=self.top_k_retrieval)
            self.last_retrieved_context = current_context
            self.generated_tokens_since_last_retrieval = 0
            self.first_turn = False
        else:
            current_context = self.last_retrieved_context

        prompt = self._build_prompt(user_query, current_context)
        llm_response, num_generated_tokens = self._get_llm_response(prompt)
        
        self.generated_tokens_since_last_retrieval += num_generated_tokens
        self.conversation_history.append((user_query, llm_response))
        
        return llm_response

if __name__ == "__main__":
    kb_docs = [
        "Our return policy allows returns within 30 days of purchase with a valid receipt. Items must be in original condition.",
        "Shipping usually takes 5-7 business days for standard delivery within the country. Expedited options are available.",
        "You can track your order using the tracking number provided in your shipping confirmation email.",
        "We accept Visa, MasterCard, American Express, and PayPal for all purchases.",
        "Our customer service is available Monday to Friday, 9 AM to 5 PM EST.",
        "The warranty for electronics is 1 year from the purchase date, covering manufacturing defects.",
        "To apply a discount code, enter it in the 'promo code' field at checkout.",
        "Our physical stores are open from 10 AM to 8 PM daily. Check our website for specific store locations."
    ]

    print("Initializing Chatbot with Retrieval Stride Optimization (stride=4)...")
    chatbot = DynamicFAQChatbot(kb_docs, retrieval_stride=4)
    print("Chatbot Ready. Type 'exit' to end the conversation.")

    while True:
        user_input = input("Customer: ")
        if user_input.lower() == 'exit':
            break
        
        response = chatbot.chat(user_input)
        print(f"Bot: {response}")
        print(f"[Debug] Tokens generated since last retrieval: {chatbot.generated_tokens_since_last_retrieval}")
        print(f"[Debug] Last retrieved context: {chatbot.last_retrieved_context}")
        print("-" * 30)
