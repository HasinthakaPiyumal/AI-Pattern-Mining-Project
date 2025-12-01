from typing import Any, List, Dict, Optional
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import faiss
import torch

class KVCacheManager:
    def __init__(self):
        self.gpu_cache: Dict[str, Any] = {}
        self.host_cache: Dict[str, Any] = {}
        self.critical_keys: set[str] = set()

    def set(self, key: str, value: Any, is_critical: bool = False):
        self.gpu_cache[key] = value
        if is_critical:
            self.critical_keys.add(key)
            self.host_cache[key] = value

    def get(self, key: str) -> Optional[Any]:
        if key in self.gpu_cache:
            return self.gpu_cache[key]
        elif key in self.host_cache:
            return self.host_cache[key]
        return None

    def replicate_all_critical(self):
        for key in self.critical_keys:
            if key in self.gpu_cache:
                self.host_cache[key] = self.gpu_cache[key]

    def simulate_gpu_failure(self):
        print("Simulating GPU failure: Clearing GPU cache...")
        self.gpu_cache.clear()

    def recover_from_failure(self):
        print("Recovering from failure: Restoring critical keys from host cache...")
        for key in self.critical_keys:
            if key in self.host_cache:
                self.gpu_cache[key] = self.host_cache[key]

    def is_critical(self, key: str) -> bool:
        return key in self.critical_keys

class RAGSystem:
    def __init__(self, knowledge_base_docs: List[str], model_name: str = "distilgpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llm_model = AutoModelForCausalLM.from_pretrained(model_name)
        self.documents = knowledge_base_docs

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.vector_store = self._build_vector_store(knowledge_base_docs)

    def _get_document_embeddings(self, texts: List[str]) -> np.ndarray:
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.llm_model(**inputs, output_hidden_states=True)
            # Use the mean of the last hidden state for a simple embedding
            embeddings = outputs.hidden_states[-1].mean(dim=1).cpu().numpy()
        return embeddings

    def _build_vector_store(self, docs: List[str]):
        print("Building FAISS vector store...")
        doc_embeddings = self._get_document_embeddings(docs)
        d = doc_embeddings.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(doc_embeddings)
        print("FAISS vector store built.")
        return index

    def retrieve(self, query: str, top_k: int = 2) -> List[str]:
        query_embedding = self._get_document_embeddings([query])
        _, indices = self.vector_store.search(query_embedding, top_k)
        retrieved_docs = [self.documents[i] for i in indices[0] if i < len(self.documents)]
        return retrieved_docs

    def generate_response(self, prompt: str, retrieved_context: List[str]) -> str:
        full_prompt = "\n".join(retrieved_context) + "\nUser: " + prompt + "\nAssistant:"
        inputs = self.tokenizer(full_prompt, return_tensors="pt", max_length=1024, truncation=True)
        with torch.no_grad():
            outputs = self.llm_model.generate(
                inputs["input_ids"],
                max_new_tokens=100,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt itself from the response if present
        if response.startswith(full_prompt):
            response = response[len(full_prompt):].strip()
        return response

class CustomerSupportAssistant:
    def __init__(self, knowledge_base_docs: List[str]):
        self.kv_cache_manager = KVCacheManager()
        self.rag_system = RAGSystem(knowledge_base_docs)
        self.conversation_history: List[Dict[str, str]] = []

        # Initialize critical KV cache nodes
        system_prompt = "You are a helpful customer support assistant. Provide concise and accurate answers."
        self.kv_cache_manager.set("system_prompt", system_prompt, is_critical=True)
        print("System prompt loaded as critical KV cache node.")

    def process_query(self, user_query: str) -> str:
        # Update KV cache with current conversation context (critical)
        current_context_key = f"conversation_context_{len(self.conversation_history)}"
        current_context_value = {"user": user_query}
        self.kv_cache_manager.set(current_context_key, current_context_value, is_critical=True)
        print(f"User query stored in KV cache (critical): {user_query}")

        # Retrieve relevant documents
        retrieved_context = self.rag_system.retrieve(user_query)
        print(f"Retrieved context: {retrieved_context}")

        # Construct prompt for LLM
        system_prompt = self.kv_cache_manager.get("system_prompt")
        conversation_string = "\n".join([f"{turn['role']}: {turn['content']}" for turn in self.conversation_history])

        full_llm_prompt = f"{system_prompt}\n\nConversation History:\n{conversation_string}\n\nRetrieved Knowledge:\n{'\n'.join(retrieved_context)}\n\nUser Query: {user_query}"

        # Generate response
        llm_response = self.rag_system.generate_response(full_llm_prompt, retrieved_context)
        
        # Update KV cache with LLM's response (critical)
        self.kv_cache_manager.set(f"llm_response_{len(self.conversation_history)}", llm_response, is_critical=True)
        print(f"LLM response stored in KV cache (critical).")

        # Append to conversation history
        self.conversation_history.append({"role": "User", "content": user_query})
        self.conversation_history.append({"role": "Assistant", "content": llm_response})

        return llm_response

    def start_chat(self):
        print("\n--- Starting Customer Support Chat (type 'exit' to end) ---")
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                break
            response = self.process_query(user_input)
            print(f"Assistant: {response}")
        print("--- Chat Ended ---")

    def simulate_fault_and_recover(self):
        print("\n--- Demonstrating Fault Tolerance and Recovery ---")

        # Process a query to populate GPU cache
        initial_query = "What are your operating hours?"
        print(f"\nProcessing initial query: {initial_query}")
        _ = self.process_query(initial_query)
        print(f"Current GPU cache keys: {list(self.kv_cache_manager.gpu_cache.keys())}")
        print(f"Current Host cache keys: {list(self.kv_cache_manager.host_cache.keys())}")

        # Simulate GPU failure
        self.kv_cache_manager.simulate_gpu_failure()
        print(f"GPU cache after failure: {list(self.kv_cache_manager.gpu_cache.keys())}")

        # Attempt to get a critical item after failure (should be missing from GPU but available in Host)
        print(f"Attempting to retrieve 'system_prompt' from cache after GPU failure: {self.kv_cache_manager.get('system_prompt') is not None}")

        # Recover from failure
        self.kv_cache_manager.recover_from_failure()
        print(f"GPU cache after recovery: {list(self.kv_cache_manager.gpu_cache.keys())}")

        # Process another query to show continuity
        recovery_query = "Can I speak to a human agent?"
        print(f"\nProcessing query after recovery: {recovery_query}")
        recovery_response = self.process_query(recovery_query)
        print(f"Assistant (after recovery): {recovery_response}")
        print(f"GPU cache keys after recovery query: {list(self.kv_cache_manager.gpu_cache.keys())}")
        print(f"Host cache keys after recovery query: {list(self.kv_cache_manager.host_cache.keys())}")
        print("--- Fault Tolerance Demonstration Complete ---")

if __name__ == "__main__":
    knowledge_base = [
        "Our operating hours are Monday to Friday, 9 AM to 5 PM EST.",
        "For technical support, please visit our website and open a support ticket.",
        "You can reach a human agent by calling our hotline at 1-800-123-4567 during business hours.",
        "We offer a 30-day money-back guarantee on all products.",
        "Shipping usually takes 3-5 business days for domestic orders.",
        "Product returns can be initiated through your account's order history page."
    ]

    assistant = CustomerSupportAssistant(knowledge_base)
    assistant.start_chat()
    assistant.simulate_fault_and_recover()