import collections
import threading
import time
import gradio as gr
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss

class GPUCache:
    def __init__(self, capacity=100):
        self.cache = collections.OrderedDict()
        self.capacity = capacity
        self.is_healthy = True
        self.lock = threading.Lock()

    def put(self, key, value):
        with self.lock:
            if not self.is_healthy:
                return False
            self.cache[key] = value
            self.cache.move_to_end(key)
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)
            return True

    def get(self, key):
        with self.lock:
            if not self.is_healthy:
                return None
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
            return None

    def fail(self):
        with self.lock:
            print("Simulating GPU cache failure...")
            self.cache.clear()
            self.is_healthy = False

    def restore(self):
        with self.lock:
            print("Restoring GPU cache health...")
            self.is_healthy = True

    def items(self):
        with self.lock:
            return list(self.cache.items())


class HostCache:
    def __init__(self):
        self.cache = {}
        self.lock = threading.Lock()

    def put(self, key, value):
        with self.lock:
            self.cache[key] = value

    def get(self, key):
        with self.lock:
            return self.cache.get(key)

    def items(self):
        with self.lock:
            return list(self.cache.items())


class FaultTolerantKVCache:
    def __init__(self, gpu_capacity=100):
        self.gpu_cache = GPUCache(capacity=gpu_capacity)
        self.host_cache = HostCache()
        self.critical_nodes = set()
        self.lock = threading.Lock()
        self.system_prompt_key = "system_prompt_kv"

    def put(self, key, value, critical=False):
        with self.lock:
            gpu_success = self.gpu_cache.put(key, value)
            if critical or key == self.system_prompt_key:
                self.critical_nodes.add(key)
                self.host_cache.put(key, value)
            return gpu_success

    def get(self, key):
        with self.lock:
            value = self.gpu_cache.get(key)
            if value is None and not self.gpu_cache.is_healthy:
                print(f"GPU cache unhealthy or key {key} not found in GPU. Falling back to Host cache.")
                value = self.host_cache.get(key)
                if value is not None:
                    # Try to restore to GPU cache if found in host and GPU is healthy now
                    if self.gpu_cache.is_healthy: # Ensure GPU is healthy before putting back
                         self.gpu_cache.put(key, value)
            elif value is None and key in self.critical_nodes:
                # If not in GPU but marked as critical, try host cache
                print(f"Key {key} not found in healthy GPU cache but is critical. Checking Host cache.")
                value = self.host_cache.get(key)
                if value is not None and self.gpu_cache.is_healthy:
                    self.gpu_cache.put(key, value)
            return value

    def replicate_critical_nodes(self):
        with self.lock:
            print("Replicating critical nodes from GPU to Host cache...")
            for key in list(self.critical_nodes):
                if self.gpu_cache.is_healthy: # Only replicate if GPU is healthy
                    value = self.gpu_cache.get(key)
                    if value is not None:
                        self.host_cache.put(key, value)
                else:
                    # If GPU is unhealthy, we can't get from it, but host cache already has a copy
                    pass
            print("Replication complete.")
            return "Critical nodes replicated to Host cache."

    def failover_to_host(self):
        with self.lock:
            self.gpu_cache.fail()
            print("Performing failover: clearing GPU cache and restoring from host for critical nodes.")
            # Simulate new GPU being provisioned
            self.gpu_cache = GPUCache(capacity=self.gpu_cache.capacity) 
            self.gpu_cache.restore()
            for key in list(self.critical_nodes):
                value = self.host_cache.get(key)
                if value is not None:
                    self.gpu_cache.put(key, value)
            print("Failover complete. GPU cache re-populated with critical nodes from host.")
            return "GPU failure simulated. Critical nodes restored from Host cache."

    def is_gpu_healthy(self):
        with self.lock:
            return self.gpu_cache.is_healthy


class InMemoryVectorStore:
    def __init__(self, embedding_model):
        self.documents = []
        self.embeddings = None
        self.embedding_model = embedding_model
        self.index = None

    def add_documents(self, docs):
        self.documents.extend(docs)
        if docs:
            new_embeddings = self.embedding_model.encode(docs, convert_to_tensor=True).cpu().numpy()
            if self.embeddings is None:
                self.embeddings = new_embeddings
            else:
                self.embeddings = np.vstack([self.embeddings, new_embeddings])

            if self.index is None:
                dimension = self.embeddings.shape[1]
                self.index = faiss.IndexFlatLL2(dimension)
            self.index.add(new_embeddings)

    def search(self, query, k=3):
        if self.index is None:
            return []
        query_embedding = self.embedding_model.encode([query], convert_to_tensor=True).cpu().numpy()
        D, I = self.index.search(query_embedding, k)
        results = [self.documents[i] for i in I[0] if i != -1]
        return results


class RAGModel:
    def __init__(self, kv_cache):
        self.kv_cache = kv_cache
        print("Loading LLM and Tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        self.model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.vector_store = InMemoryVectorStore(self.embedding_model)
        self._prime_vector_store()
        self._initialize_system_prompt_kv_cache()
        print("LLM, Tokenizer, Embedding Model, and Vector Store loaded.")

    def _prime_vector_store(self):
        faqs = [
            "What are your operating hours? We are open 9 AM to 5 PM, Monday to Friday.",
            "How can I reset my password? You can reset your password by clicking on 'Forgot Password' on the login page.",
            "What is your return policy? We offer a 30-day return policy for most items, provided they are in their original condition.",
            "How do I contact customer support? You can reach us via email at support@example.com or call us at 1-800-123-4567.",
            "Do you offer international shipping? Yes, we offer international shipping to selected countries. Shipping costs and delivery times vary by destination."
        ]
        self.vector_store.add_documents(faqs)
        print(f"Primed vector store with {len(faqs)} FAQ documents.")

    def _initialize_system_prompt_kv_cache(self):
        system_prompt = "You are a helpful customer support assistant. Answer user questions concisely and politely."
        # Simulate storing system prompt's KV cache (just storing the prompt itself for simplicity)
        self.kv_cache.put(self.kv_cache.system_prompt_key, system_prompt, critical=True)
        print("System prompt initialized and marked as critical in KV cache.")

    def retrieve_context(self, query, k=3):
        return self.vector_store.search(query, k=k)

    def generate_response(self, query, context, chat_history):
        # Retrieve system prompt from KV cache
        system_prompt = self.kv_cache.get(self.kv_cache.system_prompt_key)
        if system_prompt is None:
            system_prompt = "You are a helpful assistant."
            print("Warning: System prompt not found in KV cache, using default.")

        context_str = "\n".join(context)
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        for human_msg, ai_msg in chat_history:
            messages.append({"role": "user", "content": human_msg})
            messages.append({"role": "assistant", "content": ai_msg})

        # Add the current query and context
        messages.append({"role": "user", "content": f"Context: {context_str}\nQuestion: {query}"})

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Use torch.no_grad() for inference to save memory
        with torch.no_grad():
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            output = self.model.generate(**inputs, max_new_tokens=100, pad_token_id=self.tokenizer.eos_token_id)
            response = self.tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        return response


# Main application logic
kv_cache_instance = FaultTolerantKVCache(gpu_capacity=50)
rag_model_instance = RAGModel(kv_cache_instance)

def chat_interface(message, history):
    context = rag_model_instance.retrieve_context(message)
    response = rag_model_instance.generate_response(message, context, history)
    return response

def replicate_action():
    return kv_cache_instance.replicate_critical_nodes()

def simulate_failure_action():
    return kv_cache_instance.failover_to_host()

with gr.Blocks() as demo:
    gr.Markdown("# Intelligent Customer Support Assistant")
    gr.Markdown("This assistant uses a RAG-based LLM with a fault-tolerant KV cache.")
    
    with gr.Row():
        with gr.Column():
            chat_history = gr.Chatbot(label="Chat History", height=400)
            msg = gr.Textbox(label="Your Message")
            msg.submit(chat_interface, [msg, chat_history], [chat_history])
            clear = gr.Button("Clear Chat")
            clear.click(lambda: None, None, chat_history, queue=False)

        with gr.Column():
            gr.Markdown("### KV Cache Controls")
            replicate_btn = gr.Button("Replicate Critical Nodes to Host")
            simulate_failure_btn = gr.Button("Simulate GPU Failure & Restore")
            status_output = gr.Textbox(label="Cache Status", interactive=False)

            replicate_btn.click(replicate_action, inputs=None, outputs=status_output)
            simulate_failure_btn.click(simulate_failure_action, inputs=None, outputs=status_output)

    demo.launch()
