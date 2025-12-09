import time
from collections import OrderedDict

KNOWLEDGE_BASE = {
    "doc_1": "The return policy states items can be returned within 30 days of purchase, provided they are in their original condition and packaging. Electronics have a 15-day return window. Customized items are non-refundable.",
    "doc_2": "Our new smartphone model, the 'Aether X', features a 108MP camera, 12GB RAM, and a 5000mAh battery. It supports 5G connectivity and comes in Midnight Black and Aurora Silver.",
    "doc_3": "To troubleshoot Wi-Fi connection issues, first restart your router and device. Ensure your device's Wi-Fi is enabled and you're entering the correct password. If problems persist, contact our technical support.",
    "doc_4": "Shipping usually takes 3-5 business days for standard delivery. Express shipping options are available for an additional cost, delivering within 1-2 business days. International shipping times vary.",
    "doc_5": "To initiate a return, log into your account, go to 'Order History', select the item you wish to return, and follow the on-screen instructions to generate a return label. Pack the item securely with the label attached."
}

def simulate_embedding_lookup(query):
    query_lower = query.lower()
    if "return policy" in query_lower or "returns" in query_lower:
        return "doc_1"
    if "smartphone" in query_lower or "camera" in query_lower or "specs" in query_lower:
        return "doc_2"
    if "wifi" in query_lower or "troubleshoot" in query_lower:
        return "doc_3"
    if "shipping" in query_lower or "delivery" in query_lower:
        return "doc_4"
    if "how to return" in query_lower or "return item" in query_lower:
        return "doc_5"
    return "doc_1"

class RAGCache:
    def __init__(self, max_size=5):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.frequency = {}

    def _evict(self):
        if not self.cache:
            return

        max_eviction_priority = -1.0
        key_to_evict = None

        current_time = time.time()

        for key, (_, freq, last_accessed) in self.cache.items():
            time_since_last_access = current_time - last_accessed
            eviction_priority = time_since_last_access / (self.frequency.get(key, 0) + 1)

            if eviction_priority > max_eviction_priority:
                max_eviction_priority = eviction_priority
                key_to_evict = key

        if key_to_evict:
            del self.cache[key_to_evict]
            del self.frequency[key_to_evict]

    def get(self, doc_id, prefix_length=0):
        key = (doc_id, prefix_length)
        if key in self.cache:
            kv_tensors, freq, _ = self.cache[key]
            self.frequency[key] = freq + 1
            self.cache[key] = (kv_tensors, self.frequency[key], time.time())
            self.cache.move_to_end(key)
            return kv_tensors
        return None

    def put(self, doc_id, prefix_length, kv_tensors):
        key = (doc_id, prefix_length)
        if key in self.cache:
            freq = self.frequency.get(key, 0) + 1
            self.frequency[key] = freq
            self.cache[key] = (kv_tensors, freq, time.time())
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.max_size:
                self._evict()
            freq = self.frequency.get(key, 0) + 1
            self.frequency[key] = freq
            self.cache[key] = (kv_tensors, freq, time.time())
            self.cache.move_to_end(key)

def simulate_llm_inference(text_input, kv_tensors_input=None, full_inference=True):
    response = "Based on the information: "
    simulated_kv_tensors = f"KV_TENSORS_FOR_{text_input[:20].replace(' ', '_').upper()}"

    if kv_tensors_input:
        response += f" (cached knowledge used from {kv_tensors_input})."
        simulated_kv_tensors = kv_tensors_input + "_EXTENDED"
        inference_time = 0.05
    else:
        response += " (full generation performed)."
        inference_time = 0.2

    time.sleep(inference_time)

    if "return policy" in text_input.lower():
        response += "Most returns within 30 days, electronics within 15 days. Items must be in original condition."
    elif "smartphone" in text_input.lower():
        response += "The Aether X smartphone boasts a 108MP camera, 12GB RAM, and 5000mAh battery."
    elif "wifi" in text_input.lower():
        response += "To troubleshoot Wi-Fi, restart your router and device, and check your password."
    elif "shipping" in text_input.lower():
        response += "Standard shipping takes 3-5 business days. Express options are faster."
    elif "how to return" in text_input.lower():
        response += "Initiate a return through your account's 'Order History' to get a label."
    else:
        response += "I need more details to provide a precise answer."

    return response, simulated_kv_tensors

class SmartCustomerSupportAssistant:
    def __init__(self, cache_max_size=5):
        self.rag_cache = RAGCache(max_size=cache_max_size)

    def answer_query(self, query):
        retrieved_doc_id = simulate_embedding_lookup(query)
        retrieved_doc_content = KNOWLEDGE_BASE.get(retrieved_doc_id, "No relevant document found.")

        cached_kv_tensors = self.rag_cache.get(retrieved_doc_id)

        if cached_kv_tensors:
            answer, new_kv_tensors = simulate_llm_inference(
                retrieved_doc_content,
                kv_tensors_input=cached_kv_tensors,
                full_inference=False
            )
            self.rag_cache.put(retrieved_doc_id, 0, new_kv_tensors)
        else:
            answer, new_kv_tensors = simulate_llm_inference(
                retrieved_doc_content,
                full_inference=True
            )
            self.rag_cache.put(retrieved_doc_id, 0, new_kv_tensors)

        return answer

if __name__ == "__main__":
    assistant = SmartCustomerSupportAssistant(cache_max_size=3)
    queries = [
        "What is your return policy?",
        "What are the specs of the Aether X smartphone?",
        "My Wi-Fi isn't working, how do I fix it?",
        "How long does standard shipping take?",
        "I want to return an item, what should I do?",
        "What is the return policy for electronics?",
        "Tell me more about the Aether X's camera.",
        "How do I start a return?",
        "What's the process for getting a refund?",
        "How fast is express delivery?"
    ]

    for q in queries:
        print(f"\nCustomer Query: \"{q}\"")
        response = assistant.answer_query(q)
        print(f"Assistant: {response}")
