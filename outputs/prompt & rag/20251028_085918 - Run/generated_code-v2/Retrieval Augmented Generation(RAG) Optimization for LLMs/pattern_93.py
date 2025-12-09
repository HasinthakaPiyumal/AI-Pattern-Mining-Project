import random
from collections import defaultdict

knowledge_base_documents = [
    "The new 'Sonic Speed' sneakers feature a breathable mesh upper and a responsive foam sole for ultimate comfort and performance. Available in sizes 7-12.",
    "Our return policy allows for full refunds within 30 days of purchase, provided the item is in its original condition with tags attached. Electronics have a 15-day return window.",
    "Customer service can be reached via phone at 1-800-555-0199 or email at support@ecomstore.com during business hours (9 AM - 6 PM EST, Monday-Friday).",
    "The 'Everest Backpack' is made from durable, water-resistant nylon and features multiple compartments, including a padded laptop sleeve. Ideal for hiking and travel.",
    "Shipping typically takes 3-5 business days for standard delivery. Expedited shipping options are available at checkout for an additional fee.",
    "This document talks about the history of ancient Roman pottery and its cultural significance in the Mediterranean region.", 
    "Our latest smart home device, the 'Eco-Monitor', tracks energy consumption and suggests ways to reduce your carbon footprint.",
    "The 'Stellar Smartwatch' offers heart rate monitoring, GPS tracking, and notifications. Compatible with iOS and Android devices.",
    "This text discusses the advancements in quantum computing and its potential impact on cryptography.", 
    "Payment methods accepted include Visa, Mastercard, American Express, PayPal, and Google Pay. We do not accept cash on delivery.",
    "The 'SuperCharge Power Bank' provides fast charging for multiple devices and has a 20000mAh capacity. Comes with a 1-year warranty.",
    "Here is an article about the migratory patterns of monarch butterflies across North America.", 
    "Discount code 'SAVE20' gives 20% off all orders over $100. Valid until the end of the month.",
    "The 'Ergonomic Office Chair' features adjustable lumbar support, armrests, and headrest for optimal comfort during long working hours.",
    "An excerpt from a novel describing a detective's thrilling chase through the streets of Paris.", 
    "For international shipping, please allow 7-14 business days. Customs duties and taxes may apply, which are the responsibility of the customer."
]

class BM25Simulator:
    def __init__(self, corpus):
        self.corpus = corpus
        self.indexed_corpus = [doc.lower().split() for doc in corpus]

    def retrieve(self, query, top_k=3):
        query_tokens = query.lower().split()
        scores = defaultdict(float)
        for i, doc_tokens in enumerate(self.indexed_corpus):
            common_words = set(query_tokens) & set(doc_tokens)
            scores[i] = len(common_words)
        
        sorted_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [self.corpus[idx] for idx, _ in sorted_docs[:top_k]]

bm25_retriever = BM25Simulator(knowledge_base_documents)

qa_pairs = [
    {
        "question": "What is the return policy for electronics?",
        "answer": "Electronics have a 15-day return window.",
        "gold_context_id": 1 
    },
    {
        "question": "How can I contact customer service?",
        "answer": "Customer service can be reached via phone at 1-800-555-0199 or email at support@ecomstore.com.",
        "gold_context_id": 2 
    },
    {
        "question": "What are the features of the Everest Backpack?",
        "answer": "The 'Everest Backpack' is made from durable, water-resistant nylon and features multiple compartments, including a padded laptop sleeve.",
        "gold_context_id": 3 
    },
    {
        "question": "Which payment methods are accepted?",
        "answer": "Payment methods accepted include Visa, Mastercard, American Express, PayPal, and Google Pay.",
        "gold_context_id": 9 
    },
    {
        "question": "Tell me about the Sonic Speed sneakers.",
        "answer": "The new 'Sonic Speed' sneakers feature a breathable mesh upper and a responsive foam sole for ultimate comfort and performance.",
        "gold_context_id": 0 
    }
]

instruction_tuning_data = []

for qa in qa_pairs:
    gold_context = knowledge_base_documents[qa["gold_context_id"]]
    
    retrieved_contexts = bm25_retriever.retrieve(qa["question"], top_k=5)
    
    if gold_context not in retrieved_contexts:
        retrieved_contexts.insert(0, gold_context)
    
    hard_negative_candidates = [
        c for c in retrieved_contexts if c != gold_context and ("ancient Roman pottery" in c or "quantum computing" in c or "monarch butterflies" in c or "detective's thrilling chase" in c)
    ]
    
    num_hard_negatives = min(2, len(hard_negative_candidates))
    selected_hard_negatives = random.sample(hard_negative_candidates, num_hard_negatives) if hard_negative_candidates else []

    final_contexts_for_training = [gold_context] + selected_hard_negatives
    random.shuffle(final_contexts_for_training) 
    
    final_contexts_for_training = list(dict.fromkeys(final_contexts_for_training)) 

    combined_context = " ".join(final_contexts_for_training)
    
    instruction_tuning_data.append({
        "instruction": f"Given the following context: {combined_context}\nAnswer the question: {qa['question']}",
        "response": qa["answer"]
    })

class DummyLLM:
    def __init__(self, model_name="dummy_llm"):
        self.model_name = model_name
        self.knowledge = {}

    def fine_tune(self, training_data):
        for item in training_data:
            self.knowledge[item["instruction"]] = item["response"]
        return "Dummy LLM fine-tuning complete, learned from provided instructions."

    def generate(self, prompt, retrieved_context=""):
        
        for qa in qa_pairs:
            if qa["question"] in prompt and qa["answer"] in retrieved_context:
                return qa["answer"]
        
        if prompt in self.knowledge:
            return self.knowledge[prompt]
        
        return "I'm sorry, I don't have enough information to answer that based on the provided context."

class RAGSystem:
    def __init__(self, llm, retriever, knowledge_base):
        self.llm = llm
        self.retriever = retriever
        self.knowledge_base = knowledge_base

    def answer_query(self, query):
        retrieved_contexts = self.retriever.retrieve(query, top_k=3)
        
        combined_context = " ".join(retrieved_contexts)
        
        prompt = f"Given the following context: {combined_context}\nAnswer the question: {query}"
        
        answer = self.llm.generate(prompt, retrieved_context=combined_context)
        return answer


my_llm = DummyLLM()

print(my_llm.fine_tune(instruction_tuning_data))

rag_system = RAGSystem(my_llm, bm25_retriever, knowledge_base_documents)

query1 = "How do I return electronics?"
print(f"\nQuery: {query1}")
response1 = rag_system.answer_query(query1)
print(f"Answer: {response1}")

query2 = "What are the specifications of the Sonic Speed sneakers?"
print(f"\nQuery: {query2}")
response2 = rag_system.answer_query(query2)
print(f"Answer: {response2}")

query3 = "Tell me about the payment options."
print(f"\nQuery: {query3}")
response3 = rag_system.answer_query(query3)
print(f"Answer: {response3}")

query4 = "What is the material of the Everest Backpack?"
print(f"\nQuery: {query4}")
response4 = rag_system.answer_query(query4)
print(f"Answer: {response4}")