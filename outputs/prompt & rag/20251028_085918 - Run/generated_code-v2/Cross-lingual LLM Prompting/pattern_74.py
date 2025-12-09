import torch
import faiss
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import gradio as gr
import numpy as np
import json

# Mock data for demonstration
mock_inclt_examples = [
    {"source_lang": "es", "source_query": "¿Dónde está mi pedido?", "target_lang": "en", "target_response": "Where is my order?"},
    {"source_lang": "es", "source_query": "Quiero devolver un producto.", "target_lang": "en", "target_response": "I want to return a product."},
    {"source_lang": "fr", "source_query": "Comment puis-je suivre ma commande ?", "target_lang": "en", "target_response": "How can I track my order?"},
    {"source_lang": "fr", "source_query": "Quel est votre politique de remboursement ?", "target_lang": "en", "target_response": "What is your refund policy?"},
    {"source_lang": "de", "source_query": "Meine Lieferung ist verspätet.", "target_lang": "en", "target_response": "My delivery is delayed."},
    {"source_lang": "de", "source_query": "Wie kann ich den Kundendienst kontaktieren?", "target_lang": "en", "target_response": "How can I contact customer service?"},
]

mock_knowledge_base = [
    {"title": "Order Tracking", "content": "You can track your order using the tracking number provided in your shipping confirmation email. Visit our website and enter the number in the 'Track Order' section."},
    {"title": "Returns Policy", "content": "Our return policy allows for returns within 30 days of purchase. Items must be in their original condition. Please initiate a return through your account."},
    {"title": "Refund Process", "content": "Refunds are processed within 5-7 business days after the returned item is received and inspected. The refund will be issued to your original payment method."},
    {"title": "Contact Customer Service", "content": "You can reach our customer service team via live chat on our website, email at support@example.com, or by calling +1-800-123-4567."},
    {"title": "Shipping Delays", "content": "Shipping delays can occur due to various reasons. Please check your tracking information for the latest updates. If you have concerns, contact customer service."},
    {"title": "Product Warranty", "content": "Most of our products come with a one-year manufacturer's warranty. Details can be found on the product page or by contacting support."},
]

class EmbeddingModelService:
    def __init__(self, model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)

    def get_embeddings(self, texts):
        return self.model.encode(texts, convert_to_numpy=True)

class LLMService:
    def __init__(self, model_name="facebook/mbart-large-50", device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

    def generate_response(self, prompt, max_length=150):
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=max_length, num_beams=5, early_stopping=True)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

class InCLTManager:
    def __init__(self, embedding_service, inclt_examples, top_k=2):
        self.embedding_service = embedding_service
        self.inclt_examples = inclt_examples
        self.top_k = top_k
        self._build_index()

    def _build_index(self):
        source_queries = [ex["source_query"] for ex in self.inclt_examples]
        embeddings = self.embedding_service.get_embeddings(source_queries)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

    def retrieve_examples(self, query):
        query_embedding = self.embedding_service.get_embeddings([query])
        D, I = self.index.search(query_embedding, self.top_k)
        return [self.inclt_examples[i] for i in I[0]]

class KnowledgeBaseRetriever:
    def __init__(self, embedding_service, knowledge_base_articles, top_k=3):
        self.embedding_service = embedding_service
        self.knowledge_base_articles = knowledge_base_articles
        self.top_k = top_k
        self._build_index()

    def _build_index(self):
        article_contents = [article["content"] for article in self.knowledge_base_articles]
        embeddings = self.embedding_service.get_embeddings(article_contents)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

    def retrieve_articles(self, query):
        query_embedding = self.embedding_service.get_embeddings([query])
        D, I = self.index.search(query_embedding, self.top_k)
        return [self.knowledge_base_articles[i] for i in I[0]]

class PromptBuilder:
    def build_prompt(self, customer_query, inclt_examples, kb_articles):
        prompt_parts = []

        prompt_parts.append("You are a helpful multilingual customer support assistant for an e-commerce platform.")
        prompt_parts.append("Please answer the customer's question concisely and accurately.")

        if inclt_examples:
            prompt_parts.append("\nHere are some examples of similar questions and their English answers (source language-target language transfer):")
            for ex in inclt_examples:
                prompt_parts.append(f"Source ({ex['source_lang']}): {ex['source_query']}")
                prompt_parts.append(f"Target (en): {ex['target_response']}\n")

        if kb_articles:
            prompt_parts.append("\nHere is relevant information from our knowledge base that might help:")
            for article in kb_articles:
                prompt_parts.append(f"Title: {article['title']}")
                prompt_parts.append(f"Content: {article['content']}\n")

        prompt_parts.append(f"\nCustomer Query: {customer_query}")
        prompt_parts.append("\nAssistant Response:")

        return "\n".join(prompt_parts)

class ChatbotApp:
    def __init__(self):
        self.embedding_service = EmbeddingModelService()
        self.llm_service = LLMService()
        self.inclt_manager = InCLTManager(self.embedding_service, mock_inclt_examples)
        self.kb_retriever = KnowledgeBaseRetriever(self.embedding_service, mock_knowledge_base)
        self.prompt_builder = PromptBuilder()

    def process_query(self, query):
        # Step 3: InCLTManager selects relevant cross-lingual examples
        inclt_examples = self.inclt_manager.retrieve_examples(query)

        # Step 4: KnowledgeBaseRetriever fetches relevant context
        kb_articles = self.kb_retriever.retrieve_articles(query)

        # Step 5: PromptBuilder combines components into a prompt
        prompt = self.prompt_builder.build_prompt(query, inclt_examples, kb_articles)

        # Step 6 & 7: LLMService generates and returns response
        response = self.llm_service.generate_response(prompt)
        return response

    def launch_gradio_interface(self):
        iface = gr.Interface(
            fn=self.process_query,
            inputs=gr.Textbox(lines=2, placeholder="Type your query here..."),
            outputs="text",
            title="Multilingual Customer Support Chatbot (InCLT Enhanced)",
            description="Ask questions in various languages and get support based on cross-lingual examples and knowledge base."
        )
        iface.launch()

if __name__ == "__main__":
    chatbot = ChatbotApp()
    chatbot.launch_gradio_interface()