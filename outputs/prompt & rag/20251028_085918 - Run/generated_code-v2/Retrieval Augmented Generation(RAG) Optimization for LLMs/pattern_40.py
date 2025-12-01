import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class MockKnowledgeBase:
    def __init__(self, articles):
        self.articles = articles

    def retrieve_candidates(self, query: str, k: int = 5) -> list:
        """
        Simulates initial document retrieval based on simple keyword matching.
        In a real system, this would be a more sophisticated retriever (e.g., BM25, vector search).
        """
        query_words = set(query.lower().split())
        scored_articles = []
        for i, article in enumerate(self.articles):
            article_words = set(article.lower().split())
            overlap = len(query_words.intersection(article_words))
            scored_articles.append((overlap, article, i)) # Store index to maintain original order if needed

        # Sort by overlap in descending order and take top k
        scored_articles.sort(key=lambda x: x[0], reverse=True)
        return [article[1] for article in scored_articles[:k]]

class ZeroShotReranker:
    def __init__(self, model_name="gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        # Ensure the tokenizer has a pad token for batch processing, if not, set it.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval() # Set model to evaluation mode

    def _calculate_nll(self, text: str) -> float:
        """Calculates the negative log-likelihood (NLL) of a given text.
           A lower NLL indicates higher probability/coherence according to the LM.
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=self.tokenizer.model_max_length)
        input_ids = inputs["input_ids"]
        labels = input_ids.clone() # For causal LMs, labels are typically the input_ids themselves for perplexity

        with torch.no_grad():
            outputs = self.model(input_ids, labels=labels)
            loss = outputs.loss
        return loss.item()

    def rerank(self, query: str, candidate_documents: list) -> str:
        """
        Reranks candidate documents by evaluating the NLL of the combined document and query.
        The document yielding the lowest NLL (highest coherence/probability) is selected.
        """
        if not candidate_documents:
            return ""

        best_document = ""
        lowest_nll = float("inf")

        for doc in candidate_documents:
            # Concatenate document and query for NLL calculation.
            # The LM will assess how 'probable' this combined sequence is.
            # A lower NLL suggests better coherence between the document and the query.
            combined_text = f"{doc}\n\nUser query: {query}"
            current_nll = self._calculate_nll(combined_text)

            if current_nll < lowest_nll:
                lowest_nll = current_nll
                best_document = doc
        return best_document

class CustomerSupportChatbot:
    def __init__(self, knowledge_base_articles: list, reranker_model_name="gpt2", generator_model_name="gpt2"):
        self.knowledge_base = MockKnowledgeBase(knowledge_base_articles)
        self.reranker = ZeroShotReranker(reranker_model_name)
        self.generator_tokenizer = AutoTokenizer.from_pretrained(generator_model_name)
        self.generator_model = AutoModelForCausalLM.from_pretrained(generator_model_name)
        if self.generator_tokenizer.pad_token is None:
            self.generator_tokenizer.pad_token = self.generator_tokenizer.eos_token


    def answer_query(self, query: str) -> str:
        # 1. Initial Retrieval
        candidate_documents = self.knowledge_base.retrieve_candidates(query, k=5)
        if not candidate_documents:
            return "I apologize, but I couldn't find any relevant articles in our knowledge base."

        # 2. Zero-Shot Reranking
        best_document = self.reranker.rerank(query, candidate_documents)

        if not best_document:
            return "I apologize, but even after reranking, I couldn't find a sufficiently relevant article."

        # 3. Generate Answer using the best document and query
        prompt = (
            f"Based on the following article, please provide a concise answer to the user's question.\n\n"
            f"Article: {best_document}\n\n"
            f"User Question: {query}\n\n"
            f"Answer:"
        )
        inputs = self.generator_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.generator_tokenizer.model_max_length)
        
        # Determine the maximum generation length relative to the prompt length
        max_output_length = min(inputs["input_ids"].shape[1] + 100, self.generator_tokenizer.model_max_length) # Generate up to 100 new tokens

        output_sequences = self.generator_model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_output_length,
            temperature=0.7,
            num_return_sequences=1,
            pad_token_id=self.generator_tokenizer.eos_token_id # Important for generation
        )
        generated_text = self.generator_tokenizer.decode(output_sequences[0], skip_special_tokens=True)

        # Extract only the generated answer part
        try:
            answer_start_index = generated_text.lower().rfind("answer:")
            if answer_start_index != -1:
                return generated_text[answer_start_index + len("answer:"):].strip()
            else:
                return generated_text.strip() # Fallback if "Answer:" not found
        except Exception:
            return generated_text.strip()


# --- Example Usage (Not part of the generated tool output, but for understanding) ---
# KNOWLEDGE_BASE_ARTICLES = [
#     "Our shipping policy states that standard shipping takes 5-7 business days. Expedited shipping is available for an extra fee and delivers in 2-3 business days. International shipping times vary.",
#     "To reset your password, go to the login page and click 'Forgot Password'. Follow the instructions sent to your registered email address. Make sure to check your spam folder.",
#     "Our return policy allows returns within 30 days of purchase, provided the item is unused and in its original packaging. Refunds are processed within 7-10 business days after we receive the returned item.",
#     "You can track your order by logging into your account and visiting the 'Order History' section. Click on the specific order to see its current status and tracking number.",
#     "For technical support, please visit our help center or open a ticket. Our support team is available Monday-Friday, 9 AM to 5 PM EST.",
#     "This article talks about the history of our company and its mission to provide quality products to customers worldwide.",
#     "Information about upcoming sales and promotions will be announced via our newsletter and social media channels. Subscribe to stay updated!",
#     "Our loyalty program offers points for every purchase, which can be redeemed for discounts on future orders. Join today to start earning rewards.",
#     "We accept major credit cards (Visa, Mastercard, Amex) and PayPal. We do not accept bank transfers or checks for online orders.",
#     "How to configure your device settings. Connect to Wi-Fi, adjust brightness, change language, and manage notifications.",
#     "Troubleshooting common device issues. Restarting the device, checking cable connections, and updating software often resolve problems."
# ]

# # Initialize the chatbot
# # Note: Using 'gpt2' for both reranking and generation for simplicity.
# # In a real application, you might use a smaller, faster model for reranking
# # and a more capable one for generation.
# # Loading these models can take time and memory.
# # For local execution, ensure you have enough RAM and a good internet connection for initial download.
# # You might want to use a smaller model if memory is an issue (e.g., "distilgpt2").
# chatbot = CustomerSupportChatbot(KNOWLEDGE_BASE_ARTICLES)

# # Example queries
# print(f"User: How do I reset my password?")
# print(f"Chatbot: {chatbot.answer_query('How do I reset my password?')}\n")

# print(f"User: What is your shipping policy?")
# print(f"Chatbot: {chatbot.answer_query('What is your shipping policy?')}\n")

# print(f"User: I need help with my device settings.")
# print(f"Chatbot: {chatbot.answer_query('I need help with my device settings.')}\n")

# print(f"User: Tell me about the company history.")
# print(f"Chatbot: {chatbot.answer_query('Tell me about the company history.')}\n")

# print(f"User: How can I return an item?")
# print(f"Chatbot: {chatbot.answer_query('How can I return an item?')}\n")
