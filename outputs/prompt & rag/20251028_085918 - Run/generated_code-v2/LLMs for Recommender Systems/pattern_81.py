import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

class ProductRecommender:
    def __init__(self, products_df):
        self.products_df = products_df
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None
        self._preprocess_data()

    def _preprocess_data(self):
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.products_df['description'])

    def get_recommendations(self, product_id, num_recommendations=5):
        if product_id not in self.products_df['id'].values:
            print(f"Product with ID {product_id} not found.")
            return []

        idx = self.products_df[self.products_df['id'] == product_id].index[0]
        cosine_similarities = cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()
        related_product_indices = cosine_similarities.argsort()[:-num_recommendations-2:-1]
        
        recommendations = []
        for i in related_product_indices:
            if i != idx:
                recommendations.append(self.products_df.iloc[i].to_dict())
        return recommendations

class LLMExplainer:
    def __init__(self, model_name='gpt2'):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_explanation(self, recommended_product_details, original_product_details=None):
        prompt_parts = ["Explain why the following product is recommended:"]
        
        if original_product_details:
            prompt_parts.append(f"Original product: {original_product_details['name']} ({original_product_details['description']}).")
        
        prompt_parts.append(f"Recommended product: {recommended_product_details['name']} ({recommended_product_details['description']}).")
        prompt_parts.append("Explanation:")
        
        prompt = " ".join(prompt_parts)
        
        inputs = self.tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=512)
        
        # Generate a response. Ensure max_new_tokens is set appropriately.
        # Adding temperature for more creative explanations and a max_new_tokens limit
        output_sequences = self.model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=100,  # Limit the length of the explanation
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            do_sample=True, # Enable sampling
            temperature=0.7, # Controls randomness, higher means more random
            top_k=50, # Consider top 50 words for sampling
            top_p=0.95, # Consider words that sum up to 95% probability
            pad_token_id=self.tokenizer.eos_token_id # Use eos_token_id for padding
        )
        
        generated_text = self.tokenizer.decode(output_sequences[0], skip_special_tokens=True)
        
        # Post-process to extract only the explanation part after 