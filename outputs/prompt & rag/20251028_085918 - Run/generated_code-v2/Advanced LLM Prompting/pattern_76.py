import pandas as pd
import re
import nltk
from collections import Counter

try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')

class PromptOptimizer:
    def __init__(self):
        self.stopwords = set(nltk.corpus.stopwords.words('english'))

    def load_historical_data(self, filepath):
        return pd.read_csv(filepath)

    def identify_successful_interactions(self, data, success_column='successful', text_column='text'):
        if success_column in data.columns:
            return data[data[success_column] == True][text_column].tolist()
        else:
            # If no success column, assume all are 'successful' for demonstration or handle as per specific logic
            print("Warning: 'successful' column not found. Assuming all interactions are relevant for prompt mining.")
            return data[text_column].tolist()

    def preprocess_text(self, text):
        text = text.lower()  # Lowercasing
        text = re.sub(r'[^a-z0-9\s]', '', text)  # Remove punctuation and special characters
        tokens = nltk.word_tokenize(text)  # Tokenization
        tokens = [word for word in tokens if word.isalpha() and word not in self.stopwords]  # Remove stopwords and non-alphabetic tokens
        return tokens

    def find_ngrams(self, tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

    def mine_prompts(self, successful_interactions_text, n_gram_range=(2, 4), top_k=10):
        all_ngrams = Counter()
        for text in successful_interactions_text:
            preprocessed_tokens = self.preprocess_text(text)
            for n in range(n_gram_range[0], n_gram_range[1] + 1):
                ngrams = self.find_ngrams(preprocessed_tokens, n)
                all_ngrams.update(ngrams)
        
        # Convert n-grams back to strings for better readability as templates
        top_prompts = []
        for ngram, count in all_ngrams.most_common(top_k):
            top_prompts.append({"template": " ".join(ngram), "frequency": count})
        
        return top_prompts

    def run_optimizer(self, filepath, success_column='successful', text_column='text', n_gram_range=(2, 4), top_k=10):
        print("Loading historical data...")
        data = self.load_historical_data(filepath)
        
        print("Identifying successful interactions...")
        successful_texts = self.identify_successful_interactions(data, success_column, text_column)
        
        if not successful_texts:
            print("No successful interactions found or identified. Cannot mine prompts.")
            return []

        print(f"Mining prompts from {len(successful_texts)} successful interactions...")
        mined_templates = self.mine_prompts(successful_texts, n_gram_range, top_k)
        
        print("Prompt mining complete.")
        return mined_templates

if __name__ == "__main__":
    # Example Usage:
    # Create a dummy CSV file for demonstration
    dummy_data = {
        'text': [
            "I need help with my billing statement, it seems incorrect.",
            "My internet is not working, how can I fix it?",
            "Can you tell me about the latest promotions for existing customers?",
            "I want to upgrade my plan, what are the options?",
            "My bill shows an extra charge, please explain this charge.",
            "The network is down, please provide an update.",
            "I'd like to change my address, what is the procedure?",
            "My service is slow, how to improve my internet speed?",
            "Explain the new data plan features, please.",
            "I have a question about my recent payment."
        ],
        'successful': [
            True, True, True, True, True,
            False, False, False, False, True
        ]
    }
    dummy_df = pd.DataFrame(dummy_data)
    dummy_filepath = "customer_support_logs.csv"
    dummy_df.to_csv(dummy_filepath, index=False)

    optimizer = PromptOptimizer()
    mined_prompts = optimizer.run_optimizer(dummy_filepath, n_gram_range=(2, 3), top_k=5)

    if mined_prompts:
        print("\nTop Mined Prompt Templates:")
        for prompt in mined_prompts:
            print(f"  Template: '{prompt['template']}' (Frequency: {prompt['frequency']})")
    else:
        print("No prompt templates were mined.")
