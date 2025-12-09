import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from collections import Counter
import re

# Ensure NLTK resources are downloaded (only needed once)
try:
    nltk.data.find('punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')
try:
    nltk.data.find('stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')

class PromptMiner:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text):
        """
        Cleans and tokenizes the input text.
        - Converts to lowercase.
        - Removes punctuation.
        - Removes numbers.
        - Removes stopwords.
        """
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)  # Remove non-alphabetic characters
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in self.stop_words and len(word) > 1] # Remove stopwords and single char tokens
        return tokens

    def mine_prompts(self, interactions, n_gram_range=(1, 3), num_suggestions=5):
        """
        Mines common n-gram phrases from a list of customer interactions.
        
        Args:
            interactions (list of str): A list of successful customer interaction texts.
            n_gram_range (tuple): A tuple (min_n, max_n) for n-gram generation.
            num_suggestions (int): The number of top common prompts to return.

        Returns:
            list of str: A list of the most common prompt structures/n-grams.
        """
        all_ngrams = []
        for interaction in interactions:
            tokens = self.preprocess_text(interaction)
            for n in range(n_gram_range[0], n_gram_range[1] + 1):
                all_ngrams.extend([' '.join(gram) for gram in ngrams(tokens, n)])

        # Count frequencies of n-grams
        ngram_counts = Counter(all_ngrams)

        # Get the most common n-grams
        most_common_prompts = [prompt for prompt, _ in ngram_counts.most_common(num_suggestions)]
        return most_common_prompts

    def suggest_prompt(self, base_query, mined_prompts):
        """
        Suggests an enhanced prompt based on a base query and mined prompts.
        This is a simple concatenation for demonstration.
        """
        if mined_prompts:
            # For simplicity, just append the most relevant (or first) mined prompt
            # In a real system, this would involve more sophisticated matching
            return f"{base_query} {mined_prompts[0]}"
        return base_query
