import re
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.util import ngrams
from nltk.probability import FreqDist

import nltk
try:
    nltk.data.find("corpora/stopwords")
except nltk.downloader.DownloadError:
    nltk.download("stopwords")
try:
    nltk.data.find("tokenizers/punkt")
except nltk.downloader.DownloadError:
    nltk.download("punkt")

class CorpusLoader:
    def load_corpus(self, data_source):
        if isinstance(data_source, list):
            return data_source
        elif isinstance(data_source, str) and data_source.endswith(".txt"):
            with open(data_source, "r", encoding="utf-8") as f:
                return f.readlines()
        else:
            raise ValueError("Unsupported data source. Provide a list of strings or a .txt file path.")

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words("english"))

    def preprocess(self, text):
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", "", text)  # Remove punctuation and special characters
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in self.stop_words]
        return " ".join(tokens)

class NgramExtractor:
    def extract_ngrams(self, text, n=2, top_k=100):
        tokens = word_tokenize(text)
        n_grams = list(ngrams(tokens, n))
        freq_dist = FreqDist(n_grams)
        return freq_dist.most_common(top_k)

class PatternDiscoverer:
    def discover_patterns(self, corpus, top_k=20):
        sentence_starters = Counter()
        sentence_enders = Counter()
        
        for doc in corpus:
            sentences = sent_tokenize(doc)
            for sentence in sentences:
                words = word_tokenize(sentence.lower())
                if len(words) > 0:
                    sentence_starters[words[0]] += 1
                if len(words) > 0:
                    sentence_enders[words[-1]] += 1
        
        return {
            "starters": [word for word, count in sentence_starters.most_common(top_k)],
            "enders": [word for word, count in sentence_enders.most_common(top_k)]
        }

class PromptTemplateGenerator:
    def generate_templates(self, ngrams, patterns, min_ngram_freq=5):
        templates = set()
        
        # Add frequent n-grams as templates
        for ngram, freq in ngrams:
            if freq >= min_ngram_freq:
                templates.add(" ".join(ngram))
        
        # Add patterns based on common starters and enders
        for starter in patterns.get("starters", []):
            templates.add(f"{starter} [USER_QUERY]")
            templates.add(f"{starter}, how can I assist you?")
        for ender in patterns.get("enders", []):
            templates.add(f"[USER_QUERY] {ender}")
            templates.add(f"I can help with [USER_QUERY] {ender}")

        return list(templates)

class ResponseTemplateMatcher:
    def match_template(self, user_query, templates):
        user_query_lower = user_query.lower()
        best_match = None
        max_overlap = 0

        for template in templates:
            template_lower = template.replace("[USER_QUERY]", "").strip().lower()
            
            # Simple keyword overlap for matching
            query_words = set(word_tokenize(user_query_lower))
            template_words = set(word_tokenize(template_lower))
            
            overlap = len(query_words.intersection(template_words))
            
            if overlap > max_overlap:
                max_overlap = overlap
                best_match = template

        return best_match if best_match else "How can I assist you with [USER_QUERY]?"

class ResponseGenerator:
    def generate_response(self, user_query, template):
        if "[USER_QUERY]" in template:
            return template.replace("[USER_QUERY]", user_query)
        else:
            return f"{template} Regarding: {user_query}"


# Main Application Flow
if __name__ == "__main__":
    # 1. Define a simulated corpus of customer support interactions
    simulated_corpus = [
        "Hello, I need help with my internet connection. It's not working.",
        "My bill seems incorrect. Can you check the charges for last month?",
        "I want to upgrade my data plan. What are the available options?",
        "My service is down, and I can't access any websites. Please help.",
        "I would like to inquire about the new promotional offers for existing customers.",
        "My account shows an overdue payment, but I paid it last week.",
        "Can you provide information on setting up a new Wi-Fi router?",
        "I need to change my billing address. How can I do that?",
        "My internet is slow. What troubleshooting steps can I take?",
        "I am looking for a better deal on my mobile plan. Do you have any recommendations?"
    ]

    print("--- Starting Chatbot Response Optimizer ---")

    # 2. Preprocess the corpus
    corpus_loader = CorpusLoader()
    raw_corpus = corpus_loader.load_corpus(simulated_corpus)
    
    text_preprocessor = TextPreprocessor()
    preprocessed_corpus = [text_preprocessor.preprocess(doc) for doc in raw_corpus]
    full_preprocessed_text = " ".join(preprocessed_corpus)

    print("\n--- Corpus Preprocessing Complete ---")
    print(f"Sample preprocessed text: {full_preprocessed_text[:200]}...")

    # 3. Instantiate NgramExtractor and PatternDiscoverer to identify frequent phrases and structural patterns
    ngram_extractor = NgramExtractor()
    # Extract bigrams and trigrams
    frequent_bigrams = ngram_extractor.extract_ngrams(full_preprocessed_text, n=2, top_k=50)
    frequent_trigrams = ngram_extractor.extract_ngrams(full_preprocessed_text, n=3, top_k=50)
    all_frequent_ngrams = frequent_bigrams + frequent_trigrams
    
    pattern_discoverer = PatternDiscoverer()
    discovered_patterns = pattern_discoverer.discover_patterns(raw_corpus, top_k=10)

    print("\n--- Prompt Mining Complete ---")
    print(f"Top 5 Ngrams: {all_frequent_ngrams[:5]}")
    print(f"Discovered Patterns (Starters): {discovered_patterns['starters'][:5]}")

    # 4. Use PromptTemplateGenerator to consolidate these findings into actionable templates
    prompt_template_generator = PromptTemplateGenerator()
    optimized_templates = prompt_template_generator.generate_templates(all_frequent_ngrams, discovered_patterns)

    print(f"\n--- Generated {len(optimized_templates)} Optimized Templates ---")
    for i, template in enumerate(optimized_templates[:10]):
        print(f"Template {i+1}: {template}")

    # 5. Demonstrate ResponseTemplateMatcher and ResponseGenerator with example user queries
    response_template_matcher = ResponseTemplateMatcher()
    response_generator = ResponseGenerator()

    print("\n--- Demonstrating Chatbot Response Optimization ---")
    example_queries = [
        "My internet is down. Can you help?",
        "I want to know about my bill.",
        "Upgrade my plan.",
        "What are the new offers?",
        "I need assistance with my Wi-Fi."
    ]

    for query in example_queries:
        print(f"\nUser Query: {query}")
        matched_template = response_template_matcher.match_template(query, optimized_templates)
        print(f"Matched Template: {matched_template}")
        generated_response = response_generator.generate_response(query, matched_template)
        print(f"Optimized Response: {generated_response}")

    print("\n--- Chatbot Response Optimizer Demo Complete ---")
