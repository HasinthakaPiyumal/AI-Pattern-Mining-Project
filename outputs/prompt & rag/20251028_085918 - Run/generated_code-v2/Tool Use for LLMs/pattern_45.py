from concurrent.futures import ThreadPoolExecutor
from transformers import pipeline
import re

class ContentEnrichmentPlatform:
    def __init__(self):
        self.summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        self.sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    def _summarize_content(self, text):
        summary = self.summarizer(text, max_length=130, min_length=30, do_sample=False)
        return summary[0]["summary_text"]

    def _extract_keywords(self, text):
        words = re.findall(r'\b\w+\b', text.lower())
        stop_words = set(["the", "and", "a", "is", "in", "it", "of", "for", "to", "with", "on", "as", "by", "at", "from", "an", "was", "are", "that", "this", "will", "be", "have", "has", "would", "could", "should", "can", "you", "your", "we", "our", "they", "their", "he", "she", "it", "him", "her", "us", "me", "my", "his", "her", "its", "them", "their", "what", "where", "when", "why", "how", "who", "which", "whom", "whose", "am", "i", "not", "but", "or", "so", "if", "than", "then", "up", "down", "out", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "only", "own", "same", "so", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"])
        filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
        word_counts = {}
        for word in filtered_words:
            word_counts[word] = word_counts.get(word, 0) + 1
        sorted_keywords = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)
        return [word for word, count in sorted_keywords[:5]]

    def _suggest_images(self, keywords):
        return [f"image_of_{kw}" for kw in keywords[:2]] + ["generic_stock_photo"]

    def _search_related_articles(self, text):
        return ["related_article_1_url", "related_article_2_url"]

    def _check_grammar_style(self, text):
        return {"grammar_errors": ["Potential typo: 'teh' -> 'the'"], "style_suggestions": ["Consider using shorter sentences."]}

    def _analyze_sentiment(self, text):
        sentiment = self.sentiment_analyzer(text)
        return sentiment[0]["label"]

    def enrich_content(self, article_draft):
        with ThreadPoolExecutor(max_workers=6) as executor:
            future_summary = executor.submit(self._summarize_content, article_draft)
            future_keywords = executor.submit(self._extract_keywords, article_draft)
            future_sentiment = executor.submit(self._analyze_sentiment, article_draft)

            keywords = future_keywords.result() # Get keywords result before image suggestion

            future_images = executor.submit(self._suggest_images, keywords)
            future_related_articles = executor.submit(self._search_related_articles, article_draft)
            future_grammar_style = executor.submit(self._check_grammar_style, article_draft)

            results = {
                "summary": future_summary.result(),
                "keywords": keywords,
                "image_suggestions": future_images.result(),
                "related_articles": future_related_articles.result(),
                "grammar_style_check": future_grammar_style.result(),
                "sentiment": future_sentiment.result(),
            }
        return results

if __name__ == "__main__":
    platform = ContentEnrichmentPlatform()
    sample_article = (
        "The quick brown fox jumps over the lazy dog. This is a very interesting article about ",
        "the benefits of parallel processing in AI systems. Sequential execution can be slow, ",
        "but parallel approaches dramatically improve efficiency. Machine learning models ",
        "often benefit from concurrent data processing. We should also consider the style and ",
        "grammar of the document. Teh document contains several valuable insights."
    )
    sample_article_text = " ".join(sample_article)

    print("\n--- Enriching Sample Article ---")
    enriched_data = platform.enrich_content(sample_article_text)
    for key, value in enriched_data.items():
        print(f"{key}: {value}")

    sample_article_negative = (
        "This product is terrible. I am extremely disappointed with its performance.",
        "It constantly crashes and the user interface is very frustrating. I would not",
        "recommend it to anyone. What a waste of money and time. Very bad experience."
    )
    sample_article_negative_text = " ".join(sample_article_negative)

    print("\n--- Enriching Negative Article ---")
    enriched_data_negative = platform.enrich_content(sample_article_negative_text)
    for key, value in enriched_data_negative.items():
        print(f"{key}: {value}")

    sample_article_positive = (
        "This new smartphone is absolutely fantastic! The camera quality is superb, ",
        "and the battery life is incredibly long. I love the sleek design and the ",
        "smooth performance. Highly recommend this device to everyone. Great job!"
    )
    sample_article_positive_text = " ".join(sample_article_positive)

    print("\n--- Enriching Positive Article ---")
    enriched_data_positive = platform.enrich_content(sample_article_positive_text)
    for key, value in enriched_data_positive.items():
        print(f"{key}: {value}")