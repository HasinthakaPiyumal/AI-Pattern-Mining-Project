import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import logging
import re
import os

# --- Configuration ---
class Config:
    # Replace 'YOUR_NEWS_API_KEY' with an actual key from a news API (e.g., NewsAPI.org)
    # or ensure your environment variable NEWS_API_KEY is set.
    # For this demonstration, the NewsFetcher.search_news method is mocked.
    NEWS_API_KEY = os.getenv("NEWS_API_KEY", "YOUR_NEWS_API_KEY")
    BLACKLISTED_DOMAINS = ["malicious-site.com", "phishing-link.net", "example-bad-site.org"]
    LOG_FILE = "security_log.txt"

# --- Security Monitor ---
class SecurityMonitor:
    def __init__(self, log_file):
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def check_url(self, url):
        """Checks if a URL is blacklisted to prevent access to unsafe sites."""
        for domain in Config.BLACKLISTED_DOMAINS:
            if domain in url:
                self.log_incident("BLACKLISTED_URL_ACCESS", f"Attempted access to blacklisted domain: {url}")
                return False
        return True

    def check_query_for_malice(self, query):
        """Basic check for potentially malicious keywords in an AI-generated query (tripwire test)."""
        malicious_keywords = ["exploit", "sql injection", "script", "delete database", "rm -rf"]
        for keyword in malicious_keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', query, re.IGNORECASE):
                self.log_incident("MALICIOUS_QUERY_ATTEMPT", f"Potentially malicious query detected: '{query}'")
                return False
        return True

    def log_incident(self, incident_type, details):
        """Logs a security incident and prints an alert."""
        self.logger.warning(f"SECURITY INCIDENT [{incident_type}]: {details}")
        print(f"!!! SECURITY ALERT !!! {incident_type}: {details}")  # Print to console for immediate visibility

# --- News Fetcher (Controlled Web Access Layer) ---
class NewsFetcher:
    def __init__(self, security_monitor: SecurityMonitor):
        self.security_monitor = security_monitor
        self.headers = {
            "User-Agent": "SecureAINewsSummarizer/1.0 (https://github.com/your-repo-link)"
        }

    def _make_request(self, url):
        """Handles HTTP GET requests safely, checking against blacklisted URLs."""
        if not self.security_monitor.check_url(url):
            return None
        try:
            print(f"[NewsFetcher] Making request to: {url}")
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
            return response.text
        except requests.exceptions.RequestException as e:
            self.security_monitor.log_incident("HTTP_REQUEST_ERROR", f"Failed to fetch {url}: {e}")
            return None

    def _extract_text(self, html_content):
        """Extracts readable text from HTML, stripping out scripts, styles, and other non-content elements.
        This prevents the execution of dynamic content."""
        if not html_content:
            return ""
        soup = BeautifulSoup(html_content, 'html.parser')
        # Remove script and style elements to prevent execution of dynamic content
        for script_or_style in soup(["script", "style", "header", "footer", "nav"]):
            script_or_style.decompose()
        # Get text, preserving paragraph breaks
        text = soup.get_text(separator=' ', strip=True)
        return text

    def search_news(self, query, language='en'):
        """Simulates searching for news articles. In a real application, this would integrate
        with a trusted news search API (e.g., NewsAPI.org, Google News API).
        It includes a query malicousness check."""
        if not self.security_monitor.check_query_for_malice(query):
            return [] # Do not proceed with potentially malicious queries

        print(f"[NewsFetcher] Searching for news related to: '{query}'")
        # --- MOCK API CALL (for demonstration) ---
        # Replace this section with actual integration with a news API.
        # Example using NewsAPI.org (requires an API key and proper error handling):
        # news_api_url = f"https://newsapi.org/v2/everything?q={query}&language={language}&sortBy=relevancy&apiKey={Config.NEWS_API_KEY}"
        # try:
        #     response_data = requests.get(news_api_url, headers=self.headers, timeout=10).json()
        #     articles = [{
        #         "title": art['title'], 
        #         "url": art['url'], 
        #         "description": art.get('description', '')
        #     } for art in response_data.get('articles', [])[:5]]
        # except requests.exceptions.RequestException as e:
        #     self.security_monitor.log_incident("NEWS_API_ERROR", f"Failed to query NewsAPI for '{query}': {e}")
        #     articles = []
        # return articles

        # For this demonstration, returning dummy articles with a valid Wikipedia link for content fetching
        dummy_articles = [
            {"title": f"Breaking News: {query.title()} Latest", "url": "https://en.wikipedia.org/wiki/Artificial_intelligence" if "AI" in query.upper() else "https://en.wikipedia.org/wiki/Quantum_computing"},
            {"title": f"Analysis of {query.title()} Impact", "url": "https://en.wikipedia.org/wiki/Machine_learning" if "AI" in query.upper() else "https://en.wikipedia.org/wiki/Space_exploration"},
            {"title": f"Global Trends in {query.title()}", "url": "https://en.wikipedia.org/wiki/News_aggregation"},
        ]
        print(f"[NewsFetcher] Found {len(dummy_articles)} mock articles.")
        return dummy_articles[:3]  # Limit for demo

    def fetch_article_content(self, article_url):
        """Fetches and extracts clean text content of a single article from a URL."""
        print(f"[NewsFetcher] Attempting to fetch content from: {article_url}")
        html_content = self._make_request(article_url)
        if html_content:
            return self._extract_text(html_content)
        return None

# --- AI Summarizer ---
class Summarizer:
    def __init__(self):
        try:
            # Using 'sshleifer/distilbart-cnn-12-6' for summarization
            # This model is a smaller, faster alternative to full BART, suitable for demonstration.
            # Requires 'transformers' and 'torch' to be installed.
            self.summarizer_pipeline = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
            print("[Summarizer] Summarization model loaded successfully.")
        except Exception as e:
            print(f"[Summarizer] Failed to load summarization model: {e}. Please ensure 'transformers' and 'torch' are installed.")
            print("[Summarizer] Running without a summarization model. Summaries will be mocked.")
            self.summarizer_pipeline = None  # Indicate that the model is not loaded

    def summarize(self, text, max_length=150, min_length=50):
        """Generates a concise summary of the provided text."""
        if not self.summarizer_pipeline:
            return f"MOCKED SUMMARY: This is a summary of the article content for demonstration purposes. Original text length: {len(text)} characters."
        if not text or len(text.strip()) < min_length: # Basic check for meaningful text
            return "No sufficient content to summarize."
        try:
            # Truncate text if too long for the model's context window (e.g., 1024 tokens for BART variants)
            # A very rough estimate for token count is text_length / 4. Max input for distilbart is 1024 tokens.
            max_model_input_chars = 1024 * 3  # Approximate character limit to stay within token limit
            if len(text) > max_model_input_chars:
                print(f"[Summarizer] Warning: Article text is very long ({len(text)} chars). Truncating for summarization.")
                text = text[:max_model_input_chars]

            summary = self.summarizer_pipeline(
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False  # For more deterministic output
            )[0]['summary_text']
            return summary
        except Exception as e:
            print(f"[Summarizer] Error during summarization: {e}")
            return f"Failed to generate summary due to an error. Original text length: {len(text)} characters."

# --- Main Application Logic ---
class SecureAINewsSummarizer:
    def __init__(self):
        self.security_monitor = SecurityMonitor(Config.LOG_FILE)
        self.news_fetcher = NewsFetcher(self.security_monitor)
        self.summarizer = Summarizer()
        self.user_preferences = {
            "categories": [],  # e.g., "technology", "science"
            "keywords": []     # e.g., "AI", "machine learning"
        }
        print("\n--- Secure AI News Summarizer Initialized ---")

    def update_preferences(self, categories=None, keywords=None):
        """Updates user preferences for news filtering (though not fully implemented in mock search)."""
        if categories is not None:
            self.user_preferences["categories"] = categories
        if keywords is not None:
            self.user_preferences["keywords"] = keywords
        print(f"[App] User preferences updated: {self.user_preferences}")

    def get_personalized_summary(self, query: str):
        """Fetches relevant news articles, performs security checks, and provides summarized content."""
        print(f"\n--- Requesting personalized summary for query: '{query}' ---")

        # Step 1: Search for news articles with query validation
        articles_meta = self.news_fetcher.search_news(query)
        if not articles_meta:
            print("[App] No articles found or search failed for query.")
            return "No relevant news articles could be retrieved securely."

        summaries_output = []
        for article_index, article in enumerate(articles_meta):
            url = article.get('url')
            title = article.get('title', f'Untitled Article {article_index + 1}')

            if not url:
                summaries_output.append(f"Article {article_index + 1}: {title}\nURL: N/A\nSummary: No URL provided for this article.\n---")
                continue

            # Step 2: Fetch article content with security checks (controlled interaction)
            article_content = self.news_fetcher.fetch_article_content(url)

            if article_content:
                # Step 3: Summarize the content using the AI model
                summary_text = self.summarizer.summarize(article_content)
                summaries_output.append(f"Title: {title}\nURL: {url}\nSummary: {summary_text}\n---")
            else:
                summaries_output.append(f"Title: {title}\nURL: {url}\nSummary: Content could not be fetched securely or was empty.\n---")

        return "\n".join(summaries_output)

# --- Example Usage ---
if __name__ == "__main__":
    # Install necessary libraries: pip install requests beautifulsoup4 transformers torch
    # For better performance, consider installing: pip install accelerate

    app = SecureAINewsSummarizer()

    # Example 1: Successful news summary for 