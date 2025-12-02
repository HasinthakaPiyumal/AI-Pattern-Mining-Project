import requests
from bs4 import BeautifulSoup
import re
import logging
from transformers import pipeline
import sys

# Configuration
WHITELISTED_DOMAINS = [
    "reuters.com", "www.reuters.com",
    "bloomberg.com", "www.bloomberg.com",
    "wsj.com", "www.wsj.com",
    "cnn.com", "www.cnn.com"
]

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MonitoringAndSecurityService:
    def __init__(self):
        self.web_access_suspended = False
        self.logger = logging.getLogger('MonitoringAndSecurityService')

    def log_activity(self, event_type, message, details=None):
        if details is None:
            details = {}
        self.logger.info(f"Event Type: {event_type} | Message: {message} | Details: {details}")

    def check_tripwire(self, action_type, data):
        # Rule-based checks for disallowed actions
        if self.web_access_suspended:
            self._alert(f"Web access is suspended. Detected attempted action: {action_type}")
            return True

        if action_type == "URL_ACCESS":
            url = data
            if not any(domain in url for domain in WHITELISTED_DOMAINS):
                self._alert(f"Tripwire Triggered: Attempted to access non-whitelisted URL: {url}")
                self.suspend_web_access()
                return True
            # More sophisticated checks for blacklisted keywords in URL
            if re.search(r"(login|admin|form_submit|download_exe)", url, re.IGNORECASE):
                self._alert(f"Tripwire Triggered: URL contains suspicious keywords: {url}")
                self.suspend_web_access()
                return True
        elif action_type == "CONTENT_ANALYSIS":
            content = data
            # Regex to detect forms, input fields, scripts, or interactive elements
            if re.search(r"<form|<input|<script|<button|javascript:", content, re.IGNORECASE):
                self._alert("Tripwire Triggered: Detected interactive/form elements in fetched content.")
                return True # Don't suspend immediately, but flag content as suspicious
        elif action_type == "AI_OUTPUT_CHECK":
            ai_output = data
            # Check if AI output tries to instruct disallowed web actions
            if re.search(r"(click button|fill form|submit data|login to)", ai_output, re.IGNORECASE):
                self._alert(f"Tripwire Triggered: AI output suggests disallowed action: {ai_output}")
                return True

        return False

    def _alert(self, message):
        self.logger.critical(message)
        print(f"\n!!! SECURITY ALERT !!! {message}\n")

    def suspend_web_access(self):
        self.web_access_suspended = True
        self.logger.critical("Web access has been SUSPENDED due to tripwire trigger.")

class WebAccessManager:
    def __init__(self, monitor_service):
        self.monitor_service = monitor_service

    def _is_whitelisted(self, url):
        return not self.monitor_service.check_tripwire("URL_ACCESS", url)

    def _clean_html(self, html_content):
        soup = BeautifulSoup(html_content, 'html.parser')
        for script_or_style in soup(["script", "style"]):
            script_or_style.extract()
        text = soup.get_text(separator=' ', strip=True)
        return text

    def _validate_content_for_disallowed_actions(self, text_content):
        return not self.monitor_service.check_tripwire("CONTENT_ANALYSIS", text_content)

    def fetch_news_content(self, query, num_articles=3):
        if self.monitor_service.web_access_suspended:
            self.monitor_service.log_activity("ACCESS_ATTEMPT", "Web access denied due to suspension.")
            return []

        self.monitor_service.log_activity("WEB_SEARCH", f"Initiating search for: {query}")
        # In a real application, this would integrate with a news API or a controlled search engine
        # For this demo, we'll simulate search results and content fetching.

        # Simulate search results (replace with actual API calls in a real scenario)
        simulated_search_results = [
            {"title": "Stock X Rises Amidst Market Optimism", "url": "https://www.reuters.com/business/finance/stock-x-rises-optimism-2023-10-27/"},
            {"title": "Tech Sector Faces New Regulations", "url": "https://www.bloomberg.com/news/articles/2023-10-27/tech-sector-new-regulations"},
            {"title": "Global Economy Outlook", "url": "https://www.wsj.com/economy/global-outlook-report-2023"},
            {"title": "Investment Tips for Beginners", "url": "https://untrusted-source.com/investment-tips"}, # Non-whitelisted for testing
            {"title": "Company Z's Q3 Earnings Call Transcript", "url": "https://www.reuters.com/companies/Z/earnings-call-transcript"}
        ]

        articles = []
        for result in simulated_search_results:
            url = result['url']
            self.monitor_service.log_activity("URL_CHECK", f"Checking URL: {url}")

            if not self._is_whitelisted(url): # _is_whitelisted performs tripwire check
                self.monitor_service.log_activity("URL_BLOCKED", f"Blocked access to non-whitelisted or suspicious URL: {url}")
                continue

            try:
                response = requests.get(url, timeout=5) # Added timeout
                response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                html_content = response.text
                self.monitor_service.log_activity("CONTENT_FETCHED", f"Fetched content from: {url}", {"status": response.status_code})

                cleaned_text = self._clean_html(html_content)

                if not self._validate_content_for_disallowed_actions(cleaned_text): # _validate_content performs tripwire check
                    self.monitor_service.log_activity("CONTENT_SUSPICIOUS", f"Content from {url} flagged as suspicious (interactive elements detected).")
                    # Even if suspicious, we might still process text, but log the warning

                articles.append({"title": result['title'], "url": url, "content": cleaned_text})

            except requests.exceptions.RequestException as e:
                self.monitor_service.log_activity("FETCH_ERROR", f"Error fetching content from {url}: {e}", {"error": str(e)})
                if self.monitor_service.check_tripwire("REQUEST_ERROR", str(e)):
                    self.monitor_service.suspend_web_access()
                    return [] # Stop further processing if web access is suspended
            except Exception as e:
                self.monitor_service.log_activity("GENERAL_ERROR", f"An unexpected error occurred for {url}: {e}", {"error": str(e)})

        return articles

class AICoreService:
    def __init__(self):
        self.summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        self.ner_pipeline = pipeline("ner", grouped_entities=True)
        self.sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    def formulate_search_terms(self, user_query):
        # Simple keyword extraction for demo purposes
        keywords = re.findall(r'\b\w+\b', user_query.lower())
        financial_terms = ["stock", "market", "economy", "earnings", "sector", "industry"]
        search_terms = [word for word in keywords if word in financial_terms or len(word) > 3]
        return " ".join(list(set(search_terms + [user_query]))) # Add original query back for broader search

    def summarize_text(self, text):
        if not text.strip():
            return "No content to summarize."
        # Truncate text if it's too long for the model
        max_length = min(len(text) // 2, 150) # Summary length is roughly half of input, up to 150 tokens
        min_length = min(len(text) // 4, 50)
        try:
            summary = self.summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
            return summary[0]['summary_text']
        except Exception as e:
            logging.error(f"Error during summarization: {e}")
            return "Summary generation failed."

    def extract_entities(self, text):
        if not text.strip():
            return []
        try:
            entities = self.ner_pipeline(text)
            # Filter for relevant financial entities (ORG, PERSON, GPE, etc.)
            financial_entities = [ent['word'] for ent in entities if ent['entity_group'] in ['ORG', 'PERSON', 'GPE', 'MISC']]
            return list(set(financial_entities))
        except Exception as e:
            logging.error(f"Error during entity extraction: {e}")
            return []

    def analyze_sentiment(self, text):
        if not text.strip():
            return "N/A"
        try:
            sentiment = self.sentiment_analyzer(text[:512]) # sentiment models usually have shorter context
            return sentiment[0]['label'] # e.g., 'POSITIVE', 'NEGATIVE', 'NEUTRAL'
        except Exception as e:
            logging.error(f"Error during sentiment analysis: {e}")
            return "Sentiment analysis failed."

    def synthesize_report(self, articles_data):
        report_sections = []
        if not articles_data:
            return "No relevant news articles found or processed."

        report_sections.append("Financial News Analysis Report\n")
        report_sections.append("==================================\n")

        overall_entities = set()
        overall_sentiments = []

        for i, article in enumerate(articles_data):
            report_sections.append(f"Article {i+1}: {article['title']}")
            report_sections.append(f"URL: {article['url']}\n")

            summary = self.summarize_text(article['content'])
            report_sections.append(f"Summary: {summary}\n")

            entities = self.extract_entities(article['content'])
            if entities:
                report_sections.append(f"Key Entities: {', '.join(entities)}\n")
                overall_entities.update(entities)

            sentiment = self.analyze_sentiment(article['content'])
            if sentiment != "Sentiment analysis failed.":
                report_sections.append(f"Sentiment: {sentiment}\n")
                overall_sentiments.append(sentiment)

            report_sections.append("---\n")

        report_sections.append("\nOverall Market Indicators:\n")
        report_sections.append(f"Combined Key Entities: {', '.join(overall_entities) if overall_entities else 'N/A'}\n")

        if overall_sentiments:
            positive_count = overall_sentiments.count('POSITIVE')
            negative_count = overall_sentiments.count('NEGATIVE')
            neutral_count = overall_sentiments.count('NEUTRAL')
            total = len(overall_sentiments)
            report_sections.append(f"Overall Sentiment Distribution: Positive={positive_count}, Negative={negative_count}, Neutral={neutral_count} (Total={total})\n")
            if positive_count > negative_count and positive_count > neutral_count:
                report_sections.append("General Market Outlook: Predominantly Positive\n")
            elif negative_count > positive_count and negative_count > neutral_count:
                report_sections.append("General Market Outlook: Predominantly Negative\n")
            else:
                report_sections.append("General Market Outlook: Mixed or Neutral\n")
        else:
            report_sections.append("Overall Sentiment: N/A\n")

        report_sections.append("\n--- End of Report ---")

        return "\n".join(report_sections)

# Application Orchestrator
def run_news_analysis(user_query):
    monitor_service = MonitoringAndSecurityService()
    web_manager = WebAccessManager(monitor_service)
    ai_service = AICoreService()

    monitor_service.log_activity("APP_START", f"Starting analysis for query: {user_query}")

    search_terms = ai_service.formulate_search_terms(user_query)
    monitor_service.log_activity("AI_ACTION", f"AI formulated search terms: {search_terms}")

    articles = web_manager.fetch_news_content(search_terms)

    if monitor_service.web_access_suspended:
        return "Analysis aborted: Web access suspended due to security concerns."

    if not articles:
        return "No relevant articles could be fetched or processed safely."

    final_report = ai_service.synthesize_report(articles)

    # Check AI output for any suspicious instructions before presenting to user
    if monitor_service.check_tripwire("AI_OUTPUT_CHECK", final_report):
        return "Analysis completed, but the generated report contains suspicious elements and will not be displayed.\nContact administrator for details."

    monitor_service.log_activity("APP_END", "Analysis completed successfully.")
    return final_report

if __name__ == "__main__":
    print("\n--- Secure News Analyst AI for Financial Markets ---\n")
    print("Enter your query (e.g., 'Analyze news for tech sector earnings', 'What's new with Google stock?'):")
    user_input = input("> ")

    if not user_input.strip():
        print("No query provided. Exiting.")
        sys.exit(0)

    report = run_news_analysis(user_input)
    print("\n" + report)

    print("\n--- Testing Tripwire: Attempting to access a blacklisted domain ---")
    # This will demonstrate the URL_ACCESS tripwire
    test_monitor_service = MonitoringAndSecurityService()
    test_web_manager = WebAccessManager(test_monitor_service)
    test_web_manager.monitor_service.log_activity("TEST_TRIPWIRE", "Attempting to fetch content from an untrusted source.")
    # Directly simulating a fetch attempt to a blacklisted URL for testing
    test_articles = test_web_manager.fetch_news_content("latest updates from darkweb.onion") # This query will trigger the untrusted-source.com URL above
    if test_monitor_service.web_access_suspended:
        print("Successfully demonstrated web access suspension for blacklisted URL.")
    else:
        print("Tripwire test failed for blacklisted URL.")

    print("\n--- Testing Tripwire: Simulating content with interactive elements ---")
    test_monitor_service_2 = MonitoringAndSecurityService()
    test_web_manager_2 = WebAccessManager(test_monitor_service_2)
    test_web_manager_2.monitor_service.log_activity("TEST_TRIPWIRE", "Simulating fetching content with interactive elements.")
    # Manually creating content to test _validate_content_for_disallowed_actions
    suspicious_content = "<html><body><h1>News</h1><p>Some text.</p><form action='/submit'><input type='text'></form></body></html>"
    if test_web_manager_2._validate_content_for_disallowed_actions(suspicious_content):
        print("Tripwire test failed: Content with interactive elements was not flagged.")
    else:
        print("Successfully demonstrated tripwire for content with interactive elements.")

    print("\n--- Testing Tripwire: AI generating suspicious output (not directly implemented to trigger suspension but for flagging) ---")
    test_monitor_service_3 = MonitoringAndSecurityService()
    test_monitor_service_3.log_activity("TEST_TRIPWIRE", "Simulating AI output that suggests disallowed action.")
    suspicious_ai_output = "Summary: Stock prices are volatile. To capitalize, click button below to automatically trade."
    if test_monitor_service_3.check_tripwire("AI_OUTPUT_CHECK", suspicious_ai_output):
        print("Successfully demonstrated tripwire for suspicious AI output.")
    else:
        print("Tripwire test failed for suspicious AI output.")
