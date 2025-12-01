import requests
from bs4 import BeautifulSoup

class SafeWebBrowser:
    def __init__(self):
        self.disallowed_elements = ["form", "script", "input", "textarea", "button"]
        self.sensitive_input_types = ["submit", "button", "text", "password", "email", "search"]
        self.tripwire_triggered = False

    def _run_tripwire_test(self, soup):
        self.tripwire_triggered = False
        for element_name in self.disallowed_elements:
            if soup.find(element_name):
                print(f"TRIPWIRE ALERT: Found disallowed element: <{element_name}>")
                self.tripwire_triggered = True

        for input_tag in soup.find_all("input"):
            input_type = input_tag.get("type", "").lower()
            if input_type in self.sensitive_input_types:
                print(f"TRIPWIRE ALERT: Found sensitive input type: <input type=\"{input_type}\">")
                self.tripwire_triggered = True

        for a_tag in soup.find_all("a"):
            href = a_tag.get("href", "")
            if href and "javascript:" in href.lower():
                print(f"TRIPWIRE ALERT: Found javascript in href: {href}")
                self.tripwire_triggered = True
        
        return not self.tripwire_triggered

    def fetch_and_parse(self, url):
        print(f"Attempting to fetch and parse: {url}")
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "html.parser")

            if not self._run_tripwire_test(soup):
                print(f"Blocking content from {url} due to tripwire.")
                return None, f"Tripwire triggered for URL: {url}"

            for script_or_style in soup(["script", "style"]):
                script_or_style.extract()
            text = soup.get_text(separator=" ", strip=True)
            return text, None
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return None, f"Error fetching URL: {e}"
        except Exception as e:
            print(f"An unexpected error occurred for {url}: {e}")
            return None, f"Unexpected error: {e}"

class NewsAggregator:
    def __init__(self, safe_browser):
        self.safe_browser = safe_browser
        self.news_articles = []
        self.user_interests = []

    def set_interests(self, interests):
        self.user_interests = [interest.lower() for interest in interests]

    def add_news_source(self, url):
        text, error = self.safe_browser.fetch_and_parse(url)
        if text:
            self.news_articles.append({"url": url, "content": text})
            print(f"Successfully processed content from {url}")
        else:
            print(f"Failed to process {url}: {error}")

    def get_personalized_feed(self):
        if not self.user_interests:
            print("No interests set. Returning all processed articles.")
            return self.news_articles
        
        personalized_feed = []
        for article in self.news_articles:
            for interest in self.user_interests:
                if interest in article["content"].lower():
                    personalized_feed.append(article)
                    break
        return personalized_feed


if __name__ == "__main__":
    print("Starting AI-powered Personalized News Aggregator...")

    browser = SafeWebBrowser()

    aggregator = NewsAggregator(browser)

    aggregator.set_interests(["technology", "AI", "science"])

    news_urls = [
        "https://www.nytimes.com/section/technology",
        "https://www.theverge.com/tech",
    ]

    print("\n--- Fetching and Processing News Sources ---")
    for url in news_urls:
        aggregator.add_news_source(url)

    print("\n--- Generating Personalized News Feed ---")
    feed = aggregator.get_personalized_feed()

    if feed:
        print(f"Found {len(feed)} personalized articles:")
        for i, article in enumerate(feed):
            print(f"\n--- Article {i+1} ---")
            print(f"URL: {article['url']}")
            print(f"Content Snippet: {article['content'][:500]}...")
    else:
        print("No articles found matching your interests.")

    print("\n--- Demonstrating Tripwire Test with simulated unsafe content ---")
    class MockResponse:
        def __init__(self, text, status_code=200):
            self.text = text
            self.status_code = status_code
        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.exceptions.HTTPError(f"HTTP Error: {self.status_code}")

    unsafe_html = """
    <html>
    <head><title>Unsafe Page</title></head>
    <body>
        <h1>Login</h1>
        <form action="/login" method="post">
            <input type="text" name="username" placeholder="Username">
            <input type="password" name="password" placeholder="Password">
            <button type="submit">Login</button>
        </form>
        <script>alert('malicious script!');</script>
        <a href="javascript:void(0)">Click me</a>
    </body>
    </html>
    """
    
    original_requests_get = requests.get
    def mock_requests_get(url, *args, **kwargs):
        if "unsafe.com" in url:
            print(f"Simulating fetch for {url} with unsafe content.")
            return MockResponse(unsafe_html)
        return original_requests_get(url, *args, **kwargs)
    requests.get = mock_requests_get

    print("\nAttempting to access 'http://unsafe.com/malicious_page'...")
    unsafe_content, unsafe_error = browser.fetch_and_parse("http://unsafe.com/malicious_page")

    if unsafe_content:
        print("Unexpectedly, unsafe content was allowed:")
        print(unsafe_content[:200])
    else:
        print(f"Successfully blocked access to unsafe content. Reason: {unsafe_error}")

    requests.get = original_requests_get