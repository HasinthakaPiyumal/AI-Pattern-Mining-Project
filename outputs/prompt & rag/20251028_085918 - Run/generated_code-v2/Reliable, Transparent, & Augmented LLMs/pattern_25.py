import newspaper
import feedparser
from bs4 import BeautifulSoup
import re
import nltk
from nltk.tokenize import sent_tokenize

class NewsScraper:
    def scrape_article(self, url):
        try:
            article = newspaper.Article(url)
            article.download()
            article.parse()
            return {
                "url": url,
                "title": article.title,
                "text": article.text,
                "authors": article.authors,
                "publish_date": article.publish_date,
                "source": article.source_url
            }
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return None

class RSSFeedReader:
    def read_feeds(self, feed_urls):
        articles_meta = []
        for url in feed_urls:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries:
                    articles_meta.append({
                        "title": entry.title,
                        "link": entry.link,
                        "published": entry.published if hasattr(entry, 'published') else None,
                        "summary": entry.summary if hasattr(entry, 'summary') else None
                    })
            except Exception as e:
                print(f"Error reading RSS feed {url}: {e}")
        return articles_meta

class TextCleaner:
    def clean_text(self, html_content):
        if not html_content:
            return ""
        soup = BeautifulSoup(html_content, "html.parser")
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()
        text = soup.get_text()

        text = re.sub(r"\n+", "\n", text).strip()
        text = re.sub(r" +", " ", text)
        return text

class InformationExtractor:
    def __init__(self):
        try:
            nltk.data.find('tokenizers/punkt')
        except nltk.downloader.DownloadError:
            nltk.download('punkt')

    def segment_sentences(self, text):
        return sent_tokenize(text)

    def extract_claims(self, sentences, topic="a general controversial topic"):
        extracted_claims = []
        for i, sentence in enumerate(sentences):
            stance = "for" if i % 2 == 0 else "against"
            extracted_claims.append({
                "claim": sentence,
                "stance": stance,
                "source_sentence_index": i
            })
        return extracted_claims

class ControversyCompassAggregator:
    def __init__(self):
        self.scraper = NewsScraper()
        self.rss_reader = RSSFeedReader()
        self.text_cleaner = TextCleaner()
        self.info_extractor = InformationExtractor()
        self.processed_articles = []

    def aggregate_news(self, rss_feeds, initial_urls):
        rss_article_meta = self.rss_reader.read_feeds(rss_feeds)

        urls_to_scrape = [item["link"] for item in rss_article_meta]
        urls_to_scrape.extend(initial_urls)
        urls_to_scrape = list(set(urls_to_scrape))

        for url in urls_to_scrape:
            article_data = self.scraper.scrape_article(url)
            if article_data and article_data["text"]:
                cleaned_text = self.text_cleaner.clean_text(article_data["text"])
                if cleaned_text:
                    article_data["cleaned_text"] = cleaned_text
                    sentences = self.info_extractor.segment_sentences(cleaned_text)
                    claims = self.info_extractor.extract_claims(sentences)
                    article_data["sentences"] = sentences
                    article_data["claims"] = claims
                    self.processed_articles.append(article_data)

        return self.processed_articles

    def generate_summary(self, topic="a controversial topic"):
        pro_claims = []
        con_claims = []
        
        for article in self.processed_articles:
            for claim_data in article.get("claims", []):
                if claim_data["stance"] == "for":
                    pro_claims.append(f"- {claim_data['claim']} (Source: {article.get('title', 'N/A')}, {article.get('source', 'N/A')})")
                else:
                    con_claims.append(f"- {claim_data['claim']} (Source: {article.get('title', 'N/A')}, {article.get('source', 'N/A')})")
        
        summary_parts = []
        summary_parts.append(f"## Debate-Style Summary on: {topic}\n")
        
        if pro_claims:
            summary_parts.append("### Arguments For:\n")
            summary_parts.extend(pro_claims)
            summary_parts.append("\n")
            
        if con_claims:
            summary_parts.append("### Arguments Against:\n")
            summary_parts.extend(con_claims)
            summary_parts.append("\n")

        if not pro_claims and not con_claims:
            summary_parts.append("No clear arguments for or against found for this topic from the processed articles.")
            
        return "\n".join(summary_parts)
