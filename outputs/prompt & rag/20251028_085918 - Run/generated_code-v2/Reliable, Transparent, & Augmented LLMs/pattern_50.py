import random

def fetch_news_articles(topic: str, num_articles: int = 3) -> list[str]:
    """
    Simulates fetching news articles related to a given topic.
    In a real application, this would involve web scraping or using news APIs.
    """
    print(f"  [News Fetcher] Simulating fetching {num_articles} articles for topic: \'{topic}\'")
    # Example articles (hardcoded for demonstration)
    if topic == "Impact of AI on Job Market":
        return [
            "Article 1: AI is expected to automate many routine tasks, potentially displacing jobs in manufacturing, customer service, and data entry. However, it will also create new jobs in AI development, maintenance, and ethical oversight. The key is reskilling the workforce.",
            "Article 2: Studies suggest that while AI will transform job roles, it's more likely to augment human capabilities rather than fully replace humans. Collaboration between humans and AI could lead to increased productivity and new economic opportunities. Concerns about massive unemployment may be overstated.",
            "Article 3: Economists warn of significant job losses in sectors vulnerable to automation, urging governments and educational institutions to prepare for a future where traditional job security is diminished. The ethical implications of AI-driven unemployment require careful consideration and policy changes."
        ]
    else:
        return [
            f"Generic article about {topic} 1.",
            f"Generic article about {topic} 2."
        ]
