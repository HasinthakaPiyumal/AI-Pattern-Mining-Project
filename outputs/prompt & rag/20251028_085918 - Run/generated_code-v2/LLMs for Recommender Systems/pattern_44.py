import pandas as pd
import numpy as np

def generate_news_data(num_articles=100):
    categories = ["Technology", "Politics", "Sports", "Science", "Entertainment"]
    news_articles = []
    for i in range(num_articles):
        category = np.random.choice(categories)
        title = f"Headline for {category} Article {i+1}"
        content = f"This is the full content of the {category} article number {i+1}. It contains detailed information about recent developments in {category.lower()} and relevant events. This article is designed to be interesting for readers who follow {category.lower()} news."
        news_articles.append({"article_id": f"article_{i+1}", "title": title, "content": content, "category": category})
    return pd.DataFrame(news_articles)

def generate_user_behavior(news_df, num_users=20, articles_per_user=10):
    user_behavior = []
    unique_article_ids = news_df["article_id"].tolist()
    for i in range(num_users):
        user_id = f"user_{i+1}"
        read_articles = np.random.choice(unique_article_ids, min(articles_per_user, len(unique_article_ids)), replace=False)
        for article_id in read_articles:
            user_behavior.append({"user_id": user_id, "article_id": article_id, "timestamp": pd.Timestamp.now()})
    return pd.DataFrame(user_behavior)

if __name__ == "__main__":
    print("Generating simulated news data...")
    news_df = generate_news_data(num_articles=500)
    news_df.to_csv("simulated_news_articles.csv", index=False)
    print(f"Generated {len(news_df)} news articles and saved to simulated_news_articles.csv")

    print("Generating simulated user behavior data...")
    user_behavior_df = generate_user_behavior(news_df, num_users=50, articles_per_user=15)
    user_behavior_df.to_csv("simulated_user_behavior.csv", index=False)
    print(f"Generated {len(user_behavior_df)} user interactions and saved to simulated_user_behavior.csv")