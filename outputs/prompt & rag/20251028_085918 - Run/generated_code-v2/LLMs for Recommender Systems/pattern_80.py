import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class NewsRecommender:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.articles_df = pd.DataFrame()
        self.article_embeddings = None
        self.user_interactions = {}

    def load_articles(self, articles_data):
        self.articles_df = pd.DataFrame(articles_data)
        self.articles_df['article_id'] = self.articles_df.index
        self._generate_article_embeddings()

    def _generate_article_embeddings(self):
        article_contents = self.articles_df['content'].tolist()
        self.article_embeddings = self.model.encode(article_contents, show_progress_bar=False)

    def record_user_interaction(self, user_id, article_id):
        if user_id not in self.user_interactions:
            self.user_interactions[user_id] = []
        if article_id not in self.user_interactions[user_id]:
            self.user_interactions[user_id].append(article_id)

    def _get_user_embedding(self, user_id):
        if user_id not in self.user_interactions or not self.user_interactions[user_id]:
            return None
        interacted_article_ids = self.user_interactions[user_id]
        interacted_embeddings = self.article_embeddings[interacted_article_ids]
        return np.mean(interacted_embeddings, axis=0)

    def recommend_articles(self, user_id, num_recommendations=5):
        user_embedding = self._get_user_embedding(user_id)

        if user_embedding is None:
            # Cold-start: recommend popular/random articles if no interaction history
            return self.articles_df.sample(num_recommendations)['article_id'].tolist()

        read_article_ids = set(self.user_interactions.get(user_id, []))
        unread_articles_df = self.articles_df[~self.articles_df['article_id'].isin(read_article_ids)]

        if unread_articles_df.empty:
            return []

        unread_article_embeddings = self.article_embeddings[unread_articles_df['article_id'].tolist()]

        similarities = cosine_similarity([user_embedding], unread_article_embeddings)[0]

        top_indices = similarities.argsort()[-num_recommendations:][::-1]
        recommended_article_ids = unread_articles_df.iloc[top_indices]['article_id'].tolist()

        return recommended_article_ids


if __name__ == '__main__':
    # Simulate News Articles
    news_data = [
        {"content": "Tech giant releases new smartphone with advanced AI camera features."},
        {"content": "Scientists discover new exoplanet with potential for life."},
        {"content": "Financial markets react to latest interest rate hike."},
        {"content": "New study shows benefits of mindfulness for stress reduction."},
        {"content": "Local elections results announced, new mayor takes office."},
        {"content": "Breakthrough in renewable energy could power cities."},
        {"content": "Art exhibition opens featuring works by renowned contemporary artists."},
        {"content": "Healthy recipes for a balanced diet and active lifestyle."},
        {"content": "Software update brings performance improvements and new privacy features."},
        {"content": "Global summit addresses climate change and sustainable development goals."}
    ]

    recommender = NewsRecommender()
    recommender.load_articles(news_data)

    # Simulate User Interactions
    user1_id = 1
    user2_id = 2
    user3_id = 3 # Cold-start user

    recommender.record_user_interaction(user1_id, 0) # User 1 reads Tech article
    recommender.record_user_interaction(user1_id, 8) # User 1 reads Software update article

    recommender.record_user_interaction(user2_id, 1) # User 2 reads Science article
    recommender.record_user_interaction(user2_id, 5) # User 2 reads Renewable energy article
    recommender.record_user_interaction(user2_id, 9) # User 2 reads Climate change article

    # Get Recommendations
    print(f"Recommendations for User {user1_id} (Tech/Software focused):\n")
    user1_recs = recommender.recommend_articles(user1_id)
    for rec_id in user1_recs:
        print(f"  - Article ID: {rec_id}, Content: {recommender.articles_df.loc[rec_id, 'content']}")

    print(f"\nRecommendations for User {user2_id} (Science/Environment focused):\n")
    user2_recs = recommender.recommend_articles(user2_id)
    for rec_id in user2_recs:
        print(f"  - Article ID: {rec_id}, Content: {recommender.articles_df.loc[rec_id, 'content']}")

    print(f"\nRecommendations for User {user3_id} (Cold-start user):\n")
    user3_recs = recommender.recommend_articles(user3_id)
    for rec_id in user3_recs:
        print(f"  - Article ID: {rec_id}, Content: {recommender.articles_df.loc[rec_id, 'content']}")
