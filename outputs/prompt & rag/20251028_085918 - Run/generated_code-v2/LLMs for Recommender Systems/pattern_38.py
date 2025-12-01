import numpy as np
from sentence_transformers import SentenceTransformer

class NewsRecommendationSystem:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.articles = {}
        self.article_embeddings = {}
        self.users = {}

    def add_article(self, article_id, title, content, categories=None):
        text = title + " " + content
        embedding = self.model.encode(text, convert_to_tensor=True).cpu().numpy()
        self.articles[article_id] = {
            "title": title,
            "content": content,
            "categories": categories if categories else [],
            "embedding": embedding
        }
        self.article_embeddings[article_id] = embedding

    def update_user_interactions(self, user_id, interacted_article_ids):
        if user_id not in self.users:
            self.users[user_id] = {"read_articles": set()}
        self.users[user_id]["read_articles"].update(interacted_article_ids)

    def _get_user_embedding(self, user_id):
        if user_id not in self.users or not self.users[user_id]["read_articles"]:
            # Cold-start user: return a zero vector or an average of popular articles
            # For simplicity, returning a zero vector. In a real system, you might recommend popular items.
            if not self.article_embeddings:
                return np.zeros(self.model.get_sentence_embedding_dimension())
            # For a more useful cold-start, could average embeddings of most popular articles
            # For this example, if no interactions, we'll return a generic 'popular' embedding if available
            # or a zero vector if no articles at all.
            all_embeddings = list(self.article_embeddings.values())
            if all_embeddings:
                return np.mean(all_embeddings, axis=0)
            else:
                return np.zeros(self.model.get_sentence_embedding_dimension())

        read_article_embeddings = []
        for article_id in self.users[user_id]["read_articles"]:
            if article_id in self.article_embeddings:
                read_article_embeddings.append(self.article_embeddings[article_id])
        
        if not read_article_embeddings:
            return np.zeros(self.model.get_sentence_embedding_dimension())

        return np.mean(read_article_embeddings, axis=0)

    def _cosine_similarity(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0
        return dot_product / (norm_vec1 * norm_vec2)

    def get_recommendations(self, user_id, num_recommendations=5):
        user_embedding = self._get_user_embedding(user_id)
        
        if user_embedding.sum() == 0 and not self.article_embeddings: # Truly cold-start: no user history, no articles
            return []

        scores = []
        read_articles = self.users.get(user_id, {}).get("read_articles", set())

        for article_id, article_data in self.articles.items():
            if article_id not in read_articles:
                similarity = self._cosine_similarity(user_embedding, article_data["embedding"])
                scores.append((article_id, similarity))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        
        recommended_articles = []
        for article_id, _ in scores[:num_recommendations]:
            recommended_articles.append(self.articles[article_id]["title"])
            
        return recommended_articles

# Example Usage:
if __name__ == "__main__":
    recommender = NewsRecommendationSystem()

    # Add some articles
    recommender.add_article("article1", "AI Revolutionizes Healthcare", "Artificial intelligence is transforming diagnostics and drug discovery.", ["AI", "Healthcare"])
    recommender.add_article("article2", "New Breakthrough in Quantum Computing", "Scientists have achieved a major milestone in quantum entanglement.", ["Technology", "Quantum Computing"])
    recommender.add_article("article3", "The Future of Renewable Energy", "Solar and wind power are becoming increasingly efficient and affordable.", ["Energy", "Environment"])
    recommender.add_article("article4", "Exploring Mars: Recent Discoveries", "NASA's rover has sent back fascinating data from the Red Planet.", ["Space", "Science"])
    recommender.add_article("article5", "Impact of Machine Learning on Finance", "ML algorithms are now used for fraud detection and algorithmic trading.", ["AI", "Finance"])
    recommender.add_article("article6", "Sustainable Cities Initiatives", "Urban planners are designing eco-friendly cities for the future.", ["Environment", "Urban Planning"])

    # User 1 interactions
    print("\n--- User 1 (AI & Healthcare enthusiast) ---")
    recommender.update_user_interactions("user1", ["article1", "article5"])
    print("Recommendations for user1:", recommender.get_recommendations("user1", num_recommendations=3))

    # User 2 interactions (more interested in Space/Tech)
    print("\n--- User 2 (Space & Tech enthusiast) ---")
    recommender.update_user_interactions("user2", ["article2", "article4"])
    print("Recommendations for user2:", recommender.get_recommendations("user2", num_recommendations=3))

    # Cold-start user 3 (no prior interactions)
    print("\n--- User 3 (Cold-start user) ---")
    print("Recommendations for user3:", recommender.get_recommendations("user3", num_recommendations=3))

    # User 1 reads a new article and gets updated recommendations
    print("\n--- User 1 reads a new article ---")
    recommender.add_article("article7", "Latest Advances in Medical Robotics", "Robots are assisting surgeons with unprecedented precision.", ["Healthcare", "Robotics"])
    recommender.update_user_interactions("user1", ["article7"])
    print("Updated recommendations for user1:", recommender.get_recommendations("user1", num_recommendations=3))

    # Cold-start user with no articles in the system initially
    print("\n--- User 4 (Cold-start user, no articles initially) ---")
    empty_recommender = NewsRecommendationSystem()
    print("Recommendations for user4 (empty system):", empty_recommender.get_recommendations("user4"))
    empty_recommender.add_article("article_empty1", "Test Article", "This is a test.")
    print("Recommendations for user4 (after adding one article):", empty_recommender.get_recommendations("user4"))
    empty_recommender.update_user_interactions("user4", ["article_empty1"])
    print("Recommendations for user4 (after interaction):", empty_recommender.get_recommendations("user4"))


