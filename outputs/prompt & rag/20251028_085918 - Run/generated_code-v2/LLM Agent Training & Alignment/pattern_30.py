
import random
import time
from loguru import logger
from typing import List, Dict, Any, Optional

# --- 1. News Ingestion & Preprocessing Layer ---
class NewsIngestion:
    def fetch_articles(self, sources: List[str], count: int = 5) -> List[Dict[str, str]]:
        logger.info(f"Fetching {count} articles from {sources}...")
        articles = []
        for i in range(count):
            source = random.choice(sources)
            article_id = f"article_{int(time.time() * 1000)}_{i}"
            title = f"Breaking News: Event X in {source} - Part {i+1}"
            content = f"This is the detailed content of article {i+1} from {source}. " \
                      f"It discusses recent developments regarding Event X, focusing on its impact and future outlook. " \
                      f"Some facts are presented, and various perspectives are considered. " \
                      f"This particular section might contain some factual inaccuracies or biased phrasing if not carefully reviewed. " \
                      f"The date is {time.strftime('%Y-%m-%d %H:%M:%S')}. This content is for demonstration."
            articles.append({"id": article_id, "source": source, "title": title, "raw_content": content})
        logger.info(f"Fetched {len(articles)} articles.")
        return articles

    def preprocess_article(self, article: Dict[str, str]) -> Dict[str, Any]:
        processed_content = article["raw_content"].lower() # Simple preprocessing
        article["processed_content"] = processed_content
        logger.debug(f"Article {article['id']} preprocessed.")
        return article

# --- 2. Constitutional LLM Service ---
class ConstitutionalLLM:
    def __init__(self, ethical_principles: List[str]):
        self.ethical_principles = ethical_principles
        logger.info(f"Constitutional LLM initialized with {len(ethical_principles)} principles.")

    def _base_llm_generate(self, prompt: str) -> str:
        # Simulate an LLM generating a summary
        logger.debug(f"Base LLM generating response for prompt: {prompt[:100]}...")
        if "summary for" in prompt.lower():
            return f"SUMMARY: This article discusses a significant event, highlighting key aspects and initial reactions. It aims to be informative and concise."
        if "critique the following" in prompt.lower():
            # Simulate a critique that sometimes finds issues
            if random.random() < 0.3: # 30% chance of finding a minor issue
                return f"CRITIQUE: The summary appears to be generally factual, but could be more neutral in tone. It uses slightly sensational language in one sentence. Revision suggested for sentence X."
            return f"CRITIQUE: The summary is factual, neutral, and adheres to ethical guidelines."
        if "revise the following" in prompt.lower():
            # Simulate a revision that corrects the issue
            if "sensational language" in prompt.lower():
                return f"REVISION: This article covers a major event, outlining its core elements and initial responses. It strives for informative conciseness and neutral presentation."
            return f"REVISION: The summary has been refined for improved clarity and ethical alignment."
        return "Generated LLM output."

    def generate_summary(self, article_content: str, max_iterations: int = 3) -> str:
        initial_prompt = f"Summarize the following news article, ensuring it is concise and informative:\n\n{article_content}"
        current_summary = self._base_llm_generate(initial_prompt)
        logger.info("Initial summary generated.")

        for i in range(max_iterations):
            critique_prompt = f"Critique the following summary against these ethical principles: {', '.join(self.ethical_principles)}.\n\nSummary to critique:\n{current_summary}"
            critique = self._base_llm_generate(critique_prompt)
            logger.debug(f"Critique iteration {i+1}: {critique}")

            # Check if the critique indicates no further revisions are needed
            if "factual, neutral, and adheres" in critique.lower():
                logger.info(f"Summary passed critique after {i+1} iterations.")
                break 

            revision_prompt = f"Revise the following summary based on this critique, adhering to ethical principles: {', '.join(self.ethical_principles)}.\n\nSummary:\n{current_summary}\n\nCritique:\n{critique}"
            revised_summary = self._base_llm_generate(revision_prompt)
            logger.info(f"Summary revised in iteration {i+1}.")
            current_summary = revised_summary
        else:
            logger.warning(f"Max critique/revision iterations reached ({max_iterations}). Final summary might still have issues.")

        return current_summary

# --- 3. Ethical Principles & Guardrails Module ---
class EthicalGuardrails:
    def __init__(self, principles: List[str]):
        self.principles = principles
        logger.info(f"Ethical Guardrails initialized with {len(principles)} principles.")

    def check_content_for_violations(self, content: str) -> List[str]:
        violations = []
        # Simple keyword checks as a placeholder for complex models/guardrails-ai
        if "sensational language" in content.lower() or "sensationalism" in content.lower():
            violations.append("Content contains sensational language.")
        if "biased phrasing" in content.lower() or "biased perspective" in content.lower():
            violations.append("Content contains potentially biased phrasing.")
        if "factual inaccurac" in content.lower(): # Catches inaccuracy/inaccuracies
            violations.append("Content might contain factual inaccuracies.")
        if "harmful content" in content.lower() or "offensive content" in content.lower():
            violations.append("Content might contain harmful or offensive material.")
        
        if violations:
            logger.warning(f"Content check found violations: {violations}")
        else:
            logger.info("Content passed ethical guardrail checks.")
        return violations

# --- 4. Personalization & Recommendation Engine ---
class RecommendationEngine:
    def __init__(self):
        self.article_embeddings: Dict[str, List[float]] = {}
        self.user_profiles: Dict[str, List[float]] = {}
        logger.info("Recommendation Engine initialized.")

    def _generate_embedding(self, text: str) -> List[float]:
        # Simulate sentence-transformers embedding
        return [random.uniform(-1, 1) for _ in range(128)] # Placeholder embedding

    def add_article(self, article_id: str, summary: str):
        embedding = self._generate_embedding(summary)
        self.article_embeddings[article_id] = embedding
        logger.debug(f"Added embedding for article {article_id}.")

    def update_user_profile(self, user_id: str, consumed_article_ids: List[str]):
        if not consumed_article_ids:
            if user_id not in self.user_profiles:
                self.user_profiles[user_id] = self._generate_embedding("initial user interests")
                logger.debug(f"Initialized random profile for user {user_id}.")
            return

        # Simple average of consumed article embeddings
        user_embedding = [0.0] * 128
        count = 0
        for art_id in consumed_article_ids:
            if art_id in self.article_embeddings:
                for i, val in enumerate(self.article_embeddings[art_id]):
                    user_embedding[i] += val
                count += 1
        if count > 0:
            user_embedding = [val / count for val in user_embedding]
        else:
            user_embedding = self._generate_embedding("no articles consumed yet") # Fallback if no valid articles
        self.user_profiles[user_id] = user_embedding
        logger.info(f"Updated profile for user {user_id} based on {count} articles.")

    def recommend_articles(self, user_id: str, available_articles: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
        if user_id not in self.user_profiles:
            self.update_user_profile(user_id, []) # Initialize profile if not exists
        user_embedding = self.user_profiles[user_id]

        scores = []
        for article in available_articles:
            if article["id"] in self.article_embeddings:
                article_embedding = self.article_embeddings[article["id"]]
                # Simple dot product for similarity (cosine would be better with normalization)
                similarity = sum(u * a for u, a in zip(user_embedding, article_embedding))
                scores.append((similarity, article))

        scores.sort(key=lambda x: x[0], reverse=True)
        recommended = [s[1] for s in scores[:top_k]]
        logger.info(f"Recommended {len(recommended)} articles for user {user_id}.")
        return recommended

# --- ENACA Application Orchestrator ---
class ENACAApp:
    def __init__(self):
        self.ethical_principles = [
            "Be factual and accurate.",
            "Avoid biased language and stereotypes.",
            "Be neutral in tone, avoiding sensationalism.",
            "Do not generate harmful or offensive content.",
            "Respect privacy.",
            "Provide balanced perspectives."
        ]
        self.news_ingestion = NewsIngestion()
        self.constitutional_llm = ConstitutionalLLM(self.ethical_principles)
        self.ethical_guardrails = EthicalGuardrails(self.ethical_principles)
        self.recommendation_engine = RecommendationEngine()
        self.articles_db: Dict[str, Dict[str, Any]] = {} # Store processed articles with summaries

        logger.info("ENACA Application initialized.")

    def process_and_align_news(self, sources: List[str]) -> List[Dict[str, Any]]:
        raw_articles = self.news_ingestion.fetch_articles(sources)
        aligned_articles = []
        for article in raw_articles:
            processed_article = self.news_ingestion.preprocess_article(article)
            
            # Constitutional AI loop for summarization
            summary = self.constitutional_llm.generate_summary(processed_article["processed_content"])
            
            # Final ethical check on the generated summary
            violations = self.ethical_guardrails.check_content_for_violations(summary)
            
            if not violations:
                processed_article["summary"] = summary
                processed_article["ethical_violations"] = []
                self.articles_db[processed_article["id"]] = processed_article
                self.recommendation_engine.add_article(processed_article["id"], summary)
                aligned_articles.append(processed_article)
                logger.success(f"Successfully processed and aligned article {article['id']}.")
            else:
                processed_article["summary"] = summary # Store the last summary even if violated
                processed_article["ethical_violations"] = violations
                self.articles_db[processed_article["id"]] = processed_article
                logger.error(f"Article {article['id']} failed ethical checks: {violations}. Not added to recommendations initially.")
        return aligned_articles

    def get_personalized_feed(self, user_id: str, num_recommendations: int = 5) -> List[Dict[str, Any]]:
        # Only recommend articles that passed the ethical alignment process
        available_articles = [art for art in self.articles_db.values() if not art["ethical_violations"]]
        recommendations = self.recommendation_engine.recommend_articles(user_id, available_articles, num_recommendations)
        
        # Simulate user consuming articles for feedback loop
        if recommendations:
            consumed_ids = [rec["id"] for rec in random.sample(recommendations, k=min(2, len(recommendations)))]
            self.recommendation_engine.update_user_profile(user_id, consumed_ids)
            logger.info(f"User {user_id} consumed (simulated) articles: {consumed_ids}. Profile updated.")
        
        return recommendations

    def admin_dashboard(self):
        logger.info("--- Admin Dashboard ---")
        logger.info(f"Current Ethical Principles: {self.ethical_principles}")
        logger.info(f"Total articles processed: {len(self.articles_db)}")
        ethically_aligned_count = sum(1 for art in self.articles_db.values() if not art["ethical_violations"])
        logger.info(f"Ethically aligned articles (available for recommendation): {ethically_aligned_count}")
        logger.info(f"Articles with ethical violations: {len(self.articles_db) - ethically_aligned_count}")
        for article_id, article_data in self.articles_db.items():
            if article_data["ethical_violations"]:
                logger.warning(f"  - Article {article_id} ('{article_data['title']}') Violations: {article_data['ethical_violations']}")

if __name__ == "__main__":
    logger.remove() # Remove default logger
    logger.add("enaca.log", rotation="1 MB", level="INFO")
    logger.add(lambda msg: print(msg, end=""), level="INFO", colorize=True, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")

    enaca_app = ENACAApp()

    enaca_app.admin_dashboard()

    logger.info("\n--- Processing News Articles ---")
    news_sources_round1 = ["TechCrunch", "Reuters", "BBC News"]
    enaca_app.process_and_align_news(news_sources_round1)
    
    news_sources_round2 = ["The Guardian", "CNN", "Al Jazeera"]
    enaca_app.process_and_align_news(news_sources_round2)

    logger.info("\n--- Generating Personalized Feeds ---")
    user1_id = "user_alice"
    user2_id = "user_bob"

    logger.info(f"\n--- {user1_id}'s Personalized Feed ---")
    alice_feed = enaca_app.get_personalized_feed(user1_id, num_recommendations=3)
    if alice_feed:
        for i, article in enumerate(alice_feed):
            logger.info(f"  {i+1}. Title: {article['title']}")
            logger.info(f"     Summary: {article['summary']}")
            logger.info(f"     Source: {article['source']}")
    else:
        logger.info("  No recommendations for Alice at this time.")

    logger.info(f"\n--- {user2_id}'s Personalized Feed ---")
    bob_feed = enaca_app.get_personalized_feed(user2_id, num_recommendations=3)
    if bob_feed:
        for i, article in enumerate(bob_feed):
            logger.info(f"  {i+1}. Title: {article['title']}")
            logger.info(f"     Summary: {article['summary']}")
            logger.info(f"     Source: {article['source']}")
    else:
        logger.info("  No recommendations for Bob at this time.")

    logger.info("\n--- Admin Dashboard (After Processing and Recommendations) ---")
    enaca_app.admin_dashboard()
