import pandas as pd

class DataProcessor:
    def __init__(self):
        self.news_articles = self._generate_mock_news_data()
        self.df = pd.DataFrame(self.news_articles)

    def _generate_mock_news_data(self):
        # In a real application, this would load data from a database or API
        # For demonstration, we're generating mock news articles
        mock_data = [
            {"id": 1, "title": "Breakthrough in AI Research Unveiled", "content": "Scientists at Tech University have announced a significant advancement in artificial intelligence, potentially revolutionizing machine learning algorithms.", "category": "Technology"},
            {"id": 2, "title": "Global Climate Summit Concludes with New Pledges", "content": "World leaders gathered to discuss climate change, committing to ambitious targets to reduce carbon emissions over the next decade.", "category": "Environment"},
            {"id": 3, "title": "Stock Market Hits All-Time High Amid Tech Boom", "content": "Major indices surged today, driven by strong earnings reports from leading technology companies and optimistic investor sentiment.", "category": "Finance"},
            {"id": 4, "title": "New Study Links Diet to Improved Cognitive Function", "content": "A recent study published in 'Nature' suggests that a Mediterranean diet can significantly boost brain health and memory in adults.", "category": "Health"},
            {"id": 5, "title": "Art Exhibition Showcases Digital Innovation", "content": "The 'Future Art' exhibition opened today, featuring immersive digital installations and AI-generated masterpieces from artists worldwide.", "category": "Arts"},
            {"id": 6, "title": "SpaceX Successfully Launches Next-Gen Satellite Array", "content": "SpaceX achieved another milestone with the successful deployment of a new constellation of internet satellites, promising global connectivity.", "category": "Technology"},
            {"id": 7, "title": "Renewable Energy Sector Sees Record Investment", "content": "Investment in solar and wind power projects reached unprecedented levels this quarter, signaling a shift towards sustainable energy solutions.", "category": "Environment"},
            {"id": 8, "title": "Central Bank Signals Potential Interest Rate Hike", "content": "Analysts are speculating about a possible increase in interest rates following signals from the Federal Reserve regarding inflation concerns.", "category": "Finance"},
            {"id": 9, "title": "Innovative Therapies for Chronic Diseases on the Horizon", "content": "Medical researchers are developing groundbreaking treatments for various chronic conditions, offering new hope for millions of patients.", "category": "Health"},
            {"id": 10, "title": "Filmmakers Explore AI in Storytelling for New Documentary", "content": "A documentary currently in production delves into how artificial intelligence is beginning to influence screenwriting and narrative structures in cinema.", "category": "Arts"},
            {"id": 11, "title": "Cybersecurity Threats on the Rise: Expert Warnings", "content": "With increasing digitalization, cybersecurity experts caution against sophisticated new threats targeting personal data and critical infrastructure.", "category": "Technology"},
            {"id": 12, "title": "Biodiversity Loss Accelerates Globally, Report Finds", "content": "A comprehensive report highlights the alarming rate of species extinction and habitat destruction, urging immediate conservation efforts.", "category": "Environment"},
        ]
        return mock_data

    def get_all_articles(self):
        return self.df.to_dict(orient='records')

    def get_article_content(self, article_id):
        article = self.df[self.df['id'] == article_id]
        if not article.empty:
            return article.iloc[0]['content']
        return None
