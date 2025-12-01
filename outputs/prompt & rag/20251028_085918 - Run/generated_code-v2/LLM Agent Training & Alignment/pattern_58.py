from transformers import pipeline

# 1. Configuration Module (config.py)
CONSTITUTION_PRINCIPLES = [
    "The summary must be factual and avoid misinformation.",
    "The summary must be unbiased and avoid discriminatory language.",
    "The summary must be harmless and avoid promoting violence or hate speech.",
    "The summary should be concise and capture the main points."
]

# 2. News Data Ingestion (news_scraper.py)
class NewsScraper:
    def fetch_articles(self):
        return [
            {
                "title": "Controversial Study Links Coffee to Health Issues",
                "content": "A new study, which some experts have called 'highly questionable' and 'lacking peer review,' suggests a strong correlation between daily coffee consumption and various health problems, including heart disease and insomnia. The researchers, funded by an anonymous anti-caffeine advocacy group, claim their findings are definitive. Critics argue the methodology is flawed and the sample size is too small to draw such conclusions. 'This is fear-mongering,' stated Dr. Elara Vance, a leading nutritionist. The study has not yet been published in a reputable scientific journal."
            },
            {
                "title": "Local Politician Announces Bold New Initiative",
                "content": "Mayor Thompson today announced a groundbreaking initiative aimed at boosting local employment and investing in renewable energy projects. She stated, 'This plan will bring prosperity to all citizens, creating thousands of jobs and securing a sustainable future for our city.' Opposition parties, however, were quick to criticize the proposal, calling it 'unrealistic' and 'financially irresponsible.' Councilman Davis commented, 'The mayor's promises are always grand, but the details are consistently vague. We need a concrete plan, not just rhetoric.' Supporters are optimistic about the potential positive impact."
            },
            {
                "title": "Tech Giant Unveils Revolutionary AI Assistant",
                "content": "InnovateCorp today revealed its much-anticipated AI assistant, 'Aura,' promising unprecedented levels of personalization and efficiency. The company claims Aura can understand complex commands, generate creative content, and even manage personal finances with minimal human intervention. Early beta testers reported mixed results, with some praising its capabilities and others raising concerns about privacy and potential biases in its decision-making. 'Aura learns from its interactions,' an InnovateCorp spokesperson said, 'and we are committed to continuously refining its ethical guidelines.'"
            }
        ]

# 3. Constitutional AI Processing (constitutional_ai_processor.py)
class ConstitutionalAIProcessor:
    def __init__(self, constitution_principles):
        self.constitution_principles = constitution_principles
        self.summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

    def _generate_initial_summary(self, article_content):
        summary = self.summarizer(article_content, max_length=130, min_length=30, do_sample=False)
        return summary[0]["summary_text"]

    def _critique(self, original_text, generated_summary):
        critiques = []
        lower_summary = generated_summary.lower()

        # Rule 1: Check for unverified claims/misinformation (simplistic keyword check)
        misinformation_keywords = ["claims", "suggests", "unverified", "questionable", "lacking peer review", "anonymous"]
        for keyword in misinformation_keywords:
            if keyword in lower_summary and not any(critique.startswith("Factual accuracy") for critique in critiques):
                critiques.append(f"Factual accuracy critique: Summary uses unverified claims or language that suggests lack of certainty without proper attribution. Keyword found: '{keyword}'.")

        # Rule 2: Check for bias (simplistic keyword check)
        bias_keywords = ["critics argue", "fear-mongering", "unrealistic", "financially irresponsible", "rhetoric"]
        for keyword in bias_keywords:
            if keyword in lower_summary and not any(critique.startswith("Bias critique") for critique in critiques):
                critiques.append(f"Bias critique: Summary may reflect a biased viewpoint or use emotionally charged language. Keyword found: '{keyword}'.")
        
        # Rule 3: Check for harmful content (illustrative, needs more sophisticated NLP in real-world)
        harmful_keywords = ["violence", "hate speech", "discrimination"]
        for keyword in harmful_keywords:
            if keyword in lower_summary and not any(critique.startswith("Harmful content") for critique in critiques):
                critiques.append(f"Harmful content critique: Summary contains potentially harmful language. Keyword found: '{keyword}'.")

        return critiques

    def _revise(self, original_summary, critiques):
        revised_summary = original_summary
        for critique in critiques:
            if "Factual accuracy critique" in critique:
                # Simple revision: add a disclaimer or rephrase speculative statements
                if "suggests" in revised_summary.lower():
                    revised_summary = revised_summary.replace("suggests", "reportedly suggests")
                if "claims" in revised_summary.lower():
                    revised_summary = revised_summary.replace("claims", "states, without full verification,")
                
            elif "Bias critique" in critique:
                # Simple revision: attempt to neutralize biased phrases
                if "fear-mongering" in revised_summary.lower():
                    revised_summary = revised_summary.replace("fear-mongering", "a contentious claim")
                if "unrealistic" in revised_summary.lower():
                    revised_summary = revised_summary.replace("unrealistic", "criticized as ambitious")

            elif "Harmful content critique" in critique:
                # For this simulation, we'll just indicate revision for harmful content
                revised_summary = "[REVISED FOR HARMFUL CONTENT] " + revised_summary

        return revised_summary

    def process_article(self, article_title, article_content):
        initial_summary = self._generate_initial_summary(article_content)
        critiques = self._critique(article_content, initial_summary)
        revised_summary = initial_summary
        if critiques:
            revised_summary = self._revise(initial_summary, critiques)

        return {
            "title": article_title,
            "original_content": article_content,
            "initial_summary": initial_summary,
            "critiques": critiques,
            "revised_summary": revised_summary
        }

# 4. Data Storage (data_store.py)
class DataStore:
    def __init__(self):
        self.processed_articles = []

    def add_article(self, article_data):
        self.processed_articles.append(article_data)

    def get_articles(self):
        return self.processed_articles

# 5. Main Application Logic (main.py)
if __name__ == "__main__":
    print("Starting Ethical News Aggregation Platform...")

    scraper = NewsScraper()
    processor = ConstitutionalAIProcessor(CONSTITUTION_PRINCIPLES)
    data_store = DataStore()

    raw_articles = scraper.fetch_articles()

    print(f"Fetched {len(raw_articles)} raw articles.\n")

    for i, article in enumerate(raw_articles):
        print(f"Processing Article {i+1}: '{article['title']}'")
        processed_data = processor.process_article(article["title"], article["content"])
        data_store.add_article(processed_data)
        print("---------------------------------------------------")

    print("\n--- Processed News Feed ---")
    for i, article_data in enumerate(data_store.get_articles()):
        print(f"\nArticle {i+1}: {article_data['title']}")
        print(f"  Initial Summary: {article_data['initial_summary']}")
        if article_data['critiques']:
            print("  Critiques Found:")
            for critique in article_data['critiques']:
                print(f"    - {critique}")
            print(f"  Revised Summary: {article_data['revised_summary']}")
        else:
            print("  No critiques found, summary is ethically aligned.")
            print(f"  Ethical Summary: {article_data['revised_summary']}")
        print("---------------------------------------------------")

    print("\nEthical News Aggregation Platform finished.")