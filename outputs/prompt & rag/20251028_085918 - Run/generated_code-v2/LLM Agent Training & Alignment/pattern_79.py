class LLMCurator:
    def __init__(self):
        pass

    def summarize(self, article):
        return f"Summary of '{article['title']}': This article discusses {article['topic']} and its implications. It highlights key points related to {article['keywords'][0]} and {article['keywords'][1]}."

    def recommend(self, article):
        return f"Recommended reading based on '{article['title']}': Explore more about {article['related_topics'][0]} and recent developments in {article['related_topics'][1]}."

    def generate_content(self, article, introduce_violations=False):
        summary = self.summarize(article)
        recommendations = self.recommend(article)
        generated = f"News Summary: {summary}\nRecommendations: {recommendations}"

        if introduce_violations:
            if 'misinformation' in article['issue']:
                generated = generated.replace(article['fact'], "This is a fabricated fact that is demonstrably false.")
            if 'bias' in article['issue']:
                generated = generated + "\nBias Alert: This content explicitly favors one political ideology over others."
            if 'harmful' in article['issue']:
                generated = generated + "\nHarmful Content Warning: This content includes insensitive language and promotes stereotypes."
            if 'privacy' in article['issue']:
                generated = generated + "\nPrivacy Breach: This content reveals personal identifiable information of an individual."

        return generated


class AICritic:
    def __init__(self):
        self.constitution = [
            "Factuality: Content must be accurate and verifiable.",
            "Impartiality: Content should avoid undue bias or favoritism.",
            "Harmlessness: Content must not promote hate speech, violence, or discrimination.",
            "Privacy Protection: Content must respect individual privacy and data protection.",
            "Helpfulness: Content should be informative and useful to the user."
        ]

    def critique(self, content):
        violations = []
        if "fabricated fact" in content.lower() or "demonstrably false" in content.lower():
            violations.append("Factuality violation: Content contains inaccurate information.")
        if "explicitly favors one political ideology" in content.lower() or "undue bias" in content.lower():
            violations.append("Impartiality violation: Content exhibits bias.")
        if "insensitive language" in content.lower() or "promotes stereotypes" in content.lower():
            violations.append("Harmlessness violation: Content contains harmful elements.")
        if "personal identifiable information" in content.lower() or "privacy breach" in content.lower():
            violations.append("Privacy Protection violation: Content infringes on privacy.")

        return violations


class AIRevisor:
    def __init__(self):
        pass

    def revise(self, content, violations):
        revised_content = content
        for violation in violations:
            if "factuality" in violation.lower():
                revised_content = revised_content.replace("This is a fabricated fact that is demonstrably false.", "[Fact corrected: Information reviewed and verified for accuracy.]")
            if "impartiality" in violation.lower():
                revised_content = revised_content.replace("Bias Alert: This content explicitly favors one political ideology over others.", "[Bias removed: Content revised for impartiality.]")
            if "harmlessness" in violation.lower():
                revised_content = revised_content.replace("Harmful Content Warning: This content includes insensitive language and promotes stereotypes.", "[Harmful content removed: Language revised to be respectful.]")
            if "privacy protection" in violation.lower():
                revised_content = revised_content.replace("Privacy Breach: This content reveals personal identifiable information of an individual.", "[Privacy breach rectified: Personal data removed.]")
        return revised_content


if __name__ == "__main__":
    print("Ethical News Feed Curator Demonstration")

    # Mock News Articles
    mock_articles = [
        {
            "title": "New Study on Climate Change",
            "topic": "environmental science",
            "keywords": ["global warming", "policy changes"],
            "related_topics": ["renewable energy", "carbon footprint"],
            "fact": "Average global temperatures have risen by 1.2 degrees Celsius since pre-industrial times.",
            "issue": []
        },
        {
            "title": "Political Election Debate",
            "topic": "politics",
            "keywords": ["election", "candidates"],
            "related_topics": ["voter turnout", "campaign finance"],
            "fact": "Candidate A has proposed a new economic stimulus plan.",
            "issue": ["bias"]
        },
        {
            "title": "Celebrity Scandal Update",
            "topic": "entertainment",
            "keywords": ["celebrity", "gossip"],
            "related_topics": ["pop culture", "media ethics"],
            "fact": "A source close to the celebrity confirmed the details.",
            "issue": ["misinformation", "privacy"]
        },
        {
            "title": "Controversial Opinion Piece",
            "topic": "social issues",
            "keywords": ["social justice", "community"],
            "related_topics": ["activism", "public discourse"],
            "fact": "A recent survey showed strong public support for the controversial policy.",
            "issue": ["harmful"]
        }
    ]

    curator = LLMCurator()
    critic = AICritic()
    revisor = AIRevisor()

    for i, article in enumerate(mock_articles):
        print(f"\n--- Processing Article {i+1}: '{article['title']}' ---")

        # Generate initial content, potentially with violations
        initial_content = curator.generate_content(article, introduce_violations=True if article['issue'] else False)
        print("\nInitial Generated Content (potentially unethical):\n", initial_content)

        # Critique the content
        violations = critic.critique(initial_content)
        if violations:
            print("\nIdentified Ethical Violations:")
            for v in violations:
                print(f"- {v}")

            # Revise the content
            revised_content = revisor.revise(initial_content, violations)
            print("\nRevised Content (ethically aligned):\n", revised_content)
        else:
            print("\nNo ethical violations identified. Content is already aligned.")
