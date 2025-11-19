
import random
import json

# Mock LLM for demonstration purposes
class MockLLM:
    def __init__(self, model_name="mock-llm"):
        self.model_name = model_name

    def invoke(self, prompt: str):
        # Simulate LLM response based on prompt content
        if "summary of" in prompt.lower():
            if "cultural context" in prompt.lower():
                return f"This is a culturally adapted summary for '{prompt[prompt.find('context for ')+len('context for '):prompt.find('.')].strip()}' considering local customs and values."
            elif "pros and cons" in prompt.lower() or "balanced overview" in prompt.lower():
                return f"Here's a balanced view on the topic. Pros: [simulated pro argument]. Cons: [simulated con argument]."
            else:
                return f"This is a general summary of the article based on: {prompt}"
        elif "headline" in prompt.lower():
            return f"Mock Headline: {prompt[prompt.find('text:')+len('text:'):].strip()[:50]}..."
        return "Mock LLM response for: " + prompt[:100] + "..."


class NewsAggregator:
    def __init__(self, llm):
        self.llm = llm

    def fetch_articles(self):
        # Simulate fetching articles
        articles = [
            {"title": "Local Election Results", "content": "Detailed analysis of recent local election outcomes, voter turnout, and candidate statements."},
            {"title": "New Tech Gadget Launch", "content": "Review of the latest smartphone, focusing on its camera, battery life, and innovative features."},
            {"title": "Global Climate Summit Progress", "content": "Updates from the international climate conference, discussing new pledges and challenges in emission reduction."},
        ]
        return articles

    def _call_llm(self, prompt_template: str, article_content: str, **kwargs):
        full_prompt = prompt_template.format(article_content=article_content, **kwargs)
        return self.llm.invoke(full_prompt)

    def generate_summary_dense(self, article_content: str) -> str:
        """
        Applies Demonstration Ensembling (DENSE) conceptually.
        Generates summaries using slightly varied prompts (different "demonstrations" or instructions)
        and aggregates them (here, simply taking the first for brevity, but in real DENSE,
        one would aggregate outputs, e.g., by voting or averaging).
        """
        prompts = [
            "Summarize the following article concisely: {article_content}",
            "Provide a brief overview of the main points in this article: {article_content}",
            "Extract the key information from the following text: {article_content}"
        ]
        
        all_summaries = []
        for prompt_template in prompts:
            summary = self._call_llm(prompt_template, article_content)
            all_summaries.append(summary)
        
        # In a real DENSE implementation, aggregation logic would go here.
        # For demonstration, we'll just return a concatenated version or the first one.
        return f"DENSE Aggregated Summary (showing first output):\n{all_summaries[0]}\n"

    def generate_culturally_aware_summary(self, article_content: str, cultural_context: str) -> str:
        """
        Applies Cultural Awareness pattern by injecting specific cultural instructions into the prompt.
        """
        prompt_template = (
            "Summarize the following article, adapting the language and focus to be culturally relevant "
            "for a {cultural_context} audience. Ensure sensitivity to local customs and values. "
            "Article: {article_content}"
        )
        return self._call_llm(prompt_template, article_content, cultural_context=cultural_context)

    def generate_balanced_overview(self, article_content: str) -> str:
        """
        Applies Debate-Style Evidence Aggregation.
        Prompts the LLM to provide arguments both for and against a potential claim or topic
        derived from the article. This also implicitly touches on "Selecting Balanced Demonstrations"
        by aiming for a neutral, comprehensive perspective.
        """
        prompt_template = (
            "Analyze the following article and present a balanced overview, highlighting both potential "
            "positive aspects/arguments and negative aspects/counter-arguments related to the main topic. "
            "Article: {article_content}"
        )
        return self._call_llm(prompt_template, article_content)

    def run(self):
        print("Fetching articles...")
        articles = self.fetch_articles()

        for i, article in enumerate(articles):
            print(f"\n--- Article {i+1}: {article['title']} ---")

            # Demonstration Ensembling (DENSE)
            dense_summary = self.generate_summary_dense(article["content"])
            print(f"DENSE Summary: {dense_summary}")

            # Cultural Awareness
            cultural_context = random.choice(["European", "Asian", "North American"]) # Example contexts
            culturally_aware_summary = self.generate_culturally_aware_summary(article["content"], cultural_context)
            print(f"Culturally Aware Summary ({cultural_context} context): {culturally_aware_summary}")

            # Debate-Style Evidence Aggregation & Balanced Demonstrations (conceptual)
            balanced_overview = self.generate_balanced_overview(article["content"])
            print(f"Balanced Overview (Debate-Style): {balanced_overview}")

            print("-" * 50)

if __name__ == "__main__":
    mock_llm = MockLLM()
    aggregator = NewsAggregator(mock_llm)
    aggregator.run()
