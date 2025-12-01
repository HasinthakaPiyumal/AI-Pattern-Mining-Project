class PromptConstructor:
    def create_unbiased_summary_prompt(self, article_text):
        bias_mitigation_instruction = "Please provide a neutral and unbiased summary/rephrasing of the following text, avoiding stereotypes, subjective interpretations, or any form of bias."
        prompt = f"{bias_mitigation_instruction}\n\nArticle: {article_text}\n\nSummary:"
        return prompt

class LLMClient:
    def generate_text(self, prompt):
        print(f"\n--- Mock LLM Call ---")
        print(f"Prompt sent to LLM:\n{prompt[:200]}...")
        if "example of an aggressive executive" in prompt:
            return "The executive, known for their decisive leadership, implemented a new strategy to improve company performance. This move was met with varied reactions from stakeholders, highlighting the challenges of corporate restructuring."
        elif "benefits of a plant-based diet" in prompt:
            return "A summary of the article discusses the potential health advantages associated with consuming a plant-based diet, such as improved cardiovascular health and weight management, based on research findings. It also notes considerations for ensuring nutritional completeness."
        else:
            return f"[MOCK] Unbiased summary of the article content based on the prompt instructions: {prompt.split('Article:')[-1].split('Summary:')[0].strip()}"

class NewsProcessor:
    def __init__(self, prompt_constructor, llm_client):
        self.prompt_constructor = prompt_constructor
        self.llm_client = llm_client

    def process_article(self, article_text):
        prompt = self.prompt_constructor.create_unbiased_summary_prompt(article_text)
        unbiased_content = self.llm_client.generate_text(prompt)
        return unbiased_content

class Reporter:
    def display_results(self, original_article, processed_content):
        print(f"\n====================================================")
        print(f"Original Article:\n{original_article}")
        print(f"----------------------------------------------------")
        print(f"Bias-Mitigated Summary/Rephrasing:\n{processed_content}")
        print(f"====================================================")

if __name__ == "__main__":
    sample_articles = [
        {
            "title": "Tech CEO's Aggressive Stance on Competition",
            "content": "The CEO of a leading tech firm made headlines today for their unusually aggressive stance on competitors, vowing to 'crush anyone who stands in our way.' Critics are calling this an example of an aggressive executive mentality prevalent in the industry."
        },
        {
            "title": "New Study on Diet Trends",
            "content": "A recent study highlights the increasing popularity and purported benefits of a plant-based diet, claiming it leads to superior health outcomes compared to traditional diets. Some nutritionists are concerned about potential nutritional deficiencies."
        },
        {
            "title": "Local Politician's Controversial Statement",
            "content": "A local politician sparked outrage with a controversial statement regarding immigrant communities, suggesting they are solely responsible for recent economic downturns. This statement has been widely condemned as divisive and prejudiced."
        }
    ]

    prompt_constructor = PromptConstructor()
    llm_client = LLMClient()
    news_processor = NewsProcessor(prompt_constructor, llm_client)
    reporter = Reporter()

    print("\n--- Running News Aggregation with Bias Mitigation ---")

    for article_data in sample_articles:
        original_content = article_data["content"]
        processed_summary = news_processor.process_article(original_content)
        reporter.display_results(original_content, processed_summary)

    print("\n--- News Aggregation Complete ---")