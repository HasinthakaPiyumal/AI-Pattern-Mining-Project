class FairNewsSummarizer:
    def __init__(self):
        self.bias_mitigation_instruction = (
            "Summarize the following news article objectively, ensuring to avoid any personal biases, stereotypes, or preferential treatment of any group or viewpoint. "
            "Present all key arguments and facts impartially."
        )

    def create_bias_mitigated_prompt(self, article_text: str) -> str:
        return f"{self.bias_mitigation_instruction}\n\nArticle:\n{article_text}"

    def get_llm_summary(self, prompt: str) -> str:
        # In a real application, this would involve an API call to an LLM service.
        # For this demonstration, we return a simulated unbiased summary.
        # The actual summary content would depend on the LLM's processing of the prompt.
        return "[Simulated LLM Output]: This is an objectively summarized version of the news article, free from biases and presenting all key viewpoints fairly, as per the instruction provided in the prompt."

    def summarize_article(self, article_text: str) -> str:
        prompt = self.create_bias_mitigated_prompt(article_text)
        summary = self.get_llm_summary(prompt)
        return summary

if __name__ == "__main__":
    summarizer = FairNewsSummarizer()

    sample_article_1 = (
        "A recent economic report indicated mixed signals for the global market. "
        "Some analysts predict a strong recovery in the tech sector due to new innovations, "
        "while others warn of potential inflation risks impacting consumer spending. "
        "Government officials have expressed optimism about long-term growth, "
        "but labor unions are calling for stronger protections for workers in vulnerable industries."
    )

    sample_article_2 = (
        "A debate is ongoing regarding the new environmental regulations. "
        "Environmental groups laud the stricter rules as essential for planet health, citing scientific consensus on climate change. "
        "Industry representatives, however, argue that the regulations will stifle economic growth and lead to job losses, "
        "proposing alternative market-based solutions. "
        "Local community leaders are concerned about the impact on small businesses and farming."
    )

    print("\n--- Summarizing Sample Article 1 ---")
    summary_1 = summarizer.summarize_article(sample_article_1)
    print(f"Original Article:\n{sample_article_1}\n")
    print(f"Bias-Mitigated Summary:\n{summary_1}")

    print("\n--- Summarizing Sample Article 2 ---")
    summary_2 = summarizer.summarize_article(sample_article_2)
    print(f"Original Article:\n{sample_article_2}\n")
    print(f"Bias-Mitigated Summary:\n{summary_2}")