class MockLLM:
    def generate(self, prompt: str) -> str:
        if "summarize this in a neutral, objective, and unbiased manner" in prompt:
            return "This is a neutral and objective summary of the provided article, carefully avoiding any biases or personal opinions.\n\n[Simulated summary content based on the input article, but without actual parsing]"
        return "This is a summary without explicit bias mitigation instructions.\n\n[Simulated summary content based on the input article]"


class BiasAwareSummarizer:
    def __init__(self, llm_model):
        self.llm_model = llm_model

    def summarize_article(self, article_text: str) -> str:
        bias_mitigation_instruction = "Please summarize the following news article in a neutral, objective, and unbiased manner, avoiding any stereotypes, personal opinions, or perpetuation of biases present in the original text. Focus purely on the factual content and present it impartially:\n\n"
        prompt = bias_mitigation_instruction + article_text
        summary = self.llm_model.generate(prompt)
        return summary


if __name__ == "__main__":
    mock_llm = MockLLM()
    summarizer = BiasAwareSummarizer(mock_llm)

    sample_article = (
        "A recent report highlighted the growing trend of young people engaging in [activity]. "
        "Critics argue this [activity] is detrimental, leading to [negative consequence], "
        "while proponents emphasize its benefits for [positive aspect]. "
        "The article quoted a 'leading expert' from [organization] stating, 'It's clear that this [activity] will shape the future in [specific way].'"
    )

    print("--- Original Article ---")
    print(sample_article)
    print("\n--- Bias-Aware Summary ---")
    bias_aware_summary = summarizer.summarize_article(sample_article)
    print(bias_aware_summary)

    print("\n--- Example without explicit bias instruction (for comparison) ---")
    # Simulate a direct LLM call without the bias mitigation instruction
    # In a real scenario, you'd have another method or direct LLM call here.
    # For this mock, we're just showing what the mock LLM *would* return if the instruction wasn't there.
    print(mock_llm.generate(sample_article))
