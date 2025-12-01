from llm_mock import MockLLM
from retriever_mock import MockNewsRetriever

class SmartNewsReporter:
    """
    Orchestrates the iterative retrieval augmentation process to generate news articles.
    """
    def __init__(self):
        self.llm = MockLLM()
        self.retriever = MockNewsRetriever()

    def generate_article(self, topic: str, num_sentences: int = 3) -> str:
        """
        Generates a news article by iteratively retrieving and augmenting content.
        """
        article = []
        current_context = ""

        for i in range(num_sentences):
            # Step 1: Generate a temporary sentence (content plan)
            content_plan = self.llm.generate_content_plan(topic, current_context)
            print(f"[Step 1] Content Plan: {content_plan}")

            # Step 2: Retrieve external knowledge using the temporary sentence as a query
            retrieved_knowledge = self.retriever.retrieve(content_plan)
            print(f"[Step 2] Retrieved Knowledge: {retrieved_knowledge}")

            # Step 3: Inject the retrieved knowledge into the temporary sentence
            output_sentence = self.llm.integrate_knowledge(content_plan, retrieved_knowledge)
            print(f"[Step 3] Output Sentence: {output_sentence}\n")

            article.append(output_sentence)
            current_context += " " + output_sentence # Update context for next iteration

        return " ".join(article)

if __name__ == "__main__":
    reporter = SmartNewsReporter()
    print("Generating article on 'AI advancements':")
    article_ai = reporter.generate_article(topic="AI advancements", num_sentences=3)
    print("\n--- Generated Article (AI advancements) ---")
    print(article_ai)

    print("\n\nGenerating article on 'climate change impacts':")
    article_climate = reporter.generate_article(topic="climate change impacts", num_sentences=3)
    print("\n--- Generated Article (climate change impacts) ---")
    print(article_climate)
