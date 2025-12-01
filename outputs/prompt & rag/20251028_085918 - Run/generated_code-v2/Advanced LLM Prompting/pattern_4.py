import collections
import random

# --- Mock LLM Component ---
def mock_llm_generate(prompt: str) -> str:
    """
    A mock Large Language Model function for demonstration purposes.
    In a real application, this would call an actual LLM API.
    """
    if "summarize" in prompt.lower() or "summary" in prompt.lower() or "extract the main points" in prompt.lower():
        # Extract the review part from the prompt
        # This is a very simplified extraction and might need refinement for complex prompts
        review_start_idx = prompt.find("Review: \"")
        if review_start_idx == -1:
            review_start_idx = prompt.find("review: \"")
        
        review_content = ""
        if review_start_idx != -1:
            review_start = review_start_idx + len("Review: \"")
            review_end = prompt.find("\"", review_start)
            if review_end != -1:
                review_content = prompt[review_start:review_end].strip()

        if review_content:
            # Simulate summarization based on review content length
            if len(review_content) > 70:
                return f"Summary: Key aspects include: {review_content[:60]}... (Prompt Mined)"
            else:
                return f"Summary: {review_content}... (Prompt Mined)"
    return f"Response: {prompt[:100]}... (Mock LLM generic response)"

# --- Prompt Mining and Summarization Application ---
class ProductReviewSummarizer:
    def __init__(self, llm_model_fn):
        self.llm_model_fn = llm_model_fn
        self.mined_prompt_templates = []
        self._corpus_data = []

    def _load_data_corpus(self):
        """
        Simulates loading a corpus of product reviews and their desired summaries.
        In a real scenario, this would load from a database or file.
        """
        self._corpus_data = [
            ("This product is amazing! The battery life is fantastic, and it's very easy to use. Highly recommend.",
             "Fantastic battery life, easy to use, highly recommended."),
            ("I received the item quickly, but it broke after only a week. Very disappointed with the quality.",
             "Quick delivery, but poor quality; broke quickly."),
            ("Good value for money. The screen is clear, and it performs well for basic tasks. Not for heavy gaming.",
             "Good value, clear screen, performs well for basic tasks. Not for gaming."),
            ("The sound quality is superb, but the fit is a bit uncomfortable after long periods.",
             "Superb sound quality, uncomfortable fit for long periods."),
            ("The setup was a nightmare, took hours. But once it was working, it performed flawlessly. Mixed feelings.",
             "Difficult setup, but flawless performance once working. Mixed feelings.")
        ]

    def _mine_prompts(self):
        """
        Analyzes the corpus of summaries to discover frequently occurring phrases
        that can serve as optimal 'middle words' or introductory elements for prompt templates.
        This simulates the 'Prompt Mining' pattern.
        """
        all_summary_phrases = []
        for _, summary in self._corpus_data:
            # Simple tokenization: split by spaces, remove punctuation for basic phrase extraction
            words = [word.strip(".,!?;:") for word in summary.lower().split() if word.strip(".,!?;:")]

            # Extract potential prompt components: first word, and common bigrams/trigrams
            if words:
                all_summary_phrases.append(words[0]) # First word of summary
            for i in range(len(words) - 1):
                all_summary_phrases.append(" ".join(words[i:i+2])) # Bigrams
            if len(words) >= 3:
                for i in range(len(words) - 2):
                    all_summary_phrases.append(" ".join(words[i:i+3])) # Trigrams

        # Count frequencies of extracted phrases
        phrase_counts = collections.Counter(all_summary_phrases)

        # Select a few top phrases to integrate into prompt templates
        # Filtering for phrases that seem useful as prompt starters/connectors
        useful_phrases = []
        for phrase, count in phrase_counts.most_common(10):
            # Heuristic: filter out very short or uninformative phrases
            if len(phrase.split()) > 1 and count > 1 and phrase not in ["a", "the", "is", "and", "for"]: # simple filter
                useful_phrases.append(phrase)
        
        if not useful_phrases:
            useful_phrases = ["key points", "in summary", "main highlights"]

        # Generate various prompt templates incorporating mined phrases
        base_templates = [
            "Please provide a concise summary of the following product review:\n\nReview: \"{review_text}\"\n\nSummary:",
            "Summarize the essence of this product review:\n\nReview: \"{review_text}\"\n\nKey takeaways:",
            "Given the product review: \"{review_text}\". What are the crucial points?",
        ]

        self.mined_prompt_templates.extend(base_templates)

        # Add templates enriched with mined patterns
        for phrase in useful_phrases:
            self.mined_prompt_templates.append(
                f"Extract the main points from this product review: \"{{review_text}}\"\n\n{phrase.capitalize()} from the review:"
            )
            self.mined_prompt_templates.append(
                f"Review: \"{{review_text}}\"\n\n{phrase.capitalize()}: Summarize this product review."
            )
        
        # Remove duplicates and ensure at least one template exists
        self.mined_prompt_templates = list(set(self.mined_prompt_templates))
        if not self.mined_prompt_templates:
            self.mined_prompt_templates.append("Please summarize the product review: \"{review_text}\"")

        print(f"Mined {len(self.mined_prompt_templates)} unique prompt templates for summarization.")

    def summarize_review(self, review_text: str) -> str:
        """
        Summarizes a given product review using one of the mined prompt templates
        and the integrated LLM.
        """
        if not self.mined_prompt_templates:
            raise ValueError("No prompt templates mined. Run _mine_prompts() first.")

        # Randomly select a mined prompt template for demonstration.
        # In a real system, advanced techniques (e.g., A/B testing, reinforcement learning) 
        # would be used to select the optimal template for a given review or task.
        selected_template = random.choice(self.mined_prompt_templates)

        # Format the prompt with the review text
        formatted_prompt = selected_template.format(review_text=review_text)

        print(f"\n--- Selected Prompt Template (Mined) ---\n{selected_template}")
        print(f"--- Formatted Prompt (Truncated for display) ---\n{formatted_prompt[:300]}...") # Truncate for display

        # Call the LLM with the formatted prompt
        summary = self.llm_model_fn(formatted_prompt)
        return summary

# --- Main execution block ---
if __name__ == "__main__":
    print("Starting Product Review Summarization with Prompt Mining application...")

    # Initialize the summarizer with the mock LLM
    summarizer_app = ProductReviewSummarizer(llm_model_fn=mock_llm_generate)

    # Step 1: Load simulated data corpus
    print("\nStep 1: Loading simulated data corpus...")
    summarizer_app._load_data_corpus()
    print(f"Loaded {len(summarizer_app._corpus_data)} review-summary pairs.")

    # Step 2: Mine prompts from the corpus
    print("\nStep 2: Mining prompt structures from the corpus...")
    summarizer_app._mine_prompts()

    # Step 3: Demonstrate summarization for new reviews using mined prompts
    print("\nStep 3: Demonstrating summarization for new reviews using mined prompts:")

    new_review_1 = "The customer service was excellent, they resolved my issue very quickly. However, the product itself feels a bit flimsy. Overall satisfied because of the support."
    print(f"\nReview 1: \"{new_review_1}\"\n")
    summary_1 = summarizer_app.summarize_review(new_review_1)
    print(f"\nGenerated Summary 1: {summary_1}")

    new_review_2 = "Absolutely love this gadget! It's compact, stylish, and works flawlessly. A must-have for tech enthusiasts. The battery lasts forever!"
    print(f"\nReview 2: \"{new_review_2}\"\n")
    summary_2 = summarizer_app.summarize_review(new_review_2)
    print(f"\nGenerated Summary 2: {summary_2}")

    new_review_3 = "Initially, I had high hopes, but the software is buggy and crashes frequently. Needs a lot of updates. The hardware is decent though."
    print(f"\nReview 3: \"{new_review_3}\"\n")
    summary_3 = summarizer_app.summarize_review(new_review_3)
    print(f"\nGenerated Summary 3: {summary_3}")

    new_review_4 = "This oven is incredible! Heats up fast, cooks evenly, and cleaning is a breeze. It's transformed my cooking experience. Best purchase this year!"
    print(f"\nReview 4: \"{new_review_4}\"\n")
    summary_4 = summarizer_app.summarize_review(new_review_4)
    print(f"\nGenerated Summary 4: {summary_4}")

    new_review_5 = "The headphones have excellent noise cancellation, but the earcups are too small for my ears, causing discomfort after an hour. Sound quality is top-notch."
    print(f"\nReview 5: \"{new_review_5}\"\n")
    summary_5 = summarizer_app.summarize_review(new_review_5)
    print(f"\nGenerated Summary 5: {summary_5}")