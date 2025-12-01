import textwrap

class ConstitutionalAILayer:
    """
    A simplified simulation of a Constitutional AI layer for news content.
    This class takes an initial news summary and a 'constitution' (principles)
    and attempts to "critique" and "revise" the summary to adhere to these principles.
    In a real-world scenario, this would involve further LLM calls with specific
    prompts for critique and revision, possibly iteratively.
    """
    def __init__(self, constitution: list[str]):
        self.constitution = constitution

    def _apply_critique_and_revision_rules(self, summary: str) -> str:
        """
        Simulates the AI-driven critique and revision process based on the constitution.
        This is a highly simplified, rule-based demonstration. In practice, this
        would involve an LLM acting as a 'critic' and then another LLM acting
        as a 'reviser', guided by the constitution.
        """
        revised_summary = summary

        # Rule 1: Ensure factual accuracy, remove speculative language.
        # (Simplified: Remove common speculative phrases)
        speculative_phrases = ["it is believed that ", "sources suggest ", "could potentially ", "might be "]
        for phrase in speculative_phrases:
            revised_summary = revised_summary.replace(phrase, "")

        # Rule 2: Avoid inflammatory or overly emotional language.
        # (Simplified: Replace strong negative adjectives with neutral ones)
        inflammatory_map = {
            "catastrophic": "significant",
            "outrageous": "noteworthy",
            "shocking": "surprising",
            "terrible": "unfavorable"
        }
        for inflammatory, neutral in inflammatory_map.items():
            revised_summary = revised_summary.replace(inflammatory, neutral)

        # Rule 3: Present balanced perspectives when possible.
        # (Simplified: Add a generic reminder for balance if not explicitly present)
        if "however" not in revised_summary.lower() and "on the other hand" not in revised_summary.lower():
            if len(revised_summary.split()) > 20: # Only add if summary is long enough
                 revised_summary += " Efforts are ongoing to gather diverse perspectives."

        # Further processing to clean up double spaces, etc.
        revised_summary = " ".join(revised_summary.split())
        return revised_summary

    def generate_ethically_aligned_summary(self, raw_news_summary: str, query: str = "") -> str:
        """
        Generates an ethically aligned summary by applying the constitutional principles.
        """
        print(f"--- Original Query: {query} ---")
        print(f"--- Initial Raw Summary ---")
        print(textwrap.fill(raw_news_summary, width=80))

        # In a real Constitutional AI system, this would be an iterative process:
        # 1. LLM generates initial response.
        # 2. Another LLM (critic) critiques the response against the constitution.
        # 3. The original LLM revises its response based on the critique and constitution.
        # 4. This process repeats until alignment or a stopping condition.

        # For this simulation, we apply simplified revision rules directly.
        ethically_aligned_summary = self._apply_critique_and_revision_rules(raw_news_summary)

        print(f"\n--- Constitution Used ---")
        for i, principle in enumerate(self.constitution):
            print(f"{i+1}. {principle}")

        print(f"\n--- Ethically Aligned Summary (after critique and revision simulation) ---")
        print(textwrap.fill(ethically_aligned_summary, width=80))

        return ethically_aligned_summary

def main():
    # Define the 'journalistic constitution' for Ethical NewsFeed
    journalistic_constitution = [
        "1. Ensure factual accuracy and remove speculative or unverified claims.",
        "2. Avoid inflammatory, sensational, or overly emotional language.",
        "3. Strive for balanced reporting, presenting multiple perspectives where appropriate.",
        "4. Do not promote hate speech, discrimination, or harmful content.",
        "5. Protect privacy by avoiding unnecessary disclosure of personal information."
    ]

    # Initialize the Constitutional AI layer
    constitutional_ai = ConstitutionalAILayer(journalistic_constitution)

    # Example 1: A potentially biased/sensational summary
    raw_summary_1 = (
        "BREAKING NEWS! A catastrophic economic downturn is believed to be imminent "
        "following shocking new trade policies. Sources suggest millions could potentially "
        "lose their jobs. This outrageous decision by the government will cause terrible "
        "hardship for everyone."
    )
    query_1 = "Summarize the recent economic news."
    constitutional_ai.generate_ethically_aligned_summary(raw_summary_1, query_1)

    print("\n" + "="*80 + "\n")

    # Example 2: Another summary with speculative language
    raw_summary_2 = (
        "The new environmental bill might be able to solve all climate problems, "
        "it is believed that it will usher in an era of unprecedented prosperity. "
        "However, some critics argue its impact could be limited."
    )
    query_2 = "Explain the new environmental bill."
    constitutional_ai.generate_ethically_aligned_summary(raw_summary_2, query_2)

    print("\n" + "="*80 + "\n")

    # Example 3: A relatively neutral summary to see minimal changes
    raw_summary_3 = (
        "The local council approved funding for a new community center. "
        "The project is expected to begin next quarter and aims to provide "
        "various services to residents."
    )
    query_3 = "What's new with the local council?"
    constitutional_ai.generate_ethically_aligned_summary(raw_summary_3, query_3)


if __name__ == "__main__":
    main()