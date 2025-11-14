class LLMAgent:
    """
    Simulates an LLM agent with a specific persona to evaluate text.
    In a real application, this would interface with a live LLM API.
    """
    def __init__(self, name: str, persona_prompt: str):
        self.name = name
        self.persona_prompt = persona_prompt

    def evaluate(self, article_text: str) -> dict:
        """
        Simulates the evaluation of an article based on the agent's persona.
        Returns a dictionary with evaluation details.
        """
        # In a real scenario, this would involve calling an actual LLM
        # with the persona_prompt and article_text.
        # For this simulation, we return a predefined structured response.

        print(f"Agent '{self.name}' (Persona: '{self.persona_prompt[:50]}...') is evaluating...")

        # Simulate different evaluation outputs based on agent persona
        if "fact-checker" in self.name.lower():
            return {
                "agent": self.name,
                "perspective": "Fact-Checking",
                "score": "High reliability, minor claims need verification.",
                "details": "Cross-referenced key statements with common knowledge and found no major discrepancies. Some statistics could benefit from explicit source citation or further investigation."
            }
        elif "readability expert" in self.name.lower():
            return {
                "agent": self.name,
                "perspective": "Readability and Style",
                "score": "Good clarity, slightly formal tone.",
                "details": "Average sentence length is moderate. Jargon use is minimal and explained contextually. Paragraphs are well-structured. Could improve engagement with more active voice and varied sentence structures."
            }
        elif "bias detector" in self.name.lower():
            return {
                "agent": self.name,
                "perspective": "Bias Detection",
                "score": "Neutral tone, no obvious ideological bias detected.",
                "details": "The article appears to present both sides of the issue fairly, avoiding emotionally charged language. Quotes, if present, seem balanced from different viewpoints without overt favoritism."
            }
        elif "ethical reviewer" in self.name.lower():
            return {
                "agent": self.name,
                "perspective": "Ethical Implications",
                "score": "No immediate ethical concerns found.",
                "details": "Respects privacy, avoids sensationalism, and does not promote harmful stereotypes. Focuses on objective reporting without undue speculation or moralizing. No content that could foreseeably cause harm."
            }
        else:
            return {
                "agent": self.name,
                "perspective": "General Evaluation",
                "score": "Generic assessment.",
                "details": "This agent provides a general evaluation without a specific focused perspective, indicating a need for a more defined persona."
            }

    def get_persona_description(self) -> str:
        """Returns a string description of the agent's persona."""
        return f"Agent Name: {self.name}\nPersona: {self.persona_prompt}"


class MultiPerspectiveEvaluator:
    """
    Orchestrates the evaluation of a news article using multiple LLM agents,
    each with a distinct evaluative persona.
    """
    def __init__(self):
        # Initialize LLM agents with specific personas
        self.agents = [
            LLMAgent(
                "Fact-Checker Agent",
                "Act as a meticulous fact-checker. Your primary goal is to verify the accuracy of all factual claims, statistics, and reported events in the news article. Highlight any unverified statements or potential misinformation, and suggest sources for verification."
            ),
            LLMAgent(
                "Readability Expert Agent",
                "Act as a readability expert and editor. Assess the clarity, coherence, grammar, syntax, and overall flow of the article. Provide feedback on how well the article communicates its message to a general audience, focusing on vocabulary, sentence structure, and engagement."
            ),
            LLMAgent(
                "Bias Detector Agent",
                "Act as a neutral bias detector. Analyze the language, framing, selection of information, and overall tone to identify any potential political, social, economic, or cultural biases. Point out instances where the article might be subtly promoting a particular viewpoint or omitting crucial counter-arguments."
            ),
            LLMAgent(
                "Ethical Reviewer Agent",
                "Act as an ethical journalist reviewer. Evaluate the article for adherence to journalistic ethics, including fairness, privacy, impact on vulnerable groups, sensationalism, and potential for harm. Identify any ethical red flags, misrepresentations, or irresponsible reporting."
            )
        ]

    def evaluate_article(self, article_text: str) -> list[dict]:
        """
        Distributes the article to each agent for evaluation and collects their results.
        """
        print("\n--- Starting Multi-Perspective Evaluation ---")
        results = []
        for agent in self.agents:
            evaluation = agent.evaluate(article_text)
            results.append(evaluation)
        print("--- Evaluation Complete ---")
        return results

    def generate_report(self, evaluation_results: list[dict], article_title: str = "News Article") -> str:
        """
        Generates a formatted report from the collected evaluation results.
        """
        report = f"\n### Multi-Perspective Evaluation Report for: '{article_title}' ###\n"
        report += "-------------------------------------------------------------------\n"

        for result in evaluation_results:
            report += f"\nAgent: {result['agent']}\n"
            report += f"Perspective: {result['perspective']}\n"
            report += f"Overall Score: {result['score']}\n"
            report += f"Details: {result['details']}\n"
            report += "-------------------------------------------------------------------\n"

        # Optional: Add a summary or overall conclusion from the combined perspectives
        report += "\n### Overall Summary ###\n"
        report += "This report synthesizes evaluations from multiple specialized agents. Each agent assessed the article based on its unique persona, providing a holistic view of the content's quality, biases, and ethical considerations. The combination of these perspectives aims to offer a more robust and nuanced understanding than a single evaluation."
        report += "\n-------------------------------------------------------------------\n"

        return report


# Example Usage for the Multi-Perspective News Article Quality Evaluator
if __name__ == "__main__":
    sample_article_1 = """
    A groundbreaking study published today by independent researchers reveals a strong correlation between daily consumption of exotic fruits and enhanced cognitive function in adults over 50. Dr. Elena Rodriguez, lead author, stated, "Our findings suggest a revolutionary approach to combating age-related cognitive decline, offering a natural and accessible solution." The study involved 200 participants over six months, with half consuming a specific exotic fruit blend daily. While promising, critics caution against drawing definitive conclusions from a single study, emphasizing the need for larger, long-term trials to validate the results. Funding for the study was provided by the 'Exotic Fruit Growers Association'.
    """

    sample_article_2 = """
    Local elections saw a surprising upset last night as the incumbent mayor was defeated by a newcomer advocating for radical urban development plans. Sarah Chen, the newly elected mayor, promised to transform the city's infrastructure within her first term, including controversial proposals for high-rise residential buildings in historically low-density areas. Supporters laud her vision for a modern metropolis, while opposition groups express deep concerns about gentrification and the potential loss of community character. Public reaction is divided, with protests already planned by neighborhood associations. The city council is expected to face a contentious period ahead.
    """

    evaluator = MultiPerspectiveEvaluator()

    # Evaluate Sample Article 1
    print("\n\n--- Evaluating Sample Article 1 ---")
    results_1 = evaluator.evaluate_article(sample_article_1)
    report_1 = evaluator.generate_report(results_1, "Exotic Fruits & Cognitive Function Study")
    print(report_1)

    # Evaluate Sample Article 2
    print("\n\n--- Evaluating Sample Article 2 ---")
    results_2 = evaluator.evaluate_article(sample_article_2)
    report_2 = evaluator.generate_report(results_2, "Local Elections Upset - Urban Development Plans")
    print(report_2)