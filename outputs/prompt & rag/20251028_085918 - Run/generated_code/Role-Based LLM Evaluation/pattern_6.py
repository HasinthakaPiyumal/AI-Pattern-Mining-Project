import requests
from bs4 import BeautifulSoup
from abc import ABC, abstractmethod
from typing import List, Dict, Literal
from pydantic import BaseModel, Field

# --- 1. Pydantic Models for Structured Output ---

class FactCheckReport(BaseModel):
    factual_claims_identified: List[str] = Field(description="List of key factual claims in the article.")
    verified_claims: Dict[str, bool] = Field(description="Dictionary indicating if a claim was verified (True/False).")
    contradictions_found: List[str] = Field(description="List of contradictions identified.")
    fact_check_score: float = Field(description="Overall fact-check score (0-1.0).")
    reasoning: str = Field(description="Detailed reasoning for the fact-check score.")

class BiasAnalysisReport(BaseModel):
    identified_bias_types: List[str] = Field(description="Types of bias identified (e.g., political, cultural, corporate).")
    biased_language_examples: List[str] = Field(description="Examples of biased language or framing.")
    tone: Literal["positive", "negative", "neutral", "mixed"] = Field(description="Overall tone of the article.")
    bias_score: float = Field(description="Overall bias score (0-1.0, lower is less biased).")
    reasoning: str = Field(description="Detailed reasoning for the bias score.")

class ReadabilityClarityReport(BaseModel):
    flesch_reading_ease: float = Field(description="Flesch Reading Ease score.")
    grade_level: float = Field(description="Estimated U.S. school grade level.")
    clarity_issues: List[str] = Field(description="Specific clarity, grammar, or coherence issues.")
    readability_score: float = Field(description="Overall readability and clarity score (0-1.0).")
    reasoning: str = Field(description="Detailed reasoning for the readability score.")

class EthicalImpactReport(BaseModel):
    ethical_concerns: List[str] = Field(description="List of ethical concerns (e.g., privacy, sensationalism).")
    potential_societal_impact: str = Field(description="Description of potential positive or negative societal impact.")
    responsible_reporting_score: float = Field(description="Score for responsible reporting practices (0-1.0).")
    reasoning: str = Field(description="Detailed reasoning for the ethical and impact score.")

class SentimentAnalysisReport(BaseModel):
    overall_sentiment: Literal["positive", "negative", "neutral", "mixed"] = Field(description="Overall sentiment.")
    sentiment_score: float = Field(description="Numerical sentiment score (-1.0 to 1.0).")
    key_sentiment_phrases: List[str] = Field(description="Phrases contributing most to the sentiment.")
    reasoning: str = Field(description="Detailed reasoning for the sentiment score.")

class SynthesizedReport(BaseModel):
    overall_credibility_score: float = Field(description="Aggregated overall credibility score (0-1.0).")
    summary_of_findings: str = Field(description="Comprehensive summary of all agent findings.")
    recommendations: str = Field(description="Recommendations for the reader regarding the article's credibility.")
    individual_agent_reports: Dict[str, Dict] = Field(description="Dictionary containing each agent's full report.")

# --- 2. News Article Ingester Module ---

class NewsArticleIngester:
    def fetch_article_content(self, url: str) -> str:
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            soup = BeautifulSoup(response.text, 'html.parser')

            # Attempt to find common article content containers
            paragraphs = soup.find_all('p')
            article_text = '\n'.join([p.get_text() for p in paragraphs])

            if not article_text:
                # Fallback for less structured pages, try to get all text
                article_text = soup.get_text(separator=' ', strip=True)

            return self.clean_text(article_text)
        except requests.exceptions.RequestException as e:
            print(f"Error fetching article from {url}: {e}")
            return ""

    def clean_text(self, text_content: str) -> str:
        # Remove multiple newlines and spaces, trim whitespace
        cleaned_text = ' '.join(text_content.split())
        return cleaned_text

# --- 3. LLM Agent Definitions Module ---

class BaseEvaluatorAgent(ABC):
    def __init__(self, persona: str):
        self.persona = persona
        self.llm_model = "SimulatedLLM" # Placeholder for an actual LLM instance

    @abstractmethod
    def evaluate(self, article_text: str) -> Dict:
        pass

class FactCheckerAgent(BaseEvaluatorAgent):
    def __init__(self):
        super().__init__("Fact-Checker Agent: Verifies factual claims and identifies contradictions.")
        self.persona_description = "You are an expert fact-checker. Your task is to meticulously verify factual claims within a news article, identify any inconsistencies or contradictions, and assess the overall factual accuracy. Provide a score from 0.0 to 1.0, where 1.0 is perfectly factually sound." 
        self.evaluation_prompt_template = """
        Article: {article_text}

        Based on the provided article, identify key factual claims, evaluate their veracity (simulated), and note any contradictions. Provide a fact-check score and detailed reasoning.
        Return a JSON object conforming to the FactCheckReport Pydantic model.
        """

    def evaluate(self, article_text: str) -> Dict:
        # Simulate LLM response based on article length or simple keywords
        if len(article_text) < 500:
            score = 0.6
            claims = ["Short article often lacks depth."]
            verified = {"Short article often lacks depth.": False}
            contradictions = []
            reasoning = "The article is quite short, making it difficult to find extensive factual claims to verify. Some general statements were made without direct evidence."
        else:
            score = 0.85
            claims = ["Main claim 1", "Another important fact."]
            verified = {"Main claim 1": True, "Another important fact.": True}
            contradictions = []
            reasoning = "The article presents several verifiable claims which, for the purpose of this simulation, are considered accurate. No obvious contradictions were found."
        
        return FactCheckReport(
            factual_claims_identified=claims,
            verified_claims=verified,
            contradictions_found=contradictions,
            fact_check_score=score,
            reasoning=reasoning
        ).dict()

class BiasAnalystAgent(BaseEvaluatorAgent):
    def __init__(self):
        super().__init__("Bias Analyst Agent: Analyzes language for various forms of bias, tone, and framing.")
        self.persona_description = "You are a sophisticated bias analyst. Your role is to scrutinize the language, tone, and framing of a news article to detect any political, cultural, corporate, or other biases. Assess the neutrality and provide a bias score (0.0 to 1.0, lower is less biased)."
        self.evaluation_prompt_template = """
        Article: {article_text}

        Analyze the article for any forms of bias, including language, tone, and framing. Provide examples and an overall bias score. 
        Return a JSON object conforming to the BiasAnalysisReport Pydantic model.
        """

    def evaluate(self, article_text: str) -> Dict:
        # Simulate LLM response
        if "government" in article_text.lower() and "criticism" in article_text.lower():
            bias_types = ["political"]
            examples = ["uses strong negative verbs when describing government actions."]
            tone = "negative"
            score = 0.7
            reasoning = "The article shows a clear negative leaning against government policies through specific word choices and framing."
        else:
            bias_types = []
            examples = []
            tone = "neutral"
            score = 0.2
            reasoning = "The article appears to maintain a relatively neutral stance without overt signs of strong bias in its language or framing."
        
        return BiasAnalysisReport(
            identified_bias_types=bias_types,
            biased_language_examples=examples,
            tone=tone,
            bias_score=score,
            reasoning=reasoning
        ).dict()

class ReadabilityClarityAgent(BaseEvaluatorAgent):
    def __init__(self):
        super().__init__("Readability & Clarity Expert Agent: Assesses the article's clarity, coherence, grammar, and overall readability.")
        self.persona_description = "You are an expert in communication and linguistics. Evaluate the news article for its clarity, grammatical correctness, sentence structure, and overall readability for a general audience. Provide a score from 0.0 to 1.0, where 1.0 is excellent clarity and readability."
        self.evaluation_prompt_template = """
        Article: {article_text}

        Assess the article's readability, clarity, grammar, and coherence. Provide relevant metrics (simulated) and a readability score. 
        Return a JSON object conforming to the ReadabilityClarityReport Pydantic model.
        """

    def evaluate(self, article_text: str) -> Dict:
        # Simulate LLM response
        # Simple simulation for readability based on length and sentence structure guess
        num_words = len(article_text.split())
        num_sentences = article_text.count('.') + article_text.count('!') + article_text.count('?')

        if num_sentences == 0: # Avoid division by zero
            flesch = 0.0
            grade = 18.0
        else:
            avg_words_per_sentence = num_words / num_sentences
            # Very crude simulation of Flesch and Grade Level
            flesch = max(0, 100 - avg_words_per_sentence * 5 - (num_words / 100) * 10) # Lower avg words per sentence -> higher score
            grade = min(18, max(5, avg_words_per_sentence / 2 + (num_words / 1000))) # Higher avg words per sentence -> higher grade
        
        clarity_issues = []
        if "long and complex sentence" in article_text.lower():
            clarity_issues.append("Some sentences are excessively long and complex.")

        score = (flesch / 100 + (1 if not clarity_issues else 0.5)) / 2 # Combine flesch and clarity issues
        score = max(0.0, min(1.0, score))

        return ReadabilityClarityReport(
            flesch_reading_ease=round(flesch, 2),
            grade_level=round(grade, 1),
            clarity_issues=clarity_issues,
            readability_score=round(score, 2),
            reasoning="Simulated readability metrics suggest the article is generally clear, with some potential for simplification based on length and assumed sentence complexity."
        ).dict()

class EthicalImpactAgent(BaseEvaluatorAgent):
    def __init__(self):
        super().__init__("Ethical & Impact Reviewer Agent: Evaluates potential societal impact and ethical reporting aspects.")
        self.persona_description = "You are an ethical review board member and societal impact analyst. Your task is to evaluate the news article for any ethical concerns (e.g., privacy violations, sensationalism, incitement) and assess its potential positive or negative societal impact. Provide a score for responsible reporting (0.0 to 1.0)."
        self.evaluation_prompt_template = """
        Article: {article_text}

        Evaluate the article for ethical concerns and its potential societal impact. Assign a responsible reporting score.
        Return a JSON object conforming to the EthicalImpactReport Pydantic model.
        """

    def evaluate(self, article_text: str) -> Dict:
        # Simulate LLM response
        ethical_concerns = []
        societal_impact = "Neutral potential impact."
        score = 0.9

        if "sensational" in article_text.lower() or "shocking" in article_text.lower():
            ethical_concerns.append("Potential sensationalism.")
            score -= 0.2
        if "privacy" in article_text.lower() and ("breach" in article_text.lower() or "leak" in article_text.lower()):
            ethical_concerns.append("Raises privacy concerns.")
            societal_impact = "Could lead to public concern over privacy."
            score -= 0.3
        
        score = max(0.0, min(1.0, score))

        return EthicalImpactReport(
            ethical_concerns=ethical_concerns,
            potential_societal_impact=societal_impact,
            responsible_reporting_score=round(score, 2),
            reasoning="Simulated analysis indicates generally responsible reporting, with some minor ethical flags depending on keywords."
        ).dict()

class SentimentAnalystAgent(BaseEvaluatorAgent):
    def __init__(self):
        super().__init__("Sentiment Analyst Agent: Identifies and quantifies the emotional tone and sentiment.")
        self.persona_description = "You are an advanced sentiment analysis engine. Your task is to accurately determine the overall emotional tone and sentiment expressed in the news article. Quantify the sentiment and identify key phrases contributing to it. Provide an overall sentiment (-1.0 to 1.0)."
        self.evaluation_prompt_template = """
        Article: {article_text}

        Identify the overall sentiment (positive, negative, neutral, mixed), a numerical sentiment score, and key phrases. 
        Return a JSON object conforming to the SentimentAnalysisReport Pydantic model.
        """

    def evaluate(self, article_text: str) -> Dict:
        # Simulate LLM response
        sentiment_score = 0.0
        overall_sentiment = "neutral"
        key_phrases = []

        if "positive" in article_text.lower() or "good news" in article_text.lower():
            sentiment_score += 0.5
            key_phrases.append("positive outlook")
        if "negative" in article_text.lower() or "bad news" in article_text.lower() or "crisis" in article_text.lower():
            sentiment_score -= 0.5
            key_phrases.append("negative development")

        if sentiment_score > 0.3:
            overall_sentiment = "positive"
        elif sentiment_score < -0.3:
            overall_sentiment = "negative"
        elif -0.3 <= sentiment_score <= 0.3 and key_phrases:
            overall_sentiment = "mixed" # if some keywords were found, it's mixed
        else:
            overall_sentiment = "neutral"

        return SentimentAnalysisReport(
            overall_sentiment=overall_sentiment,
            sentiment_score=round(sentiment_score, 2),
            key_sentiment_phrases=key_phrases,
            reasoning="Simulated sentiment analysis based on keyword detection and article length."
        ).dict()

# --- 4. Evaluation Orchestrator Module ---

class NewsCredibilityEvaluator:
    def __init__(self, agents: List[BaseEvaluatorAgent]):
        self.agents = agents

    def orchestrate_evaluation(self, article_text: str) -> Dict[str, Dict]:
        all_reports = {}
        for agent in self.agents:
            print(f"[*] {agent.persona} is evaluating...")
            report = agent.evaluate(article_text)
            all_reports[agent.__class__.__name__] = report
        return all_reports

    def synthesize_results(self, individual_reports: Dict[str, Dict]) -> SynthesizedReport:
        # This method would typically use another LLM call or complex logic to synthesize
        # For simulation, we'll do a simple aggregation and generate a summary.

        overall_score_sum = 0.0
        num_scores = 0
        summary_parts = []

        for agent_name, report_dict in individual_reports.items():
            report_model = None
            if agent_name == "FactCheckerAgent": report_model = FactCheckReport(**report_dict)
            elif agent_name == "BiasAnalystAgent": report_model = BiasAnalysisReport(**report_dict)
            elif agent_name == "ReadabilityClarityAgent": report_model = ReadabilityClarityReport(**report_dict)
            elif agent_name == "EthicalImpactAgent": report_model = EthicalImpactReport(**report_dict)
            elif agent_name == "SentimentAnalystAgent": report_model = SentimentAnalysisReport(**report_dict)
            
            if report_model:
                summary_parts.append(f"**{agent_name.replace('Agent', '')} Report:**\n- Score: {getattr(report_model, 'fact_check_score', '') or getattr(report_model, 'bias_score', '') or getattr(report_model, 'readability_score', '') or getattr(report_model, 'responsible_reporting_score', '') or getattr(report_model, 'sentiment_score', '')}\n- Reasoning: {report_model.reasoning}\n")
                
                # Aggregate scores (simple average for simulation)
                if hasattr(report_model, 'fact_check_score'): 
                    overall_score_sum += report_model.fact_check_score
                    num_scores += 1
                elif hasattr(report_model, 'bias_score'): 
                    overall_score_sum += (1 - report_model.bias_score) # Invert bias score for credibility (lower bias -> higher credibility)
                    num_scores += 1
                elif hasattr(report_model, 'readability_score'):
                    overall_score_sum += report_model.readability_score
                    num_scores += 1
                elif hasattr(report_model, 'responsible_reporting_score'):
                    overall_score_sum += report_model.responsible_reporting_score
                    num_scores += 1
                # Sentiment score is +/- 1, not directly a credibility score

        overall_credibility_score = round(overall_score_sum / num_scores, 2) if num_scores > 0 else 0.5
        
        summary_of_findings = "\n".join(summary_parts)
        recommendations = f"Based on the multi-perspective evaluation, this article has an overall credibility score of {overall_credibility_score}. Readers should consider the detailed findings from each expert agent for a comprehensive understanding. Focus particularly on areas highlighted by the Fact-Checker and Bias Analyst agents."

        return SynthesizedReport(
            overall_credibility_score=overall_credibility_score,
            summary_of_findings=summary_of_findings,
            recommendations=recommendations,
            individual_agent_reports=individual_reports
        )

# --- Main Application Logic ---

def main():
    print("\n--- Advanced News Credibility Evaluator ---\n")

    # Example Usage
    article_url = "https://www.bbc.com/news/world-us-canada-67990740"
    # Or provide raw text
    # sample_article_text = "The government announced today a groundbreaking new policy that will unequivocally solve all economic problems. Critics, however, raised some minor concerns which are easily dismissed. This is truly the best news of the year."

    ingester = NewsArticleIngester()
    print(f"[*] Fetching article from: {article_url}")
    article_content = ingester.fetch_article_content(article_url)

    if not article_content:
        print("Error: Could not fetch article content. Exiting.")
        return
    
    print(f"[*] Article content fetched (first 200 chars): {article_content[:200]}...")

    # Initialize agents
    agents = [
        FactCheckerAgent(),
        BiasAnalystAgent(),
        ReadabilityClarityAgent(),
        EthicalImpactAgent(),
        SentimentAnalystAgent()
    ]

    evaluator = NewsCredibilityEvaluator(agents)
    
    print("\n[*] Starting multi-perspective evaluation...")
    individual_reports = evaluator.orchestrate_evaluation(article_content)

    print("\n[*] Synthesizing results...")
    final_report = evaluator.synthesize_results(individual_reports)

    print("\n--- Final Credibility Report ---")
    print(f"Overall Credibility Score: {final_report.overall_credibility_score:.2f}/1.00")
    print("\nSummary of Findings:")
    print(final_report.summary_of_findings)
    print("\nRecommendations:")
    print(final_report.recommendations)
    # Optionally, print full individual reports
    # print("\nFull Individual Reports:")
    # for agent_name, report in final_report.individual_agent_reports.items():
    #     print(f"\n--- {agent_name.replace('Agent', '')} ---")
    #     print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
