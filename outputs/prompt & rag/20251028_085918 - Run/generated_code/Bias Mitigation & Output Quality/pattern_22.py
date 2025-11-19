
import os
import requests
from dotenv import load_dotenv
from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate, FewShotPromptTemplate, PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_community.llms import FakeListLLM

load_dotenv()

# --- 1. Config (Integrated from config.py) ---

class Config:
    # Mock API key for demonstration purposes
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "mock-openai-key")
    NEWS_API_KEY = os.getenv("NEWS_API_KEY", "mock-news-api-key")

# --- 2. Prompts and Few-Shot Examples (Integrated from prompts/) ---

# Base Prompts
SUMMARY_PROMPT_TEMPLATE = """
Summarize the following news article concisely and neutrally. Focus on the main facts and events.
Article: {article_content}
Summary:
"""

CULTURAL_AWARENESS_PROMPT_TEMPLATE = """
Adapt the following news summary for a {target_culture} audience. Consider relevant cultural nuances, preferred terminology, and areas of interest, while maintaining factual accuracy.
Original Summary: {original_summary}
Cultural Context: {target_culture}
Adapted Summary:
"""

BIAS_ANALYSIS_PROMPT_TEMPLATE = """
Analyze the following news summary for potential biases (e.g., political, nationalistic, emotional language). Identify specific phrases or omissions that suggest bias and explain why.
News Summary: {news_summary}
Bias Analysis:
"""

DEBATE_PROMPT_TEMPLATE = """
Given the claim: "{claim}"

Present evidence and arguments FOR the claim, followed by evidence and arguments AGAINST the claim. Conclude with a balanced perspective.

FOR:
AGAINST:
Balanced Perspective:
"""

# Few-Shot Examples
few_shot_summarization_examples = [
    {
        "article": "Scientists have discovered a new species of deep-sea fish near the Mariana Trench. The fish, named 'Abyssal Glimmerwing', has bioluminescent fins and can withstand extreme pressure. This discovery opens new avenues for marine biology research.",
        "summary": "A new deep-sea fish, 'Abyssal Glimmerwing', with bioluminescent fins and pressure resistance, has been discovered near the Mariana Trench, offering new marine research opportunities."
    },
    {
        "article": "The city council approved a new budget plan today, allocating significant funds to public transport upgrades and renewable energy projects. Critics argue it doesn't address housing shortages effectively, while supporters laud its environmental focus.",
        "summary": "The city council passed a budget prioritizing public transport and renewable energy. It faces criticism for not addressing housing but is praised for its environmental aspects."
    }
]

few_shot_bias_detection_examples = [
    {
        "text": "The incumbent party's flawless economic policies have led to unprecedented prosperity, clearly demonstrating their superior governance.",
        "bias_analysis": "Bias detected: Strong positive political bias. Uses 'flawless,' 'unprecedented prosperity,' and 'superior governance' which are highly subjective and partisan terms."
    },
    {
        "text": "Our glorious nation's athletes effortlessly dominated the global sporting event, once again proving our inherent strength.",
        "bias_analysis": "Bias detected: Nationalistic bias. Phrases like 'glorious nation,' 'effortlessly dominated,' and 'inherent strength' promote national pride over objective reporting of athletic performance."
    }
]

# --- 3. News Fetching Module (news_fetcher.py) ---

class NewsFetcher:
    def fetch_article_content(self, url: str) -> str:
        # In a real application, this would scrape the article or use a news API.
        # For this demo, we'll return a placeholder or a simple mock based on URL.
        print(f"[NewsFetcher] Simulating fetching content from: {url}")
        if "example.com/tech" in url:
            return "\n\nHeadline: Breakthrough in AI Chatbots Achieved\n\nScientists at Tech Innovations Inc. have announced a significant breakthrough in conversational AI, allowing chatbots to understand complex human emotions with 95% accuracy. This development is expected to revolutionize customer service and mental health support. The new model, named 'EmpathyEngine', utilizes a novel neural network architecture combined with advanced sentiment analysis algorithms. Early trials have shown remarkable results in user satisfaction and reduced resolution times for complex queries. The company plans to release a beta version for enterprise clients next quarter.\n\n"
        elif "example.com/politics" in url:
            return "\n\nHeadline: Government Unveils Controversial New Policy\n\nThe ruling party today introduced a contentious new economic policy aimed at boosting national exports. Critics from the opposition warn of potential job losses and increased inflation, describing the measures as 'reckless and short-sighted.' However, government spokespersons assert that the policy will stimulate growth and create long-term stability, citing expert analyses that support their projections. Public opinion remains divided, with protests planned for next week in major cities.\n\n"
        elif "example.com/science" in url:
            return "\n\nHeadline: New Exoplanet Discovered with Potential for Life\n\nAstronomers using the latest generation of space telescopes have identified a new exoplanet, 'Kepler-186f v2', located in the habitable zone of its star. Initial atmospheric analyses suggest the presence of water vapor and a stable temperature range, making it a strong candidate for supporting extraterrestrial life. Further observations are underway to confirm these findings and understand the planet's geological composition. This discovery reignites the search for life beyond Earth.\n\n"
        else:
            return f"Could not fetch content for {url}. This is a mock response.\n\nArticle text: Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."

# --- 4. LLM Service Module (llm_service.py) ---

class LLMService:
    def __init__(self, llm_model: Any = None):
        # Using FakeListLLM for demonstration without actual API calls
        # In a real app, this would be ChatOpenAI or a Hugging Face model
        if llm_model is None:
            self.llm = FakeListLLM(responses=[
                "Mock Summary: This is a neutral summary of the article.",
                "Mock Summary: A culturally adapted summary for the specified region.",
                "Mock Analysis: No significant bias detected in this text.",
                "Mock Analysis: This text shows a slight positive bias towards topic X due to Y.",
                "Mock Fact-Check: FOR: Evidence A. AGAINST: Evidence B. Balanced: Complex issue."
            ])
        else:
            self.llm = llm_model

    def _run_llm_chain(self, prompt_template: PromptTemplate, inputs: Dict[str, Any]) -> str:
        chain = prompt_template | self.llm | StrOutputParser()
        response = chain.invoke(inputs)
        # For FakeListLLM, we need to manually advance responses if multiple calls are made
        if isinstance(self.llm, FakeListLLM):
            self.llm.i += 1 # Manually advance to the next response in the list
            if self.llm.i >= len(self.llm.responses): # Reset if we run out of responses
                self.llm.i = 0
        return response

    def apply_dense_summarization(self, article_content: str, num_ensembles: int = 3) -> str:
        print(f"[LLMService] Applying DENSE summarization with {num_ensembles} ensembles.")
        summaries = []
        for i in range(num_ensembles):
            # Varying prompts slightly or using different few-shot examples for each ensemble member
            # For simplicity, we'll just run the base summary prompt multiple times here,
            # but in a real scenario, few-shot examples or prompt phrasing would be varied.
            if few_shot_summarization_examples and i < len(few_shot_summarization_examples):
                current_examples = [few_shot_summarization_examples[i]]
            else:
                current_examples = few_shot_summarization_examples[:1] if few_shot_summarization_examples else []

            example_prompt = PromptTemplate(input_variables=["article", "summary"], template="Article: {article}\nSummary: {summary}")
            few_shot_prompt = FewShotPromptTemplate(
                examples=current_examples,
                example_prompt=example_prompt,
                prefix="Summarize the following news article. Here are some examples:\n",
                suffix="Article: {article_content}\nSummary:",
                input_variables=["article_content"],
            )
            prompt = ChatPromptTemplate.from_messages([
                ("human", few_shot_prompt.template)
            ])
            
            inputs = {"article_content": article_content}
            summary = self._run_llm_chain(prompt, inputs)
            summaries.append(summary)
            print(f"[LLMService] DENSE ensemble {i+1} generated summary: {summary}")

        # Simple aggregation: taking the first non-empty summary or concatenating
        # A more advanced aggregation might involve voting, averaging, or a meta-LLM.
        aggregated_summary = summaries[0] if summaries else "No summary generated."
        print(f"[LLMService] DENSE aggregated summary: {aggregated_summary}")
        return aggregated_summary

    def get_balanced_demonstrations(self, article_topic: str, all_examples: List[Dict[str, str]]) -> List[Dict[str, str]]:
        print(f"[LLMService] Selecting balanced demonstrations for topic: {article_topic}")
        # This is a simplified example. In reality, you'd have metadata about examples
        # (e.g., political leaning, cultural origin) and select based on that.
        # For demo, we just pick some diverse ones.
        balanced_examples = []
        if article_topic == "politics":
            balanced_examples.extend([ex for ex in all_examples if "city council" in ex.get("article", "").lower()])
        elif article_topic == "science":
            balanced_examples.extend([ex for ex in all_examples if "scientists" in ex.get("article", "").lower()])
        else:
            balanced_examples = all_examples[:2] # Fallback to first two if no specific match

        if not balanced_examples and all_examples:
            balanced_examples = all_examples[:1] # Ensure at least one if possible

        print(f"[LLMService] Selected {len(balanced_examples)} balanced demonstrations.")
        return balanced_examples

    def apply_cultural_awareness(self, original_summary: str, target_culture: str) -> str:
        print(f"[LLMService] Applying cultural awareness for '{target_culture}'.")
        prompt = ChatPromptTemplate.from_template(CULTURAL_AWARENESS_PROMPT_TEMPLATE)
        return self._run_llm_chain(prompt, {"original_summary": original_summary, "target_culture": target_culture})

    def apply_attr_prompt(self, base_text: str, attribute: str, variation: str) -> str:
        print(f"[LLMService] Applying AttrPrompt to vary '{attribute}' with '{variation}'.")
        # For demonstration, we'll simply prepend/append the variation.
        # In a real LLM, the prompt would instruct the LLM to rewrite based on attribute.
        prompt_template = f"""
        Rewrite the following text to emphasize the attribute: {attribute} with a {variation} tone/focus.
        Text: {base_text}
        Rewritten Text:
        """
        prompt = ChatPromptTemplate.from_template(prompt_template)
        return self._run_llm_chain(prompt, {"base_text": base_text, "attribute": attribute, "variation": variation})

    def apply_bias_aware_mitigation(self, text_to_check: str) -> str:
        print("[LLMService] Applying bias-aware mitigation strategy.")
        # Simulate a check and then a re-prompt if bias is detected.
        # In a real scenario, this would involve a multi-step process.
        bias_analysis_prompt = ChatPromptTemplate.from_template(BIAS_ANALYSIS_PROMPT_TEMPLATE)
        analysis_result = self._run_llm_chain(bias_analysis_prompt, {"news_summary": text_to_check})
        print(f"[LLMService] Initial bias analysis: {analysis_result}")

        if "bias detected" in analysis_result.lower():
            print("[LLMService] Bias detected, attempting mitigation with a revised prompt.")
            mitigation_prompt_template = f"""
            The following text has been identified as potentially biased. Please rewrite it to be completely neutral and objective, removing any subjective language or loaded terms.
            Original Text: {text_to_check}
            Revised Neutral Text:
            """
            mitigation_prompt = ChatPromptTemplate.from_template(mitigation_prompt_template)
            mitigated_text = self._run_llm_chain(mitigation_prompt, {"text_to_check": text_to_check})
            print(f"[LLMService] Mitigated text: {mitigated_text}")
            return mitigated_text
        else:
            return text_to_check # No mitigation needed

    def apply_debate_style_fact_checking(self, claim: str) -> str:
        print(f"[LLMService] Applying debate-style fact-checking for claim: '{claim}'.")
        prompt = ChatPromptTemplate.from_template(DEBATE_PROMPT_TEMPLATE)
        return self._run_llm_chain(prompt, {"claim": claim})

# --- 5. Summarization Module (summarizer.py) ---

class Summarizer:
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def summarize_article(self, article_content: str, use_dense: bool = True) -> str:
        print("[Summarizer] Summarizing article.")
        if use_dense:
            return self.llm_service.apply_dense_summarization(article_content)
        else:
            prompt = ChatPromptTemplate.from_template(SUMMARY_PROMPT_TEMPLATE)
            return self.llm_service._run_llm_chain(prompt, {"article_content": article_content})

# --- 6. Bias Analysis Module (bias_analyzer.py) ---

class BiasAnalyzer:
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def analyze_for_bias(self, text: str) -> str:
        print("[BiasAnalyzer] Analyzing text for bias.")
        example_prompt = PromptTemplate(input_variables=["text", "bias_analysis"], template="Text: {text}\nBias Analysis: {bias_analysis}")
        few_shot_prompt = FewShotPromptTemplate(
            examples=few_shot_bias_detection_examples,
            example_prompt=example_prompt,
            prefix="Identify any biases in the following text. Here are some examples:\n",
            suffix="Text: {news_summary}\nBias Analysis:",
            input_variables=["news_summary"],
        )
        prompt = ChatPromptTemplate.from_messages([
            ("human", few_shot_prompt.template)
        ])

        return self.llm_service._run_llm_chain(prompt, {"news_summary": text})

# --- 7. Fact-Checking Module (fact_checker.py) ---

class FactChecker:
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def check_claim(self, claim: str) -> str:
        print("[FactChecker] Checking claim using debate-style aggregation.")
        return self.llm_service.apply_debate_style_fact_checking(claim)

# --- 8. API/CLI Interface (main.py) ---

def main():
    print("\n--- Global News Analysis and Summarization Platform ---\n")

    news_fetcher = NewsFetcher()
    llm_service = LLMService()
    summarizer = Summarizer(llm_service)
    bias_analyzer = BiasAnalyzer(llm_service)
    fact_checker = FactChecker(llm_service)

    # Example 1: Fetch, Summarize (with DENSE), and Culturally Adapt
    print("\n### Example 1: Fetch, Summarize (DENSE), and Culturally Adapt ###")
    article_url_1 = "http://example.com/tech-breakthrough"
    article_content_1 = news_fetcher.fetch_article_content(article_url_1)
    
    initial_summary_1 = summarizer.summarize_article(article_content_1, use_dense=True)
    print(f"Original Summary (DENSE): {initial_summary_1}")

    cultural_adapted_summary_1 = llm_service.apply_cultural_awareness(initial_summary_1, "Japanese")
    print(f"Culturally Adapted Summary (Japanese): {cultural_adapted_summary_1}")

    # Example 2: Summarize (without DENSE), Bias Analysis, and Bias Mitigation
    print("\n### Example 2: Summarize, Bias Analysis, and Mitigation ###")
    article_url_2 = "http://example.com/politics-controversy"
    article_content_2 = news_fetcher.fetch_article_content(article_url_2)

    initial_summary_2 = summarizer.summarize_article(article_content_2, use_dense=False)
    print(f"Original Summary: {initial_summary_2}")

    bias_analysis_result_2 = bias_analyzer.analyze_for_bias(initial_summary_2)
    print(f"Bias Analysis Result: {bias_analysis_result_2}")
    
    mitigated_summary_2 = llm_service.apply_bias_aware_mitigation(initial_summary_2) # This will re-analyze and mitigate if needed
    print(f"Mitigated Summary: {mitigated_summary_2}")

    # Example 3: Fact-Checking with Debate-Style Evidence Aggregation
    print("\n### Example 3: Fact-Checking ###")
    claim_3 = "AI will replace all human jobs by 2030."
    fact_check_result_3 = fact_checker.check_claim(claim_3)
    print(f"Fact Check for '{claim_3}': {fact_check_result_3}")

    # Example 4: AttrPrompt for varying output style
    print("\n### Example 4: AttrPrompt ###")
    base_text_4 = "The new government policy aims to improve public health."
    varied_text_4_optimistic = llm_service.apply_attr_prompt(base_text_4, "tone", "optimistic")
    print(f"Optimistic rewrite: {varied_text_4_optimistic}")

    varied_text_4_skeptical = llm_service.apply_attr_prompt(base_text_4, "tone", "skeptical")
    print(f"Skeptical rewrite: {varied_text_4_skeptical}")

    # Example 5: Selecting Balanced Demonstrations (demonstrated within DENSE/Bias analysis logic)
    print("\n### Example 5: Demonstrating Balanced Demonstrations Selection ###")
    # This is implicitly handled when fetching few-shot examples inside LLMService methods.
    # For a direct call:
    selected_demos = llm_service.get_balanced_demonstrations( "science", few_shot_summarization_examples)
    print(f"Selected balanced demos for science: {[d['article'][:30] + '...' for d in selected_demos]}")


if __name__ == "__main__":
    main()
