import streamlit as st
import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel
from typing import List, Dict
import time

# --- 1. Pydantic Models ---
class LLMEvaluation(BaseModel):
    persona: str
    evaluation: str

class CredibilityReport(BaseModel):
    overall_summary: str
    individual_evaluations: List[LLMEvaluation]

# --- 2. Persona Definitions ---
PERSONA_DEFINITIONS: Dict[str, str] = {
    "Fact-Checker": (
        "As a meticulous Fact-Checker, your role is to verify all factual claims within the news article. "
        "Identify key assertions and state whether they are verifiable, unsubstantiated, or potentially false, "
        "based on general knowledge or common sense. Do not invent facts, but point out areas needing verification. "
        "Focus purely on objective truth claims."
    ),
    "Bias Analyst": (
        "You are a discerning Bias Analyst. Analyze the article for any signs of political, ideological, or emotional bias. "
        "Look for loaded language, selective reporting, framing choices, or omissions that might sway reader opinion. "
        "Explain what biases you detect and how they manifest in the text."
    ),
    "Ethical Reviewer": (
        "Acting as an Ethical Reviewer, assess the article for ethical considerations. "
        "Consider potential sensationalism, privacy violations, incitement of hatred, accuracy of representation, "
        "or any content that could cause undue harm. Provide an ethical perspective on the content."
    ),
    "Public Opinion Simulator": (
        "As a Public Opinion Simulator, predict how different demographics (e.g., liberal, conservative, youth, elderly) "
        "might perceive this article and its potential impact on public discourse. "
        "Highlight aspects that could be polarizing or widely accepted across different groups."
    ),
    "Historical Contextualizer": (
        "You are a Historical Contextualizer. Provide relevant historical background or precedent to the topic discussed in the article. "
        "Explain how understanding the past sheds light on the current news event and its broader implications. "
        "Connect the dots between historical events and the present narrative."
    ),
}

# --- 3. Simulated LLM Function ---
def simulate_llm_response(persona: str, prompt: str, article_content: str) -> str:
    """
    Simulates an LLM's response based on the persona, prompt, and article content.
    In a real application, this would involve calling an actual LLM API.
    """
    st.spinner(f"\n{persona} thinking...")
    time.sleep(1) # Simulate network latency or processing time

    if persona == "Fact-Checker":
        return (
            f"As a Fact-Checker, I've reviewed the article. Many claims appear to be presented as facts. "
            f"For instance, the statement '{article_content[:100]}...' would require cross-verification with official sources. "
            f"The article lacks specific citations for several key figures, which makes direct fact-checking challenging. "
            f"Overall, the factual basis seems plausible but warrants deeper investigation."
        )
    elif persona == "Bias Analyst":
        return (
            f"My analysis as a Bias Analyst reveals potential framing bias in the article. "
            f"The use of terms like '{article_content[10:50]}...' tends to evoke a specific emotional response. "
            f"There also appears to be a selective focus on certain aspects while downplaying others, "
            f"suggesting a leaning towards a particular narrative. More balanced language could mitigate this bias."
        )
    elif persona == "Ethical Reviewer":
        return (
            f"From an ethical standpoint, this article raises concerns regarding sensationalism. "
            f"The vivid descriptions of '{article_content[20:70]}...' might exploit reader emotions "
            f"rather than providing objective information. There's also a potential for misrepresentation "
            f"if key dissenting voices are omitted. Adherence to journalistic ethics could be improved by focusing on impartiality."
        )
    elif persona == "Public Opinion Simulator":
        return (
            f"Simulating public opinion, I anticipate this article would be well-received by readers aligned with certain viewpoints, "
            f"especially given its focus on '{article_content[30:80]}...'. However, "
            f"others might find it polarizing due to its strong stance. The article has the potential to reinforce existing beliefs "
            f"within specific demographics, while potentially alienating others."
        )
    elif persona == "Historical Contextualizer":
        return (
            f"Putting this article into historical context, the events described, particularly '{article_content[40:90]}...', "
            f"bear resemblance to historical occurrences in [mention a vague historical period/event]. "
            f"Understanding the precedents of such situations can offer valuable insights into the potential long-term "
            f"implications of the current news. History often rhymes, and this article touches on familiar societal dynamics."
        )
    return f"Simulated LLM response for {persona}: This is a general evaluation of the provided content: '{article_content[:150]}...'"

# --- 4. Content Extraction Function ---
def extract_article_text(url: str) -> str:
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Attempt to find common article content containers
        paragraphs = soup.find_all('p')
        article_text = ' '.join([p.get_text() for p in paragraphs])

        if not article_text:
            # Fallback to body text if specific paragraphs are not found
            article_text = soup.body.get_text(separator=' ', strip=True) if soup.body else ''

        # Basic cleaning: remove excessive whitespace and newlines
        article_text = ' '.join(article_text.split())

        # Limit text length to avoid overwhelming simulated LLMs and display
        return article_text[:4000] # Limit to 4000 characters for practicality
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching URL: {e}")
        return ""
    except Exception as e:
        st.error(f"Error parsing article content: {e}")
        return ""

# --- 5. Evaluation Manager Function ---
def run_multi_perspective_evaluation(article_content: str) -> List[LLMEvaluation]:
    evaluations: List[LLMEvaluation] = []
    for persona, instructions in PERSONA_DEFINITIONS.items():
        prompt = f"Persona: {persona}\nInstructions: {instructions}\n\nNews Article:\n{article_content}\n\nYour Evaluation:"
        evaluation_text = simulate_llm_response(persona, prompt, article_content)
        evaluations.append(LLMEvaluation(persona=persona, evaluation=evaluation_text))
    return evaluations

# --- 6. Synthesis Engine Function ---
def synthesize_evaluations(evaluations: List[LLMEvaluation]) -> CredibilityReport:
    # In a real scenario, another LLM could do this synthesis for a more nuanced report.
    # Here, we'll do a simple aggregation.
    summary_parts = []
    for eval_item in evaluations:
        summary_parts.append(f"- The {eval_item.persona} noted: {eval_item.evaluation}")

    overall_summary = (
        "Based on the multi-perspective evaluation, here's a synthesized view:\n\n"
        "The article has been reviewed from several angles, highlighting various aspects of its content, "
        "potential biases, ethical considerations, public reception, and historical relevance. "
        "Each persona offered a unique insight, contributing to a comprehensive understanding of the news piece. "
        "While some perspectives might identify areas for improvement or caution, "
        "the collective analysis aims to provide a well-rounded assessment of its credibility and implications."
    )

    return CredibilityReport(
        overall_summary=overall_summary,
        individual_evaluations=evaluations
    )

# --- 7. Streamlit App ---
st.set_page_config(layout="wide", page_title="AI News Credibility Evaluator")
st.title("📰 AI-Powered News Credibility Evaluator")
st.markdown("--- Developed with Multi-Perspective LLM Evaluation Pattern ---")

st.write(
    "This application employs multiple AI personas to evaluate news articles "
    "from various perspectives, offering a comprehensive credibility report."
)

input_method = st.radio(
    "Choose input method:",
    ("Enter Article Text", "Enter Article URL"),
    index=0,
)

article_content = ""
if input_method == "Enter Article Text":
    article_content = st.text_area(
        "Paste your news article text here:",
        height=300,
        placeholder="E.g., 'Breaking News: New study reveals...'"
    )
else:
    article_url = st.text_input(
        "Enter the news article URL:",
        placeholder="E.g., 'https://www.example.com/news-article'"
    )
    if article_url:
        st.info("Extracting content from URL...")
        article_content = extract_article_text(article_url)
        if article_content:
            st.success("Content extracted successfully!")
            with st.expander("View Extracted Article Text"): # Allow user to see what was extracted
                st.text(article_content)
        else:
            st.warning("Could not extract meaningful content from the URL. Please try pasting the text instead.")


if st.button("Evaluate Article Credibility", type="primary"):
    if not article_content.strip():
        st.error("Please provide news article content to evaluate.")
    else:
        with st.status("Initiating multi-perspective evaluation...", expanded=True) as status:
            st.write("Running individual LLM persona evaluations...")
            individual_evaluations = run_multi_perspective_evaluation(article_content)
            st.write("Synthesizing insights from all perspectives...")
            credibility_report = synthesize_evaluations(individual_evaluations)
            status.update(label="Evaluation Complete!", state="complete", expanded=False)

        st.subheader("📊 Credibility Report")
        st.markdown(credibility_report.overall_summary)

        st.markdown("### Individual Persona Evaluations")
        for eval_item in credibility_report.individual_evaluations:
            with st.expander(f"**{eval_item.persona}**"): # Use expanders for cleaner display
                st.write(eval_item.evaluation)

st.markdown("---")
st.info("Note: LLM responses in this demo are simulated and do not reflect actual AI model outputs.")
