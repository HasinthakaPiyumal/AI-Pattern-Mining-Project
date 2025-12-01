from langchain.prompts import PromptTemplate

def mock_llm_response(prompt: str) -> str:
    if "investigative journalist" in prompt:
        return "As an investigative journalist, I'd focus on the sources cited, any potential conflicts of interest, and whether all claims are thoroughly substantiated with evidence. The article seems to rely heavily on official statements without much independent verification."
    elif "opinion columnist" in prompt:
        return "From an opinion columnist's viewpoint, this piece presents a clear narrative, but it lacks a strong, provocative stance. It's informative, but doesn't inspire much thought or debate. I'd add more personal interpretation and perhaps a call to action."
    elif "fact-checker" in prompt:
        return "As a fact-checker, I'd need to verify every numerical claim, quote, and reported event. The article appears generally factual based on the information provided, but without external cross-referencing, a definitive judgment is difficult. Specific figures and dates would be my priority."
    elif "human-interest reporter" in prompt:
        return "From a human-interest perspective, the article misses the emotional impact on individuals. Who are the people affected by this news, and what are their stories? It's too dry and impersonal. I'd seek out interviews and personal anecdotes to bring it to life."
    else:
        return "As a general evaluator, I find this article to be informative but somewhat lacking in depth."

def fetch_news_article() -> str:
    return """Title: Local Council Approves New City Park Development

Date: October 26, 2023

City Hall announced today the approval of a major new park development project in the downtown district. The project, which has been under discussion for over two years, received unanimous approval from the City Council during last night's session. The new park, named "Green Oasis," is expected to cover approximately 10 acres and will feature walking trails, a children's play area, and an outdoor amphitheater. Construction is slated to begin in Spring 2024 and is projected to be completed by late 2025. Funding for the project will come from a combination of municipal bonds and private donations. Councilwoman Sarah Chen stated, "This park will be a vital green space for our residents and a testament to our commitment to urban sustainability." Opponents of the project had raised concerns about its cost and potential impact on local traffic during construction, but these were ultimately outweighed by the perceived benefits.
"""

def preprocess_text(text: str) -> str:
    return text.strip()

def main():
    print("--- AI-Powered News Aggregator with Role-based Evaluation ---")

    # 1. Fetch News Article
    article_content = fetch_news_article()
    print("\nOriginal Article:\n")
    print(article_content)
    print("\n" + "-" * 70 + "\n")

    # 2. Preprocess Text
    processed_article = preprocess_text(article_content)

    # 3. Role-based LLM Evaluation
    journalistic_roles = [
        "investigative journalist",
        "opinion columnist",
        "fact-checker",
        "human-interest reporter",
    ]

    evaluation_prompt_template = PromptTemplate(
        input_variables=["role", "article"],
        template="Act as a {role}. Evaluate the following news article and provide your assessment based on your role's perspective. Focus on what you would typically analyze or look for.\n\nArticle: {article}\n\nYour Evaluation:",
    )

    for role in journalistic_roles:
        print(f"Evaluating as a: {role.replace('-', ' ').title()}")
        prompt = evaluation_prompt_template.format(role=role, article=processed_article)
        evaluation = mock_llm_response(prompt)
        print(f"{evaluation}\n")
        print("\n" + "-" * 70 + "\n")

if __name__ == "__main__":
    main()