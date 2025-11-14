import autogen
import os

# --- Configuration --- #
# It's recommended to set your OpenAI API key as an environment variable (OPENAI_API_KEY)
# or replace 'os.getenv("OPENAI_API_KEY")' with your actual key.
config_list = [
    {
        "model": "gpt-4-turbo", # Or "gpt-3.5-turbo" for a cheaper option
        "api_key": os.getenv("OPENAI_API_KEY"),
    }
]

llm_config = {
    "timeout": 60,
    "cache_seed": 42, # For reproducible results
    "config_list": config_list,
    "temperature": 0.1,
}

# --- Agent Definitions --- #

# User Proxy Agent: Initiates the conversation and acts as a human user.
user_proxy_agent = autogen.UserProxyAgent(
    name="User_Proxy_Agent",
    human_input_mode="NEVER", # Set to "ALWAYS" or "TERMINATE" for human intervention
    max_consecutive_auto_reply=10, # Allow agents to reply multiple times
    is_termination_msg=lambda x: "content" in x and x["content"] is not None and x["content"].rstrip().endswith("FINAL REPORT") or x["content"].rstrip().endswith("exit"),
    code_execution_config=False, # No code execution needed for this app
)

# Fact Checker Agent: Focuses on factual accuracy.
fact_checker_agent = autogen.AssistantAgent(
    name="Fact_Checker_Agent",
    llm_config=llm_config,
    system_message=(
        "You are a meticulous Fact-Checker. Your role is to critically analyze the provided news article for factual accuracy, "
        "verify sources mentioned, and identify any unsubstantiated claims or misleading information. "
        "Provide a concise assessment of the article's factual integrity. End your assessment with 'Fact Check Complete.'"
    ),
)

# Bias Analyst Agent: Identifies potential biases.
bias_analyst_agent = autogen.AssistantAgent(
    name="Bias_Analyst_Agent",
    llm_config=llm_config,
    system_message=(
        "You are an unbiased Bias Analyst. Evaluate the news article for any overt or subtle biases, "
        "including political, sensationalist, or framing biases. Assess the article's neutrality and "
        "whether it presents a balanced view. Conclude your assessment with 'Bias Analysis Complete.'"
    ),
)

# Literary Critic Agent: Assesses writing style and coherence.
literary_critic_agent = autogen.AssistantAgent(
    name="Literary_Critic_Agent",
    llm_config=llm_config,
    system_message=(
        "You are a discerning Literary Critic. Examine the news article's writing style, tone, "
        "grammatical correctness, logical flow, coherence, and rhetorical effectiveness. "
        "Comment on its overall readability and professionalism. Finish your critique with 'Literary Critique Complete.'"
    ),
)

# Reader Perspective Agent: Evaluates clarity and potential for misinterpretation.
reader_perspective_agent = autogen.AssistantAgent(
    name="Reader_Perspective_Agent",
    llm_config=llm_config,
    system_message=(
        "You represent the average reader. Evaluate the news article's clarity, ease of understanding, "
        "and whether it effectively conveys its message to a broad audience. Identify any parts that "
        "might be confusing or open to misinterpretation. Conclude your thoughts with 'Reader Perspective Complete.'"
    ),
)

# Reporter Agent: Synthesizes all assessments into a comprehensive report.
reporter_agent = autogen.AssistantAgent(
    name="Reporter_Agent",
    llm_config=llm_config,
    system_message=(
        "You are the chief Credibility Reporter. Your task is to read through all the assessments "
        "from the Fact-Checker, Bias Analyst, Literary Critic, and Reader Perspective agents. "
        "Based on their collective input, synthesize a single, comprehensive credibility report for the news article. "
        "Highlight key findings from each perspective and provide an overall credibility score or judgment. "
        "Ensure your report is well-structured and easy to understand. End your report with 'FINAL REPORT'."
    ),
)

# --- Group Chat Setup --- #

groupchat = autogen.GroupChat(
    agents=[user_proxy_agent, fact_checker_agent, bias_analyst_agent, literary_critic_agent, reader_perspective_agent, reporter_agent],
    messages=[],
    max_round=15, # Allow enough rounds for discussion and reporting
    speaker_selection_method="auto", # Let autogen decide who speaks next based on recent messages and system prompts
    allow_repeat_speaker=True # Allow agents to speak multiple times if needed
)

manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

# --- Evaluation Function --- #
def evaluate_news_article(news_article_text: str) -> str:
    """
    Evaluates a news article using multiple LLM agents and returns a comprehensive credibility report.
    """
    print("\n--- Initiating News Credibility Evaluation ---")
    print("Article:", news_article_text[:200] + "..." if len(news_article_text) > 200 else news_article_text)
    
    chat_result = user_proxy_agent.initiate_chat(
        manager,
        message=f"Please evaluate the following news article for credibility: \n\n{news_article_text}",
    )

    # The final message from the reporter should be the comprehensive report
    # We need to extract the last message from the reporter that contains 'FINAL REPORT'
    final_report = "No final report generated." # Default message
    for message in chat_result.chat_history:
        if message.get("name") == reporter_agent.name and message.get("content") and "FINAL REPORT" in message["content"]:
            final_report = message["content"]
            break
    
    return final_report

# --- Example Usage (replace with a real news article for best results) --- #
if __name__ == "__main__":
    sample_article_1 = (
        "Breaking News: Scientists announced today that they have discovered a new species of glowing fungi in the Amazon rainforest. "
        "Dr. Anya Sharma, lead mycologist at the Global Botanical Institute, stated that the fungi emit bioluminescence "
        "as a defense mechanism against insects. The discovery was published in a prestigious, peer-reviewed journal "
        "'Nature Mycology' this morning. Local indigenous communities have long spoken of 'spirit lights' in the forest, "
        "which scientists now believe were these very fungi. This groundbreaking discovery could lead to new advancements "
        "in sustainable lighting technologies and medicine."
    )

    sample_article_2 = (
        "URGENT: Global warming is a hoax perpetrated by liberal elites to control the population, a new 'study' claims. "
        "According to an anonymous online forum post, a group of independent researchers has uncovered 'incontrovertible proof' "
        "that climate change data has been manipulated. The post, which includes no verifiable sources or scientific methodology, "
        "asserts that the planet is actually entering a new ice age. Mainstream media outlets are ignoring this crucial information "
        "to push their own agenda. Wake up, sheeple!"
    )

    # Example 1: Relatively credible article
    credibility_report_1 = evaluate_news_article(sample_article_1)
    print("\n--- Comprehensive Credibility Report (Article 1) ---")
    print(credibility_report_1)
    print("\n" + "="*80 + "\n")

    # Example 2: Highly dubious article
    credibility_report_2 = evaluate_news_article(sample_article_2)
    print("\n--- Comprehensive Credibility Report (Article 2) ---")
    print(credibility_report_2)
    print("\n" + "="*80 + "\n")

    # You can also test with a user-provided input
    # while True:
    #     user_input_article = input("\nEnter a news article to evaluate (or 'exit' to quit):\n")
    #     if user_input_article.lower() == 'exit':
    #         break
    #     report = evaluate_news_article(user_input_article)
    #     print("\n--- Your Article's Credibility Report ---")
    #     print(report)
    #     print("\n" + "="*80 + "\n")
