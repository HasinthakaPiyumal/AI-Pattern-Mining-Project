import autogen
import os

# Load OpenAI API key from environment variable
# Make sure to set OPENAI_API_KEY in your environment variables
# For example: export OPENAI_API_KEY='your_api_key_here'
openai_api_key = os.environ.get("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

# Configuration for the LLM
config_list = [
    {
        "model": "gpt-4",  # You can change this to "gpt-3.5-turbo" or other available models
        "api_key": openai_api_key,
    }
]

# 1. Initialize the User Proxy Agent
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",  # Set to "ALWAYS" for interactive input, "NEVER" for automation
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("EXIT"),
    code_execution_config=False,  # No code execution in this scenario
    llm_config={"config_list": config_list},
)

# 2. Initialize the specialized Assistant Agents with their personas
content_expert = autogen.AssistantAgent(
    name="Content_Expert",
    llm_config={"config_list": config_list},
    system_message=(
        "You are a highly experienced academic content expert and subject matter specialist. "
        "Your primary role is to evaluate the factual accuracy, depth of analysis, logical coherence, "
        "and relevance of the information presented in an essay. Provide constructive feedback focusing "
        "on content quality, arguments, and evidence. Always end your turn by asking if another expert should review or by saying 'EXIT' if all evaluations are complete."
    ),
)

grammar_specialist = autogen.AssistantAgent(
    name="Grammar_Specialist",
    llm_config={"config_list": config_list},
    system_message=(
        "You are a meticulous grammar and linguistics expert. Your task is to thoroughly review "
        "the essay for any grammatical errors, spelling mistakes, punctuation issues, "
        "sentence structure problems, and clarity of expression. Offer precise corrections "
        "and suggestions for improving linguistic quality. Always end your turn by asking if another expert should review or by saying 'EXIT' if all evaluations are complete."
    ),
)

creativity_judge = autogen.AssistantAgent(
    name="Creativity_Judge",
    llm_config={"config_list": config_list},
    system_message=(
        "You are a seasoned literary critic and creativity judge. Your role is to assess the essay's "
        "originality, imaginative elements, unique writing style, captivating voice, and overall ability "
        "to engage the reader. Provide feedback on creative aspects and areas for enhancing impact. "
        "Always end your turn by asking if another expert should review or by saying 'EXIT' if all evaluations are complete."
    ),
)

# Example essay to be graded
example_essay = """
Title: The Impact of Artificial Intelligence on Society

Artificial intelligence, often referred to as AI, is rapidly transforming various aspects of human society. From automating mundane tasks to revolutionizing healthcare and transportation, its influence is profound and far-reaching. However, the rise of AI also brings significant ethical considerations and potential challenges that need careful navigation.

One of the most significant benefits of AI is its ability to enhance efficiency and productivity. AI-powered systems can process vast amounts of data at speeds impossible for humans, leading to breakthroughs in scientific research, financial analysis, and personalized education. For instance, in medicine, AI algorithms assist in diagnosing diseases earlier and more accurately, thereby saving countless lives. Similarly, self-driving cars, a product of advanced AI, promise to reduce traffic accidents and optimize urban mobility.

Nevertheless, the widespread adoption of AI is not without its drawbacks. A primary concern is job displacement. As AI systems become more sophisticated, they are capable of performing tasks traditionally done by humans, leading to anxieties about future employment. Moreover, ethical dilemmas surrounding data privacy, algorithmic bias, and autonomous decision-making in critical applications demand robust regulatory frameworks. If AI systems are trained on biased datasets, they can perpetuate and even amplify existing societal inequalities.

In conclusion, artificial intelligence presents a double-edged sword. While its potential to improve human welfare is immense, its development and deployment must be guided by careful ethical considerations and proactive policy-making. Ensuring that AI serves humanity's best interests requires a collaborative effort from technologists, policymakers, and the public to harness its power responsibly and mitigate its risks.
"""

print("\n--- Starting Multi-Perspective Essay Evaluation ---")
print(f"\nEssay to be evaluated:\n{example_essay}\n")

# Orchestrate the conversation to get feedback from all agents
# The user_proxy initiates the conversation by asking the Content_Expert to evaluate the essay.
# Then, it can route the conversation to other experts for their input.

# For a more structured debate, one could define a groupchat manager.
# For sequential evaluation, we can simply pass the baton by adding prompts.

# Let's start with a sequential evaluation for simplicity

# Step 1: Content Expert evaluates
print("\n--- Content Expert's Evaluation ---")
user_proxy.initiate_chat(
    content_expert,
    message=f"Please evaluate the following essay for its content, factual accuracy, depth of analysis, and logical coherence. Provide detailed feedback and suggest improvements.\n\nEssay:\n{example_essay}"
)

# Step 2: Grammar Specialist evaluates
print("\n--- Grammar Specialist's Evaluation ---")
user_proxy.send(
    recipient=grammar_specialist,
    message=f"Now, please review the same essay for grammar, spelling, punctuation, sentence structure, and overall clarity of expression. Provide specific corrections and suggestions.\n\nEssay:\n{example_essay}",
    request_reply=True
)

# Step 3: Creativity Judge evaluates
print("\n--- Creativity Judge's Evaluation ---")
user_proxy.send(
    recipient=creativity_judge,
    message=f"Finally, evaluate the essay for its originality, writing style, voice, and engagement factor. How creative and impactful is it? Provide your assessment and ideas for enhancing its creative aspects.\n\nEssay:\n{example_essay}",
    request_reply=True
)

print("\n--- Multi-Perspective Essay Evaluation Complete ---")
print("\nAggregated feedback is available in the chat history above.")

# You can access the full chat history if needed
# print(user_proxy.chat_messages)
