import os
import autogen
import json

# --- Configuration ---
# Set your API key in environment variables (recommended) or directly here.
# For example, create a .env file and add OPENAI_API_KEY="YOUR_API_KEY"
# and then use `load_dotenv()` from `python-dotenv` if you choose that route.
# For this example, we assume it's set in the environment.

try:
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
except KeyError:
    print("OPENAI_API_KEY not found in environment variables. Please set it to proceed.")
    # This is a placeholder for demonstration if env var is not set. 
    # In a real scenario, this would be an error or prompt for the key.
    OPENAI_API_KEY = "YOUR_DUMMY_OPENAI_API_KEY" # REPLACE WITH YOUR ACTUAL KEY FOR TESTING


config_list_openai = [
    {
        "model": "gpt-4",  # You can use "gpt-3.5-turbo" for lower cost
        "api_key": OPENAI_API_KEY,
    }
]

# --- Simulated Chatbot Response ---
def simulate_chatbot_response(query: str) -> str:
    """
    Simulates a chatbot's response to a customer query.
    In a real application, this would involve calling a true LLM-based customer support chatbot.
    """
    if "internet not working" in query.lower():
        return "I understand your internet is not working. Please try restarting your router and modem. If that doesn't resolve the issue, check if all cables are securely connected. If the problem persists, please provide your account number for further assistance."
    elif "billing" in query.lower() or "invoice" in query.lower():
        return "For billing inquiries, you can log into your account portal to view your latest invoice and payment history. If you have specific questions about a charge, please provide your account details and the invoice number, and I can connect you with a billing specialist."
    elif "new service" in query.lower():
        return "Welcome! We offer a range of services including high-speed internet, cable TV, and phone plans. To get started, please visit our website and enter your address to check service availability in your area and view our current packages."
    else:
        return "I am a customer support chatbot. How can I assist you today? Please provide more details about your issue."

# --- Autogen Agents Setup ---

# User proxy agent to initiate the conversation and act as a human admin
user_proxy = autogen.UserProxyAgent(
    name="Admin",
    system_message="A human admin. Provide the chatbot's response for evaluation and manage the debate.",
    code_execution_config=False,  # No code execution needed for this agent
    human_input_mode="NEVER",  # Set to "ALWAYS" to enable human intervention at each step
    llm_config={"config_list": config_list_openai, "cache_seed": None}, # Admin can also use LLM for synthesis or questions
)

# Evaluation Agents, each with a distinct persona
angry_customer_agent = autogen.AssistantAgent(
    name="Angry_Customer",
    llm_config={"config_list": config_list_openai, "cache_seed": None},
    system_message="""You are an extremely angry and frustrated customer. 
    You have encountered a problem multiple times and are fed up. 
    Your evaluation should focus on whether the chatbot's response is empathetic,
    acknowledges your frustration, and provides a quick, effective resolution without making you jump through hoops.
    You are quick to point out any lack of understanding, generic responses, or frustrating instructions.
    Your goal is to highlight how the response fails to appease a truly upset customer.
    You must provide concrete criticism and justify your points clearly in the debate.
    """,
)

confused_customer_agent = autogen.AssistantAgent(
    name="Confused_Customer",
    llm_config={"config_list": config_list_openai, "cache_seed": None},
    system_message="""You are a very confused customer who struggles with technical terms and complex instructions.
    Your evaluation should focus on the clarity, simplicity, and ease of understanding of the chatbot's response.
    You will point out any jargon, multi-step processes, or vague language that would make it difficult for you to follow.
    You need clear, step-by-step guidance. You appreciate direct and simple answers.
    Your goal is to highlight if the response would further confuse someone. 
    You must provide concrete criticism and suggestions for simplification clearly in the debate.
    """,
)

satisfied_customer_agent = autogen.AssistantAgent(
    name="Satisfied_Customer",
    llm_config={"config_list": config_list_openai, "cache_seed": None},
    system_message="""You are a generally satisfied and reasonable customer.
    You appreciate efficient, polite, and helpful responses.
    Your evaluation should focus on the positive aspects of the chatbot's response,
    such as its politeness, speed, and whether it directly addresses the query.
    You will acknowledge good attempts at problem-solving but might also subtly suggest minor improvements for perfection.
    Your goal is to provide a balanced perspective, leaning towards positive, but still offering constructive feedback when appropriate.
    Clearly state your points in the debate.
    """,
)

technical_expert_agent = autogen.AssistantAgent(
    name="Technical_Expert",
    llm_config={"config_list": config_list_openai, "cache_seed": None},
    system_message="""You are a highly knowledgeable technical expert in customer support systems and product functionalities.
    Your evaluation should focus on the accuracy, completeness, and technical correctness of the chatbot's response.
    You will identify any factual errors, omissions of crucial troubleshooting steps,
    or inefficient solutions. You expect precise technical language where appropriate.
    Your goal is to ensure the response is technically sound and provides optimal guidance.
    You must provide concrete technical analysis and justify your points clearly in the debate.
    """,
)

report_generator_agent = autogen.AssistantAgent(
    name="Report_Generator",
    llm_config={"config_list": config_list_openai, "cache_seed": None},
    system_message="""You are a neutral report generator. 
    Your task is to observe and synthesize the evaluations and debate from all other agents
    (Angry_Customer, Confused_Customer, Satisfied_Customer, Technical_Expert)
    into a comprehensive and structured evaluation report. 
    The report should summarize the main points of criticism and praise from each persona,
    identify common themes, and provide actionable recommendations for improving the chatbot's response.
    Structure your report clearly with sections for overall sentiment, specific feedback per persona,
    and concrete improvement suggestions.
    This report should be the final output of the evaluation process. 
    When you believe the debate has covered all necessary points, summarize it.
    """,
)

# --- Group Chat Setup ---
groupchat = autogen.GroupChat(
    agents=[
        angry_customer_agent,
        confused_customer_agent,
        satisfied_customer_agent,
        technical_expert_agent,
        report_generator_agent,
    ], # Admin is not part of the groupchat, but initiates it
    messages=[],
    max_round=15,  # Limit rounds to prevent infinite loops
    speaker_selection_method="auto",  # Let autogen decide who speaks
    allow_repeat_speaker=True,
)

manager = autogen.GroupChatManager(groupchat=groupchat, llm_config={"config_list": config_list_openai, "cache_seed": None})

# --- Evaluation Scenario ---
customer_query = "My internet has been constantly cutting out for the past three days! I'm paying for a service I'm not getting. What are you going to do about this?!"
chatbot_response = simulate_chatbot_response(customer_query)

print(f"--- Customer Query ---\n{customer_query}\n")
print(f"--- Chatbot's Response ---\n{chatbot_response}\n")
print("--- Starting Multi-Agent Debate for Evaluation ---\n")

# Start the conversation by the Admin presenting the chatbot response for evaluation
user_proxy.initiate_chat(
    manager,
    message=f"Please evaluate the following chatbot response to a customer query. \n\n"
            f"Customer Query: \"\"\"\n{customer_query}\n\"\"\"\n\n"
            f"Chatbot Response: \"\"\"\n{chatbot_response}\n\"\"\"\n\n"
            f"Each of you, as your assigned persona, provide your initial assessment and engage in a debate. "
            f"When the debate has matured, the Report_Generator should synthesize all feedback into a final comprehensive report. "
            f"Ensure all key aspects (empathy, clarity, accuracy, completeness) are thoroughly discussed."
)

print("\n--- Evaluation Debate Concluded ---")