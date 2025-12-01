import os
from typing import Any, Dict, List, Optional, Union

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory

# Mock Tools
@tool
def create_support_ticket(issue_description: str, user_email: str) -> str:
    """Creates a support ticket with the given issue description and user email.
    Requires both issue_description and user_email to be valid.
    Example: create_support_ticket("My internet is down", "user@example.com")
    """
    if not issue_description or not user_email or "@" not in user_email:
        return f"ERROR: Failed to create ticket. Invalid or missing issue_description or user_email. Provided: issue_description='{issue_description}', user_email='{user_email}'"
    return f"SUCCESS: Support ticket created for '{issue_description}' with email '{user_email}'. Ticket ID: {hash(issue_description + user_email)}"

@tool
def get_faq_answer(query: str) -> str:
    """Retrieves an answer from the FAQ database based on the query.
    Example: get_faq_answer("how to reset password")
    """
    faqs = {
        "reset password": "You can reset your password by visiting our website and clicking 'Forgot Password'.",
        "internet down": "Please try restarting your router. If the issue persists, contact technical support.",
        "billing inquiry": "You can view your billing information and pay your bills on your account dashboard.",
        "contact support": "You can contact our support team via phone at 1-800-555-1234 or email at support@example.com."
    }
    for keyword, answer in faqs.items():
        if keyword in query.lower():
            return answer
    return "I'm sorry, I couldn't find an answer to your question in the FAQ. Would you like to create a support ticket?"

# Initialize LLM
# Ensure you have OPENAI_API_KEY set in your environment variables
llm = ChatOpenAI(temperature=0, model="gpt-4o")

# Define the tools the agent can use
tools = [create_support_ticket, get_faq_answer]

# Define the prompt for the agent
prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            "You are a helpful customer support chatbot. Your goal is to assist users with their queries, provide information, and help create support tickets if needed. You have access to tools to achieve this."
            "If a tool call fails, analyze the error message, reflect on what went wrong (e.g., missing parameters, incorrect format), and try to correct your action. You might need to ask the user for clarification."
            "If the user expresses dissatisfaction (e.g., 'that's wrong', 'no, I meant something else'), acknowledge the feedback and attempt to re-evaluate their request and provide a better response or action."
            "Always try to be helpful and persistent in resolving the user's issue, even after errors or negative feedback."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessage(content="{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# Create the Langchain agent
agent = create_react_agent(llm, tools, prompt)

# Create a memory component
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create the AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=memory, handle_parsing_errors=True)

def run_chatbot():
    print("Hello! I am your customer support chatbot. How can I assist you today? (Type 'exit' to quit)")
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() == 'exit':
            break

        # Process user input with the agent executor
        try:
            response = agent_executor.invoke({"input": user_input})
            print(f"Chatbot: {response['output']}")
        except Exception as e:
            print(f"Chatbot Error: An unexpected error occurred: {e}")
            print("Chatbot: I encountered an internal issue. Please try again or rephrase your request.")

        # Simulate user feedback for self-correction demonstration
        feedback_prompt = input("Was that helpful? (Type 'no' for negative feedback, or press Enter): ")
        if feedback_prompt.lower() == 'no':
            memory.chat_memory.add_message(HumanMessage(content="That was not helpful. Can you try again or clarify?"))
            print("Chatbot: Thank you for your feedback. I will try to understand better and adjust my approach.")

if __name__ == "__main__":
    run_chatbot()