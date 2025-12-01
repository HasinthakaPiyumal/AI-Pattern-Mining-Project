from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI # Placeholder for an actual LLM

# --- Tool Definitions ---

@tool
def create_task(task_name: str, due_date: str, assigned_to: str) -> str:
    """Creates a new task in the project management system.
    Input should be a dictionary with 'task_name', 'due_date' (format YYYY-MM-DD), and 'assigned_to'.
    """
    print(f"DEBUG: Creating task: {task_name}, Due: {due_date}, Assigned to: {assigned_to}")
    # In a real application, this would interact with a PM tool API
    return f"Task '{task_name}' created for {assigned_to} with due date {due_date}."

@tool
def schedule_meeting(topic: str, date_time: str, attendees: list) -> str:
    """Schedules a new meeting in the calendar system.
    Input should be a dictionary with 'topic', 'date_time' (format YYYY-MM-DD HH:MM), and 'attendees' (list of emails).
    """
    print(f"DEBUG: Scheduling meeting: {topic}, At: {date_time}, Attendees: {', '.join(attendees)}")
    # In a real application, this would interact with a calendar API
    return f"Meeting '{topic}' scheduled for {date_time} with {', '.join(attendees)}."

@tool
def send_email(recipient: str, subject: str, body: str) -> str:
    """Sends an email to a specified recipient.
    Input should be a dictionary with 'recipient', 'subject', and 'body'.
    """
    print(f"DEBUG: Sending email to {recipient} with subject '{subject}' and body '{body[:50]}...'")
    # In a real application, this would interact with an email API
    return f"Email sent to {recipient} with subject '{subject}'."

# List of all available tools
tools = [create_task, schedule_meeting, send_email]

# --- LLM and Agent Setup ---

# Placeholder for a real LLM. In a real scenario, you'd initialize with your API key:
# llm = ChatOpenAI(model="gpt-4", temperature=0)
# For demonstration, we'll use a mocked LLM or a simple placeholder
class MockLLM:
    def invoke(self, prompt_value):
        # A very basic mock for demonstration purposes.
        # In a real setup, this would be an actual LLM call.
        # This mock needs to be intelligent enough to 'reason' and 'act'.
        # For simplicity, we'll assume the prompt guides it to output a tool call format.
        print(f"DEBUG: LLM received prompt:\n{prompt_value}")
        if "create a task" in prompt_value.lower() and "report" in prompt_value.lower():
            return "Thought: The user wants to create a task. I need to call the `create_task` tool.\nAction: create_task\nAction Input: {\"task_name\": \"Generate monthly report\", \"due_date\": \"2024-07-31\", \"assigned_to\": \"John Doe\"}"
        elif "schedule a meeting" in prompt_value.lower() and "project review" in prompt_value.lower():
            return "Thought: The user wants to schedule a meeting. I need to call the `schedule_meeting` tool.\nAction: schedule_meeting\nAction Input: {\"topic\": \"Quarterly Project Review\", \"date_time\": \"2024-08-15 10:00\", \"attendees\": [\"alice@example.com\", \"bob@example.com\"]}"
        elif "send an email" in prompt_value.lower() and "follow up" in prompt_value.lower():
             return "Thought: The user wants to send an email. I need to call the `send_email` tool.\nAction: send_email\nAction Input: {\"recipient\": \"charlie@example.com\", \"subject\": \"Follow-up on X Project\", \"body\": \"Hi Charlie, just following up on the X project status. Please provide an update at your earliest convenience.\"}"
        else:
            return "Thought: I cannot fulfill this request with the available tools. I will respond directly.\nFinal Answer: I can help with task creation, meeting scheduling, and sending emails. How can I assist you?"

llm = MockLLM() # Using the mock LLM for this example

# Define the prompt for the agent
prompt = PromptTemplate.from_template(
    """You are an AI-powered project assistant. Your goal is to help users automate tasks by using the available tools.
    You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought:{agent_scratchpad}"""
)

# Create the ReAct agent
agent = create_react_agent(llm, tools, prompt)

# Create the AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# --- Main Interaction Loop ---
def main():
    print("AI-Powered Project Assistant (Type 'exit' to quit)")
    while True:
        user_input = input("\nHow can I help you today? ")
        if user_input.lower() == 'exit':
            break
        try:
            response = agent_executor.invoke({"input": user_input})
            print(f"Assistant: {response['output']}")
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please try rephrasing your request.")

if __name__ == "__main__":
    main()