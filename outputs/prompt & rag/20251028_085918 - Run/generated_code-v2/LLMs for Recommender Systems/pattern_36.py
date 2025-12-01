import os
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI
from langchain_community.llms import FakeListLLM

# Import custom tools
from tools import get_ecommerce_tools

def run_shopping_assistant():
    # Set up the LLM
    # It's recommended to set your OPENAI_API_KEY in your environment variables.
    # Example: os.environ["OPENAI_API_KEY"] = "your_api_key_here"
    try:
        # Using ChatOpenAI for demonstration. Replace with your preferred LLM if needed.
        # Ensure OPENAI_API_KEY is set in your environment variables or directly here.
        llm = ChatOpenAI(temperature=0)
    except Exception as e:
        print(f"Warning: Could not initialize ChatOpenAI. Ensure OPENAI_API_KEY is set. Error: {e}")
        print("Falling back to a FakeListLLM for basic demonstration without external API calls.")
        # Fallback to a mock LLM for demonstration if API key is not set or issue occurs
        llm = FakeListLLM(responses=[
            "I am a mock LLM. How can I help you today?",
            "I can help you find products or check order status.",
            "Please specify what you are looking for."
        ])

    # Initialize memory for the agent to maintain conversation history
    # 'k=5' means the agent remembers the last 5 turns of the conversation.
    memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=5)

    # Get the custom e-commerce tools defined in tools.py
    ecommerce_tools = get_ecommerce_tools()

    # Initialize the conversational agent
    # AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION uses a ReAct-style prompting strategy
    # allowing the LLM to reason and use tools in a conversational context.
    agent = initialize_agent(
        ecommerce_tools,
        llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True, # Set to True to see the agent's thought process
        memory=memory,
        handle_parsing_errors=True # To gracefully handle potential LLM output parsing issues
    )

    print("\nWelcome to your E-commerce Shopping Assistant! How can I help you today?")
    print("I can recommend products, check inventory, or track your orders.")
    print("Type 'exit' to end the conversation.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Assistant: Goodbye! Happy shopping!")
            break

        try:
            # Run the agent with the user's input
            response = agent.run(user_input)
            print(f"Assistant: {response}")
        except Exception as e:
            print(f"Assistant: An error occurred: {e}")
            print("Assistant: I'm having trouble with that request. Please try rephrasing or ask for something else.")

if __name__ == "__main__":
    run_shopping_assistant()
