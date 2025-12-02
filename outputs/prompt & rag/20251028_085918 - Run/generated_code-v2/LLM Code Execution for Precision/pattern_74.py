
import streamlit as st
from langchain.llms import OpenAI
from langchain.agents import initialize_agent, Tool
from langchain.chains import LLMMathChain
from langchain.utilities import PythonREPL
import os

# Set your OpenAI API key
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY" # Replace with your actual key or use st.secrets

st.title("Financial Advisory and Portfolio Optimization AI (PAL Prompting)")
st.write("Ask me any financial question, especially those requiring precise calculations!")

# --- LangChain Setup --- #

# Initialize OpenAI LLM
llm = OpenAI(temperature=0, openai_api_key=os.environ.get("OPENAI_API_KEY"))

# Initialize Python REPL tool for code execution
python_repl = PythonREPL()

# Define tools for the agent
tools = [
    Tool(
        name="Python REPL",
        func=python_repl.run,
        description="A Python shell. Use this to execute python commands. Input should be a valid python command."
    ),
    # You can add more specific tools here, e.g., for fetching market data
    # Tool(
    #     name="Market Data Fetcher",
    #     func=fetch_market_data_function,
    #     description="Use this to fetch real-time market data like stock prices."
    # )
]

# Initialize the agent with the LLM and tools, using a conversational agent type
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

# --- Streamlit UI --- #

user_query = st.text_area("Enter your financial question:", 
                          "What is the future value of an investment of $5000 at an annual interest rate of 6% for 10 years, compounded semi-annually?",
                          height=100)

if st.button("Get Advice"):
    if user_query:
        with st.spinner("Thinking..."):
            try:
                # The agent will decide if it needs to generate and execute Python code
                # based on the prompt.
                response = agent.run(user_query)
                st.success("Here's my advice:")
                st.info(response)
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.warning("Please ensure your OpenAI API key is set and valid, and try again with a clear financial question.")
    else:
        st.warning("Please enter a financial question.")

st.subheader("How it works (PAL Prompting):")
st.markdown(
    "This AI uses a technique called Program-Aided Language Models (PAL) Prompting. "
    "When you ask a complex financial question, especially one requiring precise calculations, "
    "the underlying Language Model (LLM) might generate Python code to perform the calculation. "
    "This code is then executed, and the numerical result is fed back to the LLM, which then "
    "uses that result to formulate a clear and accurate natural language answer for you."
)

st.markdown("**Example of LLM generating code:**")
st.code(
    "Agent will automatically detect the need for calculation and generate Python code like this:\n" 
    "\`\`\`python\n" 
    "principal = 5000\n" 
    "rate = 0.06\n" 
    "time = 10\n" 
    "n_compounds = 2  # semi-annually\n" 
    "future_value = principal * (1 + rate / n_compounds)**(n_compounds * time)\n" 
    "print(future_value)\n" 
    "\`\`\`\n" 
    "... and then use the result to answer your question.", 
    language="python"
)

