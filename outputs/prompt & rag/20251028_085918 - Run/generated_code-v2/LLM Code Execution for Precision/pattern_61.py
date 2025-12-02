import streamlit as st
import openai
import sqlite3
import pandas as pd
import numpy as np
import io
import contextlib
import os

# --- Configuration --- #
# It's recommended to set your OpenAI API key as an environment variable
# For demonstration, you can uncomment and set it directly, but do NOT do this in production
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# Check if API key is set
if "OPENAI_API_KEY" not in os.environ:
    st.error("OPENAI_API_KEY environment variable not set. Please set it to use the application.")
    st.stop()

openai.api_key = os.environ["OPENAI_API_KEY"]

# --- Database Setup (SQLite for simplicity) --- #
conn = sqlite3.connect("financial_assistant.db")
c = conn.cursor()

c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        goals TEXT,
        risk_tolerance TEXT
    )
""")
conn.commit()

def save_user_data(username, goals, risk_tolerance):
    try:
        c.execute(
            "INSERT OR REPLACE INTO users (username, goals, risk_tolerance) VALUES (?, ?, ?)",
            (username, goals, risk_tolerance)
        )
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Error saving user data: {e}")
        return False

def get_user_data(username):
    c.execute("SELECT goals, risk_tolerance FROM users WHERE username = ?", (username,))
    return c.fetchone()

# --- Code Execution Environment --- #
@contextlib.contextmanager
def capture_stdout():
    old_stdout = io.StringIO()
    with contextlib.redirect_stdout(old_stdout):
        yield old_stdout

def execute_python_code(code: str) -> str:
    try:
        # Restrict globals and locals to prevent arbitrary file system/network access for security
        # This is a basic attempt at sandboxing; a true sandbox requires more robust solutions
        safe_globals = {
            "pd": pd,
            "np": np,
            "math": __import__("math"),
            "_output": None, # To capture direct output from the script
            "__builtins__": {
                "print": print,
                "len": len,
                "str": str,
                "int": int,
                "float": float,
                "range": range,
                "dict": dict,
                "list": list,
                "tuple": tuple,
                "set": set,
                "sum": sum,
                "min": min,
                "max": max,
                "abs": abs,
                "round": round,
                "pow": pow,
                "bool": bool,
                "type": type,
            }
        }
        safe_locals = {}

        with capture_stdout() as output:
            exec(code, safe_globals, safe_locals)
        
        # If the code explicitly set a variable named '_output', use that
        # Otherwise, return the captured stdout
        if "_output" in safe_locals and safe_locals["_output"] is not None:
            return str(safe_locals["_output"])
        elif output.getvalue():
            return output.getvalue().strip()
        else:
            return "Code executed successfully, no explicit output."

    except Exception as e:
        return f"Error executing code: {e}"

# --- LLM Integration (Simplified using direct OpenAI call for demonstration) --- #
# In a full Langchain setup, you would define an agent with tools.
# For a single-file Streamlit app, we'll simulate the PAL logic more directly.

def get_financial_advice(user_prompt, user_data):
    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI-powered financial planning assistant. "
                "Your goal is to provide accurate financial advice. "
                "When complex calculations, data analysis, or numerical precision are required, "
                "you should generate Python code and output it within <python>...</python> tags. "
                "This code will be executed, and the result will be provided back to you. "
                "Integrate the code execution results into your natural language response. "
                "User data: Goals - {user_data[0]}, Risk Tolerance - {user_data[1]}."
                "Assume 'pd' for pandas and 'np' for numpy are available for code execution." 
                "For example, to calculate future value, you might output: <python>fv = 1000 * (1 + 0.05)**10\nprint(fv)</python>"
                "Make sure the python code is complete and runnable. If the result of the code should be used in your answer, assign it to a variable called `_output` in the code, or just print it." 
            ),
        },
        {"role": "user", "content": user_prompt},
    ]

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",  # or "gpt-4"
        messages=messages,
        temperature=0.7,
    )
    return response.choices[0].message.content

# --- Streamlit UI --- #
st.set_page_config(page_title="AI Financial Planner", layout="wide")
st.title("💰 AI-Powered Personalized Financial Planner")

st.sidebar.header("User Profile")
username_input = st.sidebar.text_input("Enter your username", key="username_input")

if username_input:
    st.sidebar.subheader(f"Welcome, {username_input}!")
    user_data = get_user_data(username_input)
    current_goals = user_data[0] if user_data else ""
    current_risk_tolerance = user_data[1] if user_data else ""

    with st.sidebar.form(key="profile_form"):
        goals = st.text_area("Your Financial Goals (e.g., Retirement, House Down Payment)", value=current_goals)
        risk_tolerance = st.selectbox(
            "Your Risk Tolerance",
            ["Low", "Medium", "High"], 
            index=["Low", "Medium", "High"].index(current_risk_tolerance) if current_risk_tolerance else 0
        )
        submit_profile = st.form_submit_button("Save Profile")

    if submit_profile:
        if save_user_data(username_input, goals, risk_tolerance):
            st.sidebar.success("Profile updated successfully!")
            user_data = get_user_data(username_input) # Refresh data
        else:
            st.sidebar.error("Failed to update profile.")
else:
    st.sidebar.info("Please enter a username to manage your profile.")

st.header("Financial Assistant Chat")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask me anything about your finances..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if not username_input:
        st.error("Please enter a username in the sidebar before asking questions.")
    elif not user_data:
        st.error("Please save your financial goals and risk tolerance in the sidebar first.")
    else:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            # Get LLM response
            llm_response_text = get_financial_advice(prompt, user_data)

            # Check for Python code in the LLM's response
            code_blocks = []
            last_idx = 0
            while True:
                start_tag = "<python>"
                end_tag = "</python>"
                start_idx = llm_response_text.find(start_tag, last_idx)
                if start_idx == -1:
                    break
                end_idx = llm_response_text.find(end_tag, start_idx + len(start_tag))
                if end_idx == -1:
                    break # Malformed code block

                code_blocks.append(llm_response_text[start_idx + len(start_tag):end_idx].strip())
                last_idx = end_idx + len(end_tag)
            
            # Process code blocks and integrate results
            processed_response = llm_response_text
            for i, code in enumerate(code_blocks):
                st.write(f"Executing Python code block {i+1}...")
                st.code(code, language="python")
                code_output = execute_python_code(code)
                st.info(f"Code Output:\n{code_output}")
                
                # Replace the original code block with its output in the response
                # This is a simple replacement; more sophisticated integration might be needed
                processed_response = processed_response.replace(f"<python>{code}</python>", f"\n```python\n{code}\n```\n**Code Output:**\n```\n{code_output}\n```\n", 1)

            full_response = processed_response
            message_placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})


# Close connection when app exits (Streamlit doesn't have a clean exit hook, 
# but this is good practice for non-Streamlit apps). For a persistent app, 
# manage connection lifecycle carefully.
# conn.close() # Not ideal for Streamlit's rerun model. Connection will close on app restart. 
