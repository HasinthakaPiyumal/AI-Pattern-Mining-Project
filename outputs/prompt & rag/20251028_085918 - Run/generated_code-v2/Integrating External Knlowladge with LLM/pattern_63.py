
import streamlit as st
import os
import logging
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup

from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["SERPAPI_API_KEY"] = os.getenv("SERPAPI_API_KEY")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

search = GoogleSearchAPIWrapper()

@tool
def google_search_tool(query: str) -> str:
    """Searches Google for the given query and returns the results."""
    logging.info(f"Performing Google Search for: {query}")
    try:
        results = search.run(query)
        return results
    except Exception as e:
        logging.error(f"Error during Google search for '{query}': {e}")
        return f"Error during Google search: {e}"

def safe_web_scraper(url: str) -> str:
    """
    Fetches content from a URL safely (read-only) and returns cleaned text.
    Prevents any form interactions or JavaScript execution.
    """
    logging.info(f"Attempting to scrape URL: {url}")
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        for script_or_style in soup(['script', 'style', 'header', 'footer', 'nav', 'aside']):
            script_or_style.decompose()

        text_content = ' '.join(p.get_text(separator=' ', strip=True) for p in soup.find_all(['p', 'h1', 'h2', 'h3', 'li', 'span', 'div']))
        
        cleaned_text = ' '.join(text_content.split()).strip()
        
        logging.info(f"Successfully scraped and sanitized content from {url}")
        return cleaned_text[:8000] # Limit content length for LLM context, increased from 4000
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching URL {url}: {e}")
        return f"Could not retrieve content from {url} due to a network error or invalid URL."
    except Exception as e:
        logging.error(f"Error parsing content from {url}: {e}")
        return f"Could not parse content from {url}."

@tool
def scrape_web_page_tool(url: str) -> str:
    """Scrapes the content of a given URL and returns the cleaned text. This tool is strictly read-only and prevents interaction."""
    if any(keyword in url.lower() for keyword in ["edit=", "post=", "delete=", "form=", "login", "signup"]):
        logging.warning(f"Tripwire activated: Potential state-modifying/interactive URL attempted: {url}")
        return "Access to state-modifying or interactive URLs is restricted for safety reasons."
    
    return safe_web_scraper(url)

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

tools = [google_search_tool, scrape_web_page_tool]

agent_executor = initialize_agent(
    tools,
    llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

st.set_page_config(page_title="AI Customer Support Assistant", layout="wide")
st.title("AI-Powered Customer Support Assistant")
st.markdown("Ask me anything about products, and I'll find real-time information for you safely.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("How can I help you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            chat_history = []
            for msg in st.session_state.messages[:-1]: # Exclude current prompt from chat_history for agent
                if msg["role"] == "user":
                    chat_history.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    chat_history.append(AIMessage(content=msg["content"]))
            
            try:
                response = agent_executor.invoke({"input": prompt, "chat_history": chat_history})
                moderated_response = response["output"]

                # Output Moderation Tripwire
                if any(keyword in moderated_response.lower() for keyword in ["click here", "submit form", "sign up", "login to", "enter your details"]):
                    logging.warning(f"Output moderation tripwire activated: Potentially interactive instruction detected in AI response.")
                    moderated_response = "I cannot provide interactive instructions or ask for personal details. I can only provide information based on my safe web access." 
                
                st.markdown(moderated_response)
                st.session_state.messages.append({"role": "assistant", "content": moderated_response})
            except Exception as e:
                error_message = f"An error occurred while processing your request: {e}"
                logging.error(error_message)
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
