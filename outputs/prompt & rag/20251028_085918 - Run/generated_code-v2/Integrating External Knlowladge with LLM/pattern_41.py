import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

# 1. Define Tools

def search_web(query: str) -> str:
    try:
        # This is a placeholder for a real search engine API (e.g., SerpAPI, Google Custom Search)
        # For demonstration, we'll simulate a basic search result from a generic search.
        st.info(f"Searching the web for: {query}")
        response = requests.get(f"https://www.google.com/search?q={query}", headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extracting top few links and snippets as a simplified search result
        results = []
        for g in soup.find_all('div', class_='tF2Cxc'): # Google's common search result class
            link = g.find('a')
            title = g.find('h3')
            snippet = g.find('div', class_='lEBKkf') # Simplified snippet class
            
            if link and title and snippet:
                results.append({
                    "title": title.get_text(),
                    "link": link.get('href'),
                    "snippet": snippet.get_text()
                })
            if len(results) >= 3: # Limit to top 3 for brevity
                break
        
        if not results:
            return "No relevant search results found."

        formatted_results = ""
        for i, res in enumerate(results):
            formatted_results += f"Result {i+1}:\nTitle: {res['title']}\nLink: {res['link']}\nSnippet: {res['snippet']}\n\n"
        return formatted_results
    except requests.exceptions.RequestException as e:
        return f"Error during web search: {e}"
    except Exception as e:
        return f"An unexpected error occurred during search: {e}"

def scrape_web_page(url: str) -> str:
    try:
        st.info(f"Scraping content from: {url}")
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status() # Raise an exception for bad status codes
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract relevant text from paragraphs and headings
        paragraphs = [p.get_text() for p in soup.find_all('p')]
        headings = [h.get_text() for h in soup.find_all(['h1', 'h2', 'h3'])]
        
        # Limit the scraped content to avoid overwhelming the LLM
        content = "\n".join(headings + paragraphs)
        if len(content) > 4000: # Limit to 4000 characters for context window management
            return content[:4000] + "... [Content truncated]"
        return content
    except requests.exceptions.RequestException as e:
        return f"Error during web scraping: {e}"
    except Exception as e:
        return f"An unexpected error occurred during scraping: {e}"


medsearch_tools = [
    Tool(
        name="WebSearch",
        func=search_web,
        description="Useful for performing a web search to find relevant medical information based on a query. Returns a list of titles, links, and snippets."
    ),
    Tool(
        name="WebScraper",
        func=scrape_web_page,
        description="Useful for scraping the full content of a given URL. Provide the exact URL. Returns the text content of the webpage."
    )
]

# 2. Core AI Agent (LangChain)

# Ensure OPENAI_API_KEY is set in your environment variables
if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY not found in environment variables. Please set it to use the LLM.")
    st.stop()

llm = ChatOpenAI(temperature=0, model_name="gpt-4o") # Using gpt-4o for better reasoning

# Define the agent prompt
agent_prompt_template = PromptTemplate.from_template(
    """You are a highly intelligent and ethical medical AI assistant. Your primary goal is to answer complex medical queries accurately, thoroughly, and with factual evidence, by leveraging web browsing capabilities. You must cite your sources (URLs and relevant quotes) in your final answer.

Here are the tools you have access to:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I have gathered sufficient information and can now provide a comprehensive, evidence-based answer.
Final Answer: a comprehensive, well-structured answer to the original question, including citations with URLs and direct quotes from the observed content to support key statements.

Begin!

Question: {input}
Thought:{agent_scratchpad}"""
)

agent = create_react_agent(llm, medsearch_tools, agent_prompt_template)
agent_executor = AgentExecutor(agent=agent, tools=medsearch_tools, verbose=True, handle_parsing_errors=True)

# 3. Streamlit User Interface

st.set_page_config(page_title="MedSearch AI Assistant", layout="wide")
st.title("🩺 MedSearch AI Assistant")
st.markdown("An AI assistant for healthcare professionals to get evidence-based answers to medical queries.")

medical_query = st.text_area("Enter your medical query here:", height=150)

if st.button("Get Evidence-Based Answer"):
    if medical_query:
        with st.spinner("Searching and synthesizing information..."):
            try:
                response = agent_executor.invoke({"input": medical_query})
                st.subheader("🔬 Evidence-Based Answer")
                st.write(response["output"])
            except Exception as e:
                st.error(f"An error occurred while processing your request: {e}")
                st.info("Please ensure your OpenAI API key is correctly set and try again. For detailed debugging, check the console output.")
    else:
        st.warning("Please enter a medical query to get an answer.")

st.markdown("""
---
**How it works:**
This assistant uses an advanced AI agent that can browse the web (simulated search and scrape) to find up-to-date medical information. It then synthesizes a comprehensive answer, complete with citations to support its claims. This helps healthcare professionals get accurate and reliable information for diagnosis, treatment planning, and research.

**Disclaimer:** This tool is for informational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare professional for any health concerns.
""")
