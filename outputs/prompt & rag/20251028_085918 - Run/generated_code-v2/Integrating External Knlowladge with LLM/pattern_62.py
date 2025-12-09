import os
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from loguru import logger
import streamlit as st

load_dotenv()

class BrowserAgent:
    def __init__(self, driver_path=None):
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        self.driver = webdriver.Chrome(executable_path=driver_path, options=options)
        self.current_url = "about:blank"
        self.page_text = ""
        self.references = []
        logger.info("BrowserAgent initialized.")

    def _get_page_content(self):
        soup = BeautifulSoup(self.driver.page_source, "html.parser")
        for script in soup(["script", "style"]):
            script.extract()
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        return text

    def search(self, query: str) -> str:
        logger.info(f"Searching for: {query}")
        self.driver.get("https://www.google.com")
        search_box = self.driver.find_element(By.NAME, "q")
        search_box.send_keys(query)
        search_box.send_keys(Keys.RETURN)
        self.current_url = self.driver.current_url
        self.page_text = self._get_page_content()
        logger.info(f"Current URL: {self.current_url}")
        return f"Successfully searched for '{query}'. Current page content available."

    def click_link(self, link_text: str) -> str:
        logger.info(f"Attempting to click link with text: {link_text}")
        try:
            link = self.driver.find_element(By.PARTIAL_LINK_TEXT, link_text)
            link.click()
            self.current_url = self.driver.current_url
            self.page_text = self._get_page_content()
            logger.info(f"Clicked link. Current URL: {self.current_url}")
            return f"Successfully clicked link '{link_text}'. Current page content available."
        except Exception as e:
            logger.error(f"Could not click link '{link_text}': {e}")
            return f"Failed to click link '{link_text}'. Error: {e}"

    def scroll(self, direction: str) -> str:
        logger.info(f"Scrolling: {direction}")
        if direction.lower() == "down":
            self.driver.execute_script("window.scrollBy(0, window.innerHeight);")
        elif direction.lower() == "up":
            self.driver.execute_script("window.scrollBy(0, -window.innerHeight);")
        self.page_text = self._get_page_content()
        return f"Scrolled {direction}. Current page content updated."

    def quote(self, text_to_quote: str) -> str:
        if self.current_url == "about:blank":
            return "Cannot quote from a blank page. Please browse first."
        self.references.append({"quote": text_to_quote, "url": self.current_url})
        logger.info(f"Quoted text: '{text_to_quote[:50]}...' from {self.current_url}")
        return f"Successfully quoted text. {len(self.references)} references collected."

    def get_current_state(self):
        return {
            "url": self.current_url,
            "page_text": self.page_text,
            "references_count": len(self.references),
            "references": self.references
        }

    def close(self):
        self.driver.quit()
        logger.info("BrowserAgent closed.")

llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)
browser_agent_instance = BrowserAgent(driver_path=os.getenv("CHROMEDRIVER_PATH"))

tools = [
    Tool(
        name="Search",
        func=browser_agent_instance.search,
        description="Use to search the web for information. Input should be a search query string."
    ),
    Tool(
        name="ClickLink",
        func=browser_agent_instance.click_link,
        description="Use to click a link on the current page. Input should be the exact visible text of the link."
    ),
    Tool(
        name="Scroll",
        func=browser_agent_instance.scroll,
        description="Use to scroll the current page. Input should be 'up' or 'down'."
    ),
    Tool(
        name="Quote",
        func=browser_agent_instance.quote,
        description="Use to quote a relevant piece of text from the current page. Input should be the exact text to quote."
    )
]

prompt_template = PromptTemplate.from_template(
    """You are MedAgent, an expert medical information synthesizer. You can browse the web to answer complex medical queries.
    You have access to the following tools:
    {tools}
    
    Use the following format:
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I have gathered enough information and now I will provide a final answer.
    Final Answer: a comprehensive, evidence-based answer to the original question, incorporating collected quotes and their URLs.
    
    Current browsing state:
    URL: {browser_url}
    Page Text (excerpt):
    ```
    {browser_page_text}
    ```
    Collected References ({references_count}): {references}

    Question: {input}
    Thought:{agent_scratchpad}
    """
)

agent = create_react_agent(llm, tools, prompt_template)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

st.set_page_config(layout="wide")
st.title("🔬 MedAgent: Medical Information Synthesizer")

if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "agent_instance" not in st.session_state:
    st.session_state.agent_instance = BrowserAgent(driver_path=os.getenv("CHROMEDRIVER_PATH"))

query = st.text_area("Enter your medical query:", "What are the latest treatment protocols for glioblastoma multiforme, considering recent clinical trials and drug approvals?")

if st.button("Synthesize Information") and query:
    st.session_state.conversation = []
    st.session_state.agent_instance.references = []
    st.session_state.agent_instance.current_url = "about:blank"
    st.session_state.agent_instance.page_text = ""

    st.session_state.conversation.append(f"**Question:** {query}")
    st.info("MedAgent is browsing and synthesizing information...")

    initial_state = st.session_state.agent_instance.get_current_state()
    
    try:
        response = agent_executor.invoke({
            "input": query,
            "browser_url": initial_state["url"],
            "browser_page_text": initial_state["page_text"],
            "references_count": initial_state["references_count"],
            "references": initial_state["references"]
        })
        
        final_answer = response["output"]
        st.session_state.conversation.append(f"**Final Answer:** {final_answer}")
        
        if st.session_state.agent_instance.references:
            st.session_state.conversation.append("**References:**")
            for ref in st.session_state.agent_instance.references:
                st.session_state.conversation.append(f"- {ref['quote']} ([Source]({ref['url']}))")
        
    except Exception as e:
        st.error(f"An error occurred: {e}")
        logger.error(f"Streamlit error: {e}")
    finally:
        pass

st.markdown("--- Request Log ---")
for message in reversed(st.session_state.conversation):
    st.markdown(message)

if st.button("Close Browser Session"):
    st.session_state.agent_instance.close()
    st.session_state.pop("agent_instance", None)
    st.success("Browser session closed.")