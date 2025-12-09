import streamlit as st
import requests
from bs4 import BeautifulSoup
import os
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.chat_models import ChatOpenAI
from langchain.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
import re

# Ensure you have OPENAI_API_KEY set in your environment variables for ChatOpenAI
# You can set it here directly for testing, but using environment variables is recommended for production.
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# -----------------------------------------------------------------------------
# 1. Define Pydantic models for tool inputs
# -----------------------------------------------------------------------------
class SearchInput(BaseModel):
    query: str = Field(description="The search query for medical information.")

class BrowseInput(BaseModel):
    url: str = Field(description="The URL of the webpage to browse.")

# -----------------------------------------------------------------------------
# 2. Web Browsing Tools (using requests and BeautifulSoup for content extraction)
# -----------------------------------------------------------------------------
@tool("web_search", args_schema=SearchInput)
def web_search_tool(query: str) -> str:
    """
    Searches the web for medical information based on the query and returns relevant URLs.
    In a real application, this would integrate with a search engine API (e.g., Google Custom Search, SerpAPI).
    For this example, it returns predefined URLs for demonstration.
    """
    print(f"DEBUG: Performing web search for: {query}")
    # Mocking search results based on common medical queries
    if "diabetes treatment" in query.lower():
        return "Found these relevant URLs for 'diabetes treatment': https://www.niddk.nih.gov/health-information/diabetes/overview/managing-diabetes/treatment, https://www.mayoclinic.org/diseases-conditions/diabetes/diagnosis-treatment/drc-20371451, https://www.cdc.gov/diabetes/managing/index.html"
    elif "hypertension causes" in query.lower():
        return "Found these relevant URLs for 'hypertension causes': https://www.cdc.gov/bloodpressure/causes.htm, https://www.mayoclinic.org/diseases-conditions/high-blood-pressure/symptoms-causes/syc-20373410, https://www.heart.org/en/health-topics/high-blood-pressure/what-is-high-blood-pressure/causes-of-high-blood-pressure"
    elif "cancer symptoms" in query.lower():
        return "Found these relevant URLs for 'cancer symptoms': https://www.cancer.gov/about-cancer/treatment/types, https://www.mayoclinic.org/diseases-conditions/cancer/symptoms-causes/syc-20370588, https://www.cancer.org/cancer/risk-prevention/early-detection-diagnosis/cancer-symptoms.html"
    else:
        return f"No specific medical resources directly matched '{query}'. You might try refining your query. Consider general medical sites: https://www.webmd.com, https://www.nih.gov, https://www.who.int"

@tool("web_browser", args_schema=BrowseInput)
def web_browser_tool(url: str) -> str:
    """
    Browses the given URL, extracts and returns the main textual content from the webpage.
    This uses requests and BeautifulSoup for static content. For dynamic content and full
    browser interaction (click, scroll), Selenium would be required.
    """
    print(f"DEBUG: Browsing URL: {url}")
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, timeout=15, headers=headers)
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script_or_style in soup(['script', 'style']):
            script_or_style.extract()

        # Get text from common content-holding tags
        text_elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'span', 'div'])
        visible_text = ' '.join([elem.get_text(separator=' ', strip=True) for elem in text_elements])
        
        # Clean up multiple spaces and newlines
        visible_text = re.sub(r'\s+', ' ', visible_text).strip()

        return visible_text[:8000] # Limit content for LLM context window, adjust as needed
    except requests.exceptions.RequestException as e:
        return f"Error fetching {url}: {e}. Could not browse content."
    except Exception as e:
        return f"Error parsing {url}: {e}. Could not extract text."

# -----------------------------------------------------------------------------
# 3. Language Model (LLM) Integration & Agent Orchestrator (LangChain Agent)
# -----------------------------------------------------------------------------
# Initialize the LLM
# Ensure OPENAI_API_KEY is set in your environment variables for this to work.
try:
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
except Exception as e:
    st.error(f"Failed to initialize ChatOpenAI. Make sure OPENAI_API_KEY is set correctly. Error: {e}")
    st.stop()

# Define tools for the agent
tools = [web_search_tool, web_browser_tool]

# Define the prompt for the ReAct agent
# This prompt guides the LLM on how to use the tools and format its response.
prompt = PromptTemplate.from_template("""
You are a highly intelligent Medical Research Assistant for healthcare professionals.
Your primary goal is to answer complex medical questions by efficiently searching and browsing the web for up-to-date, accurate, and evidence-based information.

**Instructions:**
1.  **Understand the Query:** Carefully analyze the medical question provided by the user.
2.  **Plan:** Decide whether you need to first `web_search` for relevant links or directly `web_browser` a known URL.
3.  **Execute:** Use the available tools:
    *   `web_search(query: str)`: Use this tool to find relevant URLs for a given medical query.
    *   `web_browser(url: str)`: Use this tool to visit a URL and extract its main textual content.
4.  **Iterate:** Continue searching and browsing until you have gathered sufficient information to formulate a comprehensive answer.
5.  **Synthesize and Cite:** Combine the gathered information into a clear, concise, and accurate medical answer.
    **Crucially, all factual statements must be supported by references.**
    Present your answer, and then provide a "Sources:" section.
    Each source in the "Sources:" section should be formatted as:
    `[Source Number] URL: [Link to URL]`
    Optionally, you can include brief, direct quotes from the page that are most relevant to your answer, right after the URL. Example:
    `[1] URL: https://example.com/page.html - "A key quote supporting the answer."`

**Example Thought Process for a query:**
*   **Thought:** The user is asking about [query]. I need to find information on the web. I should start with a `web_search`.
*   **Action:** web_search
*   **Action Input:** "search query for [query]"
*   **Observation:** Found URLs: [url1, url2, ...]
*   **Thought:** I have some URLs. I will browse [url1] to extract information.
*   **Action:** web_browser
*   **Action Input:** [url1]
*   **Observation:** Content from [url1].
*   ... (continue browsing other relevant URLs or performing more searches if needed)
*   **Thought:** I have gathered enough information. I will now synthesize the answer and cite my sources.
*   **Final Answer:** [Your comprehensive answer here]
    Sources:
    [1] URL: [URL from web_browser] - "Relevant quote..."
    [2] URL: [Another URL] - "Another relevant quote..."

Your current medical question is: {input}

{agent_scratchpad}
""")

# Create the ReAct agent
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True, 
    handle_parsing_errors=True,
    max_iterations=10, # Limit iterations to prevent endless loops
    early_stopping_method="generate" # Stop when the agent decides it has a Final Answer
)

# -----------------------------------------------------------------------------
# 4. Streamlit User Interface
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Medical Research Assistant", page_icon="👨‍⚕️")
st.title("👨‍⚕️ Medical Research Assistant for Healthcare Professionals")
st.markdown("""
This AI assistant helps healthcare professionals obtain evidence-based answers to complex medical questions
by leveraging a Large Language Model (LLM) to intelligently search and browse the web.
**Note:** A valid OpenAI API key must be set in your environment variables (`OPENAI_API_KEY`) for this application to function.
""")

user_query = st.text_area("Enter your medical question here:", height=100)

if st.button("Get Evidence-Based Answer"):
    if user_query:
        st.info("The Medical Research Assistant is actively thinking, searching, and browsing the web for your answer. Please wait...")
        
        # Placeholder for dynamic output
        response_container = st.empty()

        try:
            with st.spinner("Processing your request..."):
                # Invoke the agent. The verbose=True will print steps to console/terminal running Streamlit.
                # For more granular Streamlit progress updates during agent's thinking,
                # custom LangChain callbacks would be needed, which is more complex for a single file example.
                response = agent_executor.invoke({"input": user_query})

            final_answer = response.get("output", "Could not find a clear answer.")
            
            # Display the final answer and sources
            response_container.markdown(f"### 💡 Your Evidence-Based Answer:\n{final_answer}")
            
            # Additional logic to parse and display references if the agent does not format them perfectly
            # This is a fallback in case the LLM doesn't perfectly adhere to the prompt for sources.
            st.markdown("---")
            st.subheader("🌐 Extracted References (for verification):")
            
            # Regex to find URLs in the output
            urls_in_output = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', final_answer)
            
            if urls_in_output:
                unique_urls = list(set(urls_in_output)) # Get unique URLs
                for i, url in enumerate(unique_urls):
                    st.markdown(f"• [{url}]({url})")
            else:
                st.info("No explicit URLs were extracted from the final answer. Please check the answer content for embedded citations.")

        except Exception as e:
            st.error(f"An unexpected error occurred during processing: {e}")
            st.warning("Please ensure your `OPENAI_API_KEY` environment variable is correctly set and you have an active internet connection.")
            st.stop() # Stop the Streamlit app to prevent further execution after error
    else:
        st.warning("Please enter a medical question in the text area above to get started.")