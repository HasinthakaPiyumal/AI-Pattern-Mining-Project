import os
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
import gradio as gr

# Set your OpenAI API key from environment variables
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# 1. Simulate External Knowledge Tools
@tool
def medical_database_lookup(query: str) -> str:
    """Looks up information in a simulated medical database. Use this for specific queries about diseases, symptoms, treatments, drug interactions, or clinical trial data."""
    medical_data = {
        "persistent cough and fever": "Possible diagnoses include pneumonia, bronchitis, influenza, or COVID-19. Further tests like chest X-ray and viral swabs are recommended.",
        "pneumonia treatment": "Treatment for pneumonia typically involves antibiotics (for bacterial), antivirals (for viral), rest, and fluid intake. Severe cases may require hospitalization and oxygen therapy.",
        "ibuprofen side effects": "Common side effects of ibuprofen include nausea, vomiting, indigestion, and diarrhea. Serious side effects can include stomach ulcers, kidney problems, and cardiovascular events.",
        "diabetes type 2 symptoms": "Symptoms of Type 2 Diabetes include increased thirst, frequent urination, increased hunger, fatigue, blurred vision, slow-healing sores, and frequent infections."
    }
    return medical_data.get(query.lower(), "No specific information found in the medical database for your query. Consider rephrasing or using the research tool.")

@tool
def medical_research_search(query: str) -> str:
    """Searches for the latest medical research papers, guidelines, or contemporary medical information via a simulated search engine. Use this for broader or more up-to-date queries."""
    research_results = {
        "latest COVID-19 variants": "Recent research indicates the emergence of new Omicron sub-variants, characterized by increased transmissibility but potentially reduced severity in vaccinated individuals. Updated booster recommendations are being evaluated.",
        "advances in cancer immunotherapy": "Immunotherapy continues to evolve with novel checkpoint inhibitors and CAR T-cell therapies showing promise in various solid tumors and hematological malignancies. Combination therapies are a key area of research.",
        "rare neurological disorders": "Research into rare neurological disorders like Amyotrophic Lateral Sclerosis (ALS) and Huntington's Disease focuses on genetic therapies, neuroprotection, and symptomatic management. Clinical trials are ongoing for several experimental treatments."
    }
    return research_results.get(query.lower(), "No relevant recent research found for your query. The information might be too specific or outside the scope of this simulated search.")

# 2. Initialize the LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 3. Define the tools for Langchain
tools = [medical_database_lookup, medical_research_search]

# 4. Create the Langchain Prompt for the Agent
# The system message guides the agent's behavior and defines its role.
system_message = SystemMessage(content=(
    "You are a highly intelligent and helpful Medical Diagnostic Assistant designed to assist healthcare professionals. "
    "Your primary goal is to provide accurate and contextually relevant medical information, diagnoses, and treatment suggestions "
    "by leveraging both your internal knowledge and external medical tools. "
    "Always strive for factual accuracy and consider multiple possibilities when diagnosing. "
    "When a user asks a question, use the available tools (medical_database_lookup, medical_research_search) to find the most up-to-date and specific information. "
    "Synthesize the information from your internal knowledge and the tools to provide a comprehensive answer."
))

prompt = PromptTemplate.from_messages([
    system_message,
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

# 5. Create the Langchain Agent and AgentExecutor
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# 6. Define a function to run the agent with a query
def medical_assistant_query(user_query: str) -> str:
    try:
        response = agent_executor.invoke({"input": user_query})
        return response["output"]
    except Exception as e:
        return f"An error occurred: {e}"

# 7. Set up the Gradio Interface
iface = gr.Interface(
    fn=medical_assistant_query,
    inputs=gr.Textbox(lines=5, label="Enter your medical query or patient symptoms:", placeholder="e.g., Patient presents with persistent cough and fever, what could be the diagnosis?"),
    outputs="text",
    title="AI Medical Diagnostic Assistant",
    description="This assistant leverages an LLM augmented with external medical databases and research to provide diagnostic assistance and information to healthcare professionals. (Note: This is a simulated environment for demonstration purposes.)"
)

# 8. Launch the Gradio App
if __name__ == "__main__":
    iface.launch(share=True)
