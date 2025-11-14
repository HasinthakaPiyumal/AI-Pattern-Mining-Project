import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import BaseTool, tool
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# --- data/medical_docs.txt --- (Simulated Medical Knowledge)
MEDICAL_DOCS_CONTENT = """
Medical Article: Pneumonia
Pneumonia is an inflammatory condition of the lung primarily affecting the small air sacs known as alveoli. It is usually caused by infection with viruses or bacteria and less commonly by other microorganisms, certain medications, or conditions such as autoimmune diseases. Symptoms include cough, chest pain, fever, and difficulty breathing. Diagnosis is often based on symptoms and physical examination, confirmed by chest X-ray. Treatment depends on the cause; bacterial pneumonia is treated with antibiotics.

Medical Article: Diabetes Mellitus Type 2
Type 2 diabetes is a long-term metabolic disorder that is characterized by high blood sugar, insulin resistance, and relative lack of insulin. Common symptoms include increased thirst, frequent urination, and unexplained weight loss. Long-term complications include heart disease, strokes, diabetic retinopathy, kidney failure, and poor blood flow that may lead to limb amputations. Management includes exercise, diet modifications, and medications like metformin or insulin.

Drug Information: Metformin
Metformin is a medication used to treat type 2 diabetes, primarily in overweight people and those of normal weight who cannot take other medications. It is also used in the treatment of polycystic ovary syndrome (PCOS). Common side effects include nausea, diarrhea, and abdominal pain. Rare but serious side effects include lactic acidosis. It should not be used in individuals with severe kidney disease.

Drug Information: Amoxicillin
Amoxicillin is an antibiotic used to treat a number of bacterial infections. These include middle ear infection, strep throat, pneumonia, skin infections, and urinary tract infections. Common side effects include nausea and rash. Allergic reactions, including anaphylaxis, are possible, especially in people with penicillin allergies. It is generally safe in pregnancy.

Symptom Checker: Fever
Fever is a temporary increase in your body temperature, often due to an illness. A fever is usually caused by an infection, but it can also be caused by inflammation, certain medications, or vaccinations.

Symptom Checker: Cough
Coughing is a common reflex that clears the throat and airway of irritants, fluids, mucus, or foreign particles. It can be a symptom of many conditions, from common colds to more serious lung diseases like asthma, bronchitis, or pneumonia.
"""

# --- knowledge_base.py ---
class MedicalKnowledgeBase:
    def __init__(self, data_content: str):
        self.data_content = data_content
        self.vectorstore = self._setup_knowledge_base()

    def _setup_knowledge_base(self):
        # Create a temporary file to simulate loading from 'data/medical_docs.txt'
        temp_file_path = "temp_medical_docs.txt"
        with open(temp_file_path, "w") as f:
            f.write(self.data_content)

        loader = TextLoader(temp_file_path)
        documents = loader.load()
        
        # Clean up the temporary file
        os.remove(temp_file_path)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Ensure we're using a persistent client if running multiple times or a temporary one for this session
        # For simplicity in a single script, we'll use an in-memory client unless specified otherwise
        vectorstore = Chroma.from_documents(docs, embeddings)
        return vectorstore

    def query_knowledge_base(self, query: str) -> str:
        if not self.vectorstore:
            return "Medical knowledge base not initialized."
        docs = self.vectorstore.similarity_search(query)
        return "\n\n".join([doc.page_content for doc in docs])


# --- tools.py ---
class MedicalKnowledgeSearchTool(BaseTool):
    name: str = "MedicalKnowledgeSearch"
    description: str = (
        "Searches the medical knowledge base for relevant information "
        "based on symptoms, diseases, or drug names. "
        "Input should be a clear medical query."
    )
    knowledge_base: MedicalKnowledgeBase

    def _run(self, query: str) -> str:
        return self.knowledge_base.query_knowledge_base(query)

class DrugInteractionCheckerTool(BaseTool):
    name: str = "DrugInteractionChecker"
    description: str = (
        "Checks for potential drug-drug interactions. "
        "Input should be a comma-separated list of drug names (e.g., 'Metformin, Amoxicillin')."
        "This is a simulated tool and provides placeholder information."
    )

    def _run(self, drug_names: str) -> str:
        drugs = [d.strip().lower() for d in drug_names.split(',')]
        interactions = []
        
        if "metformin" in drugs and "amoxicillin" in drugs:
            interactions.append("No significant interaction expected between Metformin and Amoxicillin based on general guidelines.")
        elif "metformin" in drugs:
            interactions.append("Consider kidney function when prescribing Metformin, especially with other nephrotoxic drugs.")
        elif "amoxicillin" in drugs:
            interactions.append("Advise patient about potential allergic reactions to Amoxicillin, especially with penicillin allergy history.")
        else:
            interactions.append(f"No specific interactions found for {', '.join(drugs)}. Consult a pharmacist for detailed information.")
        
        return "\n".join(interactions)

class SymptomAnalyzerTool(BaseTool):
    name: str = "SymptomAnalyzer"
    description: str = (
        "Analyzes a list of patient symptoms to suggest potential conditions or areas of concern. "
        "Input should be a comma-separated list of symptoms (e.g., 'fever, cough, chest pain')."
        "This is a simulated tool and provides placeholder information."
    )

    def _run(self, symptoms: str) -> str:
        symptoms_list = [s.strip().lower() for s in symptoms.split(',')]
        suggestions = []

        if "fever" in symptoms_list and "cough" in symptoms_list and "chest pain" in symptoms_list:
            suggestions.append("Considering symptoms like fever, cough, and chest pain, consider conditions such as pneumonia or bronchitis. Further investigation with a chest X-ray might be beneficial.")
        elif "fever" in symptoms_list and "fatigue" in symptoms_list:
            suggestions.append("Fever and fatigue can indicate a viral infection, common cold, or other systemic illness. Monitor for additional symptoms.")
        elif "increased thirst" in symptoms_list and "frequent urination" in symptoms_list:
            suggestions.append("Increased thirst and frequent urination are classic symptoms that could suggest diabetes mellitus. Blood glucose testing is recommended.")
        else:
            suggestions.append(f"Analyzing symptoms: {', '.join(symptoms_list)}. More information is needed for a precise diagnosis. Consider common infections or general malaise.")
        
        return "\n".join(suggestions)


# --- llm_orchestrator.py ---
class LLMOrchestrator:
    def __init__(self, medical_knowledge_base_instance: MedicalKnowledgeBase):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=5)
        
        # Initialize tools, passing the knowledge base instance to the MedicalKnowledgeSearchTool
        self.tools = [
            MedicalKnowledgeSearchTool(knowledge_base=medical_knowledge_base_instance),
            DrugInteractionCheckerTool(),
            SymptomAnalyzerTool()
        ]

        # Define the prompt for the agent
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful medical diagnostic assistant. Use the available tools to provide accurate and helpful information to healthcare professionals. Always strive for a comprehensive answer."),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        # Create the agent
        self.agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, memory=self.memory, verbose=True)

    def run_query(self, query: str) -> str:
        try:
            response = self.agent_executor.invoke({"input": query})
            return response["output"]
        except Exception as e:
            return f"An error occurred: {e}"


# --- main.py ---
def main():
    load_dotenv() # Load environment variables from .env

    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY not found. Please set it in your environment variables or in a .env file.")
        st.stop()

    st.set_page_config(page_title="Medical Diagnostic Assistant", layout="wide")
    st.title("🩺 Medical Diagnostic Assistant")
    st.markdown("This assistant helps healthcare professionals with augmented diagnostic capabilities using an LLM Orchestration Framework.")

    # Initialize knowledge base and orchestrator, using Streamlit's cache for efficiency
    @st.cache_resource
    def initialize_components():
        st.info("Initializing medical knowledge base and LLM orchestrator...")
        kb = MedicalKnowledgeBase(MEDICAL_DOCS_CONTENT)
        orchestrator = LLMOrchestrator(kb)
        st.success("Initialization complete!")
        return orchestrator

    orchestrator = initialize_components()

    # Initialize chat history in session state if not present
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Ask a medical question or provide patient symptoms..."):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = orchestrator.run_query(prompt)
                st.markdown(response)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
