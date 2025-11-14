import os
from typing import List, Dict, Any

# Mock imports for external libraries (actual installs would be needed)
try:
    from langchain.agents import AgentExecutor, create_react_agent
    from langchain.chains import LLMChain
    from langchain_core.prompts import PromptTemplate
    from langchain_core.tools import Tool
    from langchain_community.llms import OpenAI
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import OpenAIEmbeddings
    from langchain.memory import ConversationBufferMemory
    import pandas as pd
    import requests # For mock external API calls
    import chromadb
except ImportError:
    print("Please install required libraries: langchain, openai, pandas, requests, chromadb")
    # Exit or handle gracefully in a real application
    exit()

# --- Configuration ---
# Set your OpenAI API key
# It's recommended to set this as an environment variable for security:
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
# For demonstration purposes, you can uncomment and replace with your key, but be cautious.
# os.environ["OPENAI_API_KEY"] = "sk-..."

# Initialize LLM and Embeddings (ensure OPENAI_API_KEY is set)
llm = OpenAI(temperature=0)
embeddings = OpenAIEmbeddings()


# --- External Tool Interface Module (Mocks) ---
def search_pubmed(query: str) -> str:
    """Simulates searching PubMed for medical literature."""
    print(f"DEBUG: Searching PubMed for: {query}")
    if "prion disease" in query.lower() or "creutzfeldt-jakob" in query.lower():
        return "Found articles on Creutzfeldt-Jakob disease, a rare neurodegenerative disorder. Key symptoms include rapidly progressive dementia, myoclonus, and ataxia. Diagnosis involves MRI, EEG, CSF analysis (14-3-3 protein, RT-QuIC)."
    elif "cystic fibrosis" in query.lower():
        return "Found articles on Cystic Fibrosis, a genetic disorder affecting mucus and sweat glands. Symptoms include persistent cough, lung infections, pancreatic insufficiency. Diagnosis via sweat chloride test and genetic testing (CFTR gene)."
    elif "hemophilia a" in query.lower():
        return "Found articles on Hemophilia A, an X-linked bleeding disorder. Symptoms: spontaneous bleeding, prolonged bleeding after injury. Treatment: factor VIII replacement."
    return f"No specific rare disease articles found for '{query}'. General medical literature search performed."

def search_orphanet(disease_name: str) -> str:
    """Simulates searching Orphanet for rare disease information."""
    print(f"DEBUG: Searching Orphanet for: {disease_name}")
    if "creutzfeldt-jakob" in disease_name.lower() or "cjd" in disease_name.lower():
        return "Orphanet ID: ORPHA967. Creutzfeldt-Jakob disease (CJD) is a rare, fatal neurodegenerative disorder. Incidence: 1-2 cases per million population per year. Etiology: Prion protein accumulation. Symptoms: Rapidly progressive dementia, myoclonus. Management: Supportive. No cure."
    elif "hemophilia a" in disease_name.lower():
        return "Orphanet ID: ORPHA398. Hemophilia A is a rare X-linked bleeding disorder caused by a deficiency in coagulation factor VIII. Incidence: 1 in 5,000-10,000 male births. Symptoms: Spontaneous bleeding, prolonged bleeding after injury/surgery. Management: Factor VIII replacement therapy."
    elif "cystic fibrosis" in disease_name.lower():
        return "Orphanet ID: ORPHA206. Cystic Fibrosis is a genetic disorder affecting exocrine glands. Incidence: 1 in 2,500-3,500 births (Caucasian). Symptoms: Respiratory, digestive, and reproductive issues. Management: Symptomatic, includes airway clearance and enzyme replacement."
    return f"Orphanet found no direct match for '{disease_name}'. Try a more general term or check spelling."

def run_genetic_analysis(genetic_markers: str) -> str:
    """Simulates running a genetic panel analysis tool."""
    print(f"DEBUG: Running genetic analysis for markers: {genetic_markers}")
    if "prnp" in genetic_markers.lower() and ("ataxia" in genetic_markers.lower() or "dementia" in genetic_markers.lower()):
        return "Genetic analysis indicates a high probability of PRNP gene mutation, consistent with familial CJD or GSS syndrome. Further sequencing recommended."
    elif "cftr" in genetic_markers.lower():
        return "Genetic analysis detected common CFTR gene mutations, consistent with Cystic Fibrosis. Sweat chloride test correlation advised."
    elif "f8 gene" in genetic_markers.lower() or "factor viii" in genetic_markers.lower():
        return "Genetic analysis shows mutations in the F8 gene, confirming Hemophilia A. Severity depends on the specific mutation."
    return f"Genetic analysis for '{genetic_markers}' yielded no definitive rare disease markers."

tools = [
    Tool(
        name="PubMed_Search",
        func=search_pubmed,
        description="Searches medical literature on PubMed for diagnostic information, symptoms, and treatments for various conditions, including rare diseases.",
    ),
    Tool(
        name="Orphanet_Rare_Disease_Search",
        func=search_orphanet,
        description="Searches the Orphanet database specifically for information on rare diseases, including their prevalence, symptoms, and management strategies. Requires a specific disease name.",
    ),
    Tool(
        name="Genetic_Panel_Analysis",
        func=run_genetic_analysis,
        description="Runs a simulated genetic panel analysis based on provided genetic markers or suspected gene involvement. Useful for confirming genetic disorders.",
    ),
]


# --- Memory Module ---
class MemoryModule:
    def __init__(self, embeddings_model, persist_directory="./chroma_db"):
        self.patient_data = pd.DataFrame(columns=["patient_id", "symptoms", "history", "lab_results", "diagnoses"])
        # Initialize ChromaDB client and collection
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.knowledge_base = Chroma(
            client=self.client,
            collection_name="medical_knowledge",
            embedding_function=embeddings_model,
        )
        self._initialize_knowledge_base()

    def _initialize_knowledge_base(self):
        # Populate with some initial mock medical knowledge if empty
        if self.knowledge_base._collection.count() == 0:
            print("DEBUG: Populating initial medical knowledge base...")
            self.knowledge_base.add_texts(
                texts=[
                    "Creutzfeldt-Jakob disease (CJD) is a rare, fatal neurodegenerative disease. It is caused by an infectious agent called a prion. Symptoms include rapidly progressive dementia, myoclonus, and cerebellar dysfunction. Diagnosis often involves MRI, EEG, and CSF analysis (14-3-3 protein, RT-QuIC).",
                    "Huntington's disease is a genetic disorder that causes the progressive degeneration of nerve cells in the brain. Symptoms include uncontrolled movements (chorea), cognitive decline, and psychiatric problems. It is caused by a mutation in the HTT gene.",
                    "Cystic Fibrosis is an inherited disorder that causes severe damage to the lungs, digestive system and other organs in the body. Cystic fibrosis affects the cells that produce mucus, sweat and digestive juices. It is caused by a mutation in the CFTR gene.",
                    "Hemophilia A is a genetic disorder caused by missing or defective factor VIII, a clotting protein. It primarily affects males and leads to prolonged bleeding. Treatment involves factor VIII replacement therapy.",
                    "Fabry disease is a rare, X-linked inherited disorder that results from the buildup of a particular type of fat, globotriaosylceramide (Gb3), in the body’s cells. Symptoms can include pain, kidney failure, heart disease, and strokes.",
                ],
                metadatas=[
                    {"source": "NIH", "disease": "CJD"},
                    {"source": "Mayo Clinic", "disease": "Huntington's"},
                    {"source": "CFF", "disease": "Cystic Fibrosis"},
                    {"source": "CDC", "disease": "Hemophilia A"},
                    {"source": "Fabry International", "disease": "Fabry Disease"},
                ],
                ids=[f"doc_{i}" for i in range(5)]
            )
            print("DEBUG: Medical knowledge base populated.")

    def add_patient_data(self, patient_id: str, symptoms: str, history: str, lab_results: str, diagnoses: List[str] = None):
        """Adds or updates structured patient data."""
        new_data = {
            "patient_id": patient_id,
            "symptoms": symptoms,
            "history": history,
            "lab_results": lab_results,
            "diagnoses": diagnoses if diagnoses else []
        }
        # Check if patient_id already exists to update or add new
        if patient_id in self.patient_data["patient_id"].values:
            for col, value in new_data.items():
                self.patient_data.loc[self.patient_data["patient_id"] == patient_id, col] = value
            print(f"DEBUG: Updated patient data for ID: {patient_id}")
        else:
            self.patient_data = pd.concat([self.patient_data, pd.DataFrame([new_data])], ignore_index=True)
            print(f"DEBUG: Added new patient data for ID: {patient_id}")

    def get_patient_history(self, patient_id: str) -> str:
        """Retrieves structured patient history for a given patient ID."""
        print(f"DEBUG: Retrieving patient history for ID: {patient_id}")
        patient_info = self.patient_data[self.patient_data["patient_id"] == patient_id]
        if not patient_info.empty:
            return patient_info.iloc[0].to_json()
        return f"No patient history found for ID: {patient_id}"

    def retrieve_medical_knowledge(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Retrieves relevant medical knowledge from the vector store based on a query."""
        print(f"DEBUG: Retrieving medical knowledge for query: {query}")
        docs = self.knowledge_base.similarity_search(query, k=k)
        # Format docs into a readable string for the LLM
        formatted_docs = []
        for i, doc in enumerate(docs):
            formatted_docs.append(f"Document {i+1} (Source: {doc.metadata.get('source', 'N/A')} - Disease: {doc.metadata.get('disease', 'N/A')}):\n{doc.page_content}")
        return "\n\n".join(formatted_docs) if formatted_docs else "No relevant medical knowledge found."


# --- Core LLM Orchestrator & Context Management System ---
class DiagnosticAgent:
    def __init__(self, llm, tools, memory_module):
        self.llm = llm
        self.tools = tools
        self.memory_module = memory_module
        self.conversational_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Add memory retrieval tools to the agent's available tools
        memory_retrieval_tool = Tool(
            name="Patient_History_Retriever",
            func=self.memory_module.get_patient_history,
            description="Retrieves comprehensive historical data for a given patient ID (e.g., 'P101'), including symptoms, medical history, lab results, and previous diagnoses. Input should be a patient ID string.",
        )
        medical_knowledge_retrieval_tool = Tool(
            name="Medical_Knowledge_Retriever",
            func=self.memory_module.retrieve_medical_knowledge,
            description="Retrieves relevant medical articles, research papers, and guidelines from the internal medical knowledge base based on a query. Input should be a string query related to medical concepts or diseases.",
        )
        self.all_tools = self.tools + [memory_retrieval_tool, medical_knowledge_retrieval_tool]

        # Define the agent's prompt template
        self.agent_prompt = PromptTemplate.from_template(
            """You are a highly specialized AI assistant for diagnosing rare diseases and assisting with treatment planning for clinicians.
            You have access to a patient's historical data, an extensive medical knowledge base, and external diagnostic tools.
            Your goal is to accurately diagnose rare diseases, propose further diagnostic steps, and suggest personalized treatment plans.
            Always prioritize patient safety and evidence-based medicine. Provide your reasoning step-by-step.

            If a patient ID is provided, retrieve their history first using the Patient_History_Retriever tool.
            If symptoms are provided, query the Medical_Knowledge_Retriever and Orphanet_Rare_Disease_Search tools.
            If genetic markers are mentioned, use the Genetic_Panel_Analysis tool.
            Always explain your reasoning and potential next steps to the clinician.

            TOOLS:
            {tools}

            FORMAT INSTRUCTIONS:
            {agent_scratchpad}

            BEGINNING OF CONVERSATION:
            {chat_history}
            Human: {input}
            AI:"""
        )

        # Create the LangChain React Agent
        self.agent = create_react_agent(
            llm=self.llm,
            tools=self.all_tools,
            prompt=self.agent_prompt
        )

        # Create the Agent Executor to run the agent with memory
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.all_tools,
            verbose=True,
            memory=self.conversational_memory,
            handle_parsing_errors=True # Handles potential parsing errors from LLM output
        )

    def run_diagnosis(self, patient_query: str) -> str:
        """Initiates the diagnostic process for a patient query, returning the AI's response."""
        print(f"\n--- Initiating Diagnosis for Query: '{patient_query}' ---")
        try:
            response = self.agent_executor.invoke({"input": patient_query})
            return response["output"]
        except Exception as e:
            print(f"Error during diagnosis: {e}")
            return f"An error occurred during diagnosis. Please try again. Error: {e}"


# --- High-Level Workflow Simulation ---
if __name__ == "__main__":
    # --- IMPORTANT: Set your OpenAI API key ---
    # Ensure your OpenAI API key is set as an environment variable (recommended):
    # export OPENAI_API_KEY='your_api_key_here'
    # Or uncomment the line below and replace with your key (less secure for production):
    # os.environ["OPENAI_API_KEY"] = "sk-..."

    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set. Please set it to proceed.")
        print("You can set it using: export OPENAI_API_KEY='YOUR_API_KEY'")
        exit()

    print("Initializing Rare Disease Diagnostic System...")
    try:
        llm_instance = OpenAI(temperature=0)
        embeddings_instance = OpenAIEmbeddings()
        memory_module_instance = MemoryModule(embeddings_model=embeddings_instance)
        diagnostic_system = DiagnosticAgent(llm=llm_instance, tools=tools, memory_module=memory_module_instance)
        print("System initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize the system: {e}")
        print("Please ensure all required libraries are installed (langchain, openai, pandas, requests, chromadb) and your OpenAI API key is valid.")
        exit()

    # Simulate adding patient data
    print("\n--- Adding Sample Patient Data ---")
    memory_module_instance.add_patient_data(
        patient_id="P101",
        symptoms="Rapidly progressive dementia, myoclonus, ataxia, visual disturbances.",
        history="65-year-old male, recent onset of cognitive decline. No family history of neurological disorders. Initial MRI showed cortical ribboning.",
        lab_results="CSF 14-3-3 protein elevated. EEG showed periodic sharp wave complexes."
    )
    memory_module_instance.add_patient_data(
        patient_id="P102",
        symptoms="Chronic cough, recurrent lung infections, steatorrhea, salty skin.",
        history="2-year-old female, history of meconium ileus at birth, poor weight gain.",
        lab_results="Sweat chloride test: 98 mEq/L."
    )
    memory_module_instance.add_patient_data(
        patient_id="P103",
        symptoms="Frequent joint bleeds, easy bruising, prolonged bleeding after minor cuts.",
        history="10-year-old male, maternal uncle has similar bleeding disorder. Activated partial thromboplastin time (aPTT) is prolonged.",
        lab_results="Factor VIII activity level is 5% of normal."
    )

    # Example Diagnostic Queries
    print("\n--- Running Diagnostic Queries ---")

    query_1 = "Patient P101 presents with rapidly progressive dementia, myoclonus, and ataxia. CSF 14-3-3 protein is elevated. What is the most likely rare disease, and what further diagnostic steps are recommended?"
    response_1 = diagnostic_system.run_diagnosis(query_1)
    print(f"\nAI System Response for P101:\n{response_1}")
    print("\n" + "="*80 + "\n")

    query_2 = "A 2-year-old, Patient P102, has a history of chronic cough, recurrent lung infections, and a sweat chloride test of 98 mEq/L. What is the diagnosis and what treatment considerations are important?"
    response_2 = diagnostic_system.run_diagnosis(query_2)
    print(f"\nAI System Response for P102:\n{response_2}")
    print("\n" + "="*80 + "\n")

    query_3 = "Patient P103 has a family history of bleeding disorders and low Factor VIII activity. What is the suspected diagnosis and what genetic marker should be investigated?"
    response_3 = diagnostic_system.run_diagnosis(query_3)
    print(f"\nAI System Response for P103:\n{response_3}")
    print("\n" + "="*80 + "\n")

    query_4 = "Considering Patient P101 again, if genetic tests confirm a PRNP gene mutation, how does this impact the diagnosis and what specific management strategies are advised?"
    response_4 = diagnostic_system.run_diagnosis(query_4)
    print(f"\nAI System Response for P101 with genetic data:\n{response_4}")
    print("\n" + "="*80 + "\n")

    query_5 = "I need to learn more about Fabry disease. Can you retrieve information from the medical knowledge base?"
    response_5 = diagnostic_system.run_diagnosis(query_5)
    print(f"\nAI System Response for Fabry disease knowledge:\n{response_5}")
    print("\n" + "="*80 + "\n")

    query_6 = "What are the common symptoms and diagnostic methods for Huntington's disease?"
    response_6 = diagnostic_system.run_diagnosis(query_6)
    print(f"\nAI System Response for Huntington's disease:\n{response_6}")
    print("\n" + "="*80 + "\n")