import chromadb
from sentence_transformers import SentenceTransformer
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import tool
import json

class LLMProxy:
    def invoke(self, prompt: str) -> str:
        # Simulate interaction with a blackbox LLM API
        # In a real scenario, this would make an actual API call (e.g., to OpenAI, Gemini, etc.)
        return f"[LLM BASE RESPONSE]: {prompt}"

class FactualGroundingModule:
    def __init__(self):
        self.client = chromadb.Client()  # In-memory client for demonstration
        self.collection = self.client.get_or_create_collection(name="medical_knowledge")
        self.model = SentenceTransformer("all-MiniLM-L6-v2") # A small, fast sentence transformer model
        self._load_dummy_data()

    def _load_dummy_data(self):
        docs = [
            "Symptoms of common cold include runny nose, sore throat, cough, congestion, and sometimes body aches.",
            "Treatment for common cold is symptomatic relief: rest, fluids, pain relievers (acetaminophen or ibuprofen).",
            "Diabetes Mellitus Type 2 is characterized by insulin resistance and relative insulin deficiency. Symptoms include increased thirst, frequent urination, and unexplained weight loss.",
            "Management of Type 2 Diabetes involves diet, exercise, and medications like metformin or insulin.",
            "Hypertension (High Blood Pressure) is a common condition where the long-term force of the blood against your artery walls is high enough that it may eventually cause health problems, such as heart disease.",
            "Hypertension treatment includes lifestyle changes (diet, exercise) and medications (ACE inhibitors, ARBs, diuretics, beta-blockers)."
        ]
        ids = [f"doc_{i}" for i in range(len(docs))]
        embeddings = self.model.encode(docs).tolist()
        self.collection.add(documents=docs, embeddings=embeddings, ids=ids)

    def retrieve_info(self, query: str) -> str:
        query_embedding = self.model.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=2
        )
        retrieved_docs = [doc for doc_list in results.get("documents", []) for doc in doc_list]
        return " ".join(retrieved_docs) if retrieved_docs else "No relevant medical information found."

class TreatmentRecommendationModule:
    def recommend_treatment(self, diagnosis: str, patient_conditions: list = None) -> str:
        diagnosis = diagnosis.lower()
        if "common cold" in diagnosis:
            return "Recommend rest, fluids, and over-the-counter pain relievers like acetaminophen or ibuprofen."
        elif "diabetes mellitus type 2" in diagnosis or "type 2 diabetes" in diagnosis:
            return "Recommend lifestyle modifications (diet, exercise) and consider medications such as metformin. Patient conditions: {patient_conditions}"
        elif "hypertension" in diagnosis or "high blood pressure" in diagnosis:
            return "Recommend lifestyle changes (DASH diet, regular exercise) and consider medication options like ACE inhibitors or diuretics. Patient conditions: {patient_conditions}"
        else:
            return f"No specific treatment recommendation available for '{diagnosis}' based on current module knowledge."

class PatientHistoryIntegrationModule:
    def __init__(self):
        self.client = chromadb.Client()  # In-memory client for demonstration
        self.collection = self.client.get_or_create_collection(name="patient_history")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self._load_dummy_history()

    def _load_dummy_history(self):
        # Simulate EHR data for a specific patient
        patient_id = "patient_123"
        history_snippets = [
            "Patient_123 has a history of seasonal allergies, treated with antihistamines.",
            "Patient_123's last recorded blood pressure was 130/85 mmHg.",
            "Patient_123 reported no known drug allergies.",
            "Patient_123 underwent appendectomy in 2010.",
            "Patient_123 has family history of type 2 diabetes."
        ]
        ids = [f"{patient_id}_hist_{i}" for i in range(len(history_snippets))]
        embeddings = self.model.encode(history_snippets).tolist()
        self.collection.add(documents=history_snippets, embeddings=embeddings, ids=ids, metadatas=[{"patient_id": patient_id}] * len(history_snippets))

    def get_patient_history(self, patient_id: str, query: str = None) -> str:
        if query:
            query_embedding = self.model.encode(query).tolist()
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=3,
                where={"patient_id": patient_id}
            )
            retrieved_history = [doc for doc_list in results.get("documents", []) for doc in doc_list]
            return " ".join(retrieved_history) if retrieved_history else f"No specific history found for '{query}' for {patient_id}."
        else:
            results = self.collection.get(where={"patient_id": patient_id})
            all_history = [doc for doc_list in results.get("documents", []) for doc in doc_list]
            return " ".join(all_history) if all_history else f"No history found for {patient_id}."

class UtilityModule:
    def convert_units(self, value: float, from_unit: str, to_unit: str) -> float:
        if from_unit.lower() == "cm" and to_unit.lower() == "inch":
            return value / 2.54
        elif from_unit.lower() == "kg" and to_unit.lower() == "lb":
            return value * 2.20462
        # Add more conversions as needed
        return value

class MockChatLLM(BaseChatModel):
    def _generate(self, messages, stop=None, callbacks=None, **kwargs):
        last_message_content = messages[-1].content.lower()
        tool_calls = []

        if "medical information about common cold" in last_message_content:
            tool_calls.append({"name": "retrieve_medical_info", "args": {"query": "common cold symptoms and treatment"}, "id": "call_1"})
        elif "recommend treatment for diabetes" in last_message_content:
            tool_calls.append({"name": "recommend_treatment", "args": {"diagnosis": "Diabetes Mellitus Type 2", "patient_conditions": ["high blood sugar"]}, "id": "call_2"})
        elif "patient history for patient_123 about allergies" in last_message_content:
            tool_calls.append({"name": "get_patient_history_tool", "args": {"patient_id": "patient_123", "query": "allergies"}, "id": "call_3"})
        elif "convert 10 cm to inches" in last_message_content:
            tool_calls.append({"name": "convert_units_tool", "args": {"value": 10.0, "from_unit": "cm", "to_unit": "inch"}, "id": "call_4"})

        if tool_calls:
            return AIMessage(content="", tool_calls=tool_calls)
        else:
            # Fallback for general queries, simulating LLM processing
            return AIMessage(content=f"[Mock LLM]: I processed your query: '{last_message_content}'. How else can I assist?")

    @property
    def _llm_type(self) -> str:
        return "mock-chat-llm"

class MedicalDiagnosticAssistant:
    def __init__(self):
        self.llm_proxy = LLMProxy()
        self.factual_grounding = FactualGroundingModule()
        self.treatment_recommendation = TreatmentRecommendationModule()
        self.patient_history_integration = PatientHistoryIntegrationModule()
        self.utility = UtilityModule()

        # Define tools using Langchain's @tool decorator
        @tool
        def retrieve_medical_info(query: str) -> str:
            """Retrieves factual medical information from a knowledge base."""
            return self.factual_grounding.retrieve_info(query)

        @tool
        def recommend_treatment(diagnosis: str, patient_conditions: list = None) -> str:
            """Recommends treatment options based on a diagnosis and patient conditions."""
            return self.treatment_recommendation.recommend_treatment(diagnosis, patient_conditions or [])

        @tool
        def get_patient_history_tool(patient_id: str, query: str = None) -> str:
            """Retrieves relevant patient medical history for a given patient ID and optional query."""
            return self.patient_history_integration.get_patient_history(patient_id, query)

        @tool
        def convert_units_tool(value: float, from_unit: str, to_unit: str) -> float:
            """Converts a value from one unit to another (e.g., cm to inch, kg to lb)."""
            return self.utility.convert_units(value, from_unit, to_unit)

        self.tools = [
            retrieve_medical_info,
            recommend_treatment,
            get_patient_history_tool,
            convert_units_tool
        ]

        # Initialize the Langchain-like orchestrator with the mock LLM and tools
        self.llm_for_orchestration = MockChatLLM()

    def process_query(self, query: str) -> str:
        messages = [HumanMessage(content=query)]
        
        # Simulate the LLM's decision to call tools
        llm_response = self.llm_for_orchestration.invoke(messages)

        if llm_response.tool_calls:
            tool_outputs = []
            for tool_call in llm_response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                try:
                    # Find and execute the tool
                    executed = False
                    for tool_func in self.tools:
                        if tool_func.name == tool_name:
                            result = tool_func.func(**tool_args)
                            tool_outputs.append(f"[Tool Output - {tool_name}]: {result}")
                            executed = True
                            break
                    if not executed:
                        tool_outputs.append(f"[Error]: Tool '{tool_name}' not found.")
                except Exception as e:
                    tool_outputs.append(f"[Error executing {tool_name}]: {e}")
            
            # After tool execution, potentially send tool outputs back to the LLM Proxy
            # for final synthesis, or directly return tool outputs if they are sufficient.
            # For this example, we'll return the tool outputs directly.
            return "\n".join(tool_outputs)
        else:
            # If the orchestration LLM decides not to use a tool, directly use the LLM Proxy
            return self.llm_proxy.invoke(query)

if __name__ == "__main__":
    assistant = MedicalDiagnosticAssistant()

    print("\n--- Query 1: Get medical info about common cold ---")
    response = assistant.process_query("I need medical information about common cold.")
    print(response)

    print("\n--- Query 2: Recommend treatment for diabetes ---")
    response = assistant.process_query("Can you recommend treatment for diabetes?")
    print(response)

    print("\n--- Query 3: Get patient history for patient_123 about allergies ---")
    response = assistant.process_query("What is the patient history for patient_123 about allergies?")
    print(response)

    print("\n--- Query 4: Convert units ---")
    response = assistant.process_query("Convert 10 cm to inches.")
    print(response)

    print("\n--- Query 5: General query (no tool needed) ---")
    response = assistant.process_query("What is the capital of France?")
    print(response)
