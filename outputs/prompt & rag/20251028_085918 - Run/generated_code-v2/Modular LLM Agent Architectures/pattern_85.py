import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel, Field
import json

class Document:
    """Mock for langchain_core.documents.Document"""
    def __init__(self, page_content: str, metadata: Dict = None):
        self.page_content = page_content
        self.metadata = metadata if metadata is not None else {}

    def __repr__(self):
        return f"Document(page_content='{self.page_content[:50]}...', metadata={self.metadata})"

class Embeddings:
    """Mock for langchain_community.embeddings.SentenceTransformerEmbeddings or similar"""
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        logger.debug(f"MockEmbeddings: Embedding {len(texts)} documents...")
        return [[float(i % 128) / 100 for i in range(128)] for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        logger.debug(f"MockEmbeddings: Embedding query: {text[:30]}...")
        return [float(i % 128) / 100 for i in range(128)]

class VectorStore:
    """Mock for langchain_community.vectorstores.Chroma or similar"""
    def __init__(self, embeddings: Embeddings):
        self.documents: List[Document] = []
        self.embeddings = embeddings
        self.id_counter = 0

    def add_documents(self, docs: List[Document]):
        for doc in docs:
            self.documents.append(doc)
            logger.debug(f"MockVectorStore: Added document: {doc.page_content[:50]}...")
        logger.info(f"MockVectorStore: Added {len(docs)} documents.")

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        logger.info(f"MockVectorStore: Performing similarity search for query: '{query[:50]}...' (k={k})")
        if not self.documents:
            return []
        return self.documents[:min(k, len(self.documents))]

class ChatMessage:
    """Mock for langchain_core.messages.HumanMessage, AIMessage, etc."""
    def __init__(self, content: str, role: str):
        self.content = content
        self.role = role

    def __repr__(self):
        return f"ChatMessage(role='{self.role}', content='{self.content[:50]}...')"

class BaseChatModel:
    """Mock for langchain_core.language_models.chat_models.BaseChatModel"""
    def invoke(self, messages: List[ChatMessage]) -> ChatMessage:
        raise NotImplementedError("Subclasses must implement invoke method")

class MockChatModel(BaseChatModel):
    """A simple mock LLM for testing purposes."""
    def __init__(self, response_prefix: str = "Mock LLM Response: "):
        self.response_prefix = response_prefix

    def invoke(self, messages: List[ChatMessage]) -> ChatMessage:
        logger.debug(f"MockChatModel received messages: {[m.content[:50] for m in messages]}")
        user_message_content = next((m.content for m in reversed(messages) if m.role == "user"), "No user message.")
        response_content = f"{self.response_prefix} Based on the input, I've processed your request. Original query context: '{user_message_content[:100]}...'"
        return ChatMessage(content=response_content, role="assistant")

class TreatmentStep(BaseModel):
    step_number: int = Field(..., description="Sequential number for the treatment step.")
    description: str = Field(..., description="Detailed description of the treatment action.")
    rationale: str = Field(..., description="Explanation for why this step is recommended.")

class TreatmentPlan(BaseModel):
    patient_id: str = Field(..., description="Unique identifier for the patient.")
    diagnosis: str = Field(..., description="Medical diagnosis for the patient's condition.")
    plan_overview: str = Field(..., description="A summary of the overall treatment strategy.")
    treatment_steps: List[TreatmentStep] = Field(default_factory=list, description="A list of detailed, sequential treatment steps.")
    follow_up: str = Field(..., description="Instructions or recommendations for future follow-up.")

class FastAPI:
    """A minimal mock for FastAPI to simulate endpoint calls."""
    def __init__(self):
        self.routes = {}
        self.medical_assistant_app = None

    def post(self, path: str):
        def decorator(func):
            self.routes[path] = func
            return func
        return decorator

    def run(self, medical_assistant_app_instance: Any):
        logger.info("MockFastAPI: Application started. Use `call_endpoint` to simulate requests.")
        self.medical_assistant_app = medical_assistant_app_instance

    def call_endpoint(self, path: str, data: Dict) -> Dict:
        if path in self.routes:
            logger.info(f"MockFastAPI: Simulating POST request to {path} with data: {data.get('query', 'N/A')[:50]}...")
            return self.routes[path](data)
        else:
            logger.error(f"MockFastAPI: No endpoint found for {path}")
            return {"error": "Endpoint not found"}

class MedicalKnowledgeRetrievalModule:
    def __init__(self):
        self.embeddings = Embeddings()
        self.vector_store = VectorStore(self.embeddings)
        self._load_initial_knowledge()
        logger.info("MedicalKnowledgeRetrievalModule: Initialized.")

    def _load_initial_knowledge(self):
        docs = [
            Document(page_content="""
            Aspirin is a nonsteroidal anti-inflammatory drug (NSAID) used for pain, fever, and inflammation.
            It's also an antiplatelet agent, used to prevent blood clots. Contraindications include
            bleeding disorders and allergies to NSAIDs. Side effects: GI upset, bleeding.
            """, metadata={"source": "Drugs.com", "topic": "Aspirin"}),
            Document(page_content="""
            Metformin is a first-line medication for type 2 diabetes, primarily working by
            decreasing glucose production by the liver and increasing insulin sensitivity.
            Common side effects include nausea, diarrhea. Rare but serious side effect: lactic acidosis.
            """, metadata={"source": "Mayo Clinic", "topic": "Metformin, Diabetes"}),
            Document(page_content="""
            Hypertension (high blood pressure) management involves lifestyle modifications
            (diet, exercise) and medications like ACE inhibitors (e.g., Lisinopril), ARBs,
            calcium channel blockers, and diuretics. Regular monitoring is crucial.
            """, metadata={"source": "AHA", "topic": "Hypertension"}),
            Document(page_content="""
            Penicillin allergy can manifest as hives, itching, swelling, or anaphylaxis.
            Patients with penicillin allergy should avoid all penicillins and typically cephalosporins.
            Alternative antibiotics should be considered.
            """, metadata={"source": "CDC", "topic": "Penicillin Allergy"}),
            Document(page_content="""
            Symptoms of a common cold typically include runny nose, sore throat, cough, congestion,
            slight body aches or a mild headache, and sneezing. It is caused by viruses.
            """, metadata={"source": "NIH", "topic": "Common Cold"}),
        ]
        self.vector_store.add_documents(docs)

    def retrieve_knowledge(self, query: str) -> List[str]:
        retrieved_docs = self.vector_store.similarity_search(query, k=3)
        return [doc.page_content for doc in retrieved_docs]

class StructuredTreatmentPlanningModule:
    def __init__(self):
        self.planner_llm = MockChatModel(response_prefix="Treatment Planning Draft LLM Output: ")
        logger.info("StructuredTreatmentPlanningModule: Initialized.")

    def generate_plan(self, patient_data: Dict, medical_knowledge: List[str]) -> TreatmentPlan:
        logger.info("StructuredTreatmentPlanningModule: Generating treatment plan...")
        combined_context = (
            f"Patient Data:\n{json.dumps(patient_data, indent=2)}\n\n"
            f"Relevant Medical Knowledge:\n{'- ' + '\n- '.join(medical_knowledge)}\n\n"
            "Based on the patient data and medical knowledge, propose a structured treatment plan. "
            "Ensure the plan includes a diagnosis, an overview, specific treatment steps with rationale, "
            "and follow-up recommendations. Pay attention to any stated allergies. "
            "Format your response as a JSON object strictly following the TreatmentPlan Pydantic schema."
        )

        llm_response_content = self.planner_llm.invoke([ChatMessage(content=combined_context, role="user")]).content

        try:
            diagnosis_str = "Simulated Diagnosis: Viral infection (common cold-like symptoms)"
            plan_overview_str = "Symptomatic treatment and monitoring."
            steps = [
                TreatmentStep(step_number=1, description="Rest and hydration.", rationale="Support body's immune response."),
                TreatmentStep(step_number=2, description="Over-the-counter pain relievers (e.g., acetaminophen), avoiding NSAIDs due to potential GI upset and allergy concerns.", rationale="Manage headache and fever. Penicillin allergy noted, so avoiding NSAIDs is a general precaution to avoid cross-reactivity or exacerbate potential unknown sensitivities."),
                TreatmentStep(step_number=3, description="Monitor symptoms; seek medical attention if worsening.", rationale="Identify potential complications."),
            ]
            follow_up_str = "If symptoms persist beyond 7 days or worsen significantly, consult a physician."

            if "chest pain" in patient_data.get("symptoms", "").lower() or "hypertension" in patient_data.get("medical_history", "").lower():
                 diagnosis_str = "Simulated Diagnosis: Possible Cardiac Event / Hypertension Complication"
                 plan_overview_str = "Immediate medical evaluation is critical."
                 steps = [
                     TreatmentStep(step_number=1, description="Call emergency services immediately (e.g., 911).", rationale="Chest pain with hypertension is a medical emergency."),
                     TreatmentStep(step_number=2, description="Take prescribed blood pressure medication if due, but do not self-medicate for chest pain.", rationale="Maintain current regimen, but avoid new medications without medical advice."),
                     TreatmentStep(step_number=3, description="Rest in a comfortable position and loosen tight clothing.", rationale="Reduce cardiac strain."),
                 ]
                 follow_up_str = "Follow all instructions from emergency medical personnel and hospital staff."

            if "penicillin" in patient_data.get("medical_history", "").lower() and "acetaminophen" in [s.description.lower() for s in steps]:
                for step in steps:
                    if "pain relievers" in step.description.lower():
                        step.description = "Over-the-counter pain relievers like acetaminophen (avoiding NSAIDs due to penicillin allergy and general precaution)."
                        step.rationale = "Manage headache and fever. Acetaminophen is generally safe for penicillin allergy patients; NSAIDs are avoided as a precaution."

            generated_plan = TreatmentPlan(
                patient_id=patient_data.get("id", "UNKNOWN_PATIENT"),
                diagnosis=diagnosis_str,
                plan_overview=plan_overview_str,
                treatment_steps=steps,
                follow_up=follow_up_str
            )
            logger.info("StructuredTreatmentPlanningModule: Successfully generated mock treatment plan.")
            return generated_plan
        except Exception as e:
            logger.error(f"StructuredTreatmentPlanningModule: Error creating mock TreatmentPlan: {e}. Returning fallback plan.")
            return TreatmentPlan(
                patient_id=patient_data.get("id", "UNKNOWN_PATIENT"),
                diagnosis="Fallback: Unable to generate specific diagnosis",
                plan_overview="Please consult a healthcare professional for personalized advice.",
                treatment_steps=[
                    TreatmentStep(step_number=1, description="Seek immediate medical consultation.", rationale="Complex symptoms require expert evaluation.")
                ],
                follow_up="As advised by medical professional."
            )

class MedicalAssistantApp:
    def __init__(self):
        self.retrieval_module = MedicalKnowledgeRetrievalModule()
        self.planning_module = StructuredTreatmentPlanningModule()
        self.main_llm = MockChatModel(response_prefix="Medical Assistant Final LLM Response: ")
        logger.info("MedicalAssistantApp: Initialized.")

    def process_query(self, patient_data: Dict, user_query: str) -> Dict:
        logger.info(f"MedicalAssistantApp: Processing query '{user_query[:50]}' for patient {patient_data.get('id', 'N/A')}")

        medical_knowledge_docs = self.retrieval_module.retrieve_knowledge(user_query)
        formatted_medical_knowledge = "\n".join([f"- {doc_content}" for doc_content in medical_knowledge_docs])
        logger.info(f"MedicalAssistantApp: Retrieved {len(medical_knowledge_docs)} medical knowledge documents.")

        treatment_plan = self.planning_module.generate_plan(patient_data, medical_knowledge_docs)
        logger.info("MedicalAssistantApp: Generated draft treatment plan.")

        combined_context = (
            f"Patient Information:\n{json.dumps(patient_data, indent=2)}\n\n"
            f"Relevant Medical Knowledge:\n{formatted_medical_knowledge}\n\n"
            f"Draft Treatment Plan:\n{treatment_plan.model_dump_json(indent=2)}\n\n"
            f"User Query: {user_query}\n\n"
            "Based on ALL the above information (patient data, medical knowledge, draft plan, and user query), "
            "provide a comprehensive, empathetic, and evidence-based response. "
            "Address the user's specific questions, refine the treatment plan if needed, "
            "and ensure any allergy information is respected in recommendations. "
            "Keep the language clear and accessible for a patient."
        )

        final_llm_response_message = self.main_llm.invoke([ChatMessage(content=combined_context, role="user")])
        final_llm_response_content = final_llm_response_message.content
        logger.info("MedicalAssistantApp: Main LLM generated final response.")

        return {
            "query": user_query,
            "patient_data": patient_data,
            "retrieved_knowledge": medical_knowledge_docs,
            "draft_treatment_plan": treatment_plan.model_dump(),
            "final_medical_assistant_response": final_llm_response_content
        }

app = FastAPI()
medical_assistant_app_instance = MedicalAssistantApp()

@app.post("/medical_assistant")
def medical_assistant_endpoint(request_data: Dict):
    patient_data = request_data.get("patient_data", {})
    user_query = request_data.get("query", "Provide a general medical assessment.")
    return medical_assistant_app_instance.process_query(patient_data, user_query)

if __name__ == "__main__":
    load_dotenv()

    logger.add("medical_assistant.log", rotation="10 MB", level="INFO")
    logger.add(os.sys.stderr, level="INFO")
    logger.info("Starting Medical Assistant Application (simulated execution)")

    app.run(medical_assistant_app_instance)

    sample_patient_data_1 = {
        "id": "P001",
        "name": "John Doe",
        "age": 45,
        "gender": "Male",
        "symptoms": "Headache, mild fever, body aches for 2 days. Feels like a cold.",
        "medical_history": "No major chronic diseases. Allergies: Penicillin."
    }
    sample_query_1 = "What could be the possible diagnosis for my symptoms and what is the recommended treatment plan? Also, are there any contraindications with penicillin allergy?"

    logger.info("\n--- Simulating Request 1: Common Cold Symptoms with Penicillin Allergy ---")
    response1 = app.call_endpoint(
        "/medical_assistant",
        {"patient_data": sample_patient_data_1, "query": sample_query_1}
    )
    print("\nSimulated Response 1:")
    print(json.dumps(response1, indent=2))

    logger.info("\n--- Simulating Request 2: Chest Pain with Hypertension History ---")
    sample_patient_data_2 = {
        "id": "P002",
        "name": "Jane Smith",
        "age": 60,
        "gender": "Female",
        "symptoms": "Sudden onset chest pain, shortness of breath, radiating to left arm.",
        "medical_history": "Hypertension (on Lisinopril), Type 2 Diabetes."
    }
    sample_query_2 = "I'm experiencing severe chest pain and shortness of breath. What should I do immediately, and what are potential causes given my history of hypertension and diabetes?"
    response2 = app.call_endpoint(
        "/medical_assistant",
        {"patient_data": sample_patient_data_2, "query": sample_query_2}
    )
    print("\nSimulated Response 2:")
    print(json.dumps(response2, indent=2))

    logger.info("\n--- Simulating Request 3: General Query without specific symptoms ---")
    sample_patient_data_3 = {
        "id": "P003",
        "name": "Alice Brown",
        "age": 30,
        "gender": "Female",
        "medical_history": "No known conditions."
    }
    sample_query_3 = "What are general recommendations for maintaining good health?"
    response3 = app.call_endpoint(
        "/medical_assistant",
        {"patient_data": sample_patient_data_3, "query": sample_query_3}
    )
    print("\nSimulated Response 3:")
    print(json.dumps(response3, indent=2))