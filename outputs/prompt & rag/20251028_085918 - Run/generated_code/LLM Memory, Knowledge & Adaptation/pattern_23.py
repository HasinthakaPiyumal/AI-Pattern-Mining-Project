import collections
import hashlib
import json
import random
from typing import Dict, List, Any


class MockEmbeddingModel:
    def encode(self, text: str) -> List[float]:
        hash_value = int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16)
        return [(hash_value % (i + 1)) / 1000.0 for i in range(128)]

mock_embedding_model = MockEmbeddingModel()

class MockLLM:
    def generate(self, prompt: str, complexity: str) -> str:
        if "treatment plan" in complexity:
            return f"Based on complex analysis and knowledge base, a detailed treatment plan suggestion for '{prompt}' is: [Mock Detailed Plan]."
        elif "diagnostic assistance" in complexity:
            return f"Based on your diagnostic query '{prompt}', the LLM suggests considering: [Mock Diagnostic Insights]."
        else:
            return f"Responding to your simple query: '{prompt}'. Here is some general information: [Mock General Info]."

mock_llm = MockLLM()


class QueryComplexityClassifier:
    def __init__(self):
        self.rules = {
            "treatment plan": ["treatment plan", "therapy options", "management strategy"],
            "diagnostic assistance": ["diagnose", "differential diagnosis", "symptoms suggest"],
            "simple information retrieval": ["what is", "how many", "definition of"]
        }
        self.training_data_generator_counter = 0

    def automatic_classifier_training_data_generation(self, query: str, classification: str):
        self.training_data_generator_counter += 1
        return f"Generated training data point: '{query}' classified as '{classification}' (count: {self.training_data_generator_counter})"

    def classify(self, query: str) -> str:
        query_lower = query.lower()
        if any(keyword in query_lower for keyword in self.rules["treatment plan"]):
            return "treatment plan"
        if any(keyword in query_lower for keyword in self.rules["diagnostic assistance"]):
            return "diagnostic assistance"
        return "simple information retrieval"


class MemoryManager:
    def __init__(self, max_short_term_memory_size: int = 10):
        self.short_term_memory: collections.deque = collections.deque(maxlen=max_short_term_memory_size)
        self.long_term_memory: Dict[str, List[str]] = {}

    def add_short_term_memory(self, interaction: str):
        self.short_term_memory.append(interaction)

    def get_short_term_memory(self) -> List[str]:
        return list(self.short_term_memory)

    def add_long_term_memory(self, patient_id: str, record: str):
        if patient_id not in self.long_term_memory:
            self.long_term_memory[patient_id] = []
        self.long_term_memory[patient_id].append(record)

    def get_long_term_memory(self, patient_id: str) -> List[str]:
        return self.long_term_memory.get(patient_id, [])

    def augment_memory(self, current_query: str, patient_id: str) -> str:
        short_term_context = " ".join(self.get_short_term_memory())
        long_term_context = " ".join(self.get_long_term_memory(patient_id))
        
        augmented_context = (
            f"Current query: {current_query}\n"
            f"Recent interactions: {short_term_context}\n"
            f"Patient historical data (ID: {patient_id}): {long_term_context}"
        )
        return augmented_context

    def manage_long_context(self, context: str, max_tokens: int = 500) -> str:
        words = context.split()
        if len(words) > max_tokens:
            return " ".join(words[:max_tokens]) + " [CONTEXT TRUNCATED]"
        return context


class KnowledgeManager:
    def __init__(self):
        self.parametric_memory: Dict[str, str] = {
            "aspirin_dosage_adult": "81-325 mg daily for cardiovascular prevention; 300-600 mg for pain relief.",
            "diabetes_type2_symptoms": "Increased thirst, frequent urination, increased hunger, fatigue, blurred vision.",
            "hypertension_treatment_guideline": "Lifestyle changes (diet, exercise), ACE inhibitors, ARBs, calcium channel blockers, diuretics."
        }
        self.non_parametric_memory_documents: List[str] = []
        self.non_parametric_memory_index = None
        self.current_index_version = "v1.0"
        self._build_mock_index()

    def _build_mock_index(self):
        self.non_parametric_memory_index = []
        if self.non_parametric_memory_documents:
            for doc in self.non_parametric_memory_documents:
                embedding = mock_embedding_model.encode(doc)
                self.non_parametric_memory_index.append((embedding, doc))
        else:
            initial_docs = [
                "A comprehensive review of cardiac arrest management protocols.",
                "Latest guidelines on managing chronic obstructive pulmonary disease (COPD).",
                "Understanding the efficacy of novel cancer immunotherapies.",
                "Pediatric fever management: A guide for emergency physicians."
            ]
            for doc in initial_docs:
                self.non_parametric_memory_documents.append(doc)
                embedding = mock_embedding_model.encode(doc)
                self.non_parametric_memory_index.append((embedding, doc))

    def add_parametric_fact(self, key: str, value: str):
        self.parametric_memory[key] = value

    def get_parametric_fact(self, key: str) -> str:
        return self.parametric_memory.get(key, "Fact not found in parametric memory.")

    def add_non_parametric_document(self, document: str):
        self.non_parametric_memory_documents.append(document)
        self._build_mock_index()

    def search_non_parametric_memory(self, query: str, top_k: int = 3) -> List[str]:
        query_embedding = mock_embedding_model.encode(query)
        
        if not self.non_parametric_memory_index:
            return []

        def cosine_similarity(vec1, vec2):
            dot_product = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
            magnitude1 = sum(v1**2 for v1 in vec1)**0.5
            magnitude2 = sum(v2**2 for v2 in vec2)**0.5
            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0
            return dot_product / (magnitude1 * magnitude2)

        similarities = []
        for i, (doc_embedding, doc_text) in enumerate(self.non_parametric_memory_index):
            sim = cosine_similarity(query_embedding, doc_embedding)
            similarities.append((sim, doc_text))
        
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [doc for sim, doc in similarities[:top_k]]

    def hotswap_index(self, new_index_data: List[str], version: str = "vX.Y"):
        self.non_parametric_memory_documents = new_index_data
        self._build_mock_index()
        self.current_index_version = version
        return f"Non-parametric index hotswapped to version {version}."

    def read_non_parametric_memory(self, query: str) -> str:
        results = self.search_non_parametric_memory(query, top_k=1)
        return results[0] if results else "No relevant non-parametric memory found."

    def write_non_parametric_memory(self, content_to_add: str, update_existing: bool = False, query_to_match: str = None):
        if update_existing and query_to_match:
            for i, doc in enumerate(self.non_parametric_memory_documents):
                if query_to_match.lower() in doc.lower():
                    self.non_parametric_memory_documents[i] = content_to_add
                    self._build_mock_index()
                    return f"Updated non-parametric memory matching '{query_to_match}'."
            return "No matching document found to update."
        else:
            self.add_non_parametric_document(content_to_add)
            return "Added new non-parametric memory content."


class LLMInteractionLayer:
    def __init__(self, llm_model: MockLLM):
        self.llm_model = llm_model

    def generate_response(self, query: str, context: str, knowledge_facts: List[str], complexity: str) -> str:
        base_prompt = f"As an intelligent medical assistant, provide a comprehensive response to the following query, considering the patient's context and factual knowledge provided.\n"
        
        if complexity == "treatment plan":
            instruction = "Focus on generating a detailed and actionable treatment plan recommendation."
        elif complexity == "diagnostic assistance":
            instruction = "Provide diagnostic insights and potential differential diagnoses."
        else:
            instruction = "Provide concise and accurate information."

        full_prompt = (
            f"{base_prompt}\n"
            f"Query: {query}\n"
            f"Instructions: {instruction}\n"
            f"Patient Context: {context}\n"
            f"Relevant Knowledge: {' '.join(knowledge_facts)}\n"
            f"Generate a response:"
        )
        return self.llm_model.generate(full_prompt, complexity)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

query_classifier = QueryComplexityClassifier()
memory_manager = MemoryManager()
knowledge_manager = KnowledgeManager()
llm_interaction_layer = LLMInteractionLayer(mock_llm)

class MedicalQuery(BaseModel):
    patient_id: str
    query: str

class UpdateMemoryRequest(BaseModel):
    patient_id: str
    record: str

class UpdateKnowledgeRequest(BaseModel):
    key: str = None
    value: str = None
    document: str = None
    update_existing: bool = False
    query_to_match: str = None
    index_data: List[str] = None
    index_version: str = None

@app.post("/medical_assistant/query")
async def process_medical_query(medical_query: MedicalQuery):
    patient_id = medical_query.patient_id
    query = medical_query.query

    complexity = query_classifier.classify(query)
    query_classifier.automatic_classifier_training_data_generation(query, complexity)

    memory_manager.add_short_term_memory(f"User query: {query}")
    augmented_context = memory_manager.augment_memory(query, patient_id)
    managed_context = memory_manager.manage_long_context(augmented_context)

    relevant_parametric_facts = []
    for fact_key, fact_value in knowledge_manager.parametric_memory.items():
        if any(word.lower() in query.lower() for word in fact_key.split('_')):
            relevant_parametric_facts.append(f"{fact_key}: {fact_value}")

    relevant_non_parametric_docs = knowledge_manager.search_non_parametric_memory(query)
    
    all_knowledge = relevant_parametric_facts + relevant_non_parametric_docs

    llm_response = llm_interaction_layer.generate_response(
        query=query,
        context=managed_context,
        knowledge_facts=all_knowledge,
        complexity=complexity
    )
    memory_manager.add_short_term_memory(f"Assistant response: {llm_response}")

    return {"response": llm_response, "query_complexity": complexity}

@app.post("/medical_assistant/add_patient_record")
async def add_patient_record(request: UpdateMemoryRequest):
    memory_manager.add_long_term_memory(request.patient_id, request.record)
    return {"message": f"Record added for patient {request.patient_id}"}

@app.post("/medical_assistant/update_knowledge")
async def update_knowledge_base(request: UpdateKnowledgeRequest):
    if request.key and request.value:
        knowledge_manager.add_parametric_fact(request.key, request.value)
        return {"message": f"Parametric fact '{request.key}' updated."}
    elif request.document:
        if request.update_existing:
            response = knowledge_manager.write_non_parametric_memory(request.document, update_existing=True, query_to_match=request.query_to_match)
            return {"message": response}
        else:
            knowledge_manager.add_non_parametric_document(request.document)
            return {"message": "New non-parametric document added."}
    elif request.index_data and request.index_version:
        response = knowledge_manager.hotswap_index(request.index_data, request.index_version)
        return {"message": response}
    else:
        raise HTTPException(status_code=400, detail="Invalid request for knowledge update.")

@app.get("/medical_assistant/get_knowledge")
async def get_knowledge(query: str):
    parametric_results = []
    for fact_key, fact_value in knowledge_manager.parametric_memory.items():
        if query.lower() in fact_key.lower() or query.lower() in fact_value.lower():
            parametric_results.append(f"{fact_key}: {fact_value}")
    
    non_parametric_results = knowledge_manager.search_non_parametric_memory(query)
    
    human_readable_non_parametric = knowledge_manager.read_non_parametric_memory(query)

    return {
        "parametric_knowledge": parametric_results,
        "non_parametric_search_results": non_parametric_results,
        "human_readable_non_parametric_snapshot": human_readable_non_parametric
    }