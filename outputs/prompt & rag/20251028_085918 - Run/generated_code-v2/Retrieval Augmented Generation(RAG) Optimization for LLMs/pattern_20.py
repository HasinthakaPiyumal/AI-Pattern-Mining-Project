import os
import json
from typing import List, Dict, Any

# Mocking external libraries for self-contained execution
# In a real application, you would install these:
# pip install langchain openai chromadb sentence-transformers python-dotenv

class MockOpenAI:
    def __init__(self, model_name="gpt-4", temperature=0.7):
        self.model_name = model_name
        self.temperature = temperature
        self.responses = {
            "initial": "Based on the symptoms of severe headache and stiff neck, I initially consider meningitis or subarachnoid hemorrhage. To differentiate, I need more information on \"fever presence\" and \"recent trauma\".",
            "fever_present": "Given the presence of fever along with headache and stiff neck, meningitis is strongly suggested. Could you retrieve information on typical \"meningitis diagnostic criteria\" and \"treatment protocols\"?",
            "no_fever": "Without fever, subarachnoid hemorrhage becomes more likely, especially with sudden onset severe headache. Please search for \"subarachnoid hemorrhage diagnostic imaging\" and \"management guidelines\".",
            "meningitis_docs": "The retrieved meningitis diagnostic criteria (lumbar puncture for CSF analysis) and treatment protocols (antibiotics) align with the current findings. I'm confident in a meningitis diagnosis. Consider performing a lumbar puncture for confirmation and initiating broad-spectrum antibiotics. Are there any \"contraindications for lumbar puncture\" or patient allergies?",
            "sah_docs": "The retrieved information on subarachnoid hemorrhage diagnostic imaging (CT scan) and management guidelines (aneurysm coiling/clipping) is crucial. Given the patient's sudden severe headache, a CT scan is highly recommended. Are there any \"risk factors for aneurysmal rupture\" in this patient?",
            "lumbar_puncture_contraindications": "Considering no contraindications for lumbar puncture and no known allergies, proceed with lumbar puncture and empiric antibiotics. Final Diagnosis: Suspected Meningitis. Next steps: Lumbar Puncture, empiric antibiotic therapy, neuro consult.",
            "aneurysm_risk_factors": "No specific risk factors for aneurysm rupture found. Proceed with urgent CT scan. Final Diagnosis: Suspected Subarachnoid Hemorrhage. Next steps: Urgent CT scan, neurosurgical consult, blood pressure management.",
            "default": "I need more information. Please elaborate or provide additional context. What other specific \"medical conditions\" could cause these symptoms?"
        }
        self.current_state = "initial"

    def __call__(self, prompt, **kwargs):
        if "fever presence" in prompt and self.current_state == "initial":
            if "fever is present" in prompt.lower():
                self.current_state = "fever_present"
            else:
                self.current_state = "no_fever"
        elif "meningitis diagnostic criteria" in prompt and self.current_state == "fever_present":
            self.current_state = "meningitis_docs"
        elif "subarachnoid hemorrhage diagnostic imaging" in prompt and self.current_state == "no_fever":
            self.current_state = "sah_docs"
        elif "contraindications for lumbar puncture" in prompt and self.current_state == "meningitis_docs":
            self.current_state = "lumbar_puncture_contraindications"
        elif "risk factors for aneurysmal rupture" in prompt and self.current_state == "sah_docs":
            self.current_state = "aneurysm_risk_factors"
        else:
            self.current_state = "default"

        response = self.responses.get(self.current_state, self.responses["default"])
        return {"choices": [{"message": {"content": response}}]}

class MockSentenceTransformer:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name

    def encode(self, texts, convert_to_tensor=False):
        # Simple mock: returns a hash-based vector for reproducibility in mock
        if isinstance(texts, str):
            texts = [texts]
        return [[float(ord(c)) / 100.0 for c in text[:32].ljust(32, '0')] for text in texts] # Fixed length vector for demo

class MockChromaDBClient:
    def __init__(self):
        self.collections = {}

    def get_or_create_collection(self, name):
        if name not in self.collections:
            self.collections[name] = MockChromaCollection(name)
        return self.collections[name]

class MockChromaCollection:
    def __init__(self, name):
        self.name = name
        self.documents = []
        self.metadatas = []
        self.embeddings = []
        self.ids = []
        self.id_counter = 0

    def add(self, documents: List[str], metadatas: List[Dict], embeddings: List[List[float]], ids: List[str] = None):
        for i, doc in enumerate(documents):
            current_id = ids[i] if ids else str(self.id_counter)
            self.documents.append(doc)
            self.metadatas.append(metadatas[i])
            self.embeddings.append(embeddings[i])
            self.ids.append(current_id)
            self.id_counter += 1

    def query(self, query_embeddings: List[List[float]], n_results: int = 5):
        results = []
        if not self.embeddings:
            return {'documents': [[]], 'metadatas': [[]]}

        for q_emb in query_embeddings:
            distances = []
            for i, doc_emb in enumerate(self.embeddings):
                # Simple Euclidean distance for mock
                dist = sum([(q - d)**2 for q, d in zip(q_emb, doc_emb)])**0.5
                distances.append((dist, i))
            distances.sort()

            top_results_docs = []
            top_results_metadatas = []
            for _, idx in distances[:n_results]:
                top_results_docs.append(self.documents[idx])
                top_results_metadatas.append(self.metadatas[idx])
            results.append({'documents': top_results_docs, 'metadatas': top_results_metadatas})
        return {'documents': [r['documents'] for r in results], 'metadatas': [r['metadatas'] for r in results]}


class MockChatOpenAI:
    def __init__(self, model_name: str = "gpt-4", temperature: float = 0.7, **kwargs):
        self.model_name = model_name
        self.temperature = temperature
        self.mock_openai = MockOpenAI(model_name=model_name, temperature=temperature)

    def invoke(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        # Extract the last user message as the prompt for the mock LLM
        user_prompt = ""
        for msg in reversed(messages):
            if msg["role"] == "user":
                user_prompt = msg["content"]
                break
        response = self.mock_openai(user_prompt)
        return {"content": response["choices"][0]["message"]["content"]}


class MockOpenAIEmbeddings:
    def __init__(self, model: str = "text-embedding-ada-002", **kwargs):
        self.model = model
        self.mock_st = MockSentenceTransformer()

    def embed_query(self, text: str) -> List[float]:
        return self.mock_st.encode(text)[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.mock_st.encode(texts)


# --- Knowledge Base Simulation --- #
MEDICAL_KNOWLEDGE_BASE = [
    {
        "doc": "Meningitis is an inflammation of the membranes (meninges) surrounding your brain and spinal cord. Symptoms include sudden onset of fever, headache, and stiff neck. Other symptoms may include nausea, vomiting, confusion, seizures, sleepiness, and sensitivity to light.",
        "metadata": {"source": "Medical Journal A", "topic": "Meningitis Symptoms"}
    },
    {
        "doc": "Diagnostic criteria for bacterial meningitis often involve a lumbar puncture (spinal tap) to analyze cerebrospinal fluid (CSF). Findings typically include elevated white blood cells, low glucose, and high protein in CSF. Blood cultures are also crucial.",
        "metadata": {"source": "Clinical Guidelines B", "topic": "Meningitis Diagnosis"}
    },
    {
        "doc": "Treatment for bacterial meningitis typically involves prompt administration of intravenous antibiotics. Dexamethasone may also be given to reduce inflammation. Viral meningitis often resolves on its own and may only require supportive care.",
        "metadata": {"source": "Clinical Guidelines B", "topic": "Meningitis Treatment"}
    },
    {
        "doc": "Subarachnoid hemorrhage (SAH) is bleeding into the space between your brain and the surrounding membrane (subarachnoid space). It often presents as a sudden, severe headache, often described as 'the worst headache of my life'. Other symptoms include nausea, vomiting, stiff neck, and loss of consciousness.",
        "metadata": {"source": "Medical Journal C", "topic": "SAH Symptoms"}
    },
    {
        "doc": "Diagnostic imaging for subarachnoid hemorrhage primarily involves a non-contrast CT scan of the head, which can detect blood in the subarachnoid space. If CT is negative but suspicion remains high, a lumbar puncture can be performed. CT angiography (CTA) is used to identify the source of bleeding, such as an aneurysm.",
        "metadata": {"source": "Clinical Guidelines D", "topic": "SAH Diagnosis"}
    },
    {
        "doc": "Management of aneurysmal subarachnoid hemorrhage focuses on securing the aneurysm (surgical clipping or endovascular coiling) to prevent rebleeding, managing blood pressure, and preventing complications like vasospasm. Prognosis depends on the severity of the initial bleed.",
        "metadata": {"source": "Clinical Guidelines D", "topic": "SAH Management"}
    },
    {
        "doc": "Contraindications for lumbar puncture include increased intracranial pressure (risk of herniation), local skin infection, coagulopathy, and spinal cord compression. Always assess for papilledema or focal neurological deficits before performing.",
        "metadata": {"source": "Medical Procedures E", "topic": "Lumbar Puncture Contraindications"}
    },
    {
        "doc": "Risk factors for aneurysmal rupture include hypertension, smoking, excessive alcohol consumption, cocaine use, and a family history of SAH. Larger aneurysms are also at higher risk of rupture.",
        "metadata": {"source": "Medical Journal F", "topic": "Aneurysm Risk Factors"}
    },
]

class MedicalDiagnosticAssistant:
    def __init__(self, max_iterations: int = 5):
        # Initialize LLM (Mock or OpenAI)
        # os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY" # Uncomment and set your API key
        self.llm = MockChatOpenAI(model_name="gpt-4", temperature=0.7) # Using MockChatOpenAI

        # Initialize Embedding Model (Mock or OpenAIEmbeddings)
        self.embeddings = MockOpenAIEmbeddings() # Using MockOpenAIEmbeddings

        # Initialize Vector Database (Mock or ChromaDB)
        self.client = MockChromaDBClient()
        self.collection = self.client.get_or_create_collection("medical_knowledge")
        self._populate_knowledge_base()

        self.max_iterations = max_iterations

    def _populate_knowledge_base(self):
        docs = [item["doc"] for item in MEDICAL_KNOWLEDGE_BASE]
        metadatas = [item["metadata"] for item in MEDICAL_KNOWLEDGE_BASE]
        # Generate embeddings using the mock embedding model
        embedded_docs = self.embeddings.embed_documents(docs)
        self.collection.add(documents=docs, metadatas=metadatas, embeddings=embedded_docs)
        print("\nMedical knowledge base populated with dummy data.")

    def _retrieve_documents(self, query: str, n_results: int = 3) -> List[str]:
        query_embedding = self.embeddings.embed_query(query)
        results = self.collection.query(query_embeddings=[query_embedding], n_results=n_results)
        retrieved_docs = [doc for sublist in results['documents'] for doc in sublist]
        print(f"Retrieved {len(retrieved_docs)} documents for query: '{query}'")
        return retrieved_docs

    def _refine_query(self, llm_output: str) -> List[str]:
        # Simple rule-based query refinement for demonstration.
        # In a real system, this could be another small LLM call or more sophisticated NLP.
        keywords = []
        if "need more information on" in llm_output.lower():
            parts = llm_output.split("need more information on", 1)[1].split("and")
            for part in parts:
                term = part.strip().replace('\"', '').replace('.', '').replace('?', '')
                if term: keywords.append(term)
        elif "could you retrieve information on" in llm_output.lower():
            parts = llm_output.split("could you retrieve information on", 1)[1].split("and")
            for part in parts:
                term = part.strip().replace('\"', '').replace('.', '').replace('?', '')
                if term: keywords.append(term)
        elif "please search for" in llm_output.lower():
            parts = llm_output.split("please search for", 1)[1].split("and")
            for part in parts:
                term = part.strip().replace('\"', '').replace('.', '').replace('?', '')
                if term: keywords.append(term)
        return keywords if keywords else ["general medical information"]

    def diagnose_patient(self, initial_patient_data: str) -> str:
        conversation_history = [
            {"role": "system", "content": "You are a medical diagnostic assistant. Provide diagnostic hypotheses, identify information gaps, and refine queries based on provided medical context."},
            {"role": "user", "content": f"Patient data: {initial_patient_data}"
            }
        ]
        full_context = f"Patient data: {initial_patient_data}\n"
        final_diagnosis = ""

        print(f"\nStarting diagnostic process for: {initial_patient_data}")

        for i in range(self.max_iterations):
            print(f"\n--- Iteration {i+1}/{self.max_iterations} ---")
            
            # LLM Generation
            llm_response_obj = self.llm.invoke(conversation_history)
            llm_output = llm_response_obj["content"]
            print(f"LLM says: {llm_output}")
            conversation_history.append({"role": "assistant", "content": llm_output})
            full_context += f"\nAssistant: {llm_output}\n"

            if "final diagnosis:" in llm_output.lower() or "i'm confident in a" in llm_output.lower() or "next steps:" in llm_output.lower():
                final_diagnosis = llm_output
                break

            # Query Refinement
            refined_queries = self._refine_query(llm_output)
            if not refined_queries:
                print("No new information needs identified. Concluding diagnosis.")
                final_diagnosis = llm_output
                break
            print(f"Refined queries for retrieval: {refined_queries}")

            # Retrieval
            retrieved_info_list = []
            for query in refined_queries:
                retrieved_docs = self._retrieve_documents(query)
                retrieved_info_list.extend(retrieved_docs)
            
            if not retrieved_info_list:
                print("No relevant documents found. Concluding diagnosis.")
                final_diagnosis = llm_output
                break

            retrieved_context = "\n\nRetrieved Medical Information:\n" + "\n---\n".join(retrieved_info_list)
            full_context += retrieved_context

            # Context Augmentation for next iteration
            user_message_for_next_iteration = f"Considering the following new information:\n{retrieved_context}\n\nWhat are your updated diagnostic hypotheses or further information needs?"
            conversation_history.append({"role": "user", "content": user_message_for_next_iteration})
            print("Context augmented for next LLM interaction.")

        if not final_diagnosis:
            final_diagnosis = "Diagnosis inconclusive within the given iterations. Please consult a specialist."

        print("\n--- Diagnostic Process Complete ---")
        print("Final Conclusion:")
        return final_diagnosis


# --- Main Execution --- #
if __name__ == "__main__":
    # Example Usage:
    assistant = MedicalDiagnosticAssistant(max_iterations=5)

    patient_case_1 = "Patient presents with sudden onset of severe headache, stiff neck, and light sensitivity. No history of trauma. Has a fever of 102°F."
    diagnosis_1 = assistant.diagnose_patient(patient_case_1)
    print(f"\nPatient Case 1 Final Diagnosis: {diagnosis_1}")

    print("\n" + "="*80 + "\n")

    # Reset mock LLM state for a new case if necessary
    assistant.llm.mock_openai.current_state = "initial"
    patient_case_2 = "Patient presents with an excruciating headache, described as the 'worst of my life', with sudden onset. No fever, but some nausea. History of hypertension."
    diagnosis_2 = assistant.diagnose_patient(patient_case_2)
    print(f"\nPatient Case 2 Final Diagnosis: {diagnosis_2}")

    print("\n" + "="*80 + "\n")

    # Reset mock LLM state for a new case if necessary
    assistant.llm.mock_openai.current_state = "initial"
    patient_case_3 = "Patient reports chronic mild headache, fatigue, and occasional dizziness. No acute symptoms. Looking for general consultation."
    diagnosis_3 = assistant.diagnose_patient(patient_case_3)
    print(f"\nPatient Case 3 Final Diagnosis: {diagnosis_3}")
