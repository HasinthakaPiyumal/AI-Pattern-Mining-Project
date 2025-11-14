import os
import time
import random
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

class MedicalAssistant:
    def __init__(self, persist_directory="./chroma_db"):
        """
        Initializes the Medical Assistant with RAG capabilities and simulated external tools.
        Args:
            persist_directory (str): The directory where the ChromaDB vector store is persisted.
        """
        self.persist_directory = persist_directory
        # Initialize the embedding model used for both ingesting and querying the vector store.
        self.embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        # Initialize ChromaDB as the vector store for RAG.
        # It expects the persist_directory to already contain the ingested data.
        self.vectordb = Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)
        print(f"MedicalAssistant initialized with RAG from {persist_directory}")

    def _retrieve_knowledge(self, query: str, k: int = 3) -> list[str]:
        """
        Retrieves the top-k most relevant documents from the vector database based on the query.
        Args:
            query (str): The user's medical query.
            k (int): The number of top relevant documents to retrieve.
        Returns:
            list[str]: A list of strings, each representing a relevant document's content.
        """
        print(f"Retrieving knowledge from RAG for query: '{query}'")
        # Perform similarity search to find relevant documents
        docs = self.vectordb.similarity_search(query, k=k)
        retrieved_content = [doc.page_content for doc in docs]
        return retrieved_content

    def _simulate_pubmed_data(self, query: str) -> str:
        """
        Simulates fetching medical research data from a PubMed-like database.
        In a real application, this would involve API calls to PubMed or similar services.
        """
        print(f"Simulating PubMed search for: '{query}'")
        time.sleep(0.5)  # Simulate API latency
        if "diabetes" in query.lower():
            return "PubMed: Recent studies indicate new GLP-1 receptor agonists showing promise for diabetes management beyond glycemic control, including cardiovascular benefits."
        elif "hypertension" in query.lower():
            return "PubMed: Meta-analysis highlights the importance of dietary sodium reduction in conjunction with pharmacotherapy for resistant hypertension."
        elif "cancer" in query.lower():
            return "PubMed: Advances in CAR T-cell therapy show efficacy in certain hematological malignancies. Ongoing research explores solid tumor applications."
        else:
            return "PubMed: No specific recent findings for this query found, but general medical literature is available. (Simulated data)"

    def _simulate_clinical_trials(self, query: str) -> str:
        """
        Simulates fetching information from clinical trial registries.
        """
        print(f"Simulating Clinical Trials search for: '{query}'")
        time.sleep(0.5)
        if "alzheimer" in query.lower():
            return "ClinicalTrials.gov: Phase 3 trials ongoing for amyloid-beta targeting drugs showing mixed results; new tau protein inhibitors in earlier phases. (Simulated data)"
        elif "cancer" in query.lower():
            return "ClinicalTrials.gov: Immunotherapy trials continue to expand, with promising results for several solid tumor types. Personalized vaccine trials are also active. (Simulated data)"
        else:
            return "ClinicalTrials.gov: No highly relevant trials for this query at the moment. (Simulated data)"

    def _simulate_medical_news(self, query: str) -> str:
        """
        Simulates fetching real-time medical news feeds.
        """
        print(f"Simulating Medical News search for: '{query}'")
        time.sleep(0.5)
        if "covid" in query.lower():
            return "Medical News: New variant concerns emerge; health authorities emphasize updated booster shots and continued vigilance. (Simulated data)"
        elif "vaccine" in query.lower():
            return "Medical News: Global efforts for universal vaccine access continue; research into broad-spectrum antiviral treatments progresses. (Simulated data)"
        else:
            return "Medical News: General health updates and policy discussions. (Simulated data)"

    def _simulate_web_browsing_guidelines(self, query: str) -> str:
        """
        Simulates browsing official health organization guidelines (e.g., WHO, CDC).
        """
        print(f"Simulating Web Browsing for guidelines related to: '{query}'")
        time.sleep(1.0)  # Longer latency for browsing simulation
        if "covid" in query.lower():
            return "WHO Guidelines: Latest recommendations for COVID-19 prevention, testing, and treatment emphasize integrated care and equitable access to tools. (Simulated data)"
        elif "blood pressure" in query.lower() or "hypertension" in query.lower():
            return "CDC Guidelines: Comprehensive recommendations for hypertension management, including DASH diet, physical activity, and medication adherence. (Simulated data)"
        else:
            return "General Health Org Guidelines: Consult official sources for specific disease management. (Simulated data)"

    def _simulate_ehr_data(self, patient_id: str = "patient_123", query: str = "") -> str:
        """
        Simulates fetching data from an Electronic Health Record (EHR) system.
        This is a placeholder for real EHR integration, which would require secure API access and strict protocols.
        """
        print(f"Simulating EHR data access for patient: '{patient_id}' related to: '{query}'")
        time.sleep(0.3)
        # Dummy EHR data for a specific patient
        ehr_data = {
            "patient_123": {
                "name": "Jane Doe",
                "age": 65,
                "conditions": ["Type 2 Diabetes", "Hypertension"],
                "medications": ["Metformin 500mg BID", "Lisinopril 10mg QD"],
                "allergies": ["Penicillin"],
                "last_visit_summary": "Patient presented for routine diabetes and hypertension check-up. Blood sugar and blood pressure stable. Advised on diet and exercise."
            }
        }
        if patient_id in ehr_data:
            data = ehr_data[patient_id]
            response = f"EHR Data for {data['name']} (ID: {patient_id}):\n"
            response += f"  Age: {data['age']}\n"
            response += f"  Conditions: {', '.join(data['conditions'])}\n"
            response += f"  Medications: {', '.join(data['medications'])}\n"
            response += f"  Allergies: {', '.join(data['allergies'])}\n"
            response += f"  Last Visit Summary: {data['last_visit_summary']}\n"
            return response
        else:
            return f"EHR Data: Patient {patient_id} not found. (Simulated data)"

    def _llm_process_context(self, query: str, context: list[str]) -> str:
        """
        Simulates an LLM processing the query and the retrieved/augmented context
        to generate a coherent and accurate response.
        In a real application, this would involve an actual LLM API call.
        """
        print(f"Simulating LLM processing for query: '{query}' with context...")
        time.sleep(1.5) # Simulate LLM processing time

        # Simple heuristic to demonstrate context usage
        response_parts = []
        response_parts.append(f"Based on your query regarding '{query}', here's the synthesized information:\n")

        if any("EHR Data" in c for c in context) and ("patient" in query.lower() or "ehr" in query.lower()):
            ehr_info = [c for c in context if "EHR Data" in c]
            if ehr_info:
                response_parts.append("\n--- Patient Specific Information (from EHR) ---\n")
                response_parts.extend(ehr_info)
                response_parts.append("\n")

        if any("PubMed:" in c for c in context) and ("research" in query.lower() or "latest studies" in query.lower()):
            pubmed_info = [c for c in context if "PubMed:" in c]
            if pubmed_info:
                response_parts.append("\n--- Latest Research (from PubMed) ---\n")
                response_parts.extend(pubmed_info)
                response_parts.append("\n")

        if any("ClinicalTrials.gov:" in c for c in context):
            trials_info = [c for c in context if "ClinicalTrials.gov:" in c]
            if trials_info:
                response_parts.append("\n--- Clinical Trial Updates ---\n")
                response_parts.extend(trials_info)
                response_parts.append("\n")

        if any("Medical News:" in c for c in context):
            news_info = [c for c in context if "Medical News:" in c]
            if news_info:
                response_parts.append("\n--- Real-time Medical News ---\n")
                response_parts.extend(news_info)
                response_parts.append("\n")

        if any("WHO Guidelines:" in c for c in context) or any("CDC Guidelines:" in c for c in context):
            guidelines_info = [c for c in context if "Guidelines:" in c]
            if guidelines_info:
                response_parts.append("\n--- Official Guidelines (from Web Browsing) ---\n")
                response_parts.extend(guidelines_info)
                response_parts.append("\n")

        # Add RAG retrieved content if available and not already covered by specific tool output
        rag_content = [c for c in context if not any(tool_prefix in c for tool_prefix in ["PubMed:", "ClinicalTrials.gov:", "Medical News:", "WHO Guidelines:", "CDC Guidelines:", "EHR Data:"])]
        if rag_content:
            response_parts.append("\n--- General Knowledge (from RAG) ---\n")
            response_parts.extend(rag_content)
            response_parts.append("\n")

        if not response_parts:
            response_parts.append("No specific information found for your query. Please try rephrasing.")

        return "\n".join(response_parts)

    def answer_query(self, query: str) -> str:
        """
        Main method to answer a medical query by combining RAG and external tool access.
        Args:
            query (str): The user's medical query.
        Returns:
            str: A comprehensive, augmented response.
        """
        print(f"\nProcessing query: '{query}'")
        context = []

        # Step 1: RAG for foundational knowledge
        rag_docs = self._retrieve_knowledge(query)
        context.extend(rag_docs)

        # Step 2: Integrate external tools based on query intent or always for comprehensive search
        # Simple keyword-based routing for demonstration
        if "pubmed" in query.lower() or "research" in query.lower() or "latest studies" in query.lower():
            context.append(self._simulate_pubmed_data(query))

        if "clinical trials" in query.lower() or "trials" in query.lower():
            context.append(self._simulate_clinical_trials(query))

        if "news" in query.lower() or "updates" in query.lower() or "real-time" in query.lower():
            context.append(self._simulate_medical_news(query))

        if "guidelines" in query.lower() or "who" in query.lower() or "cdc" in query.lower():
            context.append(self._simulate_web_browsing_guidelines(query))

        # Example of dynamic EHR access (could be triggered by patient ID or specific intent)
        if "patient" in query.lower() or "ehr" in query.lower():
            # Attempt to extract a dummy patient ID for simulation
            patient_id_match = random.choice(["patient_123", "patient_unknown"])
            context.append(self._simulate_ehr_data(patient_id=patient_id_match, query=query))

        # Step 3: Consolidate all retrieved knowledge and pass to (simulated) LLM
        final_response = self._llm_process_context(query, context)
        return final_response

if __name__ == "__main__":
    # This block is for testing the MedicalAssistant directly (without Gradio)
    # Ensure data_ingestion.py has been run first.
    print("\n--- Testing Medical Assistant Directly ---")
    test_assistant = MedicalAssistant()

    # Example Queries
    print("\nQuery 1: What are the latest treatments for type 2 diabetes and any recent research?")
    response1 = test_assistant.answer_query("What are the latest treatments for type 2 diabetes and any recent research?")
    print(f"\nAssistant Response: {response1}")

    print("\nQuery 2: Tell me about the current guidelines for managing hypertension from official sources.")
    response2 = test_assistant.answer_query("Tell me about the current guidelines for managing hypertension from official sources.")
    print(f"\nAssistant Response: {response2}")

    print("\nQuery 3: What are the recent findings on Alzheimer's disease trials and any related news?")
    response3 = test_assistant.answer_query("What are the recent findings on Alzheimer's disease trials and any related news?")
    print(f"\nAssistant Response: {response3}")

    print("\nQuery 4: Provide EHR data for patient_123 regarding their medications.")
    response4 = test_assistant.answer_query("Provide EHR data for patient_123 regarding their medications.")
    print(f"\nAssistant Response: {response4}")

    print("\nQuery 5: What are the general recommendations for COVID-19 prevention and latest news?")
    response5 = test_assistant.answer_query("What are the general recommendations for COVID-19 prevention and latest news?")
    print(f"\nAssistant Response: {response5}")

    print("\n--- End of Direct Testing ---")