import streamlit as st
import os


class QueryPreprocessor:
    def preprocess(self, query: str) -> str:
        return query.strip().lower()


class QueryComplexityClassifier:
    def classify(self, preprocessed_query: str) -> str:
        if "drug interaction" in preprocessed_query or "medication side effect" in preprocessed_query:
            return "Simple"
        elif "diagnosis of" in preprocessed_query or "treatment for" in preprocessed_query:
            return "Moderate"
        elif "complex case management" in preprocessed_query or "differential diagnosis for" in preprocessed_query:
            return "Complex"
        else:
            return "Moderate"


class KnowledgeBaseManager:
    def __init__(self):
        self.patient_records = {
            "patient_id_1": "Patient history: 65-year-old male with type 2 diabetes and hypertension. Recent complaint of persistent cough.",
            "patient_id_2": "Patient history: 40-year-old female, no significant past medical history, presenting with severe headache and fever."
        }
        self.medical_literature = {
            "diabetes management": "Recent studies suggest GLP-1 receptor agonists are effective in type 2 diabetes management.",
            "hypertension guidelines": "JNC 8 guidelines recommend initial pharmacologic treatment for hypertension.",
            "migraine treatment": "Triptans are often prescribed for acute migraine attacks."
        }
        self.clinical_guidelines = {
            "cough treatment": "For persistent cough, consider post-nasal drip, asthma, or GERD.",
            "headache diagnosis": "Severe headache with fever warrants investigation for meningitis."
        }
        self.drug_interactions = {
            "metformin and alcohol": "Increased risk of lactic acidosis."
        }

    def retrieve_patient_info(self, query_entity: str) -> str:
        for pid, record in self.patient_records.items():
            if query_entity.lower() in record.lower():
                return record
        return "No relevant patient record found."

    def search_medical_literature(self, keyword: str) -> str:
        for k, lit in self.medical_literature.items():
            if keyword.lower() in k.lower() or keyword.lower() in lit.lower():
                return lit
        return "No relevant medical literature found."

    def get_clinical_guideline(self, topic: str) -> str:
        for k, guide in self.clinical_guidelines.items():
            if topic.lower() in k.lower() or topic.lower() in guide.lower():
                return guide
        return "No relevant clinical guideline found."

    def get_drug_interaction(self, drug_pair: str) -> str:
        for k, interaction in self.drug_interactions.items():
            if drug_pair.lower() in k.lower():
                return interaction
        return "No known drug interaction found for the given pair."


class LLMGenerator:
    def generate_response(self, prompt: str, context: str = "") -> str:
        # This is a mock LLM. In a real scenario, this would call an actual LLM API
        # e.g., using langchain.chat_models.ChatOpenAI or a similar interface.
        if "drug interaction" in prompt.lower() and "metformin" in prompt.lower() and "alcohol" in prompt.lower():
            return "The interaction between metformin and alcohol increases the risk of lactic acidosis. Advise patient to limit alcohol intake."
        elif "diagnosis for severe headache and fever" in prompt.lower():
            return "Considering a severe headache with fever, meningitis should be ruled out. Further diagnostics like lumbar puncture are recommended."
        elif "diabetes management" in prompt.lower():
            return "Effective management of type 2 diabetes involves lifestyle modifications and pharmacological treatments. GLP-1 receptor agonists are a class of drugs used for this condition."
        elif context:
            return f"Based on the provided information: {context}. A comprehensive answer could be: ... (LLM elaborates based on context)"
        else:
            return f"I'm a helpful medical assistant. For your query '{prompt}', I can provide general information. Please provide more context or clarify if needed."


class AdaptiveRetrievalOrchestrator:
    def __init__(self, kb_manager: KnowledgeBaseManager, llm_generator: LLMGenerator):
        self.kb_manager = kb_manager
        self.llm_generator = llm_generator

    def _no_retrieval_strategy(self, query: str) -> str:
        st.write("Strategy: No Retrieval (LLM-only)")
        prompt = f"Answer the following medical question: {query}"
        return self.llm_generator.generate_response(prompt)

    def _single_step_retrieval_strategy(self, query: str) -> str:
        st.write("Strategy: Single-Step Retrieval")
        context = ""
        if "patient" in query:
            context = self.kb_manager.retrieve_patient_info(query.split("patient")[-1].strip())
        elif "literature" in query or "study" in query:
            context = self.kb_manager.search_medical_literature(query)
        elif "guideline" in query:
            context = self.kb_manager.get_clinical_guideline(query)
        elif "drug interaction" in query:
            context = self.kb_manager.get_drug_interaction(query)

        prompt = f"Given the following medical context: {context}\nAnswer the question: {query}"
        return self.llm_generator.generate_response(prompt, context)

    def _multi_step_retrieval_strategy(self, query: str) -> str:
        st.write("Strategy: Multi-Step Iterative Retrieval")
        # This is a simplified multi-step strategy
        intermediate_context = []

        # Step 1: Initial broad retrieval (e.g., medical literature)
        initial_search_term = "medicine" # Placeholder for complex query parsing
        lit_context = self.kb_manager.search_medical_literature(initial_search_term)
        intermediate_context.append(lit_context)
        st.info(f"Step 1 Retrieval: {lit_context[:100]}...")

        # Step 2: Refine based on initial context or sub-questions
        # For example, if query is about a patient, retrieve patient records
        if "patient" in query:
            patient_entity = "patient_id_2" # Mock entity extraction
            patient_context = self.kb_manager.retrieve_patient_info(patient_entity)
            intermediate_context.append(patient_context)
            st.info(f"Step 2 Retrieval (Patient): {patient_context[:100]}...")

        # Step 3: Integrate and generate
        full_context = "\n".join(intermediate_context)
        prompt = f"Synthesize the following information to answer the complex medical question: {query}\nContexts: {full_context}"
        return self.llm_generator.generate_response(prompt, full_context)

    def execute_strategy(self, query: str, complexity: str) -> str:
        if complexity == "Simple":
            return self._no_retrieval_strategy(query)
        elif complexity == "Moderate":
            return self._single_step_retrieval_strategy(query)
        elif complexity == "Complex":
            return self._multi_step_retrieval_strategy(query)
        else:
            return "Unknown complexity. Cannot provide an answer."


class ResponsePostprocessor:
    def post_process(self, raw_response: str) -> str:
        return raw_response.replace("(LLM elaborates based on context)", "(Further details would be provided by a real LLM).")


def main():
    st.set_page_config(page_title="Smart Medical Assistant")
    st.title("🧠 Smart Medical Assistant")
    st.markdown("An Adaptive RAG system for healthcare professionals.")

    query_preprocessor = QueryPreprocessor()
    qcc = QueryComplexityClassifier()
    kb_manager = KnowledgeBaseManager()
    llm_generator = LLMGenerator()
    orchestrator = AdaptiveRetrievalOrchestrator(kb_manager, llm_generator)
    response_postprocessor = ResponsePostprocessor()

    user_query = st.text_area("Enter your medical query here:", height=150)

    if st.button("Get Answer"):
        if user_query:
            st.subheader("Processing Query...")

            preprocessed_query = query_preprocessor.preprocess(user_query)
            st.write(f"Preprocessed Query: {preprocessed_query}")

            complexity = qcc.classify(preprocessed_query)
            st.success(f"Predicted Query Complexity: **{complexity}**")

            with st.spinner("Retrieving and Generating Response..."):
                raw_response = orchestrator.execute_strategy(user_query, complexity)
                final_response = response_postprocessor.post_process(raw_response)

            st.subheader("Answer:")
            st.write(final_response)
        else:
            st.warning("Please enter a query.")


if __name__ == "__main__":
    main()