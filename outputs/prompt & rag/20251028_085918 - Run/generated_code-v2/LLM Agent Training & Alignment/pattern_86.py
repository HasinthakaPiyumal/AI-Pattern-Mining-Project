import os
import uvicorn
import requests
import json
import logging
from typing import List, Dict, Any

import streamlit as st
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class HealthQuery(BaseModel):
    query: str
    user_id: str = "anonymous"

class HealthRecommendation(BaseModel):
    recommendation: str
    explanation: str
    ethical_review: Dict[str, Any]

class MockLLM:
    def __init__(self, response_map: Dict[str, str] = None, default_response: str = "Mocked LLM response."):
        self.response_map = response_map if response_map else {}
        self.default_response = default_response

    def invoke(self, prompt_text: str) -> str:
        for key, value in self.response_map.items():
            if key in prompt_text:
                return value
        return self.default_response

class ConstitutionalAIManager:
    def __init__(self, constitution_principles: List[str]):
        self.constitution_principles = constitution_principles
        self.critique_llm = MockLLM(response_map={
            "harmful": "Critique: The information suggests self-medication, which is harmful.",
            "biased": "Critique: The information exhibits bias towards a specific demographic.",
            "unfactual": "Critique: The information contains factual inaccuracies.",
            "No violations found": "No violations found."
        }, default_response="No violations found.")
        self.revision_llm = MockLLM(default_response="Revised response to address ethical concerns, now more balanced and factual.")

    def _apply_constitution(self, text: str, principles: List[str]) -> str:
        principles_list_str = "\\n".join([f"- {p}" for p in principles])
        prompt_text = f"Review the following health information for adherence to ethical principles:\n\nInformation: {text}\n\nEthical Principles to check:\n{principles_list_str}\n\nCritique:"
        return self.critique_llm.invoke(prompt_text)

    def _revise_text(self, original_text: str, critique: str, principles: List[str]) -> str:
        principles_list_str = "\\n".join([f"- {p}" for p in principles])
        prompt_text = f"Given the following original health information and a critique, revise the information to address the critique and adhere to these ethical principles:\n\nOriginal Information: {original_text}\nCritique: {critique}\nEthical Principles: {principles_list_str}\n\nRevised Information:"
        return self.revision_llm.invoke(prompt_text)

    def apply_constitutional_alignment(self, generated_text: str) -> Dict[str, Any]:
        logger.info("Applying constitutional alignment...")
        critique = self._apply_constitution(generated_text, self.constitution_principles)

        ethical_review = {"initial_critique": critique, "revised": False}
        revised_text = generated_text

        if "No violations found" not in critique:
            logger.warning(f"Violations found: {critique}. Revising...")
            revised_text = self._revise_text(generated_text, critique, self.constitution_principles)
            ethical_review["revised_text"] = revised_text
            ethical_review["revised"] = True
            ethical_review["final_critique"] = self._apply_constitution(revised_text, self.constitution_principles)
            logger.info("Revision complete.")
        else:
            logger.info("Initial text adheres to constitution.")

        return {"aligned_text": revised_text, "ethical_review_details": ethical_review}

class RAGSystem:
    def __init__(self, knowledge_base_docs: List[str]):
        self.embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = Chroma.from_texts(
            texts=knowledge_base_docs,
            embedding=self.embedding_model
        )
        self.retriever = self.vectorstore.as_retriever()
        logger.info(f"RAG System initialized with {len(knowledge_base_docs)} documents.")

    def retrieve_documents(self, query: str) -> List[str]:
        docs = self.retriever.invoke(query)
        return [doc.page_content for doc in docs]

app = FastAPI(title="Personalized Health Information System API")

MOCK_HEALTH_KNOWLEDGE_BASE = [
    "Regular exercise, about 30 minutes most days of the week, can significantly improve cardiovascular health.",
    "A balanced diet rich in fruits, vegetables, and whole grains is crucial for maintaining overall well-being and preventing chronic diseases.",
    "Adequate sleep, typically 7-9 hours per night for adults, is essential for cognitive function and physical recovery.",
    "Always consult a qualified medical professional for personalized health advice. Self-diagnosing based on internet information can be dangerous.",
    "Vaccinations are a safe and effective way to prevent many infectious diseases. Consult your doctor for appropriate vaccination schedules.",
    "Mental health is as important as physical health. Seek professional help if you are experiencing symptoms of depression or anxiety."
]
rag_system = RAGSystem(MOCK_HEALTH_KNOWLEDGE_BASE)

CONSTITUTION_PRINCIPLES = [
    "Responses must be factually accurate and based on reliable medical science.",
    "Responses must be unbiased and avoid discriminatory language.",
    "Responses must be harmless and not promote dangerous practices.",
    "Responses must emphasize consulting a medical professional for personalized advice.",
    "Responses must respect user privacy and not ask for sensitive personal health information.",
    "Responses should be helpful and empathetic."
]
constitutional_ai_manager = ConstitutionalAIManager(CONSTITUTION_PRINCIPLES)

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

class TransformersLLM:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def invoke(self, prompt_text: str) -> str:
        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=50, num_return_sequences=1)
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text[len(prompt_text):].strip()

base_llm = TransformersLLM(model, tokenizer)

@app.post("/recommendation", response_model=HealthRecommendation)
async def get_health_recommendation(query_data: HealthQuery):
    logger.info(f"Received query from user {query_data.user_id}: {query_data.query}")

    relevant_docs = rag_system.retrieve_documents(query_data.query)
    context = "\\n".join(relevant_docs)
    logger.info(f"Retrieved context: {context[:200]}...")

    generation_prompt = PromptTemplate.from_template(
        "You are a helpful health assistant. Based on the following context, answer the user's query. "
        "Always remind the user to consult a medical professional for personalized advice. "
        "Context: {context}\\n\\n"
        "User Query: {query}\\n\\n"
        "Health Recommendation:"
    )
    llm_chain = {"context": RunnablePassthrough(), "query": RunnablePassthrough()} | generation_prompt | base_llm.invoke | StrOutputParser()
    initial_recommendation = llm_chain.invoke({"context": context, "query": query_data.query})
    logger.info(f"Initial LLM recommendation: {initial_recommendation[:200]}...")

    aligned_result = constitutional_ai_manager.apply_constitutional_alignment(initial_recommendation)
    final_recommendation = aligned_result["aligned_text"]
    ethical_details = aligned_result["ethical_review_details"]
    logger.info(f"Final aligned recommendation: {final_recommendation[:200]}...")

    explanation = "This recommendation was generated using AI, augmented with information from a reliable health knowledge base, and reviewed for ethical alignment against principles like accuracy, harmlessness, and unbiasedness. Always consult a healthcare professional for personalized medical advice."

    return HealthRecommendation(
        recommendation=final_recommendation,
        explanation=explanation,
        ethical_review=ethical_details
    )

def streamlit_app():
    st.set_page_config(page_title="Constitutional Health AI")
    st.title("🩺 Personalized Health Assistant (Constitutional AI)")
    st.markdown("Ask any health-related question, and get ethically aligned recommendations.")

    user_query = st.text_area("Your health question:", height=100, placeholder="e.g., What are good exercises for heart health?")
    user_id = st.text_input("Your User ID (optional):", value="demo_user")

    if st.button("Get Recommendation"):
        if not user_query:
            st.error("Please enter a health question.")
            return

        with st.spinner("Generating and aligning recommendation..."):
            try:
                api_url = os.getenv("FASTAPI_URL", "http://localhost:8000/recommendation")
                headers = {"Content-Type": "application/json"}
                data = {"query": user_query, "user_id": user_id}

                response = requests.post(api_url, headers=headers, data=json.dumps(data))
                response.raise_for_status()

                recommendation_data = response.json()

                st.success("Recommendation Generated!")
                st.subheader("Your Personalized Health Recommendation:")
                st.write(recommendation_data["recommendation"])

                st.subheader("Explanation & Ethical Review:")
                st.write(recommendation_data["explanation"])
                st.json(recommendation_data["ethical_review"])

            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the FastAPI backend. Please ensure the backend server is running.")
            except requests.exceptions.RequestException as e:
                st.error(f"An error occurred during the request: {e}")
                if e.response:
                    st.json(e.response.json())
                else:
                    st.error("No response details.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # This block is primarily for user guidance on how to run the components.
    # The FastAPI 'app' object is directly exposed for uvicorn.
    # The 'streamlit_app' function can be called by 'streamlit run <filename>.py'.
    # No direct execution of uvicorn.run() or streamlit_app() here to avoid conflicts
    # when the script is loaded by either framework.
    pass