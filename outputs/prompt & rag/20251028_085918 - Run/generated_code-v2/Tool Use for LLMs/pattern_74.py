import json
import re
import os
from typing import List, Dict, Any, Optional

from pydantic import BaseModel, Field, ValidationError
from sentence_transformers import SentenceTransformer
import chromadb
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import OpenAI
import streamlit as st


# --- 1. API Documentation Ingestion & Standardization Module ---

class APIArgument(BaseModel):
    name: str
    type: str
    description: str
    required: bool = False
    example: Optional[str] = None

class APIDocumentation(BaseModel):
    domain: str
    framework: Optional[str] = None
    functionality: str
    apiname: str
    apicall: str
    apiarguments: List[APIArgument]
    environmentrequirements: Optional[str] = None
    examplecode: Optional[str] = None
    performance: Optional[str] = None
    description: str

    def to_text_for_embedding(self) -> str:
        arg_str = ", ".join([f"{arg.name} ({arg.type})" for arg in self.apiarguments])
        return f"Domain: {self.domain}. Functionality: {self.functionality}. API Name: {self.apiname}. API Call: {self.apicall}. Arguments: {arg_str}. Description: {self.description}"

class DocumentationFetcher:
    def fetch_openapi_spec(self, url: str) -> Dict[str, Any]:
        # In a real app, this would make an HTTP request
        # For demonstration, we return a mock OpenAPI spec
        if "stripe" in url:
            return {
                "openapi": "3.0.0",
                "info": {"title": "Stripe API", "version": "1.0.0"},
                "paths": {
                    "/charges": {
                        "post": {
                            "summary": "Create a charge",
                            "parameters": [
                                {"name": "amount", "in": "formData", "schema": {"type": "integer"}, "required": True, "description": "Amount in cents"},
                                {"name": "currency", "in": "formData", "schema": {"type": "string"}, "required": True, "description": "Currency (e.g., usd)"},
                                {"name": "source", "in": "formData", "schema": {"type": "string"}, "required": True, "description": "Payment token"},
                                {"name": "description", "in": "formData", "schema": {"type": "string"}, "required": False, "description": "Charge description"}
                            ]
                        }
                    }
                }
            }
        elif "ups" in url:
            return {
                "openapi": "3.0.0",
                "info": {"title": "UPS Shipping API", "version": "1.0.0"},
                "paths": {
                    "/shipping/rates": {
                        "post": {
                            "summary": "Get shipping rates",
                            "parameters": [
                                {"name": "weight", "in": "body", "schema": {"type": "number"}, "required": True, "description": "Package weight"},
                                {"name": "weightUnit", "in": "body", "schema": {"type": "string"}, "required": True, "description": "Weight unit (e.g., KG, LBS)"},
                                {"name": "origin", "in": "body", "schema": {"type": "string"}, "required": True, "description": "Origin address"},
                                {"name": "destination", "in": "body", "schema": {"type": "string"}, "required": True, "description": "Destination address"}
                            ]
                        }
                    }
                }
            }
        return {}

class OpenAPIParser:
    def parse(self, api_data: Dict[str, Any], domain: str) -> List[APIDocumentation]:
        docs = []
        for path, methods in api_data.get("paths", {}).items():
            for method, details in methods.items():
                if "summary" in details:
                    api_name = details.get("summary")
                    api_call = f"{method.upper()} {path}"
                    description = details.get("description", api_name)

                    args = []
                    for param in details.get("parameters", []):
                        args.append(APIArgument(
                            name=param.get("name"),
                            type=param.get("schema", {}).get("type", "string"),
                            description=param.get("description", ""),
                            required=param.get("required", False)
                        ))
                    
                    docs.append(APIDocumentation(
                        domain=domain,
                        functionality=api_name,
                        apiname=api_name,
                        apicall=api_call,
                        apiarguments=args,
                        description=description,
                        examplecode=f"# Example for {api_name}" 
                    ))
        return docs


# --- 2. Structured Knowledge Base (Vector Database) ---

class VectorStoreManager:
    def __init__(self, collection_name="api_docs_collection", model_name="all-MiniLM-L6-v2"):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.embedding_model = SentenceTransformer(model_name)

    def add_documents(self, api_docs: List[APIDocumentation]):
        documents_to_add = []
        metadatas_to_add = []
        ids_to_add = []

        for i, doc in enumerate(api_docs):
            documents_to_add.append(doc.to_text_for_embedding())
            metadatas_to_add.append(doc.dict())
            ids_to_add.append(f"doc_{doc.apiname}_{i}") 
        
        if documents_to_add:
            embeddings = self.embedding_model.encode(documents_to_add).tolist()
            self.collection.add(
                embeddings=embeddings,
                documents=documents_to_add,
                metadatas=metadatas_to_add,
                ids=ids_to_add
            )
            print(f"Added {len(documents_to_add)} documents to ChromaDB.")

    def retrieve_documents(self, query: str, n_results: int = 3) -> List[APIDocumentation]:
        query_embedding = self.embedding_model.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            include=['metadatas']
        )
        
        retrieved_docs = []
        if results and results['metadatas']:
            for metadata in results['metadatas'][0]:
                try:
                    retrieved_docs.append(APIDocumentation(**metadata))
                except ValidationError as e:
                    print(f"Error validating retrieved document: {e}")
        return retrieved_docs


# --- 3. LLM Interaction & Reasoning (RAG System) ---

class LLMIntegrationAssistant:
    def __init__(self, vector_store_manager: VectorStoreManager, openai_api_key: Optional[str] = None):
        self.vector_store_manager = vector_store_manager
        self.llm = OpenAI(openai_api_key=openai_api_key) if openai_api_key else self._mock_llm

        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are an expert AI assistant for e-commerce API integrations. Use the provided API documentation to generate accurate and constrained API calls or code snippets. Focus on the `apicall`, `apiarguments`, and `examplecode` fields."),
            ("user", "Context API Documentation:\n{context}\n\nUser Query: {query}\n\nGenerate the relevant API call or Python code snippet, explaining its purpose. Ensure all required arguments are addressed.")
        ])

        self.rag_chain = (
            {"context": self._retrieve_docs_for_rag, "query": RunnablePassthrough()}
            | self.prompt_template
            | self.llm
            | StrOutputParser()
        )

    def _retrieve_docs_for_rag(self, query: str) -> str:
        docs = self.vector_store_manager.retrieve_documents(query)
        formatted_docs = []
        for doc in docs:
            formatted_docs.append(f"APIName: {doc.apiname}\nAPICall: {doc.apicall}\nArguments: {json.dumps([arg.dict() for arg in doc.apiarguments], indent=2)}\nDescription: {doc.description}\nExample: {doc.examplecode}")
        return "\n---\n".join(formatted_docs)

    def _mock_llm(self, prompt: str) -> str:
        if "Stripe" in prompt and "recurring payments" in prompt:
            return """# Python code to create a Stripe charge
import stripe
stripe.api_key = "sk_test_YOUR_STRIPE_SECRET_KEY"

charge = stripe.Charge.create(
    amount=2000, # $20.00
    currency="usd",
    source="tok_visa", # obtained with Stripe.js
    description="Example charge for recurring payment"
)
print(charge)
"""
        elif "UPS" in prompt and "shipping rates" in prompt and "London to New York" in prompt and "5kg" in prompt:
            return """# Python code to get UPS shipping rates
# This is a conceptual example, actual UPS API integration involves more steps (auth, service codes, etc.)

def get_ups_shipping_rate(weight, weight_unit, origin, destination):
    print(f"Simulating UPS rate for {weight}{weight_unit} from {origin} to {destination}")
    # In a real scenario, this would call the UPS API
    if weight == 5 and weight_unit.lower() == "kg":
        return {"rate": 50.75, "currency": "USD"}
    return {"rate": "N/A"}

rate_info = get_ups_shipping_rate(5, "KG", "London, UK", "New York, USA")
print(f"UPS Shipping Rate: {rate_info.get('rate')} {rate_info.get('currency')}")
"""
        return "I'm a mock LLM. I can generate responses based on known patterns. Please provide a more specific query related to Stripe charges or UPS shipping rates."

    def get_response(self, query: str) -> str:
        return self.rag_chain.invoke(query)


# --- 4. API Code Generation & Validation Module ---

class APICodeValidator:
    def validate_api_call_arguments(self, generated_code: str, api_doc: APIDocumentation) -> bool:
        # This is a very basic validation. In a real system, you'd parse the generated code
        # and check if the arguments passed match the `api_doc.apiarguments`.
        # For this example, we'll check if required arguments are mentioned in the generated code string.
        
        missing_required = []
        for arg in api_doc.apiarguments:
            if arg.required:
                # Simple check for argument name in the generated code string
                if re.search(r"\b" + re.escape(arg.name) + r"\b", generated_code, re.IGNORECASE) is None:
                    missing_required.append(arg.name)
        
        if missing_required:
            print(f"Validation Warning: Missing required arguments in generated code: {', '.join(missing_required)}")
            return False
        return True


# --- Main Application Logic (Streamlit UI) ---

def main():
    st.set_page_config(page_title="E-commerce Integration Assistant", layout="wide")
    st.title("Intelligent E-commerce Integration Assistant")

    # Initialize components
    fetcher = DocumentationFetcher()
    parser = OpenAPIParser()
    vector_manager = VectorStoreManager()
    validator = APICodeValidator()

    # Ensure OpenAI API key is available if not using mock LLM
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        st.warning("OPENAI_API_KEY not found in environment variables. Using mock LLM for demonstration.")
        llm_assistant = LLMIntegrationAssistant(vector_store_manager=vector_manager)
    else:
        llm_assistant = LLMIntegrationAssistant(vector_store_manager=vector_manager, openai_api_key=openai_api_key)

    # --- Ingest and Standardize API Docs (Run once or on demand) ---
    st.sidebar.header("API Documentation Management")
    if st.sidebar.button("Ingest Mock API Docs"):
        with st.spinner("Ingesting and standardizing mock API documentation..."):
            stripe_raw_doc = fetcher.fetch_openapi_spec("https://api.stripe.com/openapi.json")
            ups_raw_doc = fetcher.fetch_openapi_spec("https://api.ups.com/openapi.json")

            stripe_docs = parser.parse(stripe_raw_doc, domain="Payment Gateway")
            ups_docs = parser.parse(ups_raw_doc, domain="Shipping Carrier")

            all_docs = stripe_docs + ups_docs
            vector_manager.add_documents(all_docs)
            st.sidebar.success(f"Ingested {len(all_docs)} API documents.")

    st.sidebar.markdown("--- Other API Docs ---")
    st.sidebar.write("Coming soon: Real-time fetching and parsing of OpenAPI specs, Postman collections, etc.")

    # --- LLM Interaction Section ---
    st.header("Ask the Assistant")
    user_query = st.text_area("Describe your E-commerce integration need (e.g., 'How do I integrate Stripe for recurring payments?', 'Generate Python code to get UPS shipping rates for a 5kg package from London to New York'):", height=100)

    if st.button("Generate Integration Code") and user_query:
        with st.spinner("Generating code and explanation..."):
            llm_response = llm_assistant.get_response(user_query)
            st.subheader("Generated Code / Explanation")
            st.code(llm_response, language="python")

            # --- Basic Validation (requires knowing which API doc was relevant)
            # In a real RAG system, the LLM might output the `apiname` or `apicall` it used
            # For this demo, we'll try to infer or re-retrieve the most relevant doc for validation
            st.subheader("Validation Check (Conceptual)")
            relevant_docs = vector_manager.retrieve_documents(user_query, n_results=1)
            if relevant_docs:
                most_relevant_doc = relevant_docs[0]
                st.write(f"Attempting to validate against: `{most_relevant_doc.apiname}`")
                if validator.validate_api_call_arguments(llm_response, most_relevant_doc):
                    st.success("Basic validation passed: Required arguments seem to be present.")
                else:
                    st.warning("Basic validation failed: Some required arguments might be missing or incorrectly formatted.")
            else:
                st.info("Could not find a relevant API document for advanced validation.")


if __name__ == "__main__":
    main()