import streamlit as st
import pandas as pd
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import json
import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

# 1. Pydantic Model for Structured Output
class ProductDescription(BaseModel):
    product_name: str = Field(description="The name of the product")
    short_description: str = Field(description="A concise, engaging short description of the product")
    features_list: List[str] = Field(description="A list of key features or selling points of the product")
    specifications: Dict[str, str] = Field(description="A dictionary of product specifications (e.g., {'Color': 'Red', 'Material': 'Plastic'})")
    price_usd: float = Field(description="The price of the product in USD")
    availability_status: str = Field(description="Current availability status (e.g., 'In Stock', 'Out of Stock', 'Pre-order')")

# 2. LLM Setup (using OpenAI)
# Ensure OPENAI_API_KEY is set in your environment variables
# st.secrets could also be used for Streamlit Cloud deployment
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OPENAI_API_KEY environment variable not set. Please set it to use the LLM.")
    st.stop()

llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0.7, openai_api_key=openai_api_key)

# 3. Langchain Components for Prompt and Parser
parser = JsonOutputParser(pydantic_object=ProductDescription)

format_instructions = parser.get_format_instructions()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert e-commerce copywriter. Your task is to generate compelling and detailed product descriptions based on provided product information. The output MUST strictly adhere to the following JSON format:\n{format_instructions}"),
        ("human", "Generate a product description for the following product:\nProduct Name: {product_name}\nCategory: {category}\nKeywords: {keywords}\nAttributes: {attributes}"),
    ]
)

# Chain to combine prompt, LLM, and parser
product_description_chain = prompt | llm | parser

# Streamlit UI
st.set_page_config(layout="wide", page_title="E-commerce Product Description Generator")
st.title("🛍️ E-commerce Product Description Generator")
st.markdown("Upload a CSV file with product details to generate structured product descriptions using AI.")

# File Uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("CSV loaded successfully!")
        st.dataframe(df.head())

        st.subheader("Generated Product Descriptions")
        generated_descriptions = []

        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, row in df.iterrows():
            status_text.text(f"Processing product {i+1}/{len(df)}: {row.get('product_name', 'Unnamed Product')}")
            progress_bar.progress((i + 1) / len(df))

            # Prepare input for the LLM
            product_input = {
                "product_name": row.get("product_name", ""),
                "category": row.get("category", "General"),
                "keywords": row.get("keywords", ""),
                "attributes": json.dumps({k: v for k, v in row.items() if k not in ['product_id', 'product_name', 'category', 'keywords'] and pd.notna(v)})
            }

            try:
                # Invoke the Langchain chain
                description_obj = product_description_chain.invoke(product_input)
                generated_descriptions.append(description_obj.dict()) # Convert Pydantic object to dict
            except Exception as e:
                st.warning(f"Could not generate description for product '{row.get('product_name', 'N/A')}': {e}")
                generated_descriptions.append({"product_name": row.get("product_name", "N/A"), "error": str(e)})
        
        status_text.text("All products processed!")
        progress_bar.empty()

        if generated_descriptions:
            st.json(generated_descriptions) # Display as JSON
            
            st.download_button(
                label="Download Generated Descriptions as JSON",
                data=json.dumps(generated_descriptions, indent=2),
                file_name="generated_product_descriptions.json",
                mime="application/json"
            )
        else:
            st.info("No descriptions were generated. Please check your CSV data and API key.")

    except Exception as e:
        st.error(f"Error processing CSV file: {e}")
        st.info("Please ensure your CSV is correctly formatted and contains relevant product columns.")

st.markdown("--- Generates structured product descriptions by leveraging an LLM and enforcing JSON output via Pydantic and Langchain. --- ")