import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

load_dotenv()

# --- LLM Initialization ---
# Primary LLM for generating product descriptions
primary_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7, openai_api_key=os.getenv("OPENAI_API_KEY"))

# Meta-LLM for optimizing prompts
# Potentially use a more capable model like GPT-4 for meta-LLM if available
meta_llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5, openai_api_key=os.getenv("OPENAI_API_KEY"))

# --- Prompt Templates ---

# Template for the primary LLM to generate product descriptions
PRODUCT_DESCRIPTION_TEMPLATE = """
Generate a compelling and SEO-friendly product description based on the following details:
Product Name: {product_name}
Features: {features}
Target Audience: {target_audience}
Keywords: {keywords}

Description should be engaging, informative, and persuade the target audience.
"""
product_description_prompt = PromptTemplate(
    input_variables=["product_name", "features", "target_audience", "keywords"],
    template=PRODUCT_DESCRIPTION_TEMPLATE,
)
product_description_chain = LLMChain(llm=primary_llm, prompt=product_description_prompt)

# Template for the meta-LLM to optimize prompts
META_PROMPT_OPTIMIZATION_TEMPLATE = """
You are an expert prompt engineer. Your task is to refine and optimize a given prompt for a product description generator.
The goal is to make the generated product descriptions better based on user feedback.

Here is the current product description generation prompt:
---
{current_prompt_text}
---

Product Details:
Product Name: {product_name}
Features: {features}
Target Audience: {target_audience}
Keywords: {keywords}

User Feedback on the last generated description:
---
{feedback}
---

Based on the product details and the user feedback, improve the 'current_prompt_text'.
The improved prompt should aim to address the feedback and produce a higher quality description.
Return ONLY the optimized prompt text. Do not include any other commentary or introductory phrases.
"""
meta_prompt_optimization_prompt = PromptTemplate(
    input_variables=["current_prompt_text", "product_name", "features", "target_audience", "keywords", "feedback"],
    template=META_PROMPT_OPTIMIZATION_TEMPLATE,
)
meta_prompt_optimization_chain = LLMChain(llm=meta_llm, prompt=meta_prompt_optimization_prompt)

# --- Streamlit UI and Logic ---
st.set_page_config(page_title="E-commerce Product Description Generator")
st.title("🛒 E-commerce Product Description Generator")
st.markdown("Automate and optimize your product descriptions using AI.")

# Input fields
st.header("Product Details")
product_name = st.text_input("Product Name")
features = st.text_area("Key Features (comma-separated or bullet points)")
target_audience = st.text_input("Target Audience (e.g., 'young professionals', 'eco-conscious consumers')")
keywords = st.text_input("SEO Keywords (comma-separated)")

# Initialize session state variables if they don't exist
if "current_description" not in st.session_state:
    st.session_state.current_description = ""
if "current_primary_prompt" not in st.session_state:
    st.session_state.current_primary_prompt = product_description_prompt.template # Store the template string
if "iteration" not in st.session_state:
    st.session_state.iteration = 0

def generate_description_and_update_state():
    if product_name and features and target_audience and keywords:
        st.session_state.iteration += 1
        st.info(f"Generating description (Iteration {st.session_state.iteration})...")
        
        # Use the stored (potentially optimized) prompt template string to create a new PromptTemplate object
        temp_prompt = PromptTemplate(
            input_variables=["product_name", "features", "target_audience", "keywords"],
            template=st.session_state.current_primary_prompt,
        )
        temp_chain = LLMChain(llm=primary_llm, prompt=temp_prompt)

        try:
            generated_text = temp_chain.run(
                product_name=product_name,
                features=features,
                target_audience=target_audience,
                keywords=keywords,
            )
            st.session_state.current_description = generated_text
            st.success("Description generated!")
        except Exception as e:
            st.error(f"Error generating description: {e}")
            st.session_state.current_description = ""
    else:
        st.warning("Please fill in all product details to generate a description.")

# Generate initial description button
if st.button("Generate Initial Description"):
    st.session_state.current_primary_prompt = product_description_prompt.template # Reset to initial template
    st.session_state.iteration = 0 # Reset iteration count
    generate_description_and_update_state()

# Display generated description
if st.session_state.current_description:
    st.subheader(f"Generated Product Description (Iteration {st.session_state.iteration})")
    st.write(st.session_state.current_description)

    st.subheader("Feedback & Refinement")
    user_feedback = st.text_area("Provide feedback to refine the description:", key="user_feedback")

    if st.button("Refine Description with Feedback"):
        if user_feedback:
            st.info(f"Optimizing prompt and regenerating description (Iteration {st.session_state.iteration + 1})...")
            try:
                optimized_prompt_text = meta_prompt_optimization_chain.run(
                    current_prompt_text=st.session_state.current_primary_prompt,
                    product_name=product_name,
                    features=features,
                    target_audience=target_audience,
                    keywords=keywords,
                    feedback=user_feedback,
                )
                st.session_state.current_primary_prompt = optimized_prompt_text # Store the optimized prompt text
                generate_description_and_update_state()
                st.session_state.user_feedback = "" # Clear feedback after use
            except Exception as e:
                st.error(f"Error optimizing prompt or regenerating description: {e}")
        else:
            st.warning("Please provide some feedback to refine the description.")

st.markdown("""
--- 
*This application uses OpenAI's GPT models. Ensure you have `OPENAI_API_KEY` set in your `.env` file.*
""")