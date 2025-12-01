import streamlit as st
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import requests
import json
import os

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- FastAPI Backend --- #

app = FastAPI()

class EssaySubmission(BaseModel):
    essay_text: str

class LikertFeedback(BaseModel):
    clarity: str
    coherence: str
    argumentation: str
    grammar: str

@app.post("/evaluate-essay", response_model=LikertFeedback)
async def evaluate_essay(submission: EssaySubmission):
    # Ensure OPENAI_API_KEY is set in environment variables
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable not set.")

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # Define the Likert scale categories
    likert_scale_categories = ["Poor", "Acceptable", "Good", "Very Good", "Excellent"]
    scale_str = ", ".join(f"'{c}'" for c in likert_scale_categories)

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an AI essay grader. Evaluate the following essay based on clarity, coherence, argumentation, and grammar. For each criterion, provide a single rating using only the following Likert scale categories: " + scale_str + ". Your response must be a JSON object with keys 'clarity', 'coherence', 'argumentation', and 'grammar'."),
            ("user", "{essay}"),
        ]
    )

    output_parser = StrOutputParser()

    chain = prompt_template | llm | output_parser

    llm_response = await chain.ainvoke({"essay": submission.essay_text})
    
    try:
        # Attempt to parse the LLM's string response as JSON
        parsed_feedback = json.loads(llm_response)
        # Validate against the Pydantic model
        feedback = LikertFeedback(**parsed_feedback)
        return feedback
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse LLM response as JSON.")
    except ValidationError as e:
        raise HTTPException(status_code=500, detail=f"LLM response did not match expected LikertFeedback schema: {e}")

# --- Streamlit Frontend --- #

st.set_page_config(page_title="Automated Essay Grader")
st.title("📝 Automated Essay Grader with Likert Scale Feedback")
st.markdown("Enter your essay below to receive qualitative feedback based on clarity, coherence, argumentation, and grammar.")

essay_input = st.text_area("Your Essay", height=300, placeholder="Paste your essay here...")

if st.button("Get Feedback"):
    if essay_input:
        st.info("Evaluating your essay...")
        try:
            # Make sure FastAPI is running on http://localhost:8000
            response = requests.post(
                "http://localhost:8000/evaluate-essay",
                json={"essay_text": essay_input}
            )
            response.raise_for_status()  # Raise an exception for HTTP errors
            feedback = response.json()

            st.subheader("Evaluation Results:")
            st.write(f"**Clarity:** {feedback.get('clarity', 'N/A')}")
            st.write(f"**Coherence:** {feedback.get('coherence', 'N/A')}")
            st.write(f"**Argumentation:** {feedback.get('argumentation', 'N/A')}")
            st.write(f"**Grammar:** {feedback.get('grammar', 'N/A')}")
            st.success("Feedback generated successfully!")

        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the FastAPI backend. Please ensure it is running (run `uvicorn essay_grader_app:app --reload` in your terminal).", icon="🚨")
        except requests.exceptions.RequestException as e:
            st.error(f"An error occurred during evaluation: {e}", icon="🚨")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}", icon="🚨")
    else:
        st.warning("Please enter an essay to get feedback.")


# Instructions for running the application
st.sidebar.markdown("## How to Run This Application")
st.sidebar.markdown("1. Save this code as `essay_grader_app.py`.")
st.sidebar.markdown("2. **Install dependencies:** `pip install fastapi uvicorn 'openai<2.0.0' langchain_openai streamlit pydantic requests`")
st.sidebar.markdown("3. **Set your OpenAI API Key:** `export OPENAI_API_KEY='your_api_key'`")
st.sidebar.markdown("4. **Start the FastAPI backend:** Open a terminal and run `uvicorn essay_grader_app:app --reload`")
st.sidebar.markdown("5. **Start the Streamlit frontend:** Open another terminal and run `streamlit run essay_grader_app.py`")
st.sidebar.markdown("6. Access the Streamlit app in your browser (usually `http://localhost:8501`).")
