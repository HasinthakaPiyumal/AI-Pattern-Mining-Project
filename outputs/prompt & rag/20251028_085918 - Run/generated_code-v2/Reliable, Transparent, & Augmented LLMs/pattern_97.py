import streamlit as st
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random

# --- 1. Global Models and Data (In-memory simulation) ---
nlp = spacy.load("en_core_web_sm")
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

candidate_data_store = {}
job_data_store = {}

class CandidateProfile(BaseModel):
    candidate_id: str
    name: str
    email: str
    resume_text: str
    demographic_group: str = "Other" # For bias simulation

class JobDescription(BaseModel):
    job_id: str
    title: str
    description: str

# --- 2. Text Preprocessing Module ---
class TextPreprocessor:
    def preprocess_text(self, text: str) -> str:
        doc = nlp(text.lower())
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and not token.is_space]
        return " ".join(tokens)

    def extract_entities(self, text: str):
        doc = nlp(text)
        entities = {ent.label_: ent.text for ent in doc.ents}
        return entities

# --- 3. Feature Engineering Module ---
class FeatureEngineer:
    def create_embeddings(self, texts: list) -> np.ndarray:
        return sentence_model.encode(texts)

    def combine_features(self, preprocessed_text: str, entities: dict) -> np.ndarray:
        text_embedding = self.create_embeddings([preprocessed_text])[0]
        # Simplified: just return text embedding for now. Can add more features later.
        return text_embedding

# --- 4. Bias Detection Module (Simplified) ---
class BiasDetector:
    def detect_statistical_bias(self, candidates_df: pd.DataFrame, job_id: str):
        # Simulate statistical bias: e.g., if one group is consistently under-represented in top ranks
        if not candidates_df.empty:
            demographic_counts = candidates_df['demographic_group'].value_counts(normalize=True)
            bias_report = {"job_id": job_id, "demographic_distribution": demographic_counts.to_dict()}
            if len(demographic_counts) > 1 and demographic_counts.min() < 0.2:
                bias_report["alert"] = "Potential demographic under-representation detected."
            return bias_report
        return {"job_id": job_id, "demographic_distribution": {}, "alert": "No candidate data to analyze."
}

    def detect_textual_bias(self, text: str) -> dict:
        biased_words = ["ninja", "guru", "rockstar", "aggressive", "competitive"]
        found_bias = [word for word in biased_words if word in text.lower()]
        if found_bias:
            return {"textual_bias": True, "keywords": found_bias, "alert": "Potentially biased language detected in job description/resume."
}
        return {"textual_bias": False, "keywords": []}

# --- 5. Bias Mitigation Module (Simplified) ---
class BiasMitigator:
    def mitigate_bias(self, candidate_scores: dict, bias_report: dict) -> dict:
        mitigated_scores = candidate_scores.copy()
        if "alert" in bias_report and "demographic under-representation" in bias_report["alert"]:
            demographic_distribution = bias_report["demographic_distribution"]
            # Simple mitigation: boost scores for under-represented groups
            min_group = min(demographic_distribution, key=demographic_distribution.get)
            for cand_id, score in mitigated_scores.items():
                candidate = candidate_data_store.get(cand_id)
                if candidate and candidate.demographic_group == min_group:
                    mitigated_scores[cand_id] = score * 1.1 # 10% boost
        elif "textual_bias" in bias_report and bias_report["textual_bias"]:
            # If textual bias detected, a recruiter might manually review. For automated mitigation,
            # we could slightly penalize job-candidate match if job has high textual bias and candidate doesn't match it
            # or flag for manual review. For now, assume it's handled by flagging.
            pass
        return mitigated_scores

# --- 6. Candidate Matching and Ranking Module ---
class CandidateMatcher:
    def __init__(self):
        self.model = LogisticRegression()
        self.preprocessor = TextPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.trained = False

    def train_model(self, candidates: dict, job_descriptions: dict):
        # This is a highly simplified training. In a real scenario, this would involve
        # labeling (e.g., historical hire data) and more complex feature engineering.
        # Here, we'll simulate a binary classification: 'good fit' or 'bad fit'.
        X, y = [], []

        for job_id, job_desc_obj in job_descriptions.items():
            job_text = self.preprocessor.preprocess_text(job_desc_obj.description)
            job_embedding = self.feature_engineer.create_embeddings([job_text])[0]

            for cand_id, cand_obj in candidates.items():
                cand_text = self.preprocessor.preprocess_text(cand_obj.resume_text)
                cand_embedding = self.feature_engineer.create_embeddings([cand_text])[0]

                # Feature vector is the concatenation of job and candidate embeddings
                feature_vector = np.concatenate((job_embedding, cand_embedding))
                X.append(feature_vector)

                # Simulate 'good fit' if candidate name starts with 'A' and job title contains 'Engineer'
                # This is purely for demonstration purposes.
                is_good_fit = 1 if cand_obj.name.startswith('A') and 'engineer' in job_desc_obj.title.lower() else 0
                y.append(is_good_fit)
        
        if not X or not y: # Handle empty data case for training
            print("Not enough data to train the model.")
            self.trained = False
            return

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        if len(set(y_train)) < 2: # Check if there's only one class in training data
             print("Only one class present in training data. Cannot train Logistic Regression.")
             self.trained = False
             return

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        print(f"Model accuracy: {accuracy_score(y_test, y_pred):.2f}")
        self.trained = True

    def rank_candidates(self, job_id: str) -> dict:
        if not self.trained:
            raise RuntimeError("Model not trained. Please train the model first.")

        job_desc_obj = job_data_store.get(job_id)
        if not job_desc_obj:
            raise HTTPException(status_code=404, detail="Job Description not found.")

        job_text = self.preprocessor.preprocess_text(job_desc_obj.description)
        job_embedding = self.feature_engineer.create_embeddings([job_text])[0]

        candidate_scores = {}
        candidate_features = {}

        for cand_id, cand_obj in candidate_data_store.items():
            cand_text = self.preprocessor.preprocess_text(cand_obj.resume_text)
            cand_embedding = self.feature_engineer.create_embeddings([cand_text])[0]
            feature_vector = np.concatenate((job_embedding, cand_embedding)).reshape(1, -1)
            candidate_features[cand_id] = feature_vector
            # Get probability of being a 'good fit'
            candidate_scores[cand_id] = self.model.predict_proba(feature_vector)[0][1]
        
        # Create a DataFrame for bias detection and mitigation
        candidate_df_for_bias = pd.DataFrame([
            {"candidate_id": c_id, "demographic_group": candidate_data_store[c_id].demographic_group, "score": score}
            for c_id, score in candidate_scores.items()
        ])
        candidate_df_for_bias.set_index('candidate_id', inplace=True)

        # Bias Detection
        bias_detector = BiasDetector()
        statistical_bias_report = bias_detector.detect_statistical_bias(candidate_df_for_bias, job_id)
        textual_bias_report = bias_detector.detect_textual_bias(job_desc_obj.description) # Check job description for bias

        # Bias Mitigation
        bias_mitigator = BiasMitigator()
        mitigated_scores = bias_mitigator.mitigate_bias(candidate_scores, statistical_bias_report)
        # Further integrate textual_bias_report if needed, e.g., for flagging or adjusting scores

        ranked_candidates = sorted(
            mitigated_scores.items(), key=lambda item: item[1], reverse=True
        )

        return {
            "job_id": job_id,
            "ranked_candidates": [{"candidate_id": c_id, "score": score} for c_id, score in ranked_candidates],
            "bias_reports": {
                "statistical_bias": statistical_bias_report,
                "textual_bias_job_description": textual_bias_report
            }
        }

# --- FastAPI Application --- 
app = FastAPI()
matcher = CandidateMatcher()

@app.post("/candidates")
async def add_candidate(candidate: CandidateProfile):
    if candidate.candidate_id in candidate_data_store:
        raise HTTPException(status_code=400, detail="Candidate with this ID already exists.")
    candidate_data_store[candidate.candidate_id] = candidate
    return {"message": "Candidate added successfully", "candidate_id": candidate.candidate_id}

@app.post("/jobs")
async def add_job_description(job: JobDescription):
    if job.job_id in job_data_store:
        raise HTTPException(status_code=400, detail="Job with this ID already exists.")
    job_data_store[job.job_id] = job
    return {"message": "Job description added successfully", "job_id": job.job_id}

@app.post("/train_model")
async def train_ml_model():
    try:
        matcher.train_model(candidate_data_store, job_data_store)
        return {"message": "Model training initiated. Check logs for status.", "trained": matcher.trained}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/rank_candidates/{job_id}")
async def get_ranked_candidates(job_id: str):
    try:
        if not matcher.trained:
            raise HTTPException(status_code=400, detail="ML model has not been trained yet.")
        return matcher.rank_candidates(job_id)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Streamlit Frontend --- 
st.set_page_config(layout="wide")
st.title("AI-Powered Bias-Aware Recruitment Platform")

st.sidebar.header("Manage Data")

with st.sidebar.expander("Add New Candidate"):
    with st.form("add_candidate_form"):
        cand_id = st.text_input("Candidate ID")
        cand_name = st.text_input("Name")
        cand_email = st.text_input("Email")
        cand_resume = st.text_area("Resume Text")
        cand_demographic = st.selectbox("Demographic Group", ["Group A", "Group B", "Group C", "Other"])
        submit_candidate = st.form_submit_button("Add Candidate")
        if submit_candidate:
            try:
                candidate = CandidateProfile(candidate_id=cand_id, name=cand_name, email=cand_email, resume_text=cand_resume, demographic_group=cand_demographic)
                candidate_data_store[cand_id] = candidate
                st.success(f"Candidate {cand_name} added!")
            except Exception as e:
                st.error(f"Error adding candidate: {e}")

with st.sidebar.expander("Add New Job Description"):
    with st.form("add_job_form"):
        job_id = st.text_input("Job ID")
        job_title = st.text_input("Job Title")
        job_description = st.text_area("Job Description Text")
        submit_job = st.form_submit_button("Add Job")
        if submit_job:
            try:
                job = JobDescription(job_id=job_id, title=job_title, description=job_description)
                job_data_store[job_id] = job
                st.success(f"Job '{job_title}' added!")
            except Exception as e:
                st.error(f"Error adding job: {e}")

st.sidebar.header("ML Model Actions")
if st.sidebar.button("Train ML Model"):
    try:
        matcher.train_model(candidate_data_store, job_data_store)
        st.sidebar.success("ML model training completed (or attempted)!")
        if not matcher.trained:
            st.sidebar.warning("Model training failed or data insufficient. Check console for details.")
    except Exception as e:
        st.sidebar.error(f"Error training model: {e}")

st.header("Current Data")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Candidates")
    if candidate_data_store:
        candidates_df = pd.DataFrame([vars(c) for c in candidate_data_store.values()])
        st.dataframe(candidates_df[['candidate_id', 'name', 'demographic_group']])
    else:
        st.info("No candidates added yet.")

with col2:
    st.subheader("Job Descriptions")
    if job_data_store:
        jobs_df = pd.DataFrame([vars(j) for j in job_data_store.values()])
        st.dataframe(jobs_df[['job_id', 'title']])
    else:
        st.info("No job descriptions added yet.")

st.header("Candidate Ranking and Bias Analysis")
selected_job_id = st.selectbox("Select a Job Description to Rank Candidates", list(job_data_store.keys()))

if st.button("Rank Candidates"):
    if selected_job_id:
        if not matcher.trained:
            st.error("Please train the ML model first before ranking candidates.")
        else:
            try:
                result = matcher.rank_candidates(selected_job_id)
                st.subheader("Ranked Candidates (Bias Mitigated)")
                ranked_df = pd.DataFrame(result["ranked_candidates"])
                st.dataframe(ranked_df)

                st.subheader("Bias Reports")
                st.json(result["bias_reports"])
            except Exception as e:
                st.error(f"Error ranking candidates: {e}")
    else:
        st.warning("Please add at least one job description and candidates to rank.")

# Instructions on how to run (for the user, not part of the generated code)
# To run this application:
# 1. Save the code as recruitment_platform.py
# 2. Install necessary libraries: pip install streamlit fastapi uvicorn pandas numpy spacy scikit-learn sentence-transformers
# 3. Download spacy model: python -m spacy download en_core_web_sm
# 4. Run the FastAPI backend: uvicorn recruitment_platform:app --reload
# 5. Run the Streamlit frontend in a separate terminal: streamlit run recruitment_platform.py