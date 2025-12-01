import streamlit as st
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import random


# --- 1. Job Description Bias Analyzer Service ---
class JobDescriptionBiasAnalyzer:
    def __init__(self):
        self.biased_words = {
            "gender": [
                "he", "she", "him", "her", "man", "woman", "male", "female",
                "masculine", "feminine", "salesman", "saleswoman", "chairman", "chairwoman",
                "aggressive", "assertive", "dominant", "competitive", "nurturing", "supportive",
                "gentle", "sympathetic", "leader", "follower"
            ],
            "age": [
                "young", "energetic", "recent graduate", "experienced", "mature", "senior",
                "junior", "proven track record", "youthful", "old"
            ],
            "ethnicity": [
                "native speaker", "fluent in x language only", "cultural fit"
            ]
            # Add more categories and words as needed
        }

    def analyze(self, job_description: str) -> dict:
        found_biases = {
            "gender": [],
            "age": [],
            "ethnicity": [],
            "overall": []
        }
        job_description_lower = job_description.lower()

        for bias_type, words in self.biased_words.items():
            for word in words:
                if re.search(r'\b' + re.escape(word) + r'\b', job_description_lower):
                    found_biases[bias_type].append(word)
                    if word not in found_biases["overall"]:
                        found_biases["overall"].append(word)
        
        suggestions = self._generate_suggestions(found_biases["overall"])

        return {
            "identified_biases": found_biases,
            "suggestion_count": len(suggestions),
            "suggestions": suggestions
        }

    def _generate_suggestions(self, biased_words_found: list) -> list:
        suggestions = []
        if "he" in biased_words_found or "she" in biased_words_found:
            suggestions.append("Use gender-neutral pronouns like 'they/them' or rephrase to avoid pronouns.")
        if "salesman" in biased_words_found or "saleswoman" in biased_words_found:
            suggestions.append("Replace with 'sales representative' or 'sales associate'.")
        if "young" in biased_words_found or "recent graduate" in biased_words_found:
            suggestions.append("Focus on skills and experience, not age-related terms.")
        if "native speaker" in biased_words_found:
            suggestions.append("Specify required language proficiency level rather than 'native speaker'.")
        if not biased_words_found:
            suggestions.append("No significant biased language detected. Good job!")
        return suggestions


# --- 2. Candidate Anonymizer Service ---
class CandidateAnonymizer:
    def __init__(self):
        # Regex patterns for common PII
        self.name_patterns = [
            r"([A-Z][a-z]+(?:\s[A-Z][a-z]+){1,2})",  # Simple names like John Doe
            r"Mr\.?\s[A-Z][a-z]+", r"Ms\.?\s[A-Z][a-z]+", r"Dr\.?\s[A-Z][a-z]+"
        ]
        self.email_pattern = r"\S+@\S+\.com"
        self.phone_pattern = r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"
        self.address_pattern = r"\d+\s[A-Za-z]+\s(?:Street|St\.?|Avenue|Ave\.?|Road|Rd\.?|Lane|Ln\.?|Boulevard|Blvd\.?)"

    def anonymize(self, text: str) -> str:
        anonymized_text = text

        # Anonymize emails
        anonymized_text = re.sub(self.email_pattern, "[EMAIL_ANONYMIZED]", anonymized_text)

        # Anonymize phone numbers
        anonymized_text = re.sub(self.phone_pattern, "[PHONE_ANONYMIZED]", anonymized_text)

        # Anonymize addresses (simple example, can be much more complex)
        anonymized_text = re.sub(self.address_pattern, "[ADDRESS_ANONYMIZED]", anonymized_text)
        
        # Anonymize names (most challenging, simple regex for demonstration)
        for pattern in self.name_patterns:
            anonymized_text = re.sub(pattern, "[NAME_ANONYMIZED]", anonymized_text)

        return anonymized_text


# --- 3. Debiased Candidate Ranker Service ---
class DebiasedCandidateRanker:
    def __init__(self):
        # Dummy model and vectorizer for demonstration
        self.vectorizer = TfidfVectorizer()
        self.model = LogisticRegression()
        self._train_dummy_model()

    def _train_dummy_model(self):
        # This is a highly simplified dummy training for structural completeness.
        # In a real scenario, this would involve a large, carefully debiased dataset.
        dummy_skills_data = [
            "python java machine learning sql",
            "data analysis statistics excel",
            "project management communication leadership",
            "web development javascript html css",
            "marketing digital marketing seo sem"
        ]
        # Simulate scores (e.g., highly skilled vs. less skilled)
        dummy_scores = [1, 0, 1, 0, 1]  # 1 for good fit, 0 for less good fit

        X = self.vectorizer.fit_transform(dummy_skills_data)
        y = dummy_scores
        self.model.fit(X, y)

    def rank_candidates(self, candidates_data: list[dict]) -> pd.DataFrame:
        if not candidates_data:
            return pd.DataFrame(columns=["candidate_id", "skills", "experience_years", "ranking_score"])

        df = pd.DataFrame(candidates_data)
        
        # Simulate skill extraction if not already present
        if 'anonymized_text' in df.columns and 'skills' not in df.columns:
            df['skills'] = df['anonymized_text'].apply(self._extract_dummy_skills)

        # Vectorize skills for ranking
        candidate_skills_vectorized = self.vectorizer.transform(df['skills'])

        # Predict scores using the dummy model
        df['ranking_score'] = self.model.predict_proba(candidate_skills_vectorized)[:, 1]

        # Add some randomness to scores to make it less deterministic in this dummy setup
        df['ranking_score'] = df['ranking_score'] + df['experience_years'] * 0.1 + (random.random() * 0.2 - 0.1)
        df['ranking_score'] = df['ranking_score'].clip(0, 1) # Keep scores between 0 and 1

        return df.sort_values(by="ranking_score", ascending=False).reset_index(drop=True)

    def _extract_dummy_skills(self, text: str) -> str:
        # A very simplistic skill extractor for demonstration
        possible_skills = ["python", "java", "sql", "machine learning", "data analysis", 
                           "communication", "leadership", "web development", "marketing"]
        found_skills = [skill for skill in possible_skills if skill in text.lower()]
        return " ".join(found_skills)


# --- 4. Bias Reporting & Insights Service ---
class BiasReportingService:
    def generate_report(self, job_bias_results: dict, ranked_candidates_df: pd.DataFrame) -> dict:
        report = {
            "job_description_bias_summary": {},
            "candidate_ranking_insights": {},
            "overall_recommendations": []
        }

        # Summarize job description biases
        if job_bias_results and job_bias_results.get("identified_biases"):
            report["job_description_bias_summary"]["overall_biased_words"] = job_bias_results["identified_biases"]["overall"]
            report["job_description_bias_summary"]["gender_bias_words"] = job_bias_results["identified_biases"]["gender"]
            report["job_description_bias_summary"]["age_bias_words"] = job_bias_results["identified_biases"]["age"]
            report["job_description_bias_summary"]["ethnicity_bias_words"] = job_bias_results["identified_biases"]["ethnicity"]
            report["job_description_bias_summary"]["suggestions"] = job_bias_results["suggestions"]
        else:
            report["job_description_bias_summary"]["message"] = "No job description bias analysis performed or no biases found."

        # Provide candidate ranking insights
        if not ranked_candidates_df.empty:
            report["candidate_ranking_insights"]["top_5_candidates"] = ranked_candidates_df.head(5).to_dict(orient="records")
            report["candidate_ranking_insights"]["average_ranking_score"] = ranked_candidates_df["ranking_score"].mean()
            report["candidate_ranking_insights"]["note"] = "Ranking is based on anonymized data to reduce bias."
        else:
            report["candidate_ranking_insights"]["message"] = "No candidate ranking performed."

        # Overall recommendations
        report["overall_recommendations"].append("Regularly review and update biased word lexicons.")
        report["overall_recommendations"].append("Conduct human audits of anonymization and ranking results.")
        report["overall_recommendations"].append("Diversify sourcing channels to attract a broader candidate pool.")
        report["overall_recommendations"].append("Train hiring managers on unconscious bias.")

        return report


# --- Streamlit Frontend Application ---
st.set_page_config(layout="wide", page_title="FairRecruit AI Platform")
st.title("FairRecruit AI Platform: Bias-Aware Recruitment")

# Initialize services
bias_analyzer = JobDescriptionBiasAnalyzer()
anonymizer = CandidateAnonymizer()
ranker = DebiasedCandidateRanker()
reporter = BiasReportingService()

# Session state for storing results
if "job_bias_results" not in st.session_state:
    st.session_state.job_bias_results = None
if "anonymized_profile" not in st.session_state:
    st.session_state.anonymized_profile = ""
if "ranked_candidates_df" not in st.session_state:
    st.session_state.ranked_candidates_df = pd.DataFrame()


# --- Sidebar Navigation ---
with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Go to",
        ("Job Description Analyzer", "Candidate Anonymizer", "Debiased Candidate Ranker", "Bias & Insights Report")
    )
    st.markdown("--- ")
    st.write("Developed with fairness in mind using AI principles.")


# --- Main Content Area ---
if page == "Job Description Analyzer":
    st.header("Job Description Bias Analyzer")
    st.write("Paste your job description below to check for potential biases.")

    job_description_input = st.text_area(
        "Enter Job Description",
        "We are looking for a young, energetic salesman with a proven track record. He should be a native English speaker and a cultural fit for our aggressive team.",
        height=300
    )

    if st.button("Analyze Job Description"):
        if job_description_input:
            st.session_state.job_bias_results = bias_analyzer.analyze(job_description_input)
            st.subheader("Analysis Results:")
            st.json(st.session_state.job_bias_results)
            if st.session_state.job_bias_results["identified_biases"]["overall"]:
                st.warning(f"Detected {len(st.session_state.job_bias_results['identified_biases']['overall'])} potential biased terms.")
                st.info("Consider the following suggestions to make your job description more inclusive:")
                for suggestion in st.session_state.job_bias_results["suggestions"]:
                    st.write(f"- {suggestion}")
            else:
                st.success("No significant biased language detected. This job description appears fair.")
        else:
            st.warning("Please enter a job description to analyze.")

elif page == "Candidate Anonymizer":
    st.header("Candidate Profile Anonymizer")
    st.write("Paste a candidate's resume/profile text to anonymize PII.")

    candidate_profile_input = st.text_area(
        "Enter Candidate Profile Text",
        "John Doe, 123 Main St, Anytown, CA 90210. Email: john.doe@example.com. Phone: (555) 123-4567. Experience: 5 years as a Software Engineer.",
        height=300
    )

    if st.button("Anonymize Profile"):
        if candidate_profile_input:
            st.session_state.anonymized_profile = anonymizer.anonymize(candidate_profile_input)
            st.subheader("Anonymized Profile:")
            st.code(st.session_state.anonymized_profile)
            st.success("Candidate profile anonymized successfully.")
        else:
            st.warning("Please enter candidate profile text to anonymize.")

elif page == "Debiased Candidate Ranker":
    st.header("Debiased Candidate Ranker")
    st.write("Upload a CSV of anonymized candidates or enter data to get a fair ranking.")
    st.info("Example candidate data: `candidate_id,anonymized_text,experience_years`")
    st.markdown("Anonymized text could be the output from the Anonymizer service. Skills are extracted from this text.")

    uploaded_file = st.file_uploader("Upload Anonymized Candidate CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            candidates_df = pd.read_csv(uploaded_file)
            if not all(col in candidates_df.columns for col in ["candidate_id", "anonymized_text", "experience_years"]):
                st.error("CSV must contain 'candidate_id', 'anonymized_text', and 'experience_years' columns.")
            else:
                st.write("Original Candidate Data (first 5 rows):")
                st.dataframe(candidates_df.head())
                if st.button("Rank Candidates"): 
                    st.session_state.ranked_candidates_df = ranker.rank_candidates(candidates_df.to_dict(orient="records"))
                    st.subheader("Debiased Candidate Ranking:")
                    st.dataframe(st.session_state.ranked_candidates_df)
                    st.success("Candidates ranked successfully based on debiased attributes.")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
    else:
        st.write("Or manually enter a few candidates (comma-separated: ID, text, years_exp)")
        manual_candidate_input = st.text_area(
            "Candidate 1\nCandidate 2", 
            "1,software engineer with python and java skills,3\n2,data analyst with strong communication and excel,5", 
            height=150
        )
        if st.button("Rank Manual Candidates"):
            if manual_candidate_input:
                manual_candidates = []
                for line in manual_candidate_input.split('\n'):
                    parts = line.split(',')
                    if len(parts) == 3:
                        try:
                            manual_candidates.append({"candidate_id": parts[0].strip(), "anonymized_text": parts[1].strip(), "experience_years": int(parts[2].strip())})
                        except ValueError:
                            st.error(f"Invalid format for line: {line}. Expected ID, text, years_exp (integer).")
                            manual_candidates = [] # Clear to prevent partial data issues
                            break
                    else:
                        st.error(f"Invalid format for line: {line}. Expected ID, text, years_exp.")
                        manual_candidates = []
                        break
                
                if manual_candidates:
                    st.session_state.ranked_candidates_df = ranker.rank_candidates(manual_candidates)
                    st.subheader("Debiased Candidate Ranking:")
                    st.dataframe(st.session_state.ranked_candidates_df)
                    st.success("Candidates ranked successfully based on debiased attributes.")
            else:
                st.warning("Please enter manual candidate data or upload a CSV.")

elif page == "Bias & Insights Report":
    st.header("Bias & Insights Report")
    st.write("Generate a comprehensive report on detected biases and recruitment fairness insights.")

    if st.button("Generate Full Report"):
        if st.session_state.job_bias_results or not st.session_state.ranked_candidates_df.empty:
            report = reporter.generate_report(
                st.session_state.job_bias_results,
                st.session_state.ranked_candidates_df
            )
            st.subheader("Comprehensive Bias Report:")
            
            st.json(report)

            st.info("### Key Recommendations for Fairer Recruitment:")
            for rec in report["overall_recommendations"]:
                st.write(f"- {rec}")
            
        else:
            st.warning("No job description analysis or candidate ranking performed yet. Please use other tabs first.")

