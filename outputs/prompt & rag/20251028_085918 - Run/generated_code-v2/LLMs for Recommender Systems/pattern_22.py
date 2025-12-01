import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import openai
import os

class MedicalLiteratureDB:
    def __init__(self, articles_df: pd.DataFrame, model_name: str = "all-MiniLM-L6-v2"):
        self.articles_df = articles_df
        self.model = SentenceTransformer(model_name)
        self.embeddings = self.model.encode(articles_df["content"].tolist(), show_progress_bar=False)
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def retrieve_candidates(self, query: str, top_k: int = 5):
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(query_embedding, top_k)
        return self.articles_df.iloc[indices[0]].to_dict(orient="records")

class LLMRecommender:
    def __init__(self, openai_api_key: str):
        openai.api_key = openai_api_key

    def _generate_llm_response(self, prompt: str, model: str = "gpt-3.5-turbo"):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error communicating with LLM: {e}"

    def generate_zero_few_shot_recommendation(self, query: str, candidates: list, few_shot_examples: list = None):
        prompt_parts = []
        prompt_parts.append(f"You are an expert medical literature recommender. Recommend relevant medical articles based on the user's query.")
        
        if few_shot_examples:
            prompt_parts.append("Here are some examples of how to recommend articles:")
            for example in few_shot_examples:
                prompt_parts.append(f"\nUser query: {example['query']}\nRelevant articles (titles): {'; '.join(example['titles'])}")

        prompt_parts.append(f"\n\nUser query: {query}\nConsider the following candidate articles:")
        for i, candidate in enumerate(candidates):
            prompt_parts.append(f"Article {i+1}: Title: {candidate['title']}. Abstract: {candidate['abstract'][:200]}...")
        
        prompt_parts.append("\nBased on the user query and candidate articles, list the most relevant articles (by title) and provide a brief reason for each recommendation. Prioritize relevance and scientific rigor.")
        
        full_prompt = "\n".join(prompt_parts)
        return self._generate_llm_response(full_prompt)

    def generate_cot_recommendation(self, query: str, candidates: list):
        prompt_parts = []
        prompt_parts.append(f"You are an expert medical literature recommender. Analyze the user's query and candidate articles step-by-step to provide a precise recommendation.")
        prompt_parts.append(f"\n\nUser query: {query}\nCandidate articles:")
        for i, candidate in enumerate(candidates):
            prompt_parts.append(f"Article {i+1}: Title: {candidate['title']}. Abstract: {candidate['abstract'][:200]}...")
        
        prompt_parts.append("\n\nLet's think step by step to find the most relevant articles:")
        prompt_parts.append("1. Identify the key medical entities and concepts in the user's query.")
        prompt_parts.append("2. For each candidate article, extract its main topic and assess its relevance to the identified key entities.")
        prompt_parts.append("3. Filter out articles that are not directly relevant or lack scientific rigor for a clinician.")
        prompt_parts.append("4. Prioritize the remaining relevant articles based on recency, study type (e.g., meta-analysis, clinical trial), and direct applicability.")
        prompt_parts.append("5. Finally, list the top recommended articles (by title) and explain the detailed reasoning for each selection.")

        full_prompt = "\n".join(prompt_parts)
        return self._generate_llm_response(full_prompt)


if __name__ == "__main__":
    # 1. Simulate Medical Literature Database
    articles_data = [
        {"title": "New Advances in Type 2 Diabetes Treatment", "abstract": "This review discusses recent breakthroughs in pharmacological and lifestyle interventions for type 2 diabetes, including SGLT2 inhibitors and GLP-1 agonists.", "content": "New Advances in Type 2 Diabetes Treatment. This review discusses recent breakthroughs in pharmacological and lifestyle interventions for type 2 diabetes, including SGLT2 inhibitors and GLP-1 agonists. It covers clinical trial results and their implications for patient care.", "keywords": "diabetes, SGLT2, GLP-1, treatment"},
        {"title": "Role of AI in Early Cancer Detection", "abstract": "Investigating the application of artificial intelligence and deep learning models in improving the accuracy and speed of early cancer diagnosis from imaging data.", "content": "Role of AI in Early Cancer Detection. Investigating the application of artificial intelligence and deep learning models in improving the accuracy and speed of early cancer diagnosis from imaging data, focusing on lung and breast cancer.", "keywords": "AI, cancer, diagnosis, imaging"},
        {"title": "Cardiovascular Risk Factors in Diabetic Patients", "abstract": "A comprehensive study on the prevalence and management of cardiovascular complications in individuals with diabetes mellitus.", "content": "Cardiovascular Risk Factors in Diabetic Patients. A comprehensive study on the prevalence and management of cardiovascular complications in individuals with diabetes mellitus. Data from a large cohort study is analyzed.", "keywords": "cardiovascular, diabetes, risk factors"},
        {"title": "Precision Medicine Approaches for Oncology", "abstract": "Exploring personalized treatment strategies for various cancers based on genomic profiling and molecular biomarkers.", "content": "Precision Medicine Approaches for Oncology. Exploring personalized treatment strategies for various cancers based on genomic profiling and molecular biomarkers. Case studies and future directions are discussed.", "keywords": "precision medicine, oncology, genomics"},
        {"title": "Impact of Diet on Gut Microbiome and Metabolic Health", "abstract": "Research on how dietary patterns influence gut microbial composition and its subsequent effects on human metabolic health, including obesity and diabetes.", "content": "Impact of Diet on Gut Microbiome and Metabolic Health. Research on how dietary patterns influence gut microbial composition and its subsequent effects on human metabolic health, including obesity and diabetes. A focus on fiber intake.", "keywords": "diet, microbiome, metabolic health"},
    ]
    articles_df = pd.DataFrame(articles_data)
    
    print("Initializing Medical Literature Database...")
    medical_db = MedicalLiteratureDB(articles_df)
    print("Database initialized with", len(articles_df), "articles.")

    # Set your OpenAI API key from environment variable
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Error: OPENAI_API_KEY environment variable not set. Please set it to use the LLM recommender.")
        exit()
    
    print("Initializing LLM Recommender...")
    llm_recommender = LLMRecommender(openai_api_key)
    print("LLM Recommender initialized.")

    # 2. User Query and Candidate Generation
    user_query = "recent treatments for type 2 diabetes and their cardiovascular benefits"
    print(f"\nUser Query: {user_query}")

    print("Generating candidate articles...")
    candidate_articles = medical_db.retrieve_candidates(user_query, top_k=3)
    print("Candidate articles retrieved:")
    for i, article in enumerate(candidate_articles):
        print(f"  {i+1}. {article['title']}")

    # 3. LLM Recommendation (Zero/Few-shot)
    print("\n--- Zero/Few-shot Recommendation ---")
    # Example few-shot (optional)
    few_shot_examples = [
        {
            "query": "articles on vaccine efficacy for flu",
            "titles": ["Seasonal Influenza Vaccine Effectiveness", "Factors Influencing Flu Vaccine Efficacy"]
        }
    ]
    zero_shot_result = llm_recommender.generate_zero_few_shot_recommendation(user_query, candidate_articles, few_shot_examples)
    print("Zero/Few-shot Recommendation Result:")
    print(zero_shot_result)

    # 4. LLM Recommendation (Chain-of-Thought)
    print("\n--- Chain-of-Thought Recommendation ---")
    cot_result = llm_recommender.generate_cot_recommendation(user_query, candidate_articles)
    print("Chain-of-Thought Recommendation Result:")
    print(cot_result)