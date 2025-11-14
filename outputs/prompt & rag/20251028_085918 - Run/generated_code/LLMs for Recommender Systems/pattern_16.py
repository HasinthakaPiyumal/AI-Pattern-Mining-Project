
import pandas as pd
import numpy as np
from sklearn.decomposition import SVD
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer

# --- 0. Configuration and Mock Data --- #

# Mock API Key for LLM (in a real scenario, load securely from .env)
MOCK_LLM_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# Dummy Data Generation
np.random.seed(42)

num_users = 100
num_products = 50

users_data = {'user_id': range(1, num_users + 1), 'age': np.random.randint(18, 65, num_users), 'gender': np.random.choice(['Male', 'Female'], num_users)}
users_df = pd.DataFrame(users_data)

products_data = {
    'product_id': range(1, num_products + 1),
    'name': [f'Product {i}' for i in range(1, num_products + 1)],
    'category': np.random.choice(['Electronics', 'Books', 'Clothing', 'Home & Kitchen'], num_products),
    'description': [
        f'High-quality Product {i} from category {np.random.choice(["Electronics", "Books", "Clothing", "Home & Kitchen"])} with unique features and excellent reviews.'
        for i in range(1, num_products + 1)
    ]
}
products_df = pd.DataFrame(products_data)

# Generate sparse interaction data (user-product ratings/purchases)
interactions = []
for _ in range(300):
    user_id = np.random.randint(1, num_users + 1)
    product_id = np.random.randint(1, num_products + 1)
    rating = np.random.randint(1, 6) # Mock ratings from 1 to 5
    interactions.append({'user_id': user_id, 'product_id': product_id, 'rating': rating})
interactions_df = pd.DataFrame(interactions)

# Create user-product matrix for collaborative filtering
user_product_matrix = interactions_df.pivot_table(index='user_id', columns='product_id', values='rating').fillna(0)

# --- I. Core Recommender System (Traditional ML) --- #

class CoreRecommender:
    def __init__(self, user_product_matrix: pd.DataFrame, products_df: pd.DataFrame):
        self.user_product_matrix = user_product_matrix
        self.products_df = products_df
        self.model = None
        self.user_factors = None
        self.product_factors = None
        self.product_index_map = {pid: i for i, pid in enumerate(self.products_df['product_id'].values)}
        self.product_id_map = {i: pid for pid, i in self.product_index_map.items()}

    def train_model(self, n_components: int = 20):
        # Using SVD for collaborative filtering
        self.model = SVD(n_components=n_components)
        self.user_factors = self.model.fit_transform(self.user_product_matrix)
        self.product_factors = self.model.components_.T # Transpose to get product factors
        print(f"Core recommender model trained with {n_components} components.")

    def get_traditional_recommendations(self, user_id: int, top_n: int = 5) -> List[int]:
        if user_id not in self.user_product_matrix.index:
            return [] # User not found

        user_idx = self.user_product_matrix.index.get_loc(user_id)
        user_ratings = self.user_product_matrix.loc[user_id]

        # Predict ratings for unrated items
        predicted_ratings = np.dot(self.user_factors[user_idx], self.product_factors.T)

        # Get product IDs of items the user has not rated
        unrated_product_indices = user_ratings[user_ratings == 0].index
        unrated_product_ids = [self.product_index_map[pid] for pid in unrated_product_indices if pid in self.product_index_map]

        # Filter predicted ratings for unrated items
        if not unrated_product_ids:
            return []

        filtered_predictions = pd.Series(predicted_ratings[unrated_product_ids], index=[self.product_id_map[i] for i in unrated_product_ids])

        # Sort and get top N product IDs
        top_recommendations = filtered_predictions.nlargest(top_n).index.tolist()
        return top_recommendations

# --- II. LLM-Enhanced Explanation and Personalization Layer --- #

class SemanticProductUnderstanding:
    def __init__(self, products_df: pd.DataFrame):
        self.products_df = products_df
        # Load a pre-trained sentence transformer model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.product_embeddings = self._generate_product_embeddings()
        print("Semantic product understanding initialized.")

    def _generate_product_embeddings(self) -> Dict[int, List[float]]:
        # Generate embeddings for product descriptions
        descriptions = self.products_df['description'].tolist()
        embeddings = self.model.encode(descriptions, show_progress_bar=False)
        return {row['product_id']: embedding.tolist() for i, row in self.products_df.iterrows() for embedding in [embeddings[i]]}

    def get_product_embedding(self, product_id: int) -> Optional[List[float]]:
        return self.product_embeddings.get(product_id)

    def extract_key_features(self, product_id: int) -> List[str]:
        # In a real LLM scenario, this would use a prompt like:
        # "Extract 3 key features from the following product description: [description]"
        description = self.products_df[self.products_df['product_id'] == product_id]['description'].iloc[0]
        # Mock extraction based on keywords
        features = []
        if "high-quality" in description.lower(): features.append("High Quality Build")
        if "unique features" in description.lower(): features.append("Unique Features")
        if "excellent reviews" in description.lower(): features.append("Excellent Customer Reviews")
        if "electronics" in description.lower(): features.append("Advanced Electronic Components")
        if not features: features.append("General Product Attributes")
        return features[:3] # Limit to top 3 mock features

class ExplanationGenerator:
    def __init__(self, products_df: pd.DataFrame, users_df: pd.DataFrame):
        self.products_df = products_df
        self.users_df = users_df
        print("Explanation generator initialized.")

    def generate_explanation(self, user_id: int, product_id: int, reason: str, key_features: List[str]) -> str:
        # In a real LLM scenario, this would involve a complex prompt like:
        # "Generate a personalized explanation for recommending [product_name] to user [user_id],
        # considering their preferences and the reason: [reason]. Highlight features: [features]."

        product_name = self.products_df[self.products_df['product_id'] == product_id]['name'].iloc[0]
        product_category = self.products_df[self.products_df['product_id'] == product_id]['category'].iloc[0]
        user_age = self.users_df[self.users_df['user_id'] == user_id]['age'].iloc[0]

        features_str = ", ".join(key_features)

        explanation = f"Based on your past activity and similar users' preferences, we think you'll love '{product_name}'. "
        explanation += f"This {product_category} item stands out with its {features_str}. "
        explanation += f"As someone around {user_age} years old, we believe these attributes align well with your likely interests."

        return explanation

class PersonalizationEngine:
    def __init__(self, products_df: pd.DataFrame, semantic_product_understanding: SemanticProductUnderstanding):
        self.products_df = products_df
        self.semantic_product_understanding = semantic_product_understanding
        print("Personalization engine initialized.")

    def refine_recommendations(self, user_id: int, traditional_recommendations: List[int], user_preferences: Optional[Dict] = None) -> List[int]:
        # In a real LLM scenario, this would involve:
        # 1. LLM interpreting user_preferences (e.g., from conversational input or implicit signals)
        # 2. Re-ranking based on semantic similarity of products to interpreted user needs.
        # For this example, we'll mock a simple re-ranking based on 'high quality' preference.

        if user_preferences and user_preferences.get("desires_high_quality"): # Mock preference
            ranked_products = []
            # Prioritize products with 'High Quality Build' mock feature
            for prod_id in traditional_recommendations:
                features = self.semantic_product_understanding.extract_key_features(prod_id)
                if "High Quality Build" in features:
                    ranked_products.insert(0, prod_id) # Put at front
                else:
                    ranked_products.append(prod_id)
            return ranked_products
        
        # Otherwise, return as is (no LLM-driven re-ranking in this mock)
        return traditional_recommendations

# --- III. API & Frontend Integration (FastAPI) --- #

app = FastAPI(
    title="E-commerce LLM-Enhanced Recommender",
    description="A system demonstrating LLM-enhanced explanations and personalization for product recommendations."
)

# Pydantic models for request and response
class RecommendationRequest(BaseModel):
    user_id: int
    top_n: int = 5
    user_preferences: Optional[Dict] = None # For LLM-driven personalization

class ProductRecommendation(BaseModel):
    product_id: int
    product_name: str
    category: str
    explanation: str

class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[ProductRecommendation]
    message: str = "Recommendations generated successfully."

# Initialize core components
core_recommender = CoreRecommender(user_product_matrix, products_df)
core_recommender.train_model()

semantic_product_understanding = SemanticProductUnderstanding(products_df)
explanation_generator = ExplanationGenerator(products_df, users_df)
personalization_engine = PersonalizationEngine(products_df, semantic_product_understanding)

@app.post("/recommend", response_model=RecommendationResponse)
async def get_llm_enhanced_recommendations(request: RecommendationRequest):
    user_id = request.user_id
    top_n = request.top_n
    user_preferences = request.user_preferences

    if user_id not in users_df['user_id'].values:
        raise HTTPException(status_code=404, detail=f"User with ID {user_id} not found.")

    # 1. Core Recommender System: Get initial recommendations
    initial_recommendations_ids = core_recommender.get_traditional_recommendations(user_id, top_n * 2) # Get more to allow for refinement

    if not initial_recommendations_ids:
        return RecommendationResponse(
            user_id=user_id,
            recommendations=[],
            message="Could not generate initial recommendations for this user."
        )

    # 2. LLM-Enhanced Personalization: Refine recommendations
    # In a real LLM scenario, user_preferences would be interpreted by an LLM.
    refined_recommendations_ids = personalization_engine.refine_recommendations(user_id, initial_recommendations_ids, user_preferences)[:top_n]

    final_recommendations: List[ProductRecommendation] = []
    for prod_id in refined_recommendations_ids:
        product_info = products_df[products_df['product_id'] == prod_id].iloc[0]
        product_name = product_info['name']
        category = product_info['category']

        # LLM-Enhanced Explanation: Generate explanation
        key_features = semantic_product_understanding.extract_key_features(prod_id)
        explanation = explanation_generator.generate_explanation(
            user_id=user_id,
            product_id=prod_id,
            reason="based on collaborative filtering and your inferred preferences", # Mock reason
            key_features=key_features
        )

        final_recommendations.append(
            ProductRecommendation(
                product_id=prod_id,
                product_name=product_name,
                category=category,
                explanation=explanation
            )
        )

    return RecommendationResponse(
        user_id=user_id,
        recommendations=final_recommendations
    )

@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "Recommender system is operational."}

# To run this application:
# 1. Save the code as `ecommerce_recommender.py`
# 2. Install necessary libraries: `pip install pandas numpy scikit-learn fastapi uvicorn sentence-transformers`
# 3. Run from your terminal: `uvicorn ecommerce_recommender:app --reload`
# 4. Access the API documentation at `http://127.0.0.1:8000/docs`
