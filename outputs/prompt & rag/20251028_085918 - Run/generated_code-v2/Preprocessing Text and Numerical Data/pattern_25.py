import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_data(reviews_df, products_df):
    # 1. Numerical Data Preprocessing
    numerical_features = ['rating', 'price', 'purchase_frequency']
    
    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    products_df[numerical_features] = imputer.fit_transform(products_df[numerical_features])
    
    # Scale numerical features
    scaler = StandardScaler()
    scaled_numerical_data = scaler.fit_transform(products_df[numerical_features])
    scaled_numerical_df = pd.DataFrame(scaled_numerical_data, columns=numerical_features, index=products_df.index)
    
    # 2. Text Data Preprocessing
    stop_words = set(stopwords.words('english'))
    
    def clean_text(text):
        if not isinstance(text, str):
            return ""
        tokens = word_tokenize(text.lower())
        filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
        return " ".join(filtered_tokens)
    
    reviews_df['cleaned_text'] = reviews_df['text'].apply(clean_text)
    
    # TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer(max_features=5000) # Limiting features for demonstration
    tfidf_features = tfidf_vectorizer.fit_transform(reviews_df['cleaned_text'])
    tfidf_df = pd.DataFrame(tfidf_features.toarray(), columns=tfidf_vectorizer.get_feature_names_out(), index=reviews_df.index)
    
    # 3. Feature Combination
    # For simplicity, let's assume a direct merge based on index if product_id is implicitly aligned
    # In a real scenario, you'd merge based on a common product ID.
    # For this example, we'll concatenate assuming reviews and products are aligned or merged beforehand based on a product ID.
    
    # If reviews_df and products_df are already aligned by product or customer:
    combined_features_df = pd.concat([scaled_numerical_df, tfidf_df], axis=1)
    
    return combined_features_df, tfidf_vectorizer, imputer, scaler

# --- Example Usage ---
if __name__ == "__main__":
    # Simulate raw data
    reviews_data = {
        'product_id': [1, 1, 2, 2, 3],
        'text': [
            'This product is amazing! I love it so much. Highly recommended.',
            'The quality is poor and it broke after a week. Very disappointed.',
            'Good value for money, but the delivery was slow.',
            'Excellent product, exactly what I needed.',
            'Not bad, but I expected more. Missing some features.'
        ]
    }
    products_data = {
        'product_id': [1, 2, 3],
        'rating': [4.5, 2.0, 3.5],
        'price': [50.0, 15.0, 120.0],
        'purchase_frequency': [100, 20, 50]
    }
    
    reviews_df = pd.DataFrame(reviews_data)
    products_df = pd.DataFrame(products_data)

    # Merge reviews and product data (conceptual, adjust based on actual data structure)
    # For this example, let's assume we align reviews to products based on product_id
    # and then process. A more robust solution would involve proper joins.
    
    # For simple demonstration, we will just pass products_df for numerical and reviews_df for text
    # and assume their resulting processed features can be combined.
    
    # To properly combine, let's first prepare the data such that each review entry also has its product metadata
    # This is a simplification; in a real system, you'd likely have a product feature matrix and review feature matrix
    # and combine them for specific tasks.
    
    # Let's create a combined dataframe that would represent the input to our preprocessor if we wanted to combine at the start
    # For this example, we'll keep them separate for preprocessing as per the architecture description
    # and then show how their output *could* be combined.
    
    print("\n--- Original Reviews Data ---")
    print(reviews_df)
    print("\n--- Original Products Data ---")
    print(products_df)

    # Call the preprocessing function
    # Note: This is a simplified call. In a real application, you'd ensure proper alignment.
    # Here, we're assuming the numerical features are for products and text features for reviews,
    # and showing how their processed outputs could conceptually be combined.
    
    # To make the combined_features_df meaningful, we'll align based on product_id conceptually.
    # Let's create a richer dataset for demonstration if needed.

    # Simplified: process numerical and text data independently and then show combination idea
    
    # Process numerical features of products_df
    numerical_cols_to_process = ['rating', 'price', 'purchase_frequency']
    imputer = SimpleImputer(strategy='mean')
    products_df_imputed = pd.DataFrame(imputer.fit_transform(products_df[numerical_cols_to_process]), columns=numerical_cols_to_process, index=products_df.index)
    scaler = StandardScaler()
    products_df_scaled = pd.DataFrame(scaler.fit_transform(products_df_imputed), columns=numerical_cols_to_process, index=products_df.index)
    products_df_processed = pd.concat([products_df[['product_id']], products_df_scaled], axis=1)

    # Process text features of reviews_df
    stop_words = set(stopwords.words('english'))
    def clean_text_for_demo(text):
        if not isinstance(text, str):
            return ""
        tokens = word_tokenize(text.lower())
        filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
        return " ".join(filtered_tokens)
    reviews_df['cleaned_text'] = reviews_df['text'].apply(clean_text_for_demo)
    
    tfidf_vectorizer = TfidfVectorizer(max_features=100) # Smaller for demo
    tfidf_features_matrix = tfidf_vectorizer.fit_transform(reviews_df['cleaned_text'])
    tfidf_df_reviews = pd.DataFrame(tfidf_features_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out(), index=reviews_df.index)
    reviews_df_processed = pd.concat([reviews_df[['product_id']], tfidf_df_reviews], axis=1)

    print("\n--- Processed Numerical Features (Products) ---")
    print(products_df_processed)
    print("\n--- Processed Text Features (Reviews - TF-IDF) ---")
    print(reviews_df_processed)

    # To combine, we would typically join on 'product_id' or align based on a common entity
    # Here's a conceptual merge if we wanted to combine features per product from reviews and its metadata
    # This part would depend heavily on the exact downstream task and data structure.
    
    # Example of merging processed data if we want to get product-level features including aggregated review features
    # This is an illustration, the actual aggregation strategy might vary.
    
    # Aggregate TF-IDF features by product_id (e.g., mean of review vectors for a product)
    # This is a simplification; a more advanced system might use embeddings or more complex aggregation.
    aggregated_review_features = reviews_df_processed.groupby('product_id').mean()
    
    final_combined_features = pd.merge(
        products_df_processed,
        aggregated_review_features,
        on='product_id',
        how='left'
    ).set_index('product_id')
    
    print("\n--- Final Combined Features (Conceptual Merge) ---")
    print(final_combined_features)
    print("Shape of combined features:", final_combined_features.shape)

    # The 'final_combined_features' DataFrame would then be ready for downstream ML models.
