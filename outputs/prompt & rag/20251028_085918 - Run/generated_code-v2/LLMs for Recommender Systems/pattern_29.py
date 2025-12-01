import pandas as pd
import numpy as np

def generate_ecommerce_data(num_users=100, num_products=500, num_interactions=1000):
    """
    Generates synthetic e-commerce data for demonstration purposes.
    Includes users, products, and user-product interactions.
    """
    # Users
    user_ids = [f"user_{i}" for i in range(num_users)]
    user_features = pd.DataFrame({
        "user_id": user_ids,
        "age": np.random.randint(18, 70, num_users),
        "gender": np.random.choice(["Male", "Female", "Other"], num_users, p=[0.45, 0.5, 0.05])
    })

    # Products
    product_ids = [f"product_{i}" for i in range(num_products)]
    categories = ["Electronics", "Clothing", "Books", "Home & Garden", "Sports"]
    product_features = pd.DataFrame({
        "product_id": product_ids,
        "category": np.random.choice(categories, num_products),
        "price": np.round(np.random.uniform(10, 500, num_products), 2)
    })

    # Interactions (e.g., purchases, clicks, views)
    interactions_data = []
    for _ in range(num_interactions):
        user_id = np.random.choice(user_ids)
        product_id = np.random.choice(product_ids)
        interaction_type = np.random.choice(["purchase", "click", "view"], p=[0.2, 0.3, 0.5])
        timestamp = pd.to_datetime(f"2023-01-01") + pd.to_timedelta(np.random.randint(0, 365), unit="D")
        interactions_data.append({
            "user_id": user_id,
            "product_id": product_id,
            "interaction_type": interaction_type,
            "timestamp": timestamp
        })
    interactions = pd.DataFrame(interactions_data)

    print(f"Generated {len(user_features)} users, {len(product_features)} products, and {len(interactions)} interactions.")
    return user_features, product_features, interactions

if __name__ == "__main__":
    users, products, interactions = generate_ecommerce_data()
    print("\nUser Features Head:")
    print(users.head())
    print("\nProduct Features Head:")
    print(products.head())
    print("\nInteractions Head:")
    print(interactions.head())