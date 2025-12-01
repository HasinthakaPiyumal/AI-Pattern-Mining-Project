"""
This module simulates loading product data from an e-commerce catalog.
"""

def load_mock_products():
    """
    Loads a mock list of fashion products.
    In a real application, this would fetch data from a database or API.
    """
    products = [
        {"id": "P001", "name": "Blue Denim Jeans", "category": "Bottoms", "tags": ["casual", "denim", "blue", "jeans", "everyday"]},
        {"id": "P002", "name": "White Cotton T-Shirt", "category": "Tops", "tags": ["casual", "cotton", "white", "t-shirt", "basic"]},
        {"id": "P003", "name": "Black Leather Jacket", "category": "Outerwear", "tags": ["casual", "leather", "black", "jacket", "trendy"]},
        {"id": "P004", "name": "Formal Grey Blazer", "category": "Outerwear", "tags": ["formal", "grey", "blazer", "office", "elegant"]},
        {"id": "P005", "name": "Red Evening Dress", "category": "Dresses", "tags": ["formal", "red", "dress", "evening", "party"]},
        {"id": "P006", "name": "Sneakers White", "category": "Footwear", "tags": ["casual", "white", "sneakers", "sporty", "comfortable"]},
        {"id": "P007", "name": "Brown Leather Boots", "category": "Footwear", "tags": ["casual", "leather", "brown", "boots", "rugged"]},
        {"id": "P008", "name": "Striped Button-Up Shirt", "category": "Tops", "tags": ["casual", "smart-casual", "striped", "shirt", "versatile"]},
        {"id": "P009", "name": "Pencil Skirt Black", "category": "Bottoms", "tags": ["formal", "black", "skirt", "office", "professional"]},
        {"id": "P010", "name": "Floral Maxi Dress", "category": "Dresses", "tags": ["casual", "summer", "floral", "dress", "maxi"]},
        {"id": "P011", "name": "Chino Trousers Beige", "category": "Bottoms", "tags": ["smart-casual", "beige", "chinos", "everyday"]},
        {"id": "P012", "name": "Cashmere Sweater Grey", "category": "Tops", "tags": ["luxury", "winter", "grey", "sweater", "warm"]},
        {"id": "P013", "name": "Denim Jacket Light Blue", "category": "Outerwear", "tags": ["casual", "denim", "light blue", "jacket"]},
        {"id": "P014", "name": "Heels Black", "category": "Footwear", "tags": ["formal", "black", "heels", "elegant", "party"]},
        {"id": "P015", "name": "Sport Leggings Black", "category": "Bottoms", "tags": ["sporty", "black", "leggings", "fitness", "activewear"]}
    ]
    return products

if __name__ == "__main__":
    # Example usage
    all_products = load_mock_products()
    print(f"Loaded {len(all_products)} mock products.")
    print(all_products[0])