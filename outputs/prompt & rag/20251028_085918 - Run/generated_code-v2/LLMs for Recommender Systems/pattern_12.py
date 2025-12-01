"""
product_catalog_api.py: Simulates an external product catalog API.
In a real application, this would interface with an e-commerce platform's product database.
"""

class ProductCatalogAPI:
    def __init__(self):
        self.products = [
            {"id": "1", "name": "Red Summer Dress", "category": "dress", "color": "red", "style": "casual", "price": 45.00},
            {"id": "2", "name": "Blue Denim Shirt", "category": "shirt", "color": "blue", "style": "casual", "price": 30.00},
            {"id": "3", "name": "Black Formal Trousers", "category": "pants", "color": "black", "style": "formal", "price": 60.00},
            {"id": "4", "name": "Floral Maxi Dress", "category": "dress", "color": "multi", "style": "boho", "price": 75.00},
            {"id": "5", "name": "White Cotton T-Shirt", "category": "shirt", "color": "white", "style": "casual", "price": 20.00},
            {"id": "6", "name": "Navy Cocktail Dress", "category": "dress", "color": "navy", "style": "formal", "price": 120.00},
            {"id": "7", "name": "Green Jogger Pants", "category": "pants", "color": "green", "style": "casual", "price": 35.00},
            {"id": "8", "name": "Striped Polo Shirt", "category": "shirt", "color": "blue", "style": "casual", "price": 28.00},
        ]

    def search_products(self, query: str):
        """Searches the product catalog for items matching the query."""
        query_lower = query.lower()
        results = []
        for product in self.products:
            if query_lower in product["name"].lower() or \
               query_lower in product["category"].lower() or \
               query_lower in product["color"].lower() or \
               query_lower in product["style"].lower():
                results.append(product)
        return results