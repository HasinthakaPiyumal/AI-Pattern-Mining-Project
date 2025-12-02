from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch

class ImageCaptioningModel:
    def __init__(self):
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

    def generate_caption(self, image_path):
        raw_image = Image.open(image_path).convert("RGB")
        inputs = self.processor(raw_image, return_tensors="pt")
        out = self.model.generate(**inputs)
        return self.processor.decode(out[0], skip_special_tokens=True)

class TextEmbeddingModel:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def get_embedding(self, text):
        return self.model.encode(text, convert_to_tensor=True)

class ProductDatabase:
    def __init__(self):
        self.products = [
            {"id": 1, "name": "Blue Denim Jeans", "description": "Classic blue denim jeans, slim fit, comfortable for everyday wear.", "image_url": "url_to_jeans_image.jpg"},
            {"id": 2, "name": "Red Casual Dress", "description": "A vibrant red casual dress, knee-length, perfect for summer outings.", "image_url": "url_to_dress_image.jpg"},
            {"id": 3, "name": "Leather Office Chair", "description": "Ergonomic leather office chair with adjustable height and lumbar support.", "image_url": "url_to_chair_image.jpg"},
            {"id": 4, "name": "Striped T-Shirt", "description": "Soft cotton t-shirt with horizontal blue and white stripes.", "image_url": "url_to_tshirt_image.jpg"},
            {"id": 5, "name": "Running Shoes", "description": "Lightweight running shoes with breathable mesh upper and responsive cushioning.", "image_url": "url_to_shoes_image.jpg"}
        ]
        self.embeddings = {}

    def precompute_embeddings(self, embedding_model):
        for product in self.products:
            self.embeddings[product["id"]] = embedding_model.get_embedding(product["description"])

class SearchRecommendationEngine:
    def __init__(self, product_db, embedding_model):
        self.product_db = product_db
        self.embedding_model = embedding_model

    def search_by_image_caption(self, image_caption, top_n=3):
        query_embedding = self.embedding_model.get_embedding(image_caption)
        similarities = []

        for product in self.product_db.products:
            product_embedding = self.product_db.embeddings[product["id"]]
            similarity = cosine_similarity(query_embedding.reshape(1, -1), product_embedding.reshape(1, -1))[0][0]
            similarities.append((product, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_n]

def main():
    print("Initializing Smart Product Search and Recommendation System...")
    captioning_model = ImageCaptioningModel()
    embedding_model = TextEmbeddingModel()
    product_db = ProductDatabase()
    product_db.precompute_embeddings(embedding_model)
    search_engine = SearchRecommendationEngine(product_db, embedding_model)

    print("\nReady for product search by image.")
    image_path = input("Enter the path to your product image: ")

    try:
        print("Generating caption for the image...")
        image_caption = captioning_model.generate_caption(image_path)
        print(f"Generated Caption: {image_caption}")

        print("Searching for similar products...")
        recommended_products = search_engine.search_by_image_caption(image_caption)

        print("\n--- Top Recommended Products ---")
        if recommended_products:
            for product, score in recommended_products:
                print(f"Product Name: {product["name"]}")
                print(f"Description: {product["description"]}")
                print(f"Similarity Score: {score:.4f}")
                print("---------------------------------")
        else:
            print("No similar products found.")

    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()