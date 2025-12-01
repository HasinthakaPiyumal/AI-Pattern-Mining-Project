import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os # For mocking API key check

# Mock OpenAI API for demonstration purposes
class MockOpenAI:
    class ChatCompletion:
        @staticmethod
        def create(model, messages, temperature):
            print(f"\n--- Mock LLM Call ---")
            print(f"Model: {model}")
            print(f"Messages: {messages}")
            print(f"Temperature: {temperature}")
            print(f"--- End Mock LLM Call ---")

            # Simulate LLM reasoning and response
            user_query = next((m["content"] for m in messages if m["role"] == "user"), "")
            candidate_items_str = next((m["content"] for m in messages if m["role"] == "system" and "candidate items" in m["content"]), "")
            
            if "chain of thought" in user_query.lower():
                reasoning = "Thinking step-by-step:\n1. Analyze user's preferences from the query.\n2. Compare preferences with candidate item descriptions.\n3. Consider category and style matching.\n4. Select the most relevant items.\n"
            else:
                reasoning = ""
            
            recommendation_response = f"Based on your request and the available options, I recommend: Item 1, Item 3. These items align with your stated preferences for [extracted preferences] among the candidates. {reasoning}"
            
            return {
                "choices": [
                    {
                        "message": {
                            "content": recommendation_response
                        }
                    }
                ]
            }

# Replace the actual openai import with the mock for this demonstration
openai = MockOpenAI()

class DataLoader:
    def __init__(self):
        self.fashion_items = [
            {"id": 1, "name": "Classic Denim Jacket", "description": "A timeless blue denim jacket, perfect for casual wear.", "category": "Jackets"},
            {"id": 2, "name": "Elegant Silk Scarf", "description": "Luxurious silk scarf with intricate floral patterns.", "category": "Accessories"},
            {"id": 3, "name": "Running Shoes Pro", "description": "High-performance running shoes with superior cushioning and support.", "category": "Footwear"},
            {"id": 4, "name": "Striped Cotton T-shirt", "description": "Comfortable and breathable cotton t-shirt with classic stripes.", "category": "Tops"},
            {"id": 5, "name": "Leather Crossbody Bag", "description": "Stylish and practical leather bag for everyday essentials.", "category": "Bags"},
            {"id": 6, "name": "Wool Blend Beanie", "description": "Warm and soft beanie, ideal for cold weather.", "category": "Accessories"},
            {"id": 7, "name": "Graphic Print Hoodie", "description": "Casual hoodie with a unique graphic print, made from soft fleece.", "category": "Tops"},
            {"id": 8, "name": "Slim Fit Chinos", "description": "Versatile slim-fit chinos suitable for smart-casual occasions.", "category": "Bottoms"},
        ]

    def get_all_items(self):
        return self.fashion_items

class EmbeddingGenerator:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def get_embeddings(self, texts: list) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True)

class CandidateGenerator:
    def __init__(self):
        self.index = None
        self.item_ids = []

    def build_faiss_index(self, item_embeddings: np.ndarray, item_ids: list):
        dimension = item_embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(item_embeddings.astype('float32'))
        self.item_ids = item_ids

    def get_candidates(self, user_query_embedding: np.ndarray, top_k: int) -> list[int]:
        if self.index is None:
            raise ValueError("FAISS index not built. Call build_faiss_index first.")
        
        D, I = self.index.search(user_query_embedding.astype('float32'), top_k)
        candidate_indices = I[0].tolist()
        return [self.item_ids[i] for i in candidate_indices if i != -1]

class LLMRecommender:
    def __init__(self, llm_model="gpt-3.5-turbo"):
        self.llm_model = llm_model

    def create_llm_prompt(self, user_query: str, candidate_items: list[dict], few_shot_examples: list[dict] = None) -> list[dict]:
        system_instruction = "You are an intelligent fashion recommender system. Your goal is to provide personalized fashion item recommendations based on user preferences and a given list of candidate items. Think step-by-step to determine the best recommendations. First, understand the user's needs. Then, evaluate each candidate item against these needs. Finally, present your top recommendations with a brief justification."
        
        messages = [
            {"role": "system", "content": system_instruction}
        ]

        if few_shot_examples:
            for example in few_shot_examples:
                messages.append({"role": "user", "content": example["query"]})
                messages.append({"role": "assistant", "content": example["recommendation"]})

        candidate_items_str = "\n".join([f"- ID: {item['id']}, Name: {item['name']}, Description: {item['description']}, Category: {item['category']}" for item in candidate_items])
        
        messages.append({"role": "system", "content": f"Available candidate items:\n{candidate_items_str}"})
        messages.append({"role": "user", "content": user_query + "\nPlease provide your top 3 recommendations. Think step-by-step before giving the final recommendations."})
        
        return messages

    def get_recommendation(self, user_query: str, candidate_items: list[dict], few_shot_examples: list[dict] = None) -> str:
        prompt_messages = self.create_llm_prompt(user_query, candidate_items, few_shot_examples)
        
        try:
            response = openai.ChatCompletion.create(
                model=self.llm_model,
                messages=prompt_messages,
                temperature=0.7,
            )
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error getting recommendation from LLM: {e}"


if __name__ == "__main__":
    # 1. Initialize Modules
    data_loader = DataLoader()
    embedding_generator = EmbeddingGenerator()
    candidate_generator = CandidateGenerator()
    llm_recommender = LLMRecommender()

    # 2. Load Data
    all_fashion_items = data_loader.get_all_items()
    item_descriptions = [item["description"] for item in all_fashion_items]
    item_ids = [item["id"] for item in all_fashion_items]

    print("Generating item embeddings...")
    item_embeddings = embedding_generator.get_embeddings(item_descriptions)
    print(f"Generated {len(item_embeddings)} item embeddings.")

    # 3. Build FAISS Index
    print("Building FAISS index...")
    candidate_generator.build_faiss_index(item_embeddings, item_ids)
    print("FAISS index built.")

    # 4. Define a user query
    user_query = "I need a comfortable top for casual wear, something with a unique design."
    # user_query = "Looking for something warm for winter, maybe a hat or jacket."

    print(f"\nUser Query: {user_query}")

    # 5. Generate embedding for user query
    user_query_embedding = embedding_generator.get_embeddings([user_query])

    # 6. Get Candidates from FAISS
    top_k_candidates = 5
    candidate_item_ids = candidate_generator.get_candidates(user_query_embedding, top_k_candidates)
    candidate_items_for_llm = [item for item in all_fashion_items if item["id"] in candidate_item_ids]
    
    print(f"\nTop {len(candidate_items_for_llm)} candidate items identified for LLM processing:")
    for item in candidate_items_for_llm:
        print(f"  - {item['name']} ({item['category']})")

    # 7. Define few-shot examples (optional)
    few_shot_examples = [
        {
            "query": "I want a stylish bag for going out.",
            "recommendation": "You might like the Elegant Silk Scarf (ID: 2) or the Leather Crossbody Bag (ID: 5). The scarf adds a touch of class, and the bag is practical yet chic."
        },
        {
            "query": "What's good for a workout?",
            "recommendation": "I suggest the Running Shoes Pro (ID: 3) for performance and comfort during your run."
        }
    ]

    # 8. Get Recommendation from LLM
    print("\nGetting recommendations from LLM with Chain-of-Thought...")
    final_recommendation = llm_recommender.get_recommendation(
        user_query=user_query,
        candidate_items=candidate_items_for_llm,
        few_shot_examples=few_shot_examples
    )

    print("\nFinal Recommendation:")
    print(final_recommendation)
