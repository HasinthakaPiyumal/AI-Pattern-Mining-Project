from typing import List, Dict

class MockImageCaptioner:
    def __init__(self):
        pass

    def generate_caption(self, image_path: str) -> str:
        # In a real application, this would use a model like BLIP or CLIP
        # from the transformers library to generate a caption from the image.
        # For this simplified example, we'll return a predefined caption based on a mock image path.
        if "product_a.jpg" in image_path:
            return "A stylish blue denim jacket with two chest pockets and metal buttons."
        elif "product_b.jpg" in image_path:
            return "A pair of comfortable black running shoes with white soles and orange accents."
        elif "product_c.jpg" in image_path:
            return "A sleek silver laptop with a large screen and a backlit keyboard."
        else:
            return "A generic product image."

class MockLLMInterface:
    def __init__(self):
        # In a real application, this would load an LLM model
        # from the transformers library (e.g., GPT-2, T5, Llama).
        pass

    def generate_response(self, prompt: str) -> str:
        # Simulate LLM response based on prompt content
        if "recommend" in prompt.lower():
            return f"Based on your interest and the product descriptions, I recommend checking out our latest collection of \"urban casual wear\" which features similar stylish items. You might also like: Black Slim Fit Jeans, White Crew Neck T-shirt."
        elif "features" in prompt.lower() and "denim jacket" in prompt.lower():
            return f"The blue denim jacket features two functional chest pockets, classic metal button closures, and a comfortable regular fit. It's made from 100% premium cotton denim."
        elif "description for seller" in prompt.lower():
            return f"Product Title: Premium Blue Denim Jacket - Stylish & Versatile. Product Description: Elevate your wardrobe with our timeless blue denim jacket. Crafted from high-quality denim, it features classic metal buttons, two convenient chest pockets, and a comfortable fit, making it perfect for any casual occasion. A must-have staple for every fashion-forward individual."
        else:
            return f"I received your query: '{prompt}'. How else can I assist you today?"

class PromptEngineer:
    def __init__(self, captioner: MockImageCaptioner, llm: MockLLMInterface):
        self.captioner = captioner
        self.llm = llm

    def create_and_query_llm(self, image_path: str, user_query: str, context_info: Dict = None) -> str:
        # 1. Image Processing and Captioning
        image_caption = self.captioner.generate_caption(image_path)

        # 2. Prompt Engineering
        full_prompt = f"Image Description: {image_caption}.\n\nUser Query: {user_query}"
        if context_info:
            for key, value in context_info.items():
                full_prompt += f"\n{key}: {value}"
        
        # 3. LLM Integration
        llm_response = self.llm.generate_response(full_prompt)
        return llm_response

    def generate_product_description_for_seller(self, image_path: str, product_name: str, additional_details: str = "") -> str:
        image_caption = self.captioner.generate_caption(image_path)
        prompt = (
            f"Generate a detailed product description for a seller based on the following information.\n"
            f"Product Name: {product_name}\n"
            f"Image Description: {image_caption}\n"
            f"Additional Details: {additional_details}\n"
            f"Please provide a compelling product title and description."
        )
        llm_response = self.llm.generate_response(prompt)
        return llm_response

# --- Demonstration of Usage ---
if __name__ == "__main__":
    # Initialize components
    captioner = MockImageCaptioner()
    llm_interface = MockLLMInterface()
    prompt_engineer = PromptEngineer(captioner, llm_interface)

    print("--- Scenario 1: Product Recommendation ---")
    image_of_viewed_product = "data/images/product_a.jpg"
    user_recommendation_query = "Recommend similar items based on this product."
    user_browsing_history = {"Last Viewed Category": "Apparel", "Preferred Style": "Casual"}
    
    recommendation = prompt_engineer.create_and_query_llm(
        image_of_viewed_product,
        user_recommendation_query,
        context_info=user_browsing_history
    )
    print(f"User Query: {user_recommendation_query}\nRecommendation: {recommendation}\n")

    print("--- Scenario 2: Answering User Query about Product Features ---")
    image_of_queried_product = "data/images/product_a.jpg"
    user_feature_query = "What are the main features of this blue denim jacket?"
    
    features_response = prompt_engineer.create_and_query_llm(
        image_of_queried_product,
        user_feature_query
    )
    print(f"User Query: {user_feature_query}\nFeatures: {features_response}\n")

    print("--- Scenario 3: Generating Product Description for Seller ---")
    seller_uploaded_image = "data/images/product_a.jpg"
    product_name_for_seller = "Blue Denim Jacket"
    seller_additional_details = "Material: 100% Cotton, Fit: Regular, Style: Classic"

    seller_description = prompt_engineer.generate_product_description_for_seller(
        seller_uploaded_image,
        product_name_for_seller,
        seller_additional_details
    )
    print(f"Seller's Request: Generate description for '{product_name_for_seller}'\nGenerated Description: {seller_description}\n")

    print("--- Scenario 4: Another Product Recommendation ---")
    image_of_viewed_product_b = "data/images/product_b.jpg"
    user_recommendation_query_b = "I'm looking for new running shoes. What do you recommend based on this image and my past purchases?"
    user_browsing_history_b = {"Last Purchased Brand": "Nike", "Preferred Color": "Black"}

    recommendation_b = prompt_engineer.create_and_query_llm(
        image_of_viewed_product_b,
        user_recommendation_query_b,
        context_info=user_browsing_history_b
    )
    print(f"User Query: {user_recommendation_query_b}\nRecommendation: {recommendation_b}\n")