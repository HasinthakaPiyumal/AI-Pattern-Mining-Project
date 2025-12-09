import random

class ProductInfo:
    def __init__(self, name: str, category: str, features: list, target_audience: str, brand_guidelines: list, seo_keywords: list):
        self.name = name
        self.category = category
        self.features = features
        self.target_audience = target_audience
        self.brand_guidelines = brand_guidelines
        self.seo_keywords = seo_keywords

class ProductDescriptionGenerator:
    def generate_description(self, prompt: str, product_info: ProductInfo) -> str:
        base_description = f"Discover the amazing {product_info.name}! Perfect for {product_info.target_audience}."
        
        feature_list_str = ", ".join(product_info.features)
        if "include features" in prompt.lower() or "features" in prompt.lower():
            base_description += f" Featuring: {feature_list_str}."
            
        if "evocative" in prompt.lower():
            base_description += " Experience unparalleled quality and innovation."
        if "benefits-driven" in prompt.lower():
            base_description += " It solves your daily challenges with ease."
            
        if random.random() > 0.5:
            base_description += " Get yours today!"
        
        return base_description.strip()

class MetaPromptingAgent:
    def generate_initial_prompt(self, product_info: ProductInfo) -> str:
        initial_prompt = (
            f"Generate an engaging product description for a {product_info.category} named "
            f"'{product_info.name}'. Target audience: {product_info.target_audience}."
            f" Key features to highlight: {', '.join(product_info.features)}. "
            f"Ensure a {product_info.brand_guidelines[0] if product_info.brand_guidelines else 'positive'} tone. "
            f"Include SEO keywords: {', '.join(product_info.seo_keywords)}."
        )
        return initial_prompt

    def refine_prompt(self, current_prompt: str, product_info: ProductInfo, feedback: dict) -> str:
        refined_prompt = current_prompt
        
        if feedback.get("readability_score", 0) < 0.6:
            refined_prompt += " Make sure the description is easy to read and understand for a broad audience."
        if feedback.get("keyword_density_score", 0) < 0.5:
            refined_prompt += f" Increase the presence of keywords: {', '.join(product_info.seo_keywords)}. Ensure natural integration."
        if feedback.get("sentiment_score", 0) < 0:
            refined_prompt += " Ensure the tone is overwhelmingly positive and enthusiastic."
        if feedback.get("uniqueness_score", 0) < 0.8:
            refined_prompt += " Add more creative and unique phrasing to make it stand out from competitors."
        if not feedback.get("brand_guideline_adherence", False):
            refined_prompt += f" Strictly adhere to brand guidelines, including {', '.join(product_info.brand_guidelines)}."
            
        if random.random() > 0.7:
            refined_prompt = refined_prompt.replace("engaging", "captivating", 1)
            
        return refined_prompt

class DescriptionEvaluator:
    def evaluate(self, description: str, product_info: ProductInfo, desired_criteria: dict) -> dict:
        scores = {
            "readability_score": self._calculate_readability(description),
            "keyword_density_score": self._calculate_keyword_density(description, product_info.seo_keywords),
            "sentiment_score": self._analyze_sentiment(description),
            "uniqueness_score": self._check_uniqueness(description),
            "brand_guideline_adherence": self._check_brand_guidelines(description, product_info.brand_guidelines),
            "overall_score": 0
        }
        
        scores["sales_conversion_potential"] = (scores["readability_score"] + scores["keyword_density_score"] +
                                                 (scores["sentiment_score"] + 1) / 2 + scores["uniqueness_score"]) / 4 
        if not scores["brand_guideline_adherence"]:
            scores["sales_conversion_potential"] *= 0.7
            
        scores["overall_score"] = scores["sales_conversion_potential"]
        
        return scores

    def _calculate_readability(self, description: str) -> float:
        words = description.split()
        if not words: return 0.0
        word_len_avg = sum(len(word) for word in words) / len(words)
        sentence_count = description.count('.') + description.count('!') + description.count('?') + 1
        score = max(0.0, 1.0 - (word_len_avg / 10.0) - (len(words) / (sentence_count * 20.0)))
        return min(1.0, score + 0.3)

    def _calculate_keyword_density(self, description: str, keywords: list) -> float:
        if not keywords: return 1.0
        description_lower = description.lower()
        total_words = len(description_lower.split())
        if total_words == 0: return 0.0

        found_keywords = 0
        for keyword in keywords:
            if keyword.lower() in description_lower:
                found_keywords += 1
        return found_keywords / len(keywords)

    def _analyze_sentiment(self, description: str) -> float:
        positive_words = ["amazing", "excellent", "great", "love", "superb", "fantastic", "easy", "innovative", "high-quality", "perfect"]
        negative_words = ["bad", "poor", "difficult", "meh", "subpar"]
        
        score = 0.0
        for word in description.lower().split():
            if word in positive_words:
                score += 0.2
            elif word in negative_words:
                score -= 0.2
        return min(1.0, max(-1.0, score))

    def _check_uniqueness(self, description: str) -> float:
        generic_phrases = ["get yours today", "discover the amazing", "perfect for"]
        count = 0
        for phrase in generic_phrases:
            if phrase in description.lower():
                count += 1
        return max(0.0, 1.0 - (count * 0.2))

    def _check_brand_guidelines(self, description: str, guidelines: list) -> bool:
        description_lower = description.lower()
        for guideline in guidelines:
            if guideline.lower() not in description_lower:
                return False
        return True

class ECommerceProductDescriptionSystem:
    def __init__(self, max_iterations: int = 5, min_overall_score: float = 0.8):
        self.product_desc_generator = ProductDescriptionGenerator()
        self.meta_prompting_agent = MetaPromptingAgent()
        self.description_evaluator = DescriptionEvaluator()
        self.max_iterations = max_iterations
        self.min_overall_score = min_overall_score

    def generate_optimized_description(self, product_info: ProductInfo) -> tuple[str, str, list[dict]]:
        current_prompt = self.meta_prompting_agent.generate_initial_prompt(product_info)
        best_description = ""
        best_prompt = ""
        highest_overall_score = -1.0
        iteration_history = []

        print(f"\n--- Starting Generation for: {product_info.name} ---")
        for i in range(self.max_iterations):
            print(f"\nIteration {i+1}/{self.max_iterations}")
            print(f"Current Prompt: {current_prompt}")

            generated_description = self.product_desc_generator.generate_description(current_prompt, product_info)
            print(f"Generated Description: {generated_description}")

            desired_criteria = {
                "min_readability": 0.6,
                "min_keyword_density": 0.5,
                "min_sentiment": 0.0,
                "min_uniqueness": 0.7,
                "require_brand_adherence": True
            }
            feedback = self.description_evaluator.evaluate(generated_description, product_info, desired_criteria)
            print(f"Evaluation Feedback: {feedback}")

            iteration_history.append({
                "iteration": i + 1,
                "prompt_used": current_prompt,
                "description_generated": generated_description,
                "evaluation_scores": feedback
            })
            
            current_overall_score = feedback.get("overall_score", 0.0)

            if current_overall_score > highest_overall_score:
                highest_overall_score = current_overall_score
                best_description = generated_description
                best_prompt = current_prompt

            if current_overall_score >= self.min_overall_score:
                print(f"\nTarget quality reached (score: {current_overall_score:.2f} >= {self.min_overall_score:.2f}) in {i+1} iterations.")
                break

            current_prompt = self.meta_prompting_agent.refine_prompt(current_prompt, product_info, feedback)
            
        print(f"\n--- Generation Finished for: {product_info.name} ---")
        print(f"Best Description (Score: {highest_overall_score:.2f}): {best_description}")
        return best_description, best_prompt, iteration_history

if __name__ == "__main__":
    product1 = ProductInfo(
        name="Smart Coffee Maker X1",
        category="Home Appliances",
        features=["Wi-Fi connectivity", "voice control", "customizable brew strength", "self-cleaning"],
        target_audience="Tech-savvy coffee enthusiasts",
        brand_guidelines=["innovative", "convenient", "premium quality"],
        seo_keywords=["smart coffee maker", "Wi-Fi coffee", "voice control brew", "automatic coffee machine"]
    )

    product2 = ProductInfo(
        name="Eco-Friendly Yoga Mat",
        category="Fitness Equipment",
        features=["sustainable material", "non-slip grip", "lightweight", "biodegradable"],
        target_audience="Environmentally conscious yogis",
        brand_guidelines=["eco-conscious", "comfort", "durability"],
        seo_keywords=["eco yoga mat", "sustainable mat", "non-slip yoga", "biodegradable fitness"]
    )
    
    system = ECommerceProductDescriptionSystem(max_iterations=7, min_overall_score=0.85)

    final_desc1, final_prompt1, history1 = system.generate_optimized_description(product1)
    print(f"\nFinal Optimized Description for {product1.name}: {final_desc1}")
    print(f"Best Prompt Used: {final_prompt1}")

    print("\n" + "="*50 + "\n")

    final_desc2, final_prompt2, history2 = system.generate_optimized_description(product2)
    print(f"\nFinal Optimized Description for {product2.name}: {final_desc2}")
    print(f"Best Prompt Used: {final_prompt2}")
