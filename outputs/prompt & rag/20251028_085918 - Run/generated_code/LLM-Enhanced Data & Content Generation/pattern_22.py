from typing import List, Dict, Any
from pydantic import BaseModel

# --- Data Models ---

class Product(BaseModel):
    id: str
    name: str
    category: str
    description: str
    price: float
    keywords: List[str] = []

class UserProfile(BaseModel):
    user_id: str
    browsing_history: List[str] = []  # List of product IDs
    past_purchases: List[str] = []   # List of product IDs
    implicit_feedback: Dict[str, Any] = {} # E.g., {'liked_categories': ['electronics'], 'disliked_brands': ['BrandX']}
    explicit_preferences: Dict[str, Any] = {} # E.g., {'style': 'modern', 'color': 'blue'}

class RecommendationResult(BaseModel):
    product_id: str
    personalized_description: str
    complementary_product_ids: List[str] = []
    relevance_score: float = 0.0

# --- Mock LLM and Feedback Systems ---

class MockLLM:
    """A mock Large Language Model to simulate content generation and intent understanding."""
    def understand_user_intent(self, user_profile: UserProfile) -> Dict[str, Any]:
        # In a real system, this would analyze text data from chat, search, etc.
        # For now, it infers based on structured profile.
        interests = set()
        for prod_id in user_profile.browsing_history + user_profile.past_purchases:
            # Simulate looking up product details to infer interests
            # This would ideally come from a product DB lookup
            if "laptop" in prod_id.lower():
                interests.add("high-tech electronics")
            elif "shirt" in prod_id.lower() or "jeans" in prod_id.lower():
                interests.add("apparel")
        interests.update(user_profile.implicit_feedback.get("liked_categories", []))
        
        print(f"[MockLLM] Understanding user {user_profile.user_id}. Inferred interests: {list(interests)}")
        return {"inferred_interests": list(interests), **user_profile.explicit_preferences}

    def generate_product_description(self, product: Product, user_interests: Dict[str, Any]) -> str:
        """Generates a personalized description for a product based on user interests."""
        interest_str = ", ".join(user_interests.get("inferred_interests", []))
        style = user_interests.get("style", "standard")
        color = user_interests.get("color", "any")

        base_description = f"Discover this {product.name}, a fantastic item in our {product.category} collection. {product.description}"

        if interest_str:
            personalized_phrase = f"Given your interest in {interest_str}, you'll appreciate its features."
            if style != "standard":
                personalized_phrase += f" It features a {style} design."
            if color != "any" and color in product.description.lower():
                personalized_phrase += f" This {color} variant is sure to impress."
            return f"{personalized_phrase} {base_description}"
        return base_description

    def generate_complementary_recommendations(self, product: Product, user_interests: Dict[str, Any], all_products: Dict[str, Product]) -> List[str]:
        """Suggests complementary product IDs based on the main product and user interests."""
        recommended_ids = []
        # Simple rule-based mock: suggest items from the same category or related keywords
        for other_product_id, other_product in all_products.items():
            if other_product_id == product.id: # Don't recommend the same product
                continue
            if other_product.category == product.category:
                if len(recommended_ids) < 2: # Limit to 2 for mock
                    recommended_ids.append(other_product_id)
            elif any(k in other_product.keywords for k in product.keywords) and len(recommended_ids) < 2:
                recommended_ids.append(other_product_id)
        
        print(f"[MockLLM] Generating complementary for {product.name}. User interests: {user_interests}. Recommended: {recommended_ids}")
        return recommended_ids

class RLHFFeedbackSystem:
    """Simulates the collection and application of RLHF."""
    def __init__(self):
        self.feedback_buffer: List[Dict[str, Any]] = []

    def collect_feedback(self, user_id: str, recommendation_result: RecommendationResult, user_action: str, sentiment: float = 0.0):
        """Collects user feedback (e.g., click, purchase, explicit rating)."""
        feedback_entry = {
            "user_id": user_id,
            "product_id": recommendation_result.product_id,
            "action": user_action, # e.g., 'click', 'purchase', 'ignore'
            "sentiment": sentiment, # e.g., 1 for positive, -1 for negative, 0 for neutral
            "description_generated": recommendation_result.personalized_description
        }
        self.feedback_buffer.append(feedback_entry)
        print(f"[RLHF] Collected feedback for {user_id}: {user_action} on {recommendation_result.product_id}")

    def apply_feedback_to_model(self, model: MockLLM):
        """Conceptual: In a real system, this would fine-tune or update the LLM weights."""
        if not self.feedback_buffer:
            print("[RLHF] No feedback to apply.")
            return
        
        positive_feedback_count = sum(1 for f in self.feedback_buffer if f['action'] in ['click', 'purchase'])
        negative_feedback_count = sum(1 for f in self.feedback_buffer if f['action'] == 'ignore')

        print(f"[RLHF] Applying feedback. Positive: {positive_feedback_count}, Negative: {negative_feedback_count}.")
        # In a real scenario, this would involve training an RL agent to optimize for rewards
        # derived from this feedback, then using that to fine-tune the generative LLM.
        self.feedback_buffer.clear() # Clear buffer after 'application'

class EngagementMetricsSystem:
    """Monitors user engagement to drive iterative refinement."""
    def __init__(self):
        self.metrics: Dict[str, Any] = {"total_recommendations": 0, "total_clicks": 0, "total_purchases": 0}

    def record_engagement(self, user_id: str, action: str):
        self.metrics["total_recommendations"] += 1
        if action == 'click':
            self.metrics["total_clicks"] += 1
        elif action == 'purchase':
            self.metrics["total_purchases"] += 1
        print(f"[Metrics] Recorded engagement for {user_id}: {action}. Current metrics: {self.metrics}")

    def get_conversion_rate(self) -> float:
        if self.metrics["total_recommendations"] == 0:
            return 0.0
        return (self.metrics["total_clicks"] + self.metrics["total_purchases"]) / self.metrics["total_recommendations"]

# --- Main Personalization System ---

class PersonalizationSystem:
    def __init__(
        self, 
        product_catalog: Dict[str, Product],
        llm: MockLLM,
        rlhf_system: RLHFFeedbackSystem,
        metrics_system: EngagementMetricsSystem
    ):
        self.product_catalog = product_catalog
        self.llm = llm
        self.rlhf_system = rlhf_system
        self.metrics_system = metrics_system

    def get_personalized_recommendation(
        self, 
        user_profile: UserProfile, 
        target_product_id: str
    ) -> RecommendationResult:
        """Generates a personalized product description and complementary items for a user."""
        if target_product_id not in self.product_catalog:
            raise ValueError(f"Product ID {target_product_id} not found in catalog.")
        
        target_product = self.product_catalog[target_product_id]

        # 1. Understand user intent and preferences
        user_interests = self.llm.understand_user_intent(user_profile)

        # 2. Generate personalized product description
        personalized_description = self.llm.generate_product_description(target_product, user_interests)

        # 3. Suggest complementary items
        complementary_ids = self.llm.generate_complementary_recommendations(
            target_product, user_interests, self.product_catalog
        )

        recommendation = RecommendationResult(
            product_id=target_product_id,
            personalized_description=personalized_description,
            complementary_product_ids=complementary_ids,
            relevance_score=0.9 # Placeholder score
        )
        return recommendation

    def process_user_action(self, user_id: str, recommendation: RecommendationResult, action: str, sentiment: float = 0.0):
        """Records user action and passes it to RLHF and metrics systems."""
        self.rlhf_system.collect_feedback(user_id, recommendation, action, sentiment)
        self.metrics_system.record_engagement(user_id, action)

    def run_iterative_refinement_cycle(self):
        """Simulates an iterative refinement cycle based on collected feedback and metrics."""
        print("\n--- Running Iterative Refinement Cycle ---")
        self.rlhf_system.apply_feedback_to_model(self.llm) # Apply RLHF
        conversion_rate = self.metrics_system.get_conversion_rate()
        print(f"Current conversion rate: {conversion_rate:.2f}")
        
        # In a real system, decision logic based on metrics would trigger
        # model retraining, A/B tests, or rule adjustments.
        if conversion_rate < 0.2: # Example threshold
            print("Conversion rate is low. Consider model re-evaluation or prompt tuning.")
        
        print("--- Refinement Cycle Complete ---")

# --- Example Usage ---
if __name__ == "__main__":
    # 1. Initialize Product Catalog
    products = {
        "prod_001": Product(id="prod_001", name="Ultimate Wireless Headphones", category="Electronics", 
                            description="Immersive sound, noise cancellation, and a sleek design.", price=299.99,
                            keywords=["audio", "wireless", "noise cancelling"]),
        "prod_002": Product(id="prod_002", name="Ergonomic Office Chair", category="Furniture", 
                            description="Comfortable and supportive for long working hours.", price=199.50,
                            keywords=["office", "comfort", "ergonomic"]),
        "prod_003": Product(id="prod_003", name="High-Performance Gaming Laptop", category="Electronics", 
                            description="Blazing fast processor and dedicated graphics for gamers.", price=1499.00,
                            keywords=["gaming", "laptop", "high performance"]),
        "prod_004": Product(id="prod_004", name="Stylish Denim Jeans", category="Apparel", 
                            description="Classic fit and durable denim for everyday wear.", price=59.99,
                            keywords=["fashion", "casual", "denim"])
    }

    # 2. Initialize User Profiles
    user_profiles = {
        "user_A": UserProfile(
            user_id="user_A", 
            browsing_history=["prod_001", "prod_003"],
            implicit_feedback={"liked_categories": ["electronics"], "preferred_brands": ["TechBrand"]},
            explicit_preferences={"style": "minimalist", "color": "black"}
        ),
        "user_B": UserProfile(
            user_id="user_B", 
            browsing_history=["prod_002", "prod_004"],
            past_purchases=["prod_004"],
            implicit_feedback={"liked_categories": ["apparel", "furniture"]}
        )
    }

    # 3. Initialize Core Systems
    mock_llm = MockLLM()
    rlhf_system = RLHFFeedbackSystem()
    metrics_system = EngagementMetricsSystem()

    personalization_system = PersonalizationSystem(
        product_catalog=products,
        llm=mock_llm,
        rlhf_system=rlhf_system,
        metrics_system=metrics_system
    )

    # --- Scenario 1: User A gets a recommendation for Headphones ---
    print("\n--- Scenario 1: User A gets personalized recommendation for Headphones ---")
    user_a = user_profiles["user_A"]
    rec_for_user_a = personalization_system.get_personalized_recommendation(user_a, "prod_001")
    print(f"\nRecommendation for {user_a.user_id} (Product: {products['prod_001'].name}):")
    print(f"Description: {rec_for_user_a.personalized_description}")
    comp_names = [products[pid].name for pid in rec_for_user_a.complementary_product_ids if pid in products]
    print(f"Complementary Items: {', '.join(comp_names)}")

    # Simulate user action: click on the headphones
    personalization_system.process_user_action(user_a.user_id, rec_for_user_a, "click", sentiment=1.0)

    # --- Scenario 2: User B gets a recommendation for Gaming Laptop ---
    print("\n--- Scenario 2: User B gets personalized recommendation for Gaming Laptop ---")
    user_b = user_profiles["user_B"]
    rec_for_user_b = personalization_system.get_personalized_recommendation(user_b, "prod_003")
    print(f"\nRecommendation for {user_b.user_id} (Product: {products['prod_003'].name}):")
    print(f"Description: {rec_for_user_b.personalized_description}")
    comp_names_b = [products[pid].name for pid in rec_for_user_b.complementary_product_ids if pid in products]
    print(f"Complementary Items: {', '.join(comp_names_b)}")

    # Simulate user action: ignore the gaming laptop (not relevant for user B)
    personalization_system.process_user_action(user_b.user_id, rec_for_user_b, "ignore", sentiment=-1.0)

    # --- Run an Iterative Refinement Cycle ---
    personalization_system.run_iterative_refinement_cycle()

    # --- Scenario 3: User A gets another recommendation (after refinement) ---
    print("\n--- Scenario 3: User A gets recommendation for Gaming Laptop (after refinement) ---")
    rec_for_user_a_again = personalization_system.get_personalized_recommendation(user_a, "prod_003")
    print(f"\nRecommendation for {user_a.user_id} (Product: {products['prod_003'].name}):")
    print(f"Description: {rec_for_user_a_again.personalized_description}")
    comp_names_a_again = [products[pid].name for pid in rec_for_user_a_again.complementary_product_ids if pid in products]
    print(f"Complementary Items: {', '.join(comp_names_a_again)}")
    personalization_system.process_user_action(user_a.user_id, rec_for_user_a_again, "purchase", sentiment=1.0)

    personalization_system.run_iterative_refinement_cycle()
