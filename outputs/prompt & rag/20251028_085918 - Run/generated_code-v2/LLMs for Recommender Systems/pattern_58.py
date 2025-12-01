import json

class FashionItem:
    def __init__(self, item_id, name, description, category, color, style, price, image_url=None):
        self.item_id = item_id
        self.name = name
        self.description = description
        self.category = category
        self.color = color
        self.style = style
        self.price = price
        self.image_url = image_url

    def to_dict(self):
        return {
            "item_id": self.item_id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "color": self.color,
            "style": self.style,
            "price": self.price,
            "image_url": self.image_url
        }

# Simulated E-commerce Catalog
mock_catalog = [
    FashionItem("001", "Blue Denim Jeans", "Classic straight-fit denim jeans.", "bottom", "blue", "casual", 59.99),
    FashionItem("002", "White Cotton T-Shirt", "Soft breathable cotton t-shirt.", "top", "white", "casual", 19.99),
    FashionItem("003", "Floral Maxi Dress", "Elegant maxi dress with a vibrant floral pattern.", "dress", "multi", "bohemian", 89.99),
    FashionItem("004", "Leather Ankle Boots", "Stylish black leather ankle boots.", "shoe", "black", "chic", 120.00),
    FashionItem("005", "Casual Blazer", "Lightweight casual blazer for everyday wear.", "outerwear", "grey", "smart-casual", 75.00),
    FashionItem("006", "Summer Hat", "Wide-brim straw hat for sun protection.", "accessory", "beige", "summer", 25.00),
    FashionItem("007", "Black Evening Gown", "Sleek black gown, perfect for formal events.", "dress", "black", "formal", 199.99),
    FashionItem("008", "Silver Hoop Earrings", "Simple elegant silver hoop earrings.", "accessory", "silver", "elegant", 35.00),
    FashionItem("009", "Red Sneakers", "Comfortable and vibrant red sneakers.", "shoe", "red", "sporty", 65.00),
    FashionItem("010", "Striped Button-Up Shirt", "Classic striped long-sleeve shirt.", "top", "blue", "smart-casual", 45.00),
    FashionItem("011", "High-Waisted Skirt", "Flowy high-waisted midi skirt.", "bottom", "green", "bohemian", 55.00),
    FashionItem("012", "Denim Jacket", "Classic blue denim jacket.", "outerwear", "blue", "casual", 79.99),
    FashionItem("013", "Gold Necklace", "Delicate gold chain necklace.", "accessory", "gold", "elegant", 49.99),
    FashionItem("014", "Workout Leggings", "High-performance black workout leggings.", "bottom", "black", "sporty", 40.00),
    FashionItem("015", "Plaid Scarf", "Warm plaid scarf for winter.", "accessory", "red", "winter", 30.00),
]

class CandidateGenerationModule:
    def __init__(self, catalog):
        self.catalog = catalog

    def generate_candidates(self, query, filters=None, limit=10):
        candidates = []
        query_keywords = query.lower().split()
        
        for item in self.catalog:
            match_score = 0
            item_text = f"{item.name} {item.description} {item.category} {item.color} {item.style}".lower()

            # Keyword matching
            for keyword in query_keywords:
                if keyword in item_text:
                    match_score += 1
            
            # Apply filters if provided
            if filters:
                if filters.get("category") and filters["category"].lower() not in item.category.lower():
                    continue
                if filters.get("color") and filters["color"].lower() not in item.color.lower():
                    continue
                if filters.get("style") and filters["style"].lower() not in item.style.lower():
                    continue

            if match_score > 0:
                candidates.append((item, match_score))
        
        # Sort by match score and return limited candidates
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [item for item, _ in candidates[:limit]]

class PromptEngineeringModule:
    def generate_zero_shot_prompt(self, user_query, candidates):
        candidate_list_str = "\n".join([f"- {c.name} ({c.category}, {c.color}, {c.style})" for c in candidates])
        prompt = f"""You are an AI fashion stylist. Based on the user's request and the available items, provide a personalized fashion recommendation.

User Request: "{user_query}"
Available Items:
{candidate_list_str}

Recommendation:"""
        return prompt

    def generate_few_shot_prompt(self, user_query, candidates, examples=None):
        if examples is None:
            examples = [
                {
                    "query": "casual outfit for a weekend",
                    "available": [
                        "White Cotton T-Shirt (top, white, casual)",
                        "Blue Denim Jeans (bottom, blue, casual)",
                        "Red Sneakers (shoe, red, sporty)"
                    ],
                    "recommendation": "For a casual weekend, I recommend pairing the White Cotton T-Shirt with the Blue Denim Jeans. Complete the look with the Red Sneakers for a comfortable and stylish outfit."
                },
                {
                    "query": "an elegant dress for an evening event",
                    "available": [
                        "Black Evening Gown (dress, black, formal)",
                        "Silver Hoop Earrings (accessory, silver, elegant)"
                    ],
                    "recommendation": "For an elegant evening event, the Black Evening Gown is a perfect choice. Accessorize with the Silver Hoop Earrings for a sophisticated touch."
                }
            ]
        
        example_str = ""
        for ex in examples:
            example_str += f"""User Request: "{ex['query']}" 
Available Items:
{ex['available']}
Recommendation: {ex['recommendation']}

"""

        candidate_list_str = "\n".join([f"- {c.name} ({c.category}, {c.color}, {c.style})" for c in candidates])
        prompt = f"""You are an AI fashion stylist. Based on the user's request and the available items, provide a personalized fashion recommendation.

{example_str}User Request: "{user_query}"
Available Items:
{candidate_list_str}

Recommendation:"""
        return prompt

    def generate_cot_prompt(self, user_query, candidates):
        candidate_list_str = "\n".join([f"- {c.name} (ID: {c.item_id}, {c.category}, {c.color}, {c.style}, Price: ${c.price:.2f})" for c in candidates])
        prompt = f"""You are an AI fashion stylist. I need your help to create a personalized fashion recommendation by thinking step-by-step.

User Request: "{user_query}"
Available Items:
{candidate_list_str}

Here's how to think step-by-step:
1. Analyze the user's request to understand the occasion, desired style, and any specific item mentions.
2. From the 'Available Items', identify primary items that directly match the core request (e.g., a dress for a wedding).
3. Consider complementary items (e.g., accessories, outerwear, shoes) that would complete the outfit based on the style and occasion.
4. Ensure the recommended items are compatible in terms of color, style, and general aesthetic.
5. Formulate a clear and concise recommendation that combines these items, explaining why they are a good fit.

Let's begin.

Thought Process: 
"""
        return prompt

class LLMProviderIntegration:
    def get_llm_response(self, prompt, use_cot=False):
        # This simulates an LLM call. In a real application, you'd use an actual LLM API.
        print("\n--- LLM Prompt ---")
        print(prompt)
        print("------------------\n")

        if use_cot:
            # Simulate CoT processing and a final recommendation
            if "summer wedding" in prompt.lower() and "dress" in prompt.lower():
                return (
                    "Thought Process: The user needs an outfit for a summer wedding. I should look for dresses, preferably light-colored or floral, and then suggest elegant accessories and suitable shoes. \n" 
                    "1. Identify suitable dresses: Floral Maxi Dress (bohemian, could work for casual wedding), Black Evening Gown (too formal). Floral Maxi Dress is the best fit. \n" 
                    "2. Suggest accessories: Gold Necklace, Silver Hoop Earrings. Gold Necklace might pair well with floral. \n" 
                    "3. Suggest shoes: Leather Ankle Boots (too heavy), Red Sneakers (not for wedding). No perfect shoe in candidates, so I will recommend a general type. \n" 
                    "Recommendation: For a summer wedding, I recommend the elegant Floral Maxi Dress. Pair it with the delicate Gold Necklace for a cohesive look. Consider light sandals or elegant flats to complete this beautiful ensemble."
                )
            elif "casual weekend" in prompt.lower() and "outfit" in prompt.lower():
                 return (
                    "Thought Process: The user wants a casual weekend outfit. I need to find comfortable tops and bottoms, and appropriate casual footwear. \n" 
                    "1. Identify casual tops: White Cotton T-Shirt, Striped Button-Up Shirt. Both are good. \n" 
                    "2. Identify casual bottoms: Blue Denim Jeans, High-Waisted Skirt. Both work. \n" 
                    "3. Identify casual shoes: Red Sneakers. Perfect. \n" 
                    "4. Consider outerwear/accessories: Denim Jacket. Good addition. \n" 
                    "Recommendation: For a casual weekend, I suggest the comfortable White Cotton T-Shirt paired with the classic Blue Denim Jeans. Complete the look with the Red Sneakers and consider layering with the Denim Jacket if it gets chilly. This is a versatile and relaxed outfit."
                 )
            else:
                return (
                    "Thought Process: Analyzing the user's request and available items to find the best match. \n" 
                    "Recommendation: Based on your request and available items, I suggest a stylish combination of ... (simulated LLM recommendation for CoT)"
                )
        else:
            # Simulate direct LLM recommendation
            if "summer wedding" in prompt.lower() and "dress" in prompt.lower():
                return "For a summer wedding, I recommend the Floral Maxi Dress paired with the Gold Necklace. It's a charming and appropriate choice."
            elif "casual weekend" in prompt.lower() and "outfit" in prompt.lower():
                return "For a casual weekend outfit, go with the White Cotton T-Shirt and Blue Denim Jeans, topped with the Denim Jacket and Red Sneakers for comfort and style."
            elif "black dress" in prompt.lower() and "accessories" in prompt.lower():
                return "To accessorize your black dress, consider the Silver Hoop Earrings and the Gold Necklace for an elegant touch."
            else:
                return "Based on your request, I recommend a great choice from our collection! (simulated LLM recommendation)"


class FashionStylist:
    def __init__(self, catalog):
        self.candidate_generator = CandidateGenerationModule(catalog)
        self.prompt_engineer = PromptEngineeringModule()
        self.llm_integrator = LLMProviderIntegration()

    def get_recommendation(self, user_query, recommendation_type="zero-shot", filters=None):
        print(f"\nUser Query: {user_query}")
        print(f"Recommendation Type: {recommendation_type}")

        # 1. Candidate Generation
        candidates = self.candidate_generator.generate_candidates(user_query, filters=filters, limit=5)
        if not candidates:
            return "I couldn't find any relevant items for your request. Please try a different query or broader filters."

        print("\n--- Generated Candidates ---")
        for c in candidates:
            print(f"- {c.name} ({c.category}, {c.color}, {c.style})")
        print("----------------------------")

        # 2. Prompt Engineering
        prompt = ""
        use_cot = False
        if recommendation_type == "zero-shot":
            prompt = self.prompt_engineer.generate_zero_shot_prompt(user_query, candidates)
        elif recommendation_type == "few-shot":
            prompt = self.prompt_engineer.generate_few_shot_prompt(user_query, candidates)
        elif recommendation_type == "cot":
            prompt = self.prompt_engineer.generate_cot_prompt(user_query, candidates)
            use_cot = True
        else:
            return "Invalid recommendation type. Choose 'zero-shot', 'few-shot', or 'cot'."

        # 3. LLM Integration (Simulated)
        llm_response = self.llm_integrator.get_llm_response(prompt, use_cot=use_cot)
        
        return llm_response


# Frontend (Simulated - Text-based Interface)
def run_stylist_interface():
    stylist = FashionStylist(mock_catalog)

    print("Welcome to the AI Fashion Stylist!")
    print("Ask me for outfit recommendations or style advice. Type 'exit' to quit.")
    print("You can also specify recommendation type: 'zero-shot', 'few-shot', or 'cot'. (default: zero-shot)")
    print("Example: 'casual outfit for a weekend, type=cot'")

    while True:
        user_input = input("\nYour request: ")
        if user_input.lower() == 'exit':
            break

        query = user_input
        rec_type = "zero-shot"
        filters = None

        # Parse for recommendation type and potential filters
        if ", type=" in user_input:
            parts = user_input.split(", type=")
            query = parts[0].strip()
            rec_type = parts[1].strip().lower()
        
        if "filter_category=" in query:
            query_parts = query.split("filter_category=")
            query = query_parts[0].strip()
            filters = {"category": query_parts[1].strip()}


        recommendation = stylist.get_recommendation(query, recommendation_type=rec_type, filters=filters)
        print("\n--- Recommendation ---")
        print(recommendation)
        print("------------------------")

if __name__ == "__main__":
    run_stylist_interface()
