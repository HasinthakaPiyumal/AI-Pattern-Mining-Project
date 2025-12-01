import json
import random
import gradio as gr

class UserProfile:
    def __init__(self):
        self.preferences = {"categories": [], "styles": [], "price_range": "any"}
        self.history = []

    def update_preferences(self, new_prefs):
        for key, value in new_prefs.items():
            if key in self.preferences and isinstance(self.preferences[key], list):
                if isinstance(value, list):
                    self.preferences[key].extend([v for v in value if v not in self.preferences[key]])
                elif value not in self.preferences[key]:
                    self.preferences[key].append(value)
            else:
                self.preferences[key] = value

    def add_history(self, interaction):
        self.history.append(interaction)

    def get_profile(self):
        return self.preferences


class ECommerceActionSimulators:
    def __init__(self):
        self.products = self._generate_sample_products()

    def _generate_sample_products(self):
        categories = ["electronics", "apparel", "home_decor", "books", "sports"]
        styles = ["casual", "formal", "trendy", "classic", "minimalist"]
        brands = ["BrandA", "BrandB", "BrandC", "BrandD"]
        products = []
        for i in range(1, 101):
            product = {
                "id": f"prod{i:03d}",
                "name": f"Product {i}",
                "category": random.choice(categories),
                "style": random.choice(styles),
                "price": round(random.uniform(10.0, 500.0), 2),
                "brand": random.choice(brands),
                "description": f"A high-quality {random.choice(styles)} item from {random.choice(brands)}."
            }
            products.append(product)
        return products

    def search_products(self, query, filters=None):
        results = []
        query = query.lower()
        if filters is None: filters = {}
        
        for product in self.products:
            match = True
            if query and query not in product["name"].lower() and query not in product["description"].lower() and query not in product["category"].lower() and query not in product["style"].lower():
                match = False
            
            for key, value in filters.items():
                if key == "price_range":
                    if isinstance(value, tuple) and (product["price"] < value[0] or product["price"] > value[1]):
                        match = False
                        break
                elif isinstance(product.get(key), str) and value.lower() != product[key].lower():
                    match = False
                    break
                elif isinstance(product.get(key), list) and all(v.lower() not in [item.lower() for item in product[key]] for v in (value if isinstance(value, list) else [value])):
                    match = False
                    break
            if match:
                results.append(product)
        return results[:5]  # Limit to top 5 results

    def get_product_details(self, product_id):
        for product in self.products:
            if product["id"] == product_id:
                return product
        return None

    def recommend_products(self, user_profile, category=None):
        preferred_categories = user_profile.get("categories", [])
        preferred_styles = user_profile.get("styles", [])
        price_range = user_profile.get("price_range", "any")

        filters = {}
        if preferred_categories: filters["category"] = preferred_categories[0] # Simplistic: take first preferred category
        if preferred_styles: filters["style"] = preferred_styles[0] # Simplistic: take first preferred style
        
        if price_range != "any":
            if isinstance(price_range, str) and "-" in price_range:
                try:
                    min_price, max_price = map(float, price_range.split("-"))
                    filters["price_range"] = (min_price, max_price)
                except ValueError:
                    pass
            

        eligible_products = []
        for product in self.products:
            if (not category or product["category"].lower() == category.lower()) and \
               (not preferred_categories or product["category"] in preferred_categories) and \
               (not preferred_styles or product["style"] in preferred_styles):
                
                if "price_range" in filters:
                    min_p, max_p = filters["price_range"]
                    if not (min_p <= product["price"] <= max_p):
                        continue
                eligible_products.append(product)

        random.shuffle(eligible_products)
        return eligible_products[:5]  # Limit to top 5 recommendations


class NLUandResponseGenerator:
    def __init__(self):
        self.keywords = {
            "search": ["look for", "find", "search", "show me", "browse"],
            "recommend": ["recommend", "suggest", "what should i buy", "give me ideas"],
            "details": ["details", "info", "tell me about"],
            "clarify": ["what do you mean", "can you explain", "clarify"],
            "preferences": ["my preferences", "my style", "i like"],
            "hello": ["hi", "hello", "hey"]
        }

    def recognize_intent(self, text):
        text_lower = text.lower()
        if "for a party" in text_lower or "gift for" in text_lower:
            return "recommend_personalized"
        for intent, kws in self.keywords.items():
            for kw in kws:
                if kw in text_lower:
                    return intent
        return "unknown"

    def extract_entities(self, text):
        entities = {}
        text_lower = text.lower()

        # Simple category extraction
        for category in ["electronics", "apparel", "home decor", "books", "sports"]:
            if category in text_lower:
                entities["category"] = category
                break
        
        # Simple style extraction
        for style in ["casual", "formal", "trendy", "classic", "minimalist"]:
            if style in text_lower:
                entities["style"] = style
                break

        # Price range extraction
        import re
        price_match = re.search(r"between (\$?\\d+\.?\d*) and (\$?\\d+\.?\d*)", text_lower)
        if price_match:
            min_price = float(price_match.group(1).replace("$", ""))
            max_price = float(price_match.group(2).replace("$", ""))
            entities["price_range"] = (min_price, max_price)
        elif re.search(r"under (\$?\\d+\.?\d*)", text_lower):
            max_price = float(re.search(r"under (\$?\\d+\.?\d*)", text_lower).group(1).replace("$", ""))
            entities["price_range"] = (0, max_price)
        elif re.search(r"over (\$?\\d+\.?\d*)", text_lower):
            min_price = float(re.search(r"over (\$?\\d+\.?\d*)", text_lower).group(1).replace("$", ""))
            entities["price_range"] = (min_price, float('inf'))

        # Product ID extraction (simplified)
        product_id_match = re.search(r"prod\\d{3}", text_lower)
        if product_id_match:
            entities["product_id"] = product_id_match.group(0)

        return entities

    def generate_response(self, intent, data=None, user_profile=None):
        if intent == "hello":
            return "Hello! How can I assist you with your shopping today?"
        elif intent == "search":
            if data and data.get("products"):
                products_str = "\n".join([f" - {p['name']} ({p['category']}) - ${p['price']:.2f} (ID: {p['id']})" for p in data["products"]])
                return f"Here are some products I found:\n{products_str}\nIs there anything specific you'd like to know about them?"
            else:
                return "I couldn't find any products matching your criteria. Can you try being more specific or adjusting your filters?"
        elif intent == "recommend":
            if data and data.get("products"):
                products_str = "\n".join([f" - {p['name']} ({p['category']}) - ${p['price']:.2f} (ID: {p['id']})" for p in data["products"]])
                return f"Based on your query, here are some recommendations:\n{products_str}\nI hope you find something you like!"
            else:
                return "I need a bit more information to give you good recommendations. What kind of products are you looking for, or for what occasion?"
        elif intent == "recommend_personalized":
            if data and data.get("products"):
                products_str = "\n".join([f" - {p['name']} ({p['category']}) - ${p['price']:.2f} (ID: {p['id']})" for p in data["products"]])
                return f"Considering your preferences and the context, here are some personalized recommendations for you:\n{products_str}\nLet me know if these catch your eye!"
            else:
                return "I'm having trouble providing personalized recommendations right now. Can you tell me more about what you're looking for, or update your preferences?"
        elif intent == "details":
            if data and data.get("product_details"):
                product = data["product_details"]
                return f"Details for {product['name']} (ID: {product['id']}):\nCategory: {product['category']}\nStyle: {product['style']}\nBrand: {product['brand']}\nPrice: ${product['price']:.2f}\nDescription: {product['description']}"
            else:
                return "I couldn't find details for that product. Please provide a valid product ID."
        elif intent == "clarify":
            return data.get("question", "Could you please rephrase that? I'm not sure I understood.")
        elif intent == "preferences":
            if data and data.get("message"):
                return data["message"]
            return f"Your current preferences are: {user_profile.get_profile()}. How would you like to update them?"
        else:
            return "I'm sorry, I didn't understand that. Could you please try again?"


class DialogueManager:
    def __init__(self, nlu_generator, simulators, user_profile):
        self.nlu_generator = nlu_generator
        self.simulators = simulators
        self.user_profile = user_profile
        self.current_context = {}

    def handle_query(self, query):
        intent = self.nlu_generator.recognize_intent(query)
        entities = self.nlu_generator.extract_entities(query)
        response_data = {}
        clarifying_question = None

        # Update user profile based on entities if applicable
        if entities.get("category") or entities.get("style") or entities.get("price_range"):
            self.user_profile.update_preferences(entities)
            response_data["message"] = "I've updated your preferences."

        if intent == "search":
            search_query = query
            filters = {}
            if entities.get("category"): filters["category"] = entities["category"]
            if entities.get("style"): filters["style"] = entities["style"]
            if entities.get("price_range"): filters["price_range"] = entities["price_range"]
            
            products = self.simulators.search_products(search_query, filters)
            response_data["products"] = products
        elif intent == "recommend" or intent == "recommend_personalized":
            category_for_rec = entities.get("category")
            products = self.simulators.recommend_products(self.user_profile.get_profile(), category=category_for_rec)
            response_data["products"] = products
        elif intent == "details":
            product_id = entities.get("product_id")
            if product_id:
                product_details = self.simulators.get_product_details(product_id)
                response_data["product_details"] = product_details
            else:
                clarifying_question = "Which product's details are you looking for? Please provide a product ID (e.g., 'prod001')."
                intent = "clarify" # Override intent for clarification
        elif intent == "preferences":
            # If there are entities related to preferences, the user profile is already updated.
            # We can respond with current preferences or a confirmation.
            pass # Handled by the generic update and response generation
        elif intent == "unknown":
            clarifying_question = "I'm not sure how to help with that. Are you looking to search for products, get recommendations, or something else?"
            intent = "clarify"
        
        if clarifying_question:
            response_data["question"] = clarifying_question

        return self.nlu_generator.generate_response(intent, response_data, self.user_profile)


# Main Application
user_profile = UserProfile()
simulators = ECommerceActionSimulators()
nlu_generator = NLUandResponseGenerator()
dialogue_manager = DialogueManager(nlu_generator, simulators, user_profile)

def chat_interface(user_input):
    response = dialogue_manager.handle_query(user_input)
    return response

if __name__ == "__main__":
    print("Welcome to your E-commerce Personal Shopping Assistant! Type 'exit' to quit.")
    
    # Console Interface
    while True:
        user_query = input("You: ")
        if user_query.lower() == 'exit':
            print("Goodbye!")
            break
        response = dialogue_manager.handle_query(user_query)
        print(f"Assistant: {response}")

    # Gradio Interface (Optional)
    # if you want to run the gradio app, uncomment the lines below and run the script
    # interface = gr.Interface(fn=chat_interface, inputs="text", outputs="text", title="E-commerce Personal Shopping Assistant")
    # interface.launch()