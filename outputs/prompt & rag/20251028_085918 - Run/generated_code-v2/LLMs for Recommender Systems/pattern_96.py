import json

class UserMemory:
    def __init__(self, user_id):
        self.user_id = user_id
        self.preferences = {
            "style": [],
            "colors": [],
            "items": [],
            "budget": "",
            "occasions": []
        }
        self.chat_history = []

    def update_preferences(self, new_prefs):
        for key, value in new_prefs.items():
            if key in self.preferences:
                if isinstance(self.preferences[key], list) and isinstance(value, list):
                    self.preferences[key].extend([v for v in value if v not in self.preferences[key]])
                else:
                    self.preferences[key] = value
        self.clean_preferences()

    def get_preferences(self):
        return self.preferences
    
    def add_to_history(self, role, message):
        self.chat_history.append({"role": role, "message": message})

    def get_history(self, n=5):
        return self.chat_history[-n:]

    def extract_and_store_facts(self, conversation_segment):
        # A very basic simulation of extracting facts from a conversation segment
        # In a real scenario, an LLM would do this more intelligently
        if "my style is" in conversation_segment.lower():
            style_match = conversation_segment.lower().split("my style is")[-1].split(".")[0].strip().split(",")
            self.update_preferences({"style": [s.strip() for s in style_match if s.strip()]})
        if "i like the color" in conversation_segment.lower():
            color_match = conversation_segment.lower().split("i like the color")[-1].split(".")[0].strip().split(",")
            self.update_preferences({"colors": [c.strip() for c in color_match if c.strip()]})
        if "looking for a" in conversation_segment.lower():
            item_match = conversation_segment.lower().split("looking for a")[-1].split(".")[0].strip().split(",")
            self.update_preferences({"items": [i.strip() for i in item_match if i.strip()]})
        if "my budget is around" in conversation_segment.lower() or "i want to spend around" in conversation_segment.lower():
            budget_match = next((s for s in conversation_segment.lower().split() if s.replace('.', '', 1).isdigit()), None)
            if budget_match: self.update_preferences({"budget": budget_match})
        if "for a" in conversation_segment.lower() and "occasion" in conversation_segment.lower():
             occasion_match = conversation_segment.lower().split("for a")[-1].split("occasion")[0].strip().split(",")
             self.update_preferences({"occasions": [o.strip() for o in occasion_match if o.strip()]})

    def clean_preferences(self):
        for key, value in self.preferences.items():
            if isinstance(value, list):
                self.preferences[key] = list(set(value))

# Mock external tools
def search_products_tool(query, filters=None):
    print(f"[TOOL CALL] Searching products for: '{query}' with filters: {filters}")
    # Simulate a database lookup
    products = {
        "dress": [{"name": "Floral Maxi Dress", "price": "$65", "style": "boho"}, {"name": "Little Black Dress", "price": "$90", "style": "classic"}],
        "shirt": [{"name": "Linen Button-Up Shirt", "price": "$40", "style": "casual"}, {"name": "Silk Blouse", "price": "$75", "style": "elegant"}],
        "jeans": [{"name": "Slim Fit Jeans", "price": "$55", "style": "casual"}, {"name": "High-Waisted Skinny Jeans", "price": "$60", "style": "trendy"}],
        "shoes": [{"name": "White Sneakers", "price": "$80", "style": "sporty"}, {"name": "Leather Loafers", "price": "$110", "style": "smart casual"}],
        "accessories": [{"name": "Gold Hoop Earrings", "price": "$25"}, {"name": "Leather Tote Bag", "price": "$150"}]
    }
    
    results = []
    query_lower = query.lower()
    for item_type, item_list in products.items():
        if query_lower in item_type or item_type in query_lower:
            results.extend(item_list)
    
    # Apply basic filtering
    if filters:
        filtered_results = []
        for product in results:
            match = True
            for key, value in filters.items():
                if key in product and value.lower() not in product[key].lower():
                    match = False
                    break
            if match:
                filtered_results.append(product)
        results = filtered_results

    return results if results else [{"name": "No matching products found"}]

def get_recommendations_tool(preferences):
    print(f"[TOOL CALL] Getting recommendations based on preferences: {preferences}")
    recommendations = []

    style = preferences.get("style", [])
    colors = preferences.get("colors", [])
    items = preferences.get("items", [])
    budget = preferences.get("budget", "")
    occasions = preferences.get("occasions", [])

    if "boho" in style or "casual" in style:
        recommendations.append("How about a flowy maxi dress in a floral print?")
    if "elegant" in style or "classic" in style:
        recommendations.append("A sophisticated silk blouse with tailored trousers would be perfect.")
    if "sporty" in style:
        recommendations.append("Consider some stylish sneakers and comfortable activewear.")
    
    if "red" in colors or "blue" in colors:
        recommendations.append("We have some stunning items in your preferred colors.")

    if "dress" in items and "party" in occasions:
        recommendations.append("For a party, a chic cocktail dress would be an excellent choice.")
    elif "dress" in items:
         recommendations.append("We have a wide range of dresses to suit any occasion.")
    
    if budget:
        recommendations.append(f"Keeping your budget around ${budget} in mind.")

    return recommendations if recommendations else ["I need more information to give tailored recommendations. What kind of style are you looking for?"]

class FashionStylistAgent:
    def __init__(self, user_id):
        self.memory = UserMemory(user_id)

    def _simulate_llm_intent_and_response(self, user_input):
        user_input_lower = user_input.lower()
        
        # First, try to extract and store facts for memory management
        self.memory.extract_and_store_facts(user_input)

        # Tool learning simulation
        if "look for" in user_input_lower or "find me" in user_input_lower or "search for" in user_input_lower:
            item_query = " ".join([word for word in user_input_lower.split() if word not in ["look", "for", "find", "me", "search"]])
            current_prefs = self.memory.get_preferences()
            filters = {}
            if current_prefs["style"]: filters["style"] = current_prefs["style"][0] # Take first style for simplicity
            if current_prefs["colors"]: filters["color"] = current_prefs["colors"][0] # Take first color
            
            products = search_products_tool(item_query, filters)
            if products and products[0].get("name") != "No matching products found":
                product_names = ", ".join([p["name"] for p in products])
                return f"I found a few options for you: {product_names}. Do any of these pique your interest?"
            else:
                return "I couldn't find any products matching your specific request. Can you try being more general?"
        
        elif "recommend" in user_input_lower or "suggestions" in user_input_lower or "what should i wear" in user_input_lower:
            current_prefs = self.memory.get_preferences()
            recommendations = get_recommendations_tool(current_prefs)
            return "Based on what I know about your preferences: " + " ".join(recommendations)

        # General conversational responses (LLM NLU/NLG simulation)
        elif "hello" in user_input_lower or "hi" in user_input_lower:
            return "Hello there! I'm your personal fashion stylist. How can I help you today?"
        elif "my style is" in user_input_lower or "i like" in user_input_lower or "i am looking for" in user_input_lower:
            current_prefs = self.memory.get_preferences()
            pref_summary = []
            if current_prefs["style"]: pref_summary.append(f"style: {', '.join(current_prefs['style'])}")
            if current_prefs["colors"]: pref_summary.append(f"colors: {', '.join(current_prefs['colors'])}")
            if current_prefs["items"]: pref_summary.append(f"items: {', '.join(current_prefs['items'])}")
            if current_prefs["budget"]: pref_summary.append(f"budget: ${current_prefs['budget']}")
            if current_prefs["occasions"]: pref_summary.append(f"occasions: {', '.join(current_prefs['occasions'])}")
            
            if pref_summary:
                return f"Got it! I've noted your preferences. Currently, I have: {'; '.join(pref_summary)}. What are you looking for today?"
            else:
                return "Thanks for sharing your preferences! Is there anything specific you'd like to find or discuss?"
        elif "thank you" in user_input_lower or "thanks" in user_input_lower:
            return "You're most welcome! Is there anything else I can assist you with?"
        elif "what do you know about me" in user_input_lower or "my preferences" in user_input_lower:
            current_prefs = self.memory.get_preferences()
            summary_parts = []
            for key, value in current_prefs.items():
                if isinstance(value, list) and value:
                    summary_parts.append(f"Your preferred {key} are: {', '.join(value)}.")
                elif isinstance(value, str) and value:
                    summary_parts.append(f"Your budget is around ${value}.")
            if summary_parts:
                return " ".join(summary_parts) + " How does that sound?"
            else:
                return "I'm still learning about your preferences. Tell me more about your style!"
        elif "bye" in user_input_lower or "goodbye" in user_input_lower:
            return "Goodbye! Come back anytime for your fashion needs!"
        else:
            return "I'm not quite sure how to help with that. Could you rephrase or tell me more about what you're looking for fashion-wise?"

    def converse(self, user_input):
        self.memory.add_to_history("user", user_input)
        response = self._simulate_llm_intent_and_response(user_input)
        self.memory.add_to_history("agent", response)
        return response

if __name__ == "__main__":
    print("Welcome to your E-commerce Fashion Stylist! Type 'exit' to end the conversation.")
    stylist_agent = FashionStylistAgent(user_id="test_user_123")

    while True:
        user_message = input("You: ")
        if user_message.lower() == 'exit':
            print("Agent: Goodbye!")
            break
        
        agent_response = stylist_agent.converse(user_message)
        print(f"Agent: {agent_response}")

        # Optional: print current preferences for debugging
        # print(f"[DEBUG] Current Preferences: {stylist_agent.memory.get_preferences()}\n")