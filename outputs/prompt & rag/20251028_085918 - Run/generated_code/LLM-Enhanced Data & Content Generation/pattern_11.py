class UserProfileManager:
    def __init__(self):
        self.users = {}

    def add_user(self, user_id, preferences=None):
        if user_id not in self.users:
            self.users[user_id] = {
                "preferences": preferences if preferences else [],
                "browsing_history": [],
                "purchase_history": [],
                "explicit_feedback": [], # e.g., {'content_id': 'prod_desc_123', 'feedback': 'liked'}
                "engagement_score": 0
            }
            print(f"User {user_id} added.")
        else:
            print(f"User {user_id} already exists.")

    def get_user_profile(self, user_id):
        return self.users.get(user_id)

    def update_browsing_history(self, user_id, product_id):
        if user_id in self.users:
            self.users[user_id]["browsing_history"].append(product_id)
            self.users[user_id]["engagement_score"] += 1 # Simple engagement metric
            print(f"User {user_id} browsed product {product_id}.")

    def update_purchase_history(self, user_id, product_id):
        if user_id in self.users:
            self.users[user_id]["purchase_history"].append(product_id)
            self.users[user_id]["engagement_score"] += 10 # Higher engagement for purchase
            print(f"User {user_id} purchased product {product_id}.")

    def add_explicit_feedback(self, user_id, content_id, feedback):
        if user_id in self.users:
            self.users[user_id]["explicit_feedback"].append({'content_id': content_id, 'feedback': feedback})
            if feedback == 'liked':
                self.users[user_id]["engagement_score"] += 5
            elif feedback == 'disliked':
                self.users[user_id]["engagement_score"] -= 2
            print(f"User {user_id} gave feedback '{feedback}' on content {content_id}.")

    def update_preferences(self, user_id, new_preferences):
        if user_id in self.users:
            for pref in new_preferences:
                if pref not in self.users[user_id]["preferences"]:
                    self.users[user_id]["preferences"].append(pref)
            print(f"User {user_id} preferences updated.")