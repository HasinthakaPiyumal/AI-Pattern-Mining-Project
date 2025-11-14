
class UserProfileManager:
    def __init__(self):
        self.user_profiles = {}

    def get_user_profile(self, user_id):
        return self.user_profiles.get(user_id, {
            "browsing_history": [],
            "past_purchases": [],
            "explicit_preferences": [],
            "feedback_history": [],
            "llm_preferences": {}
        })

    def update_browsing_history(self, user_id, item_id):
        profile = self.get_user_profile(user_id)
        profile["browsing_history"].append(item_id)
        self.user_profiles[user_id] = profile

    def add_purchase(self, user_id, item_id, price):
        profile = self.get_user_profile(user_id)
        profile["past_purchases"].append({"item_id": item_id, "price": price})
        self.user_profiles[user_id] = profile

    def add_explicit_preference(self, user_id, preference):
        profile = self.get_user_profile(user_id)
        if preference not in profile["explicit_preferences"]:
            profile["explicit_preferences"].append(preference)
        self.user_profiles[user_id] = profile

    def add_feedback(self, user_id, feedback_data):
        profile = self.get_user_profile(user_id)
        profile["feedback_history"].append(feedback_data)
        self.user_profiles[user_id] = profile

    def update_llm_preferences(self, user_id, key, value):
        profile = self.get_user_profile(user_id)
        profile["llm_preferences"][key] = value
        self.user_profiles[user_id] = profile

