
class DataLoader:
    def __init__(self):
        self.fashion_items = [
            {
                "id": "F001",
                "name": "Elegant Silk Maxi Dress",
                "category": "Dress",
                "style": "Elegant",
                "occasion": "Evening, Wedding",
                "brand": "Luxe Apparel",
                "sustainability": "Eco-friendly fabric",
                "price": 180.00,
                "description": "A stunning silk maxi dress perfect for formal occasions."
            },
            {
                "id": "F002",
                "name": "Casual Denim Jacket",
                "category": "Jacket",
                "style": "Casual",
                "occasion": "Daily, Weekend",
                "brand": "Urban Threads",
                "sustainability": "Recycled denim",
                "price": 75.00,
                "description": "A versatile denim jacket for everyday wear."
            },
            {
                "id": "F003",
                "name": "Sporty Running Shoes",
                "category": "Footwear",
                "style": "Sporty",
                "occasion": "Exercise, Casual",
                "brand": "Stride",
                "sustainability": "Vegan materials",
                "price": 120.00,
                "description": "Lightweight and comfortable running shoes."
            },
            {
                "id": "F004",
                "name": "Bohemian Floral Blouse",
                "category": "Top",
                "style": "Bohemian",
                "occasion": "Casual, Summer",
                "brand": "Free Spirit",
                "sustainability": "Organic cotton",
                "price": 55.00,
                "description": "A flowy floral blouse with a bohemian touch."
            },
            {
                "id": "F005",
                "name": "Minimalist Leather Tote Bag",
                "category": "Bag",
                "style": "Minimalist",
                "occasion": "Work, Daily",
                "brand": "Sleek Designs",
                "sustainability": "Ethically sourced leather",
                "price": 250.00,
                "description": "A spacious and elegant leather tote for daily essentials."
            },
            {
                "id": "F006",
                "name": "Wool Blend Peacoat",
                "category": "Coat",
                "style": "Classic",
                "occasion": "Winter, Formal",
                "brand": "Heritage Wear",
                "sustainability": "Recycled wool",
                "price": 320.00,
                "description": "A warm and stylish peacoat for cold weather."
            },
            {
                "id": "F007",
                "name": "Chic Jumpsuit for Events",
                "category": "Jumpsuit",
                "style": "Chic",
                "occasion": "Party, Cocktail",
                "brand": "Gala Glam",
                "sustainability": "Sustainable dyes",
                "price": 160.00,
                "description": "A fashionable jumpsuit perfect for evening events."
            },
            {
                "id": "F008",
                "name": "Comfortable Lounge Pants",
                "category": "Pants",
                "style": "Casual, Lounge",
                "occasion": "Home, Relax",
                "brand": "Cozy Comfort",
                "sustainability": "Organic cotton",
                "price": 40.00,
                "description": "Soft and comfortable pants for relaxing at home."
            }
        ]

        self.user_profiles = {
            "user_alice": {
                "preferences": {"style": ["Elegant", "Minimalist"], "occasion": ["Wedding", "Work"]},
                "recent_activity": ["F001", "F005"],
                "budget": "high"
            },
            "user_bob": {
                "preferences": {"style": ["Casual", "Sporty"], "occasion": ["Daily", "Exercise"]},
                "recent_activity": ["F002", "F003"],
                "budget": "medium"
            },
            "user_charlie": {
                "preferences": {"style": ["Bohemian", "Chic"], "occasion": ["Summer", "Party"]},
                "recent_activity": ["F004", "F007"],
                "budget": "medium"
            }
        }

    def get_fashion_items(self):
        return self.fashion_items.copy()

    def get_user_profile(self, user_id):
        return self.user_profiles.get(user_id)
