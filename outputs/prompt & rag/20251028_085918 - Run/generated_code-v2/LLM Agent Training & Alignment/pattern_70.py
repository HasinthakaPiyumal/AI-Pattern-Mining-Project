import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM

class MockEcommerceAPI:
    def __init__(self):
        self.products = {
            "p1": {"name": "Laptop Pro", "category": "Electronics", "price": 1200, "features": "Intel i7, 16GB RAM, 512GB SSD"},
            "p2": {"name": "Smartphone X", "category": "Electronics", "price": 800, "features": "AMOLED, 128GB storage, Dual Camera"},
            "p3": {"name": "Running Shoes Z", "category": "Apparel", "price": 100, "features": "Breathable, Lightweight, Cushioning"},
            "p4": {"name": "Coffee Maker Deluxe", "category": "Home Goods", "price": 150, "features": "Programmable, 12-cup, Stainless Steel"},
            "p5": {"name": "Wireless Earbuds", "category": "Electronics", "price": 150, "features": "Noise Cancelling, 24h battery, Bluetooth 5.0"},
        }

    def search_products(self, query, filters=None):
        results = []
        query_lower = query.lower()
        for pid, product in self.products.items():
            match = False
            if query_lower in product["name"].lower() or query_lower in product["features"].lower():
                match = True
            if filters:
                for key, value in filters.items():
                    if key == "category" and product["category"].lower() != value.lower():
                        match = False
                    if key == "max_price" and product["price"] > value: # Simplified price filter
                        match = False
            if match:
                results.append({"id": pid, **product})
        return results

    def get_product_details(self, product_id):
        return self.products.get(product_id)

    def compare_products(self, product_ids):
        return {pid: self.products.get(pid) for pid in product_ids if pid in self.products}

def collect_demonstration_data():
    return [
        {"state": {"query": "laptop for programming", "observations": "search_results_page"}, "action": "filter_by_ram_16gb"},
        {"state": {"query": "best running shoes", "observations": "product_list"}, "action": "select_product_p3"},
    ]

def collect_comparison_data():
    return [
        {"query": "cheap wireless earbuds", "output_A": "Wireless Earbuds (p5) - great battery", "output_B": "Budget Earphones - decent sound", "preferred_output_index": 0},
        {"query": "laptop for work", "output_A": "Laptop Pro (p1) - powerful", "output_B": "Mid-range Laptop - good value", "preferred_output_index": 0},
    ]

class BehaviorCloningModel(nn.Module):
    def __init__(self, model_name="t5-small"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def generate(self, input_ids, attention_mask, max_length=50):
        return self.model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_length)

class RewardModel(nn.Module):
    def __init__(self, model_name="roberta-base"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def predict_reward(self, query, recommendation):
        inputs = self.tokenizer(f"Query: {query} Recommendation: {recommendation}", return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.logits.squeeze().item()

def train_behavior_cloning_model(model, data):
    print("Simulating Behavior Cloning model training...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    # Simplified training loop
    # In a real scenario, you'd iterate over epochs, create data loaders, and calculate actual loss
    # For demonstration, we'll just simulate the process without full training steps.
    for epoch in range(1):
        for item in data:
            state_str = f"Query: {item['state']['query']} Observations: {item['state']['observations']}"
            action_str = item['action']
            # Encoding and a dummy forward/backward pass for demonstration
            inputs = model.tokenizer(state_str, return_tensors="pt", truncation=True, padding=True)
            labels = model.tokenizer(action_str, return_tensors="pt", truncation=True, padding=True).input_ids
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss # This would be a real loss in a full training setup
            # Dummy optimizer step
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
    print("Behavior Cloning training finished.")

def train_reward_model(model, data):
    print("Simulating Reward Model training...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    # Simplified training loop
    for epoch in range(1):
        for item in data:
            # Simulate reward prediction for demonstration, no actual gradient updates here.
            reward_A = model.predict_reward(item["query"], item["output_A"])
            reward_B = model.predict_reward(item["query"], item["output_B"])
            # In a real training, a ranking loss would be computed and backpropagated.
    print("Reward Model training finished.")

class PersonalShopperAgent:
    def __init__(self):
        self.ecommerce_api = MockEcommerceAPI()
        # Initialize models with pre-trained weights (simulated)
        self.bc_model = BehaviorCloningModel()
        self.rm_model = RewardModel()

    def _understand_request(self, query):
        # Mock LLM for request understanding
        if "laptop" in query.lower():
            return {"category": "Electronics", "keywords": ["laptop", "programming"], "intent": "find_product"}
        if "shoes" in query.lower():
            return {"category": "Apparel", "keywords": ["running", "shoes"], "intent": "find_product"}
        if "earbuds" in query.lower() and "cheap" in query.lower():
            return {"category": "Electronics", "keywords": ["earbuds", "wireless", "cheap"], "intent": "find_product", "filters": {"max_price": 200}}
        return {"category": None, "keywords": [query], "intent": "search"}

    def _explore_products_bc(self, parsed_request):
        products_found = []
        if parsed_request["intent"] == "find_product":
            filters = parsed_request.get("filters", {})
            if parsed_request["category"]:
                filters["category"] = parsed_request["category"]
            
            search_query = " ".join(parsed_request["keywords"])
            products_found = self.ecommerce_api.search_products(search_query, filters)
        return products_found

    def _generate_recommendations(self, products, query):
        recommendations = []
        if not products:
            return ["No suitable products found for recommendations."]
        for p in products[:3]: 
            recommendations.append(f"I recommend {p['name']} ({p['category']}). It costs ${p['price']} and features: {p['features']}. This is a solid choice for your needs.")
            recommendations.append(f"Consider the {p['name']} product. It's a {p['category']} item priced at ${p['price']} with {p['features']}.")
        return recommendations

    def _rank_recommendations_rm(self, query, generated_recommendations):
        if not generated_recommendations or generated_recommendations == ["No suitable products found for recommendations."]:
            return "I could not generate specific recommendations based on the available products."

        ranked_recommendations = []
        for rec in generated_recommendations:
            reward = self.rm_model.predict_reward(query, rec)
            ranked_recommendations.append((rec, reward))

        ranked_recommendations.sort(key=lambda x: x[1], reverse=True)
        return ranked_recommendations[0][0]

    def get_recommendation(self, user_query):
        print(f"User Query: {user_query}")
        parsed_request = self._understand_request(user_query)
        print(f"Parsed Request: {parsed_request}")

        explored_products = self._explore_products_bc(parsed_request)
        print(f"Explored Products: {[p['name'] for p in explored_products] if explored_products else 'None'}")

        candidate_recommendations = self._generate_recommendations(explored_products, user_query)
        print(f"Candidate Recommendations: {candidate_recommendations}")

        final_recommendation = self._rank_recommendations_rm(user_query, candidate_recommendations)
        return final_recommendation

if __name__ == "__main__":
    demonstration_data = collect_demonstration_data()
    comparison_data = collect_comparison_data()

    print("--- Training Models ---")
    # Instantiate agent to get model instances for training simulation
    agent_for_training = PersonalShopperAgent()
    train_behavior_cloning_model(agent_for_training.bc_model, demonstration_data)
    train_reward_model(agent_for_training.rm_model, comparison_data)
    print("--- Training Complete ---")

    print("\n--- Agent in Action ---")
    # Create a new agent instance for inference
    agent_instance = PersonalShopperAgent()
    print(agent_instance.get_recommendation("I need a powerful laptop for coding."))
    print("-" * 30)
    print(agent_instance.get_recommendation("What's a good pair of comfortable running shoes?"))
    print("-" * 30)
    print(agent_instance.get_recommendation("I'm looking for cheap wireless earbuds, preferably under $200."))
    print("-" * 30)
    print(agent_instance.get_recommendation("Show me something for cooking."))