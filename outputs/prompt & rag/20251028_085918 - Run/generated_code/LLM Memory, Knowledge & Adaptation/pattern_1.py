
import collections
import random
import time

class QueryClassifier:
    """Simulates a query classification module."""
    def __init__(self):
        self.categories = {
            "billing": ["bill", "invoice", "charge", "payment"],
            "technical support": ["internet", "no signal", "slow", "fix", "router", "tech"],
            "service upgrade": ["upgrade", "faster", "add channel", "new plan"],
            "general inquiry": ["hello", "hi", "question", "information"]
        }

    def classify(self, query):
        query_lower = query.lower()
        for category, keywords in self.categories.items():
            if any(keyword in query_lower for keyword in keywords):
                return category
        return "general inquiry"

class LLMService:
    """Simulates an LLM for generating responses."""
    def __init__(self):
        pass

    def generate_response(self, classified_query, short_term_memory, long_term_context):
        print(f"[DEBUG] LLM Input - Classified: {classified_query}, Short-term: {short_term_memory}, Long-term: {long_term_context}")

        response_templates = {
            "billing": [
                "Regarding your billing inquiry, could you please provide your account number?",
                "I can help you with your bill. What specific charges are you looking into?",
                "For billing assistance, please confirm your identity."
            ],
            "technical support": [
                "I understand you're having technical issues. Can you describe the problem in more detail?",
                "Let's troubleshoot your technical problem. Have you tried restarting your device?",
                "Our technical team can assist you. What kind of technical problem are you experiencing?"
            ],
            "service upgrade": [
                "Great! I can help you explore service upgrade options. What kind of service are you interested in?",
                "We have several upgrade plans available. Would you like to hear about our latest offers?",
                "To upgrade your service, please tell me what you're looking for."
            ],
            "general inquiry": [
                "How can I assist you further today?",
                "Please tell me more about what you need.",
                "I'm here to help. What's on your mind?"
            ]
        }

        base_response = random.choice(response_templates.get(classified_query, response_templates["general inquiry"]))

        if long_term_context:
            return f"{base_response} Also, I found this relevant information: {long_term_context}"
        return base_response

class KnowledgeBase:
    """Simulates a long-term non-parametric memory (vector database)."""
    def __init__(self, initial_knowledge=None):
        self.knowledge = initial_knowledge if initial_knowledge else self._load_default_knowledge()

    def _load_default_knowledge(self):
        return {
            "internet speed": "Our standard internet plan offers speeds up to 100 Mbps. Premium plans go up to 1 Gbps.",
            "router setup": "For router setup, please refer to the manual or visit our website for a step-by-step guide.",
            "latest promotion": "New customers can get a 3-month discount on our fiber optic plans.",
            "billing cycles": "Billing cycles are typically on the 1st of each month, with statements issued 7 days prior."
        }

    def search(self, query, top_k=1):
        # In a real system, this would be a semantic search against a vector database.
        # For this simulation, we'll do a simple keyword match.
        query_lower = query.lower()
        relevant_info = []
        for key, value in self.knowledge.items():
            if any(word in query_lower for word in key.split()):
                relevant_info.append(value)
        return relevant_info[:top_k]

    def update_knowledge(self, new_knowledge):
        """Simulates index hotswapping by replacing the entire knowledge base."""
        print("[INFO] Updating knowledge base...")
        self.knowledge = new_knowledge
        print("[INFO] Knowledge base updated successfully.")

class DataGenerator:
    """Generates synthetic training data for classifiers."""
    def __init__(self):
        self.templates = {
            "billing": [
                "I need to pay my {bill_type}.",
                "What's the status of my {invoice_type}?",
                "I have a question about a {charge_type} on my statement."
            ],
            "technical support": [
                "My {device_type} isn't working.",
                "I have no {service_type} connection.",
                "My {problem_type} is very slow."
            ],
            "service upgrade": [
                "I want to {action_type} my {service_category}.",
                "Can I get a {speed_increase} internet plan?",
                "Tell me about {new_feature} options."
            ]
        }
        self.keywords = {
            "bill_type": ["phone bill", "internet bill", "utility bill"],
            "invoice_type": ["latest invoice", "current invoice", "previous invoice"],
            "charge_type": ["recent charge", "unfamiliar charge", "monthly charge"],
            "device_type": ["router", "modem", "phone", "TV box"],
            "service_type": ["internet", "TV", "phone"],
            "problem_type": ["internet", "Wi-Fi"],
            "action_type": ["upgrade", "improve", "enhance"],
            "service_category": ["internet service", "TV package", "mobile plan"],
            "speed_increase": ["faster", "higher speed"],
            "new_feature": ["new channel", "streaming bundle", "data add-on"]
        }

    def generate_training_data(self, num_samples_per_category=5):
        generated_data = []
        for category, templates in self.templates.items():
            for _ in range(num_samples_per_category):
                template = random.choice(templates)
                filled_template = template
                for keyword_type, keyword_list in self.keywords.items():
                    if "{" + keyword_type + "}" in filled_template:
                        filled_template = filled_template.replace("{"+keyword_type+"}", random.choice(keyword_list))
                generated_data.append((filled_template, category))
        return generated_data

class LLMFineTuner:
    """Placeholder for LLM fine-tuning logic (e.g., LoRA/QLoRA)."""
    def __init__(self, llm_model="mock_llm"):
        self.llm_model = llm_model

    def fine_tune(self, training_data):
        print(f"[INFO] Simulating fine-tuning for {self.llm_model} with {len(training_data)} samples.")
        # In a real scenario, this would involve loading a model, preparing data,
        # and running a fine-tuning script with libraries like trl and peft.
        time.sleep(2) # Simulate training time
        print(f"[INFO] {self.llm_model} fine-tuning complete. Model weights updated.")

class CustomerSupportSystem:
    """Main orchestrator for the intelligent customer support system."""
    def __init__(self):
        self.query_classifier = QueryClassifier()
        self.llm_service = LLMService()
        self.short_term_memory = collections.deque(maxlen=5) # Stores last 5 turns
        self.knowledge_base = KnowledgeBase()
        self.data_generator = DataGenerator()
        self.llm_fine_tuner = LLMFineTuner()

    def process_query(self, user_query):
        # 1. Store query in short-term memory
        self.short_term_memory.append(user_query)

        # 2. Classify the query
        classified_category = self.query_classifier.classify(user_query)
        print(f"[SYSTEM] Query classified as: {classified_category}")

        # 3. Retrieve relevant long-term context
        long_term_context = self.knowledge_base.search(user_query)

        # 4. Generate LLM response
        llm_response = self.llm_service.generate_response(
            classified_category,
            list(self.short_term_memory), # Convert deque to list for LLM input
            long_term_context
        )

        # 5. Store LLM response in short-term memory (optional, but good for context)
        self.short_term_memory.append(llm_response)

        return llm_response

    def simulate_dynamic_updates(self):
        print("\n[SYSTEM] Initiating dynamic knowledge update simulation...")
        new_knowledge = {
            "fiber optic plans": "Our new fiber optic plans offer unmatched speeds and reliability, starting at $50/month.",
            "5G rollout": "5G is now available in major cities. Check our coverage map for details.",
            "billing cycle change": "Effective next quarter, all billing cycles will shift to the 15th of each month."
        }
        self.knowledge_base.update_knowledge(new_knowledge)
        print("[SYSTEM] Dynamic knowledge update simulated.")

    def simulate_fine_tuning(self):
        print("\n[SYSTEM] Initiating LLM fine-tuning simulation...")
        # Generate some synthetic data for fine-tuning the query classifier (or an actual LLM)
        synthetic_data = self.data_generator.generate_training_data(num_samples_per_category=3)
        print(f"[SYSTEM] Generated {len(synthetic_data)} synthetic training samples for fine-tuning.")
        self.llm_fine_tuner.fine_tune(synthetic_data)
        print("[SYSTEM] LLM fine-tuning simulation complete.")

# Main interaction loop
if __name__ == "__main__":
    system = CustomerSupportSystem()
    print("Welcome to the TeleCo Customer Support Chatbot! Type 'quit' to exit.")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'quit':
            break

        response = system.process_query(user_input)
        print(f"Bot: {response}")

        # Simulate periodic updates or fine-tuning
        if random.random() < 0.2: # 20% chance to trigger an update/fine-tune
            if random.random() < 0.5:
                system.simulate_dynamic_updates()
            else:
                system.simulate_fine_tuning()

    print("Thank you for contacting TeleCo Customer Support. Goodbye!")
