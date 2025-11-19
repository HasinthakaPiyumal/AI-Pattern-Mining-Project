import random

class MockChatbot:
    def respond(self, query):
        query = query.lower()
        if "cancel subscription" in query or "refund" in query:
            return "I understand you'd like to cancel or get a refund. Please visit our website's help section or chat with a live agent for further assistance."
        elif "technical issue" in query or "not working" in query:
            return "I apologize for the inconvenience. Could you please describe your technical issue in more detail? We also have troubleshooting guides on our support page."
        elif "premium features" in query or "upgrade" in query:
            return "Thank you for your interest in our premium features! You can find all the details and upgrade options on our pricing page."
        elif "thank you" in query or "appreciate" in query:
            return "You're very welcome! I'm glad I could help. Is there anything else I can assist you with today?"
        else:
            return "Hello! How can I assist you today? Please provide more details about your request."

class CustomerPersona:
    def __init__(self, name, description):
        self.name = name
        self.description = description

    def generate_query(self):
        raise NotImplementedError("Subclasses must implement generate_query method")

class FrustratedCustomer(CustomerPersona):
    def __init__(self):
        super().__init__("Frustrated Customer", "Easily annoyed, short-tempered, wants quick resolution.")

    def generate_query(self):
        queries = [
            "I've been waiting for an hour! Why isn't this working?!",
            "Your service is terrible. I want to cancel my subscription NOW.",
            "This is ridiculous. Fix it immediately!",
            "I'm so fed up with this constant issue."
        ]
        return random.choice(queries)

class TechnicalNoviceCustomer(CustomerPersona):
    def __init__(self):
        super().__init__("Technical Novice Customer", "Lacks technical understanding, needs simple explanations.")

    def generate_query(self):
        queries = [
            "How do I make the thingy work?",
            "My internet is slow. What should I do?",
            "I can't log in. Is there a simple way to fix it?",
            "What does 'cache' mean?"
        ]
        return random.choice(queries)

class DemandingCustomer(CustomerPersona):
    def __init__(self):
        super().__init__("Demanding Customer", "Has high expectations, wants specific solutions, might ask for extras.")

    def generate_query(self):
        queries = [
            "I expect a full refund and a complimentary upgrade for this inconvenience.",
            "I need this resolved by end of day, no excuses. Provide a direct line to a manager.",
            "What special benefits do I get for being a long-time customer?",
            "I require immediate assistance and a detailed explanation of the root cause."
        ]
        return random.choice(queries)

class LoyalCustomer(CustomerPersona):
    def __init__(self):
        super().__init__("Loyal Customer", "Generally polite, appreciative, seeks continued good service.")

    def generate_query(self):
        queries = [
            "I've been a customer for years, and I usually love your service. I'm just having a small issue with my account.",
            "Could you please help me with a minor query? I appreciate your assistance.",
            "Thank you for your consistent support. I have a quick question about billing.",
            "I just wanted to know if there are any new features coming soon for loyal users like myself?"
        ]
        return random.choice(queries)

class EvaluatorPersona:
    def __init__(self, name, description):
        self.name = name
        self.description = description

    def evaluate_response(self, query, chatbot_response):
        raise NotImplementedError("Subclasses must implement evaluate_response method")

class CustomerSupportManager(EvaluatorPersona):
    def __init__(self):
        super().__init__("Customer Support Manager", "Evaluates empathy, tone, adherence to CS guidelines, and problem resolution.")

    def evaluate_response(self, query, chatbot_response):
        score = 0
        comments = []

        if any(keyword in chatbot_response.lower() for keyword in ["apologize", "understand", "sorry"]):
            score += 1
            comments.append("Showed empathy.")
        if "further assistance" in chatbot_response.lower() or "more details" in chatbot_response.lower():
            score += 1
            comments.append("Attempted problem resolution or next steps.")
        if "terrible" in query.lower() or "ridiculous" in query.lower():
            if any(keyword in chatbot_response.lower() for keyword in ["understand", "apologize"]):
                score += 1
                comments.append("Handled frustrated tone appropriately.")
            else:
                comments.append("Could have shown more empathy for frustration.")
        if "cancel" in query.lower() and "visit our website" in chatbot_response.lower():
            score += 1
            comments.append("Provided relevant next step for cancellation.")
        
        if not comments:
            comments.append("Neutral response.")

        return {"score": score, "comments": comments, "persona": self.name}

class ProductExpert(EvaluatorPersona):
    def __init__(self):
        super().__init__("Product Expert", "Assesses factual accuracy, technical correctness, and completeness.")

    def evaluate_response(self, query, chatbot_response):
        score = 0
        comments = []

        if "technical issue" in query.lower() and ("troubleshooting guides" in chatbot_response.lower() or "describe in more detail" in chatbot_response.lower()):
            score += 1
            comments.append("Suggested relevant technical troubleshooting.")
        if "premium features" in query.lower() and "pricing page" in chatbot_response.lower():
            score += 1
            comments.append("Provided accurate information source for features.")
        if "cache" in query.lower() and "hello" not in chatbot_response.lower():
             comments.append("Did not explain technical term.")
             score -= 1 # Deduct for not addressing technical term

        if not comments:
            comments.append("Neutral response.")

        return {"score": score, "comments": comments, "persona": self.name}

class LegalComplianceOfficer(EvaluatorPersona):
    def __init__(self):
        super().__init__("Legal Compliance Officer", "Assesses adherence to legal/company policies, data privacy.")

    def evaluate_response(self, query, chatbot_response):
        score = 0
        comments = []

        if any(keyword in query.lower() for keyword in ["cancel subscription", "refund"]):
            if "website's help section" in chatbot_response.lower() or "live agent" in chatbot_response.lower():
                score += 1
                comments.append("Directed to appropriate channel for sensitive actions.")
            else:
                comments.append("Did not clearly direct to appropriate channel for sensitive actions.")

        if "personal information" in query.lower() or "account details" in query.lower():
            if "security" in chatbot_response.lower() or "privacy policy" in chatbot_response.lower():
                score += 1
                comments.append("Addressed data privacy/security.")
            elif "hello" in chatbot_response.lower():
                 comments.append("Did not address potential data privacy concern.")
                 score -= 1

        if not comments:
            comments.append("No immediate compliance issues detected.")

        return {"score": score, "comments": comments, "persona": self.name}


def main():
    chatbot = MockChatbot()

    customer_personas = [
        FrustratedCustomer(),
        TechnicalNoviceCustomer(),
        DemandingCustomer(),
        LoyalCustomer()
    ]

    evaluator_personas = [
        CustomerSupportManager(),
        ProductExpert(),
        LegalComplianceOfficer()
    ]

    interactions = []

    print("--- Simulating Customer Interactions ---")
    for customer in customer_personas:
        query = customer.generate_query()
        response = chatbot.respond(query)
        interactions.append({
            "customer_name": customer.name,
            "query": query,
            "chatbot_response": response
        })
        print(f"[{customer.name}] Query: '{query}'")
        print(f"[Chatbot] Response: '{response}'\n")

    print("\n--- Evaluating Chatbot Responses ---")
    for i, interaction in enumerate(interactions):
        print(f"\nInteraction {i+1}:")
        print(f"  Customer: {interaction['customer_name']}")
        print(f"  Query: '{interaction['query']}'")
        print(f"  Chatbot Response: '{interaction['chatbot_response']}'")
        print("  Evaluations:")
        for evaluator in evaluator_personas:
            evaluation_result = evaluator.evaluate_response(
                interaction['query'], interaction['chatbot_response']
            )
            print(f"    [{evaluator.name}] Score: {evaluation_result['score']}, Comments: {', '.join(evaluation_result['comments'])}")

if __name__ == "__main__":
    main()