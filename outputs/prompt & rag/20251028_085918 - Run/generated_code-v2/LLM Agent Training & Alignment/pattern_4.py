import random

class MockLLM:
    def generate_responses(self, query: str, num_samples: int) -> list[str]:
        base_responses = [
            f"Please restart your device and check the network cables for '{query}'.",
            f"For '{query}', kindly try troubleshooting steps like checking your internet connection.",
            f"Regarding '{query}', a quick fix might involve power cycling your equipment."
        ]
        
        responses = []
        for i in range(num_samples):
            response_template = random.choice(base_responses)
            # Simple variation for demonstration
            if i % 3 == 0:
                responses.append(response_template.replace('your device', 'your router'))
            elif i % 3 == 1:
                responses.append(response_template.replace('quick fix', 'potential solution'))
            else:
                responses.append(response_template)
        return responses

class MockRewardModel:
    def score_response(self, query: str, response: str) -> float:
        # Mock scoring logic - in a real scenario, this would be a trained model
        # For demonstration, we'll assign scores based on keywords or length
        score = 0.5 # Base score
        if "restart" in response.lower() and "modem" in response.lower():
            score += 0.35
        if "troubleshooting" in response.lower() or "steps" in response.lower():
            score += 0.2
        if "contact support" in response.lower() or "call us" in response.lower():
            score += 0.1 # Slightly less preferred for initial steps
        if len(response) > 80:
            score += 0.05 # Longer responses might be more comprehensive

        # Ensure score is within a reasonable range
        return min(1.0, max(0.0, score + random.uniform(-0.1, 0.1)))

class RejectionSamplingOrchestrator:
    def __init__(self, llm_model: MockLLM, reward_model: MockRewardModel):
        self.llm_model = llm_model
        self.reward_model = reward_model

    def get_enhanced_response(self, query: str, num_samples: int = 3) -> str:
        candidate_responses = self.llm_model.generate_responses(query, num_samples)
        
        best_response = ""
        highest_score = -1.0

        for response in candidate_responses:
            score = self.reward_model.score_response(query, response)
            if score > highest_score:
                highest_score = score
                best_response = response
        
        return best_response

if __name__ == "__main__":
    # Initialize mock models
    llm = MockLLM()
    reward_model = MockRewardModel()
    
    # Initialize the orchestrator
    orchestrator = RejectionSamplingOrchestrator(llm, reward_model)

    # Simulate a customer query
    customer_query = "My internet is not working. What should I do?"
    
    # Get the enhanced response using Rejection Sampling (Best-of-N)
    enhanced_response = orchestrator.get_enhanced_response(customer_query, num_samples=5)
    
    print(f"Customer Query: {customer_query}")
    print(f"Enhanced Response: {enhanced_response}")

    customer_query_2 = "How do I reset my password for my account?"
    enhanced_response_2 = orchestrator.get_enhanced_response(customer_query_2, num_samples=4)
    print(f"\nCustomer Query: {customer_query_2}")
    print(f"Enhanced Response: {enhanced_response_2}")