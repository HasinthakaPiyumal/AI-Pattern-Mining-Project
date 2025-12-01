import random

def generate_candidate_responses(query, num_candidates):
    base_responses = [
        f"I understand you have a question about {query}. Here's a possible solution: This is response A.",
        f"Regarding your inquiry about {query}, consider the following: This is response B.",
        f"For {query}, a good approach would be: This is response C.",
        f"Let's help you with {query}. My suggestion is: This is response D.",
        f"About {query}, here's some information that might help: This is response E."
    ]
    candidates = []
    for i in range(num_candidates):
        # Simulate diversity by adding slight variations or picking randomly
        chosen_base = random.choice(base_responses)
        variation = f" (variation {i+1} from a language model)"
        candidates.append(chosen_base.replace("This is response A.", f"This is response A{variation}")
                          .replace("This is response B.", f"This is response B{variation}")
                          .replace("This is response C.", f"This is response C{variation}")
                          .replace("This is response D.", f"This is response D{variation}")
                          .replace("This is response E.", f"This is response E{variation}"))
    return candidates

def score_response(response):
    score = 0.0
    if "solution" in response.lower() or "approach" in response.lower():
        score += 0.4
    if "help" in response.lower() or "suggestion" in response.lower():
        score += 0.3
    if len(response) > 80 and len(response) < 150:
        score += 0.2 # Prefer moderately detailed responses
    else:
        score += random.uniform(0, 0.1)
    score += random.uniform(0.1, 0.5) # Add some randomness to simulate real reward model variation
    return min(1.0, score) # Cap score at 1.0

def get_best_response(query, num_samples=5):
    candidate_responses = generate_candidate_responses(query, num_samples)
    scored_responses = []
    for response in candidate_responses:
        score = score_response(response)
        scored_responses.append((response, score))
    
    best_response = None
    max_score = -1.0

    for response, score in scored_responses:
        if score > max_score:
            max_score = score
            best_response = response
            
    return best_response, max_score

def customer_support_chatbot():
    print("Welcome to the Smart Customer Support Chatbot!")
    print("Type 'exit' to end the conversation.")
    while True:
        user_query = input("You: ")
        if user_query.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break
        
        best_response, score = get_best_response(user_query, num_samples=5) # N=5 for Best-of-N
        print(f"Chatbot (Score: {score:.2f}): {best_response}")

if __name__ == "__main__":
    customer_support_chatbot()