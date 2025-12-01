import random

def generate_candidate_answers(query, num_candidates):
    base_answers = [
        f"I understand you're asking about {query}. Here's a possible solution: ",
        f"Regarding your query about {query}, consider this approach: ",
        f"For {query}, a good way to think about it is: ",
        f"Let's break down your question on {query}. My suggestion is: ",
        f"To address {query}, try the following steps: "
    ]
    candidates = []
    for i in range(num_candidates):
        chosen_base = random.choice(base_answers)
        variation = random.choice([
            "Make sure all parameters are correctly set.",
            "Consult the official documentation for more details.",
            "Restart the application and try again.",
            "Check your internet connection if applicable.",
            "Ensure you have the latest software version.",
            "It might be a temporary issue, please wait a moment."
        ])
        candidates.append(f"{chosen_base}{variation} (Confidence: {random.randint(70, 99)}%)")
    return candidates

def score_answer(answer, query):
    score = 0
    if query.lower() in answer.lower():
        score += 20
    if "solution" in answer.lower() or "approach" in answer.lower() or "suggestion" in answer.lower() or "steps" in answer.lower():
        score += 15
    if "official documentation" in answer.lower():
        score += 10
    if "latest software version" in answer.lower():
        score += 8
    score += len(answer) / 5
    score += random.randint(1, 10)
    return score

def best_of_n_selection(query, num_candidates=5):
    candidate_answers = generate_candidate_answers(query, num_candidates)
    scored_answers = []
    for answer in candidate_answers:
        score = score_answer(answer, query)
        scored_answers.append((answer, score))
    
    best_answer = None
    max_score = -1

    for answer, score in scored_answers:
        if score > max_score:
            max_score = score
            best_answer = answer
            
    return best_answer

if __name__ == "__main__":
    print("Welcome to the Best-of-N Chatbot!")
    print("Type 'exit' to quit.")
    while True:
        user_query = input("\nYou: ")
        if user_query.lower() == 'exit':
            break
        
        final_response = best_of_n_selection(user_query, num_candidates=5)
        print(f"Chatbot: {final_response}")
