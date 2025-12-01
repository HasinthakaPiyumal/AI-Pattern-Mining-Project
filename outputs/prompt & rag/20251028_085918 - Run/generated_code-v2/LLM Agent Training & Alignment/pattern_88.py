import random

def mock_llm_generate_responses(query, N):
    base_response = f"Regarding '{query}', I can assist with..."
    responses = []
    for i in range(N):
        responses.append(f"{base_response} Option {i+1} might be: {''.join(random.choices('abcdefghijklmnopqrstuvwxyz ', k=30))}.")
    return responses

def mock_reward_model_score(query, response):
    
    relevance_score = 0.5 + random.random() * 0.5  
    
    helpfulness_keywords = ['assist', 'help', 'solution', 'resolve', 'answer']
    if any(keyword in response.lower() for keyword in helpfulness_keywords):
        helpfulness_score = 0.5 + random.random() * 0.5
    else:
        helpfulness_score = random.random() * 0.5

    
    final_score = (relevance_score + helpfulness_score) / 2
    return final_score

def best_of_n_selector(query, N=5):
    candidate_responses = mock_llm_generate_responses(query, N)
    scored_responses = []

    for response in candidate_responses:
        score = mock_reward_model_score(query, response)
        scored_responses.append((response, score))

    best_response = None
    highest_score = -1

    for response, score in scored_responses:
        if score > highest_score:
            highest_score = score
            best_response = response
            
    print(f"\n--- Debugging Candidate Responses and Scores (N={N}) ---")
    for i, (res, scr) in enumerate(scored_responses):
        print(f"Candidate {i+1}: Score = {scr:.4f}, Response = {res}")
    print("---------------------------------------------------\n")

    return best_response

def main():
    print("Welcome to the AI Customer Support Chatbot!")
    print("Type 'exit' to end the conversation.")

    while True:
        user_query = input("\nYou: ")
        if user_query.lower() == 'exit':
            print("Thank you for using the chatbot. Goodbye!")
            break

        optimized_response = best_of_n_selector(user_query, N=5)  
        print(f"Chatbot: {optimized_response}")

if __name__ == "__main__":
    main()