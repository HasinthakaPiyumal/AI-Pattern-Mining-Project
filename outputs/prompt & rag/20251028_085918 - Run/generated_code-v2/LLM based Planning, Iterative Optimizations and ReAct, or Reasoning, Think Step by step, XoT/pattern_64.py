def analyze_query_for_subproblems(query: str) -> list[str]:
    complex_keywords = ["not working", "cannot access", "problem with", "issue with", "broken", "unavailable"]
    subproblems = []
    sentences = query.split(". ")
    for sentence in sentences:
        if any(keyword in sentence.lower() for keyword in complex_keywords):
            subproblems.append(sentence.strip())
    return subproblems

def solve_subproblem(subproblem_query: str, depth: int = 0) -> str:
    max_recursion_depth = 2
    if depth >= max_recursion_depth:
        return f"(Simulated solution for '{subproblem_query}' at max depth: Further complex analysis required.)"

    # Simulate finding nested sub-subproblems within the current subproblem
    nested_subproblems_keywords = ["and also", "but also", "in addition to"]
    nested_subproblems = []
    for keyword in nested_subproblems_keywords:
        if keyword in subproblem_query.lower():
            parts = subproblem_query.split(keyword, 1)
            nested_subproblems.append(parts[1].strip())
            subproblem_query = parts[0].strip()
            break 

    # Simulate LLM response for various subproblems
    if "internet not working" in subproblem_query.lower():
        solution = "Please check your router and modem connections, and try restarting them."
    elif "cannot access email" in subproblem_query.lower():
        solution = "Verify your email password and server settings. You might need to contact your email provider."
    elif "phone line dead" in subproblem_query.lower():
        solution = "Check if your phone is properly connected. If it's a landline, ensure cables are secure. For mobile, check service coverage."
    elif "printer issue" in subproblem_query.lower():
        solution = "Ensure the printer is turned on, connected, and has ink/toner. Check the print queue."
    else:
        solution = f"(Simulated basic solution for '{subproblem_query}': We are looking into this further.)"
    
    # Recursively solve nested sub-subproblems
    for nested_problem in nested_subproblems:
        nested_solution = solve_subproblem(nested_problem, depth + 1)
        solution += f" Additionally, for '{nested_problem}': {nested_solution}"

    return solution

def integrate_subproblem_solution(main_context: str, subproblem_solution: str) -> str:
    return f"{main_context}\n\nSubproblem Resolution: {subproblem_solution}"

def main_customer_support_agent(query: str) -> str:
    main_context = f"Initial Query: {query}"
    print(f"[Agent] Received query: '{query}'")

    subproblems = analyze_query_for_subproblems(query)

    if not subproblems:
        final_answer = f"{main_context}\n\n[Agent] We understand your query and are processing it without identifying immediate complex subproblems. A standard resolution will be provided shortly."
        print("[Agent] No complex subproblems identified. Providing a standard response.")
        return final_answer

    print(f"[Agent] Identified complex subproblems: {subproblems}")

    for sp in subproblems:
        print(f"[Agent] Recursively solving subproblem: '{sp}'")
        sub_solution = solve_subproblem(sp)
        main_context = integrate_subproblem_solution(main_context, sub_solution)
        print(f"[Agent] Integrated solution for '{sp}'. Current context:\n{main_context}")

    final_answer = f"{main_context}\n\n[Agent] This is the comprehensive resolution based on our analysis and recursive problem-solving."
    print("[Agent] All subproblems addressed. Providing comprehensive resolution.")
    return final_answer

if __name__ == "__main__":
    print("--- Test Case 1: Simple Query ---")
    query1 = "My internet is not working."
    response1 = main_customer_support_agent(query1)
    print("\nFinal Response 1:\n", response1)

    print("\n--- Test Case 2: Query with Multiple Subproblems ---")
    query2 = "I have an issue with my internet connection. Also, I cannot access my email. My printer is also broken."
    response2 = main_customer_support_agent(query2)
    print("\nFinal Response 2:\n", response2)

    print("\n--- Test Case 3: Query with Nested Subproblems (simulated) ---")
    query3 = "My internet is not working and also I have a problem with slow speeds. Additionally, I cannot access my email."
    response3 = main_customer_support_agent(query3)
    print("\nFinal Response 3:\n", response3)

    print("\n--- Test Case 4: Query with no complex keywords ---")
    query4 = "I would like to know my current account balance."
    response4 = main_customer_support_agent(query4)
    print("\nFinal Response 4:\n", response4)

    print("\n--- Test Case 5: Deeper Nested Subproblem (simulated max depth) ---")
    query5 = "My internet is not working and also my WiFi keeps disconnecting and also the signal is weak."
    response5 = main_customer_support_agent(query5)
    print("\nFinal Response 5:\n", response5)
