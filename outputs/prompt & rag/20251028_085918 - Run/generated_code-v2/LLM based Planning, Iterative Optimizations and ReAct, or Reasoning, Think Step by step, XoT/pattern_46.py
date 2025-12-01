# Simulated CRM/Knowledge Base
_customer_database = {
    "Alice Smith": {
        "customer_id": "cust_001",
        "services": ["broadband", "mobile"],
        "billing_status": "current",
        "authorized_users": ["Bob Smith"],
        "plan_details": {"broadband": "Fiber 100", "mobile": "Unlimited Talk & Text"},
        "recent_interactions": ["bill inquiry", "service upgrade request"],
    },
    "Bob Smith": {
        "customer_id": "cust_001_auth_001",
        "is_authorized_user": True,
        "authorized_for_customer": "Alice Smith",
        "permissions": ["view bill", "technical support"],
    },
    "Charlie Brown": {
        "customer_id": "cust_002",
        "services": ["mobile"],
        "billing_status": "overdue",
        "authorized_users": [],
        "plan_details": {"mobile": "Basic 10GB"},
        "recent_interactions": ["payment reminder"],
    }
}

def _identify_entities(query: str) -> dict:
    entities = {"customers": [], "authorized_users": []}
    for name, data in _customer_database.items():
        if name.lower() in query.lower():
            if "is_authorized_user" in data and data["is_authorized_user"]:
                entities["authorized_users"].append(name)
            else:
                entities["customers"].append(name)
    return entities

def _establish_facts(entity_type: str, entity_name: str) -> str:
    facts = []
    if entity_type == "customer" and entity_name in _customer_database:
        customer_data = _customer_database[entity_name]
        facts.append(f"Customer Name: {entity_name}")
        facts.append(f"Customer ID: {customer_data['customer_id']}")
        facts.append(f"Services: {', '.join(customer_data['services'])}")
        facts.append(f"Billing Status: {customer_data['billing_status']}")
        if customer_data["authorized_users"]:
            facts.append(f"Authorized Users: {', '.join(customer_data['authorized_users'])}")
        facts.append(f"Plan Details: {customer_data['plan_details']}")
        facts.append(f"Recent Interactions: {', '.join(customer_data['recent_interactions'])}")
    elif entity_type == "authorized_user" and entity_name in _customer_database:
        user_data = _customer_database[entity_name]
        facts.append(f"Authorized User Name: {entity_name}")
        facts.append(f"Authorized for Customer: {user_data['authorized_for_customer']}")
        facts.append(f"Permissions: {', '.join(user_data['permissions'])}")
        if user_data["authorized_for_customer"] in _customer_database:
            customer_data = _customer_database[user_data["authorized_for_customer"]]
            facts.append(f"Customer services: {', '.join(customer_data['services'])}")
            facts.append(f"Customer billing status: {customer_data['billing_status']}")
    
    return "\n".join(facts)

def _simulate_llm_response(full_context_prompt: str) -> str:
    response = "I'm processing your request based on the available information.\n"

    if "Alice Smith" in full_context_prompt and "billing status" in full_context_prompt.lower():
        response += "Alice Smith's billing status is current."
    elif "Alice Smith" in full_context_prompt and "plan details" in full_context_prompt.lower():
        response += "Alice Smith's broadband plan is Fiber 100 and her mobile plan is Unlimited Talk & Text."
    elif "Bob Smith" in full_context_prompt and "permissions" in full_context_prompt.lower():
        response += "Bob Smith is an authorized user for Alice Smith and can view bills and get technical support."
    elif "Charlie Brown" in full_context_prompt and "billing status" in full_context_prompt.lower():
        response += "Charlie Brown's billing status is overdue."
    elif "services" in full_context_prompt.lower() and "Alice Smith" in full_context_prompt:
        response += "Alice Smith has broadband and mobile services."
    elif "services" in full_context_prompt.lower() and "Charlie Brown" in full_context_prompt:
        response += "Charlie Brown has mobile service."
    else:
        response += "I'm sorry, I couldn't find specific information to answer that question directly based on the provided context. Can you please rephrase or provide more details."

    return response

def customer_support_chatbot(query: str) -> str:
    entities = _identify_entities(query)
    
    context_facts = []
    
    for customer_name in entities["customers"]:
        context_facts.append(f"Facts about Customer {customer_name}:\n{_establish_facts('customer', customer_name)}")
        
    for auth_user_name in entities["authorized_users"]:
        context_facts.append(f"Facts about Authorized User {auth_user_name}:\n{_establish_facts('authorized_user', auth_user_name)}")

    combined_facts = "\n\n".join(context_facts)
    
    if not combined_facts:
        initial_prompt = f"The user asked: \"{query}\". No specific customer or authorized user was identified in the query or our database. Please answer based on general knowledge or ask for more details."
    else:
        initial_prompt = (
            f"Given the following facts about relevant entities:\n\n"
            f"{combined_facts}\n\n"
            f"Based *only* on these facts, answer the following customer question: \"{query}\"\n"
            f"If the information is not explicitly provided in the facts, state that you don't have enough information."
        )

    llm_answer = _simulate_llm_response(initial_prompt)
    
    return llm_answer

if __name__ == "__main__":
    print("Welcome to the Contextualized Customer Support Chatbot!")
    print("Type 'exit' to end the chat.")
    while True:
        user_query = input("\nYou: ")
        if user_query.lower() == 'exit':
            break
        
        response = customer_support_chatbot(user_query)
        print(f"Bot: {response}")