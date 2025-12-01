from customer_service_agent import CustomerServiceAgent

def main():
    agent = CustomerServiceAgent()

    complex_query_1 = "I want to return item ABC-123 and get a refund, but also order a replacement for item XYZ-789 and track my current order ORD-456."
    response_1 = agent.handle_inquiry(complex_query_1)
    print("\n" + "="*50 + "\n")
    print(response_1)

    complex_query_2 = "Can I get a refund for item DEF-456? Also, where is my order PQR-789?"
    response_2 = agent.handle_inquiry(complex_query_2)
    print("\n" + "="*50 + "\n")
    print(response_2)

    complex_query_3 = "I need to return item GHI-012."
    response_3 = agent.handle_inquiry(complex_query_3)
    print("\n" + "="*50 + "\n")
    print(response_3)

if __name__ == "__main__":
    main()