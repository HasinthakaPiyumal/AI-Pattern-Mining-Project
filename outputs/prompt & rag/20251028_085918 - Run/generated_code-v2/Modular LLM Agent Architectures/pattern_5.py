
class MockLLM:
    """A mock blackbox LLM that provides basic responses based on keywords."""
    def process_query(self, query: str) -> str:
        query_lower = query.lower()
        if "product" in query_lower and "details" in query_lower:
            return "I need to retrieve product details. What specific product are you asking about?"
        elif "refund" in query_lower:
            return "It seems you want a refund. I should check your order and initiate a refund process if eligible."
        elif "issue" in query_lower or "problem" in query_lower or "broken" in query_lower:
            return "You are reporting an issue. A support ticket might be necessary."
        elif "happy" in query_lower or "great" in query_lower:
            return "Your sentiment is positive. How else can I help?"
        elif "unhappy" in query_lower or "bad" in query_lower or "frustrated" in query_lower:
            return "Your sentiment is negative. Let me see how I can resolve this."
        return "I understand your query. Is there specific information you need or an action you want me to take?"


class InformationRetriever:
    """A module to retrieve information from a mock knowledge base."""
    def __init__(self):
        self.knowledge_base = {
            "laptop x100": "The Laptop X100 features an Intel i7 processor, 16GB RAM, 512GB SSD, and a 15-inch display. It comes with a 1-year warranty.",
            "warranty policy": "Our standard warranty covers manufacturing defects for 1 year from the purchase date. Accidental damage is not covered.",
            "shipping times": "Standard shipping takes 3-5 business days. Express shipping takes 1-2 business days.",
            "return policy": "Items can be returned within 30 days of purchase if unused and in original packaging. Refunds are processed within 7-10 business days."
        }

    def retrieve(self, query: str) -> str:
        query_lower = query.lower()
        for key, info in self.knowledge_base.items():
            # Check for exact key match or if any word in the key is present in the query
            if key in query_lower or any(word in query_lower for word in key.split()):
                return f"Retrieved info: {info}"
        return "No specific information found in the knowledge base for your query."


class SentimentAnalyzer:
    """A module to analyze the sentiment of a given text."""
    def analyze(self, text: str) -> str:
        text_lower = text.lower()
        if any(word in text_lower for word in ["problem", "issue", "unhappy", "bad", "broken", "frustrated"]):
            return "negative"
        elif any(word in text_lower for word in ["happy", "great", "excellent", "satisfied", "good"]):
            return "positive"
        return "neutral"


class ActionExecutor:
    """A module to simulate executing actions like creating tickets or initiating refunds."""
    def create_support_ticket(self, details: str) -> str:
        ticket_id = hash(details) % 100000  # Simple mock ID generation
        return f"Support ticket created with ID: {ticket_id}. Details: '{details}'"

    def initiate_refund(self, order_id: str) -> str:
        return f"Refund initiated for order ID: {order_id}. Please allow 7-10 business days for processing."


class CustomerSupportAgent:
    """The main orchestrator for the Smart Customer Support Agent.
    It uses a blackbox LLM and various plug-and-play modules to handle customer queries.
    """
    def __init__(self, llm, retriever, sentiment_analyzer, action_executor):
        self.llm = llm
        self.retriever = retriever
        self.sentiment_analyzer = sentiment_analyzer
        self.action_executor = action_executor
        self.conversation_history = [] # For potential future context management

    def handle_query(self, query: str) -> str:
        self.conversation_history.append(f"User: {query}")
        
        # Step 1: Get initial LLM insight
        llm_response = self.llm.process_query(query)
        self.conversation_history.append(f"LLM: {llm_response}")

        # Step 2: Analyze sentiment of the original query
        sentiment = self.sentiment_analyzer.analyze(query)
        self.conversation_history.append(f"Sentiment: {sentiment}")

        final_agent_response_parts = [f"LLM's initial thought: {llm_response}"]
        module_triggered = False

        # Step 3: Use modules based on LLM response and/or query keywords/sentiment
        # Information Retrieval
        if "product details" in llm_response.lower() or "product" in query.lower() or "info about" in query.lower() or "tell me about" in query.lower():
            retrieval_result = self.retriever.retrieve(query)
            final_agent_response_parts.append(f"Information Retrieval Module: {retrieval_result}")
            module_triggered = True
        
        # Action: Refund
        if "refund" in llm_response.lower() or "refund" in query.lower():
            # In a real system, we'd extract order ID from query or ask for it
            mock_order_id = "ORD" + str(hash(query) % 10000) # Simple placeholder for demo
            action_result = self.action_executor.initiate_refund(mock_order_id)
            final_agent_response_parts.append(f"Action Executor Module: {action_result}")
            module_triggered = True
        
        # Action: Create Support Ticket (triggered by LLM suggestion or negative sentiment)
        if "support ticket" in llm_response.lower() or sentiment == "negative":
            action_result = self.action_executor.create_support_ticket(query)
            final_agent_response_parts.append(f"Action Executor Module: {action_result}")
            module_triggered = True

        if not module_triggered:
            final_agent_response_parts.append("No specific plug-and-play module action was triggered based on this query or LLM response.")

        return "\n".join(final_agent_response_parts).strip()

# Example Usage (demonstrates how to instantiate and use the agent):
if __name__ == "__main__":
    # Initialize the plug-and-play modules
    llm = MockLLM()
    retriever = InformationRetriever()
    sentiment_analyzer = SentimentAnalyzer()
    action_executor = ActionExecutor()

    # Initialize the Customer Support Agent with the modules
    agent = CustomerSupportAgent(llm, retriever, sentiment_analyzer, action_executor)

    print("--- Scenario 1: User asks for product details ---")
    response1 = agent.handle_query("Tell me about the Laptop X100.")
    print(response1)
    print("\n" + "="*50 + "\n")

    print("--- Scenario 2: User requests a refund ---")
    response2 = agent.handle_query("I want a refund for my recent order. It was order number 12345.")
    print(response2)
    print("\n" + "="*50 + "\n")

    print("--- Scenario 3: User reports a broken product and is unhappy ---")
    response3 = agent.handle_query("This product is broken! I am very unhappy and frustrated with my purchase.")
    print(response3)
    print("\n" + "="*50 + "\n")

    print("--- Scenario 4: User asks a general question with positive sentiment ---")
    response4 = agent.handle_query("Hi there, I just wanted to say your service is great!")
    print(response4)
    print("\n" + "="*50 + "\n")

    print("--- Scenario 5: User asks about shipping times ---")
    response5 = agent.handle_query("What are the typical shipping times?")
    print(response5)
    print("\n" + "="*50 + "\n")

    print("--- Scenario 6: User asks about something not in knowledge base ---")
    response6 = agent.handle_query("Do you sell flying cars?")
    print(response6)
    print("\n" + "="*50 + "\n")
