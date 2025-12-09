class MockLLM:
    """A mock Large Language Model to simulate responses."""
    def generate(self, prompt: str) -> str:
        # Simulate LLM response based on prompt content
        if "summary of conversation" in prompt.lower() or "summarized response" in prompt.lower():
            return "This is a concise summary of our long conversation, focusing on your product issues and recent queries."
        elif "product x freezing" in prompt.lower() and "history of issues" in prompt.lower():
            return "Given your history with Product X freezing, let's try a system reset first. I've noted your past issues."
        elif "product y" in prompt.lower() and "purchase history" in prompt.lower():
            return "I can confirm your Product Y purchase on 2023-01-15. Your warranty is active for another year. Please ignore past Product X connectivity mention."
        return f"LLM Response (mock): I understand you're asking about '{prompt[:70]}...'. How can I help further?"

class MockVectorStore:
    """A mock vector store to simulate semantic search and retrieval of user data."""
    def __init__(self):
        self.data = {}

    def add_document(self, doc_id: str, text: str, embedding_keyword: str = None):
        self.data[doc_id] = {"text": text, "embedding_keyword": embedding_keyword if embedding_keyword else text.lower()}

    def search(self, query_keyword: str, top_k: int = 1) -> list:
        # Simulate semantic search by matching keywords
        results = []
        for doc_id, doc_info in self.data.items():
            if query_keyword.lower() in doc_info["embedding_keyword"] or doc_info["embedding_keyword"] in query_keyword.lower():
                results.append({"text": doc_info["text"], "score": 0.95}) # High score for direct match
        
        # Simple sorting and truncation for top_k
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

class ContextAwareChatbot:
    """A chatbot that manages long contexts for LLMs using prioritization, summarization, and external memory."""
    def __init__(self, llm_client: MockLLM, vector_store: MockVectorStore):
        self.llm = llm_client
        self.vector_store = vector_store
        self.conversation_history = []  # Stores turns of 'User: message' and 'Chatbot: message'
        self.user_profile_memory = {
            "user123": {
                "name": "Alice",
                "email": "alice@example.com",
                "product_ownership": ["Product X", "Product Y"],
                "past_issues": ["Product X freezing (resolved)", "Product Y connectivity (unresolved)"],
                "preferences": "prefers email updates, often asks about warranty"
            }
        }
        self.current_user_id = "user123" # Hardcoded user for this demo

    def _get_user_profile(self) -> dict:
        """Retrieves the current user's profile from internal memory."""
        return self.user_profile_memory.get(self.current_user_id, {})

    def summarize_conversation(self, conversation_snippets: list[str], max_summary_tokens: int = 100) -> str:
        """Mock summarization of conversation history."""
        full_text = " ".join(conversation_snippets)
        words = full_text.split()
        if len(words) > max_summary_tokens:
            return "Summary of conversation: " + " ".join(words[:max_summary_tokens-5]) + "... (truncated)"
        return "Summary of conversation: " + full_text

    def retrieve_from_external_memory(self, query: str, user_id: str, top_k: int = 2) -> list[str]:
        """Retrieves relevant information from the external vector store based on the query."""
        # In a real scenario, 'query' would be embedded to search the vector store.
        # Here, we use a simplified keyword-based search for demonstration.
        retrieved_docs = self.vector_store.search(query, top_k)
        return [doc["text"] for doc in retrieved_docs]

    def prioritize_context(self, conversation_context_elements: list[str], user_profile_facts: dict, max_tokens: int = 250) -> str:
        """Prioritizes and selects the most relevant context elements to fit within the LLM's window."""
        final_context_parts = []
        current_token_count = 0

        # Add user profile facts first (high importance)
        profile_string = f"User Profile: Name: {user_profile_facts.get('name', 'N/A')}, Products: {', '.join(user_profile_facts.get('product_ownership', []))}, Preferences: {user_profile_facts.get('preferences', 'N/A')}. Past Issues: {', '.join(user_profile_facts.get('past_issues', []))}."
        if current_token_count + len(profile_string.split()) <= max_tokens:
            final_context_parts.append(profile_string)
            current_token_count += len(profile_string.split())

        # Add explicitly passed conversation context elements (e.g., summary, retrieved memory, recent turns)
        for element in conversation_context_elements:
            if current_token_count + len(element.split()) <= max_tokens:
                final_context_parts.append(element)
                current_token_count += len(element.split())
            else:
                break # Stop adding if context window limit is reached

        return "\n".join(final_context_parts)

    def generate_response(self, user_query: str) -> str:
        """Generates a chatbot response using context management strategies."""
        self.conversation_history.append(f"User: {user_query}")

        user_profile = self._get_user_profile()

        # 1. Retrieve relevant information from external memory
        retrieved_memory_snippets = self.retrieve_from_external_memory(user_query, self.current_user_id)
        memory_context_str = "\nExternal Memory Retrieved: " + "\n".join(retrieved_memory_snippets) if retrieved_memory_snippets else ""

        # 2. Summarize the current conversation if it grows too long
        current_dialogue_turns = [turn for turn in self.conversation_history if turn.startswith("User:") or turn.startswith("Chatbot:")]
        # Arbitrary threshold for summarization trigger
        if len(" ".join(current_dialogue_turns).split()) > 150:
            summarized_dialogue = self.summarize_conversation(current_dialogue_turns)
        else:
            summarized_dialogue = "\n".join(current_dialogue_turns)

        # 3. Prepare context elements for prioritization
        context_elements_for_prioritization = []
        if summarized_dialogue:
            context_elements_for_prioritization.append(f"Current Conversation Context: {summarized_dialogue}")
        if memory_context_str:
            context_elements_for_prioritization.append(memory_context_str)

        # Also include a few very recent raw turns for immediate context, in case summary misses nuances
        recent_raw_turns = [turn for turn in self.conversation_history[-3:] if turn.startswith("User:") or turn.startswith("Chatbot:")]
        if recent_raw_turns:
            context_elements_for_prioritization.append(f"Recent Turns: {' | '.join(recent_raw_turns)}")

        # 4. Prioritize and select context to fit LLM window
        final_llm_input_context = self.prioritize_context(context_elements_for_prioritization, user_profile)

        # Construct the final prompt for the LLM
        prompt = f"""You are a helpful and context-aware customer support chatbot.
        {final_llm_input_context}

        User's Current Query: {user_query}

        Based on all the provided context, please respond accurately and helpfully. Focus on the most urgent aspect if multiple are present.
        """

        # Get response from the LLM
        llm_response = self.llm.generate(prompt)
        self.conversation_history.append(f"Chatbot: {llm_response}")
        return llm_response

# --- Example Usage --- #
if __name__ == "__main__":
    print("Initializing Context-Aware Customer Support Chatbot...")
    mock_llm = MockLLM()
    mock_vector_store = MockVectorStore()

    # Populate mock vector store with example user history data
    mock_vector_store.add_document("user_prod_x_issue_hist", "User has a history of issues with Product X, specifically freezing. Last reported 3 months ago.", embedding_keyword="product x freezing history")
    mock_vector_store.add_document("user_purchase_y_date", "User purchased Product Y on 2023-01-15. Warranty expires 2025-01-15.", embedding_keyword="product y purchase date warranty")
    mock_vector_store.add_document("user_pref_email_support", "User prefers email support for complex issues and values quick resolutions.", embedding_keyword="support preference email quick")
    mock_vector_store.add_document("user_prod_x_connectivity", "User contacted support last week about Product X connectivity issues.", embedding_keyword="product x connectivity issue")

    chatbot = ContextAwareChatbot(mock_llm, mock_vector_store)

    print("\n--- Starting Conversation ---\n")

    # Round 1: Simple issue, leveraging basic profile/recent context
    print("User: I have an issue with Product X. It keeps freezing.")
    response = chatbot.generate_response("I have an issue with Product X. It keeps freezing.")
    print(f"Chatbot: {response}\n")

    # Round 2: Follow-up, should recall previous turn and maybe some history
    print("User: Yes, I've had this problem before a few months ago. What can I do?")
    response = chatbot.generate_response("Yes, I've had this problem before a few months ago. What can I do?")
    print(f"Chatbot: {response}\n")

    # Round 3: Multi-topic query, testing summarization and external memory retrieval for Product Y and another Product X issue
    long_query = "Also, I was wondering about my recent purchase of Product Y. I bought it last year, maybe in January? Can you confirm the date and warranty details? I think I also contacted support about Product X connectivity issues last week, but I haven't heard back yet. The freezing issue is more urgent though."
    print(f"User: {long_query}")
    response = chatbot.generate_response(long_query)
    print(f"Chatbot: {response}\n")

    # Round 4: Focuses back on the most urgent issue, ensuring priority works
    print("User: So, what's the best way to get this freezing issue resolved quickly?")
    response = chatbot.generate_response("So, what's the best way to get this freezing issue resolved quickly?")
    print(f"Chatbot: {response}\n")

    print("\n--- Full Chatbot Conversation Log ---")
    for turn in chatbot.conversation_history:
        print(turn)
