
import random

class ExemplarOrderingChatbot:
    def __init__(self, exemplars_data):
        """
        Initializes the chatbot with a set of predefined exemplars.
        Each exemplar is a dictionary with 'query' and 'response' keys.
        """
        self.exemplars = exemplars_data

    def _calculate_simple_similarity(self, text1, text2):
        """
        Calculates a basic similarity score based on common words.
        This is a simplified approach, real-world would use embeddings.
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        common_words = len(words1.intersection(words2))
        return common_words

    def _order_exemplars(self, selected_exemplars, strategy, user_query=None):
        """
        Applies the specified ordering strategy to the selected exemplars.

        Args:
            selected_exemplars (list): A list of exemplar dictionaries.
            strategy (str): The ordering strategy ('random', 'similarity', 'default').
            user_query (str, optional): The current user query, used for 'similarity' strategy.

        Returns:
            list: The reordered list of exemplars.
        """
        if strategy == "random":
            random.shuffle(selected_exemplars)
        elif strategy == "similarity" and user_query:
            # Order by how similar the exemplar query is to the user_query
            selected_exemplars.sort(
                key=lambda x: self._calculate_simple_similarity(x['query'], user_query),
                reverse=True
            )
        # 'default' or any unrecognized strategy implies no specific ordering (e.g., insertion order)
        # For 'diversity', a more complex selection or clustering would be needed,
        # for simplicity here, we can treat default as a form of initial diversity or no reordering.
        return selected_exemplars

    def _construct_prompt(self, user_query, ordered_exemplars):
        """
        Constructs the few-shot prompt string.
        """
        prompt_parts = []
        for ex in ordered_exemplars:
            prompt_parts.append(f"Customer: {ex['query']}\nChatbot: {ex['response']}")
        
        prompt_parts.append(f"Customer: {user_query}\nChatbot:")
        return "\n\n".join(prompt_parts)

    def _simulate_llm_response(self, prompt):
        """
        Simulates an LLM generating a response based on the prompt.
        In a real application, this would involve an actual LLM API call.
        For demonstration, it just returns a generic response based on the prompt's last query.
        """
        # Extract the last query from the prompt
        last_query_start = prompt.rfind("Customer: ") + len("Customer: ")
        last_query_end = prompt.rfind("\nChatbot:")
        if last_query_start != -1 and last_query_end != -1 and last_query_start < last_query_end:
            actual_query = prompt[last_query_start:last_query_end].strip()
        else:
            actual_query = "your query"

        # Very simplistic simulation: just acknowledges the query and the ordering used.
        if "Customer: What is your refund policy?" in prompt and "response to refund" in prompt.lower():
            return "Our refund policy allows full refunds within 30 days of purchase, provided the item is unused and in its original packaging. Please see our FAQs for more details on specific product categories."
        elif "Customer: How do I track my order?" in prompt and "order tracking" in prompt.lower():
            return "You can track your order by visiting the 'My Orders' section on our website and entering your order number. A tracking link will be provided there."
        elif "Customer: Can I change my shipping address?" in prompt and "shipping address change" in prompt.lower():
            return "Shipping address changes are possible before the item is shipped. Please contact support immediately with your order number to request a change."
        
        return f"[LLM Simulation] I understand {actual_query}. Based on the examples provided with '{self.last_strategy_used}' ordering, I would provide a helpful response. Please refine the LLM integration for actual output."

    def ask(self, user_query, num_exemplars=3, ordering_strategy="default"):
        """
        Processes a user query using few-shot prompting with exemplar ordering.

        Args:
            user_query (str): The customer's query.
            num_exemplars (int): The number of exemplars to include in the prompt.
            ordering_strategy (str): The strategy to order exemplars ('random', 'similarity', 'default').

        Returns:
            str: The chatbot's response.
        """
        self.last_strategy_used = ordering_strategy

        # 1. Select exemplars (simple selection: just take the first 'num_exemplars')
        # In a real system, this would involve a more sophisticated selection based on query similarity
        # or diversity. For this pattern, the focus is on *ordering* once selected.
        selected_exemplars = self.exemplars[:num_exemplars]

        # 2. Order exemplars based on the chosen strategy
        ordered_exemplars = self._order_exemplars(selected_exemplars, ordering_strategy, user_query)

        # 3. Construct the few-shot prompt
        prompt = self._construct_prompt(user_query, ordered_exemplars)
        # print(f"\n--- Generated Prompt ({ordering_strategy} ordering) ---\n{prompt}\n---") # For debugging

        # 4. Simulate LLM response
        response = self._simulate_llm_response(prompt)
        return response

# --- Example Usage ---
if __name__ == "__main__":
    # Define a set of exemplars for customer support queries
    sample_exemplars = [
        {"query": "What is your refund policy?", "response": "Our refund policy allows full refunds within 30 days of purchase. Items must be unused."},
        {"query": "How do I track my order?", "response": "You can track your order using the link in your shipping confirmation email or via your account."},
        {"query": "Can I change my shipping address?", "response": "Shipping address changes are possible before dispatch. Please contact us immediately."},
        {"query": "My product arrived damaged, what should I do?", "response": "Please provide photos of the damaged item and packaging to our support team for a replacement or refund."},
        {"query": "Do you offer international shipping?", "response": "Yes, we offer international shipping to most countries. Rates vary by destination."},
    ]

    chatbot = ExemplarOrderingChatbot(sample_exemplars)

    print("\n--- Testing with 'default' (no specific re-ordering) strategy ---")
    response_default = chatbot.ask("I want to know about returns.", ordering_strategy="default")
    print(f"Chatbot: {response_default}")

    print("\n--- Testing with 'random' ordering strategy ---")
    response_random = chatbot.ask("I want to know about returns.", ordering_strategy="random")
    print(f"Chatbot: {response_random}")
    
    print("\n--- Testing with 'similarity' ordering strategy (for refund query) ---")
    response_similarity_refund = chatbot.ask("I need to return an item, what's the process?", ordering_strategy="similarity")
    print(f"Chatbot: {response_similarity_refund}")

    print("\n--- Testing with 'similarity' ordering strategy (for tracking query) ---")
    response_similarity_track = chatbot.ask("Where is my package? How can I check its status?", ordering_strategy="similarity")
    print(f"Chatbot: {response_similarity_track}")

    print("\n--- Testing with 'similarity' ordering strategy (for address change query) ---")
    response_similarity_address = chatbot.ask("Can you update my delivery location please?", ordering_strategy="similarity")
    print(f"Chatbot: {response_similarity_address}")

