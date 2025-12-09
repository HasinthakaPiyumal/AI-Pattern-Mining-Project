import random
import time

class CustomerData:
    def __init__(self):
        self.customer_histories = {}
        self._generate_mock_data()

    def _generate_mock_data(self):
        customers = ["Alice", "Bob", "Charlie"]
        topics = ["product inquiry", "technical support", "billing question", "feature request"]
        products = ["Smartphone", "Laptop", "Smartwatch", "Headphones"]

        for customer in customers:
            history = []
            for i in range(random.randint(3, 7)):
                topic = random.choice(topics)
                product = random.choice(products)
                interaction_type = random.choice(["chat", "email", "call"])
                timestamp = time.time() - random.randint(1, 30) * 24 * 60 * 60  # Past month

                if topic == "product inquiry":
                    detail = f"Customer inquired about the features of the {product}."
                elif topic == "technical support":
                    detail = f"Customer reported an issue with their {product} not turning on."
                elif topic == "billing question":
                    detail = f"Customer had a question regarding their last bill for {product}."
                else: # feature request
                    detail = f"Customer requested a new feature for the {product}: {random.choice(['longer battery life', 'improved camera', 'voice control'])}."
                
                history.append({"timestamp": timestamp, "type": interaction_type, "topic": topic, "detail": detail, "product": product})
            self.customer_histories[customer] = history

    def get_customer_history(self, customer_id):
        return self.customer_histories.get(customer_id, [])

class PrioritizationSelection:
    def __init__(self):
        pass

    def select_relevant_segments(self, history, current_query, max_segments=3):
        relevant_segments = []
        query_keywords = current_query.lower().split()

        # Simple scoring based on keyword match and recency
        scored_history = []
        for interaction in history:
            score = 0
            interaction_text = interaction["detail"].lower() + " " + interaction["topic"].lower() + " " + interaction["product"].lower()
            for keyword in query_keywords:
                if keyword in interaction_text:
                    score += 1
            
            # Add recency bias (more recent, higher score)
            recency_factor = (time.time() - interaction["timestamp"]) / (30 * 24 * 60 * 60) # Normalize to 0-1 over a month
            score -= recency_factor * 0.5 # Penalize older interactions
            
            scored_history.append((score, interaction))
        
        # Sort by score (descending) and take top N
        scored_history.sort(key=lambda x: x[0], reverse=True)
        relevant_segments = [item[1] for item in scored_history[:max_segments]]
        
        return relevant_segments

class SummarizationCompression:
    def __init__(self, max_length=150):
        self.max_length = max_length

    def summarize(self, text_segments):
        summaries = []
        for segment in text_segments:
            full_text = f"Type: {segment['type']}, Topic: {segment['topic']}, Product: {segment['product']}, Detail: {segment['detail']}"
            if len(full_text) > self.max_length:
                # Simple truncation as a form of compression
                summaries.append(full_text[:self.max_length-3] + "...")
            else:
                summaries.append(full_text)
        return summaries

class MemoryModule:
    def __init__(self):
        self.memory_store = {}
        self.memory_id_counter = 0

    def store_memory(self, customer_id, memory_content):
        if customer_id not in self.memory_store:
            self.memory_store[customer_id] = {}
        
        memory_id = f"mem_{self.memory_id_counter}"
        self.memory_store[customer_id][memory_id] = memory_content
        self.memory_id_counter += 1
        return memory_id

    def retrieve_memory(self, customer_id, query, top_k=2):
        if customer_id not in self.memory_store:
            return []
        
        relevant_memories = []
        query_lower = query.lower()

        # Simple keyword matching for retrieval
        scored_memories = []
        for mem_id, content in self.memory_store[customer_id].items():
            score = 0
            content_lower = str(content).lower() # Convert dict to string for search
            if query_lower in content_lower:
                score += 1 # Basic match
            
            for keyword in query_lower.split():
                if keyword in content_lower:
                    score += 0.5
            scored_memories.append((score, content))
        
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        relevant_memories = [item[1] for item in scored_memories[:top_k]]
        return relevant_memories

class LLMAgent:
    def __init__(self, customer_data, prioritization_selection, summarization_compression, memory_module):
        self.customer_data = customer_data
        self.prioritization_selection = prioritization_selection
        self.summarization_compression = summarization_compression
        self.memory_module = memory_module

    def _mock_llm_response(self, prompt):
        # Simulate an LLM response based on keywords in the prompt
        prompt_lower = prompt.lower()
        response = "I'm an AI assistant here to help you. "

        if "technical issue" in prompt_lower or "not working" in prompt_lower:
            response += "It sounds like you're experiencing a technical issue. Could you please describe the problem in more detail?"
        elif "billing" in prompt_lower or "invoice" in prompt_lower:
            response += "I can help with billing questions. Please provide your account details so I can look into your invoice."
        elif "product" in prompt_lower or "features" in prompt_lower:
            response += "I can provide information about our products. Which product are you interested in?"
        elif "memory" in prompt_lower or "history" in prompt_lower:
            response += "I'm checking your past interactions to provide a personalized response."
        else:
            response += "How can I assist you today?"

        if "previous interaction" in prompt_lower:
            response += " I see from your history that you previously had a related inquiry."

        return response

    def handle_query(self, customer_id, current_query):
        print(f"\n--- Handling query for {customer_id}: '{current_query}' ---")
        
        # 1. Retrieve full customer history
        full_history = self.customer_data.get_customer_history(customer_id)
        print(f"Full history retrieved ({len(full_history)} interactions).")

        # 2. Prioritize and Select relevant segments
        selected_segments = self.prioritization_selection.select_relevant_segments(full_history, current_query)
        print(f"Selected {len(selected_segments)} relevant historical segments.")

        # 3. Summarize and Compress selected segments
        summarized_history = self.summarization_compression.summarize(selected_segments)
        print(f"Summarized history: {summarized_history}")

        # 4. Store/Retrieve from External Memory (MemoryModule)
        # For demonstration, let's store the current summarized history and also retrieve existing relevant memories.
        for summary in summarized_history:
            self.memory_module.store_memory(customer_id, summary)

        retrieved_memories = self.memory_module.retrieve_memory(customer_id, current_query)
        print(f"Retrieved {len(retrieved_memories)} memories from long-term storage.")

        # 5. Construct LLM Prompt
        prompt_parts = []
        prompt_parts.append(f"You are an intelligent customer support agent for {customer_id}.")
        prompt_parts.append("Current User Query:")
        prompt_parts.append(current_query)
        
        if summarized_history:
            prompt_parts.append("Recent Relevant History:")
            for s in summarized_history:
                prompt_parts.append(f"- {s}")
        
        if retrieved_memories:
            prompt_parts.append("Long-Term Customer Profile/History (from memory):")
            for rm in retrieved_memories:
                prompt_parts.append(f"- {rm}")
        
        final_prompt = "\n".join(prompt_parts)
        print(f"\n--- Constructed LLM Prompt (excerpt) ---\n{final_prompt[:500]}...\n")

        # 6. Get LLM Response
        llm_response = self._mock_llm_response(final_prompt)
        print(f"LLM Response: {llm_response}")
        return llm_response

if __name__ == "__main__":
    print("Initializing Intelligent Customer Support Agent...")
    
    customer_data = CustomerData()
    prioritization_selection = PrioritizationSelection()
    summarization_compression = SummarizationCompression()
    memory_module = MemoryModule()
    llm_agent = LLMAgent(customer_data, prioritization_selection, summarization_compression, memory_module)

    # Simulate customer interactions
    print("\nSimulating Customer Interactions...")
    
    # Interaction 1: General inquiry about a product, no memory yet
    llm_agent.handle_query("Alice", "I want to know more about the new Smartphone.")
    
    # Interaction 2: Technical issue, should pull relevant history
    llm_agent.handle_query("Bob", "My Laptop is not turning on. I remember having an issue with it last month.")
    
    # Interaction 3: Billing question, check if past billing questions are retrieved
    llm_agent.handle_query("Charlie", "Can you check my last bill? I think there's a discrepancy.")

    # Interaction 4: Follow-up on a previous issue for Alice, should use memory
    llm_agent.handle_query("Alice", "Regarding my previous inquiry about the Smartphone, what are its camera specifications?")

    # Interaction 5: New feature request, should still benefit from general history
    llm_agent.handle_query("Bob", "I'd like to suggest a feature for the Smartwatch - better health tracking.")

    print("\nDemonstration complete.")