class ChatbotAgent:
    """Simulates a chatbot agent with browsing and answer generation capabilities."""
    def __init__(self, name="RLChatbot"):
        self.name = name

    def browse_knowledge_base(self, query, knowledge_base):
        """Simulates browsing a knowledge base to find relevant references."""
        print(f"[{self.name}] Browsing knowledge base for query: \"{query}\"")
        retrieved_references = []
        # Simple keyword-based retrieval for demonstration
        for topic, content in knowledge_base.items():
            if query.lower() in topic.lower() or any(word in content.lower() for word in query.lower().split()):
                retrieved_references.append(f"Reference from {topic}: {content[:100]}...") # Truncate for display
        
        if not retrieved_references:
            retrieved_references.append("No specific references found, providing general info.")
            retrieved_references.append(f"General knowledge: We offer support for various topics.")
        return retrieved_references

    def generate_answer(self, query, references):
        """Simulates generating an answer based on a query and provided references."""
        print(f"[{self.name}] Generating answer for query: \"{query}\" with references.")
        
        if not references:
            return "I'm sorry, I couldn't find enough information to answer that based on the provided references."
        
        # Simple rule-based answer generation based on keywords in query and references
        if "order status" in query.lower():
            if any("shipping" in ref.lower() or "delivery" in ref.lower() for ref in references):
                return "Your order is currently being processed and is expected to ship soon. Please check your order details for tracking information."
            else:
                return "I need more information about your order to check its status. Could you provide an order ID?"
        elif "refund" in query.lower() or "return" in query.lower():
            if any("return policy" in ref.lower() or "refund" in ref.lower() for ref in references):
                return "Our return policy allows returns within 30 days of purchase. Please refer to our full policy for details on how to initiate a return."
            else:
                return "For refund or return inquiries, please visit our help center or contact support directly."
        elif "problem" in query.lower() or "technical" in query.lower():
            if any("technical support" in ref.lower() or "faq" in ref.lower() for ref in references):
                return "For technical issues, please visit our product support pages. Our FAQs cover common problems, or you can open a support ticket."
            else:
                return "Could you please describe your technical problem in more detail?"
        else:
            # Default answer if no specific rule matches, incorporating the first reference
            return f"Based on the provided information (e.g., {references[0].split(':')[0]}): I can assist you with your query about '{query}'."

class RLTrainer:
    """Orchestrates the Reinforcement Learning training process, incorporating reference reuse."""
    def __init__(self, agent, knowledge_base):
        self.agent = agent
        self.knowledge_base = knowledge_base
        self.episode_log = []

    def run_full_episode(self, query):
        """Runs a complete multi-phase episode (browsing + answering)."""
        print("\n--- Running FULL Episode (Browsing + Answering) ---")
        print(f"User Query: \"{query}\"")
        
        # Phase 1: Browsing
        references = self.agent.browse_knowledge_base(query, self.knowledge_base)
        print(f"Retrieved references: {references}")
        
        # Phase 2: Answering
        answer = self.agent.generate_answer(query, references)
        print(f"Generated Answer: \"{answer}\"")
        
        # Simulate reward calculation (placeholder for actual RL reward logic)
        # A simple heuristic: higher reward for more comprehensive answers or specific keywords.
        reward = 1.0 
        if "tracking information" in answer.lower() or "return policy" in answer.lower():
            reward += 0.5
        print(f"Simulated Episode Reward: {reward}")
        
        self.episode_log.append({
            "type": "full",
            "query": query,
            "references": references,
            "answer": answer,
            "reward": reward
        })
        return references, reward

    def run_answering_only_episode(self, query, pre_collected_references):
        """Runs an 'answering-only' episode using pre-collected references to optimize answer generation."""
        print("\n--- Running ANSWERING-ONLY Episode ---")
        print(f"User Query: \"{query}\" (using pre-collected references from a full episode)")
        
        # Phase 2: Answering (skips browsing phase to save computational cost)
        answer = self.agent.generate_answer(query, pre_collected_references)
        print(f"Generated Answer: \"{answer}\"")
        
        # Simulate reward calculation for answering-only phase (can be different from full episode)
        # Focus on answer quality given fixed references.
        reward = 0.8 
        if "tracking information" in answer.lower() or "return policy" in answer.lower():
            reward += 0.7 # Potentially higher reward for focused answer improvement
        print(f"Simulated Episode Reward: {reward}")

        self.episode_log.append({
            "type": "answering_only",
            "query": query,
            "references": pre_collected_references,
            "answer": answer,
            "reward": reward
        })
        return reward

    def train_chatbot(self, queries, num_answering_only_per_full_episode=15):
        """Simulates the RL training process for the chatbot.
        
        Args:
            queries (list): A list of user queries to simulate training episodes.
            num_answering_only_per_full_episode (int): Number of additional answering-only episodes 
                                                      to run after each full episode.
        """
        print("\n--- Starting RL Training with Reference Reuse Strategy ---")
        total_episodes = 0
        total_simulated_reward = 0

        for i, query in enumerate(queries):
            print(f"\n----- Training Round {i+1}/{len(queries)} (Processing Query: \"{query}\") -----")
            
            # Step 1: Run a full episode (browsing + answering)
            collected_references, full_ep_reward = self.run_full_episode(query)
            total_episodes += 1
            total_simulated_reward += full_ep_reward
            
            # Step 2: Generate additional 'answering-only' episodes using collected references
            print(f"\n----- Generating {num_answering_only_per_full_episode} Answering-Only Episodes for current query -----")
            for _ in range(num_answering_only_per_full_episode):
                ao_ep_reward = self.run_answering_only_episode(query, collected_references)
                total_episodes += 1
                total_simulated_reward += ao_ep_reward
        
        print(f"\n--- RL Training Simulation Complete ---")
        print(f"Total Simulated Episodes Run: {total_episodes}")
        print(f"Total Accumulated Simulated Reward: {total_simulated_reward:.2f}")
        if total_episodes > 0:
            print(f"Average Simulated Reward per Episode: {(total_simulated_reward / total_episodes):.2f}")
        print("\nNote: In a real Reinforcement Learning setup, these rewards would be used to update the agent's policy and value functions, leading to improved behavior over time.")

# Main execution block to demonstrate the pattern
if __name__ == "__main__":
    # 1. Define a simulated knowledge base for the chatbot
    knowledge_base = {
        "Order Status and Tracking": "To check your order status, please log into your account and navigate to 'My Orders'. You will find the current shipping status and estimated delivery date, along with tracking information if available.",
        "Return and Refund Policy": "Our return policy allows for returns within 30 days of purchase for a full refund. Items must be in their original condition and packaging. Please initiate returns through our online portal, providing your order number.",
        "Technical Support for Product Issues": "For technical issues with our products, please visit our dedicated product support pages. Our comprehensive FAQs cover common problems, or you can open a support ticket for personalized assistance.",
        "Accepted Payment Methods": "We accept all major credit cards (Visa, MasterCard, American Express, Discover), PayPal, and secure bank transfers. All transactions are processed securely and are encrypted.",
        "General Shipping Information": "Shipping costs vary by destination and selected speed. Standard shipping typically takes 3-5 business days. Expedited shipping options are available at checkout for faster delivery."
    }

    # 2. Initialize the Chatbot Agent
    agent = ChatbotAgent("CustomerSupportBot")

    # 3. Initialize the RL Trainer with the agent and knowledge base
    trainer = RLTrainer(agent, knowledge_base)

    # 4. Define example user queries to simulate the training process
    training_queries = [
        "Where is my order?",
        "How can I return a product?",
        "My device is not working properly.",
        "What are the payment options?"
    ]

    # 5. Run the simulated RL training with reference reuse
    # The 'num_answering_only_per_full_episode' parameter controls the sample efficiency boost.
    trainer.train_chatbot(training_queries, num_answering_only_per_full_episode=5) 
    # In a real scenario, this would be higher (e.g., 15 as per the pattern description) 
    # but is reduced here for cleaner output demonstration.
