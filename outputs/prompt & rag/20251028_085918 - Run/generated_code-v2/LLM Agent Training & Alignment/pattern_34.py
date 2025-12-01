import random

class KnowledgeBase:
    def __init__(self):
        self.faqs = {
            "shipping status": "You can check your shipping status by entering your order number on our 'Track Order' page.",
            "return policy": "Our return policy allows returns within 30 days of purchase with a valid receipt.",
            "payment methods": "We accept major credit cards, PayPal, and bank transfers.",
            "contact support": "You can contact our support team via email at support@example.com or call us at 1-800-123-4567.",
            "product warranty": "All our products come with a one-year manufacturer's warranty.",
            "account login issues": "If you are having trouble logging into your account, please try resetting your password or contact support."
        }

    def get_documents(self):
        return list(self.faqs.items())

class ReferenceRetrievalModule:
    def retrieve_references(self, query, knowledge_base_documents):
        references = []
        query_lower = query.lower()
        for faq_question, faq_answer in knowledge_base_documents:
            if any(keyword in query_lower for keyword in faq_question.split()):
                references.append(f"FAQ: {faq_question} - {faq_answer}")
        return references if references else ["No direct references found, providing general information."]

class AnswerGenerationModule:
    def generate_answer(self, query, references):
        if "No direct references found" in references[0]:
            return "I'm sorry, I couldn't find a direct answer to your question. Can you please rephrase it or provide more details?"

        combined_references = " ".join(references)
        if "shipping status" in query.lower():
            return f"Based on the references: {combined_references}. Please visit our 'Track Order' page to check your shipping status."
        elif "return policy" in query.lower():
            return f"Based on the references: {combined_references}. Our return policy allows returns within 30 days of purchase."
        elif "payment methods" in query.lower():
            return f"Based on the references: {combined_references}. We accept various payment methods, including major credit cards and PayPal."
        else:
            return f"Here is some information from our knowledge base: {combined_references}. If this doesn't fully answer your question, please let me know."

class SimulatedRewardFunction:
    def calculate_reward(self, query, generated_answer, references):
        reward = 0
        query_lower = query.lower()
        answer_lower = generated_answer.lower()
        
        # Reward for using relevant references
        for ref in references:
            if any(keyword in ref.lower() for keyword in query_lower.split() if len(keyword) > 2):
                reward += 1
        
        # Reward for keyword overlap between query and answer
        for keyword in query_lower.split():
            if len(keyword) > 2 and keyword in answer_lower:
                reward += 0.5
        
        # Penalty for generic answers if specific references were available
        if "No direct references found" in references[0] and "couldn't find a direct answer" not in answer_lower:
             reward -= 2 # Should not happen with current logic, but good for robustness
        elif "No direct references found" not in references[0] and "couldn't find a direct answer" in answer_lower:
            reward -= 5

        return max(0, reward)

class RLEpisodeManager:
    def __init__(self, knowledge_base, ref_retrieval_module, answer_gen_module, reward_func):
        self.knowledge_base = knowledge_base
        self.ref_retrieval_module = ref_retrieval_module
        self.answer_gen_module = answer_gen_module
        self.reward_func = reward_func
        self.reference_replay_buffer = [] # Stores (query, references) for answering-only episodes
        self.full_episode_log = [] # Stores (query, references, generated_answer, reward)

    def run_full_episode(self, query):
        print(f"\n--- Running Full Episode for Query: '{query}' ---")
        # Browsing Phase
        knowledge_base_documents = self.knowledge_base.get_documents()
        references = self.ref_retrieval_module.retrieve_references(query, knowledge_base_documents)
        print(f"Retrieved References: {references}")

        # Answering Phase
        generated_answer = self.answer_gen_module.generate_answer(query, references)
        print(f"Generated Answer: {generated_answer}")

        # Simulate Reward
        reward = self.reward_func.calculate_reward(query, generated_answer, references)
        print(f"Episode Reward: {reward}")

        # Store for full episode logging and replay buffer
        self.full_episode_log.append({
            "query": query,
            "references": references,
            "answer": generated_answer,
            "reward": reward
        })
        self.reference_replay_buffer.append({"query": query, "references": references})
        return generated_answer, reward

    def run_answering_only_episodes(self, num_episodes=5):
        if not self.reference_replay_buffer:
            print("No full episodes completed yet, cannot run answering-only episodes.")
            return

        print(f"\n--- Running {num_episodes} Answering-Only Episodes ---")
        for i in range(num_episodes):
            # Sample a (query, references) pair from the buffer
            episode_data = random.choice(self.reference_replay_buffer)
            query = episode_data["query"]
            references = episode_data["references"]
            
            print(f"  [Answering-Only Episode {i+1}] for Query: '{query}' (reusing references)")
            
            # Answering Phase (browsing phase skipped)
            generated_answer = self.answer_gen_module.generate_answer(query, references)
            
            # Simulate Reward
            reward = self.reward_func.calculate_reward(query, generated_answer, references)
            print(f"  Generated Answer: {generated_answer}")
            print(f"  Episode Reward: {reward}")
            # In a real RL setup, this reward would drive optimization of the answer generation model

class ChatbotInterface:
    def __init__(self, rl_episode_manager):
        self.rl_episode_manager = rl_episode_manager

    def start(self):
        print("\nWelcome to the Sample-Efficient Customer Support Chatbot!")
        print("Type 'exit' to quit.")

        while True:
            user_query = input("\nYour question: ")
            if user_query.lower() == 'exit':
                print("Goodbye!")
                break
            
            # Run a full episode
            generated_answer, full_episode_reward = self.rl_episode_manager.run_full_episode(user_query)
            
            # After each full episode, run a few answering-only episodes
            self.rl_episode_manager.run_answering_only_episodes(num_episodes=3)
            
            print(f"\nChatbot says (from full episode): {generated_answer}")

if __name__ == "__main__":
    kb = KnowledgeBase()
    rrm = ReferenceRetrievalModule()
    agm = AnswerGenerationModule()
    srf = SimulatedRewardFunction()

    rl_manager = RLEpisodeManager(kb, rrm, agm, srf)
    chatbot = ChatbotInterface(rl_manager)
    chatbot.start()
