"""
customer_support_env.py: Simulates the multi-stage customer support environment.
"""

import random

class CustomerSupportEnv:
    def __init__(self, knowledge_base, customer_data):
        self.knowledge_base = knowledge_base
        self.customer_data = customer_data
        self.current_query = None
        self.current_references = []
        self.episode_step = 0
        self.max_browsing_steps = 3

    def reset(self, query=None):
        # For simplicity, queries are fixed or randomly chosen from customer_data keys
        if query is None:
            self.current_query = random.choice(list(self.customer_data.keys()))
        else:
            self.current_query = query
        self.current_references = []
        self.episode_step = 0
        print(f"Environment Reset. New query: '{self.current_query}'")
        return {"query": self.current_query, "references": []}

    def _browse_knowledge_base(self, query_keywords):
        # Simulate finding relevant articles based on keywords in the query
        found_references = []
        query_keywords_set = set(query_keywords.lower().split())
        for topic, content in self.knowledge_base.items():
            if any(keyword in topic.lower() for keyword in query_keywords_set) or \
               any(keyword in content.lower() for keyword in query_keywords_set):
                found_references.append(f"Ref: {topic} - {content[:50]}...") # Snippet of content
        return list(set(found_references)) # Return unique references

    def _evaluate_answer(self, original_query, generated_answer, references_used):
        # Simplified reward function:
        # - Higher reward for including keywords from the query
        # - Higher reward for using provided references
        # - Penalty for very short answers
        reward = 0
        done = False

        original_query_keywords = set(original_query.lower().split())
        generated_answer_keywords = set(generated_answer.lower().split())

        # Reward for addressing query keywords
        common_keywords = original_query_keywords.intersection(generated_answer_keywords)
        reward += len(common_keywords) * 5

        # Reward for using relevant references (simple check if reference text is in answer)
        for ref in references_used:
            if ref.split(' - ')[1].lower()[:20] in generated_answer.lower(): # Check for a snippet match
                reward += 10

        # Penalty for very short or irrelevant answers
        if len(generated_answer) < 30:
            reward -= 20
        if "i don't know" in generated_answer.lower():
            reward -= 15

        # Simulate reaching a satisfactory resolution
        if reward > 30 and len(generated_answer) > 50:
            done = True
            reward += 50 # Bonus for resolving

        return reward, done

    def step_browsing(self, agent_action_keywords):
        self.episode_step += 1
        if self.episode_step > self.max_browsing_steps:
            print("Max browsing steps reached.")
            return self.current_references, {"query": self.current_query, "references": self.current_references}, True # Done with browsing

        print(f"Browsing step {self.episode_step} with keywords: {agent_action_keywords}")
        new_references = self._browse_knowledge_base(agent_action_keywords)
        self.current_references.extend(new_references)
        self.current_references = list(set(self.current_references)) # Ensure unique references
        
        # State after browsing includes the query and gathered references
        state = {"query": self.current_query, "references": self.current_references}
        return self.current_references, state, False # Not done with browsing yet

    def step_answering(self, references, agent_proposed_answer):
        # This is the final step where the agent provides an answer
        # The `references` here are what the agent *decided* to use, or all it gathered
        print(f"Answering phase. Proposed answer: '{agent_proposed_answer[:100]}...' ")
        reward, done = self._evaluate_answer(self.current_query, agent_proposed_answer, references)
        # For simplicity, after answering, the episode is always done.
        return reward, done, {"query": self.current_query, "references": references, "answer": agent_proposed_answer}
