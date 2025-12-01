import random

class KnowledgeBase:
    def __init__(self, documents):
        self.documents = documents

    def search(self, query, top_k=3):
        # Simulate searching for relevant documents
        # In a real system, this would involve embedding and similarity search
        relevant_docs = []
        for doc_id, doc_content in enumerate(self.documents):
            if query.lower() in doc_content.lower():
                relevant_docs.append((doc_id, doc_content))
        # Sort by some relevance score (here, just return first top_k found)
        return relevant_docs[:top_k]

class CustomerSupportEnv:
    def __init__(self, knowledge_base_docs):
        self.knowledge_base = KnowledgeBase(knowledge_base_docs)
        self.current_query = ""
        self.collected_references = []
        self.browsing_step = 0
        self.max_browsing_steps = 5
        self.state_dim = 10  # Simplified state dimension
        self.action_space = {
            "browsing": ["search_relevant", "search_irrelevant", "finish_browsing"],
            "answering": ["generate_good_answer", "generate_bad_answer"]
        }

    def reset(self, user_query=None):
        if user_query is None:
            # Simulate a new user query
            possible_queries = ["problem with login", "how to reset password", "billing issue", "product features"]
            self.current_query = random.choice(possible_queries)
        else:
            self.current_query = user_query

        self.collected_references = []
        self.browsing_step = 0
        print(f"\nEnvironment reset. New query: {self.current_query}")
        # Initial state representation (simplified)
        return self._get_state(), {}

    def _get_state(self):
        # A very simplified state representation
        # In a real scenario, this would involve query embeddings, history, etc.
        return [len(self.current_query), len(self.collected_references), self.browsing_step, random.random(), random.random(), random.random(), random.random(), random.random(), random.random(), random.random()]

    def step_browsing(self, action):
        self.browsing_step += 1
        reward = 0
        done = False
        info = {"phase": "browsing"}

        if action == "search_relevant":
            found_docs = self.knowledge_base.search(self.current_query, top_k=2)
            if found_docs:
                self.collected_references.extend([doc for _, doc in found_docs])
                reward = 0.5  # Reward for finding relevant info
            else:
                reward = -0.1 # Penalty for searching but finding nothing
        elif action == "search_irrelevant":
            # Simulate searching for something not directly related
            found_docs = self.knowledge_base.search("random_topic_not_related", top_k=1)
            if found_docs:
                # Still add to references, but maybe not as useful
                self.collected_references.extend([doc for _, doc in found_docs])
            reward = -0.2 # Penalty for irrelevant search
        elif action == "finish_browsing":
            done = True # End browsing phase
            if not self.collected_references:
                reward = -0.5 # Penalty for finishing without references

        if self.browsing_step >= self.max_browsing_steps and not done:
            done = True
            # If browsing steps exhausted, force finish
            if not self.collected_references:
                reward = -0.7 # Heavier penalty for failing to collect references within steps

        next_state = self._get_state()
        return next_state, reward, done, info

    def step_answering(self, action, fixed_references=None):
        # Use fixed_references if provided (for reuse episodes), otherwise use self.collected_references
        references_to_use = fixed_references if fixed_references is not None else self.collected_references

        reward = 0
        done = True  # Answering phase is typically a single step
        info = {"phase": "answering"}

        if action == "generate_good_answer":
            if references_to_use and self.current_query.lower() in " ".join(references_to_use).lower():
                reward = 1.0  # High reward for good answer with relevant references
            else:
                reward = 0.2  # Some reward for trying, but not optimal
        elif action == "generate_bad_answer":
            reward = -1.0 # Penalty for a bad answer
        
        # If no references were provided but a good answer was attempted, a small positive reward
        # but much less than if references were used correctly.
        if not references_to_use and action == "generate_good_answer":
            reward = 0.1

        next_state = self._get_state()
        return next_state, reward, done, info