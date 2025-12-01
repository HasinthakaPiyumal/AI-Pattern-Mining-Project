import random

class KnowledgeBase:
    def __init__(self):
        self.documents = {
            "doc_1": "Information about product A: features, pricing, and availability.",
            "doc_2": "Troubleshooting guide for common issues with product B.",
            "doc_3": "Return policy details for all products.",
            "doc_4": "Contact information for technical support and sales.",
            "doc_5": "FAQs about account management and billing."
        }

    def retrieve_documents(self, query, keywords):
        relevant_docs = {}
        query_lower = query.lower()
        for doc_id, content in self.documents.items():
            if any(k.lower() in content.lower() for k in keywords) or any(q_word in content.lower() for q_word in query_lower.split()):
                relevant_docs[doc_id] = content
        return relevant_docs

class BrowsingPhaseEnvironment:
    def __init__(self, user_query, knowledge_base):
        self.user_query = user_query
        self.knowledge_base = knowledge_base
        self.state = self.reset()
        self.available_actions = ["search_general", "search_specific", "read_doc_1", "read_doc_2", "read_doc_3", "read_doc_4", "read_doc_5", "finish_browsing"]

    def reset(self):
        return {"query": self.user_query, "current_references": {}, "browsing_progress": 0}

    def step(self, action):
        reward = -0.1 # Cost per action
        done = False
        info = {}
        next_state = self.state.copy()

        if action == "search_general":
            keywords = self.user_query.split()
            found_docs = self.knowledge_base.retrieve_documents(self.user_query, keywords)
            next_state["current_references"].update(found_docs)
            reward += 0.5 if found_docs else 0
            next_state["browsing_progress"] += 1
            info["message"] = f"Performed general search, found {len(found_docs)} documents."
        elif action == "search_specific":
            specific_keywords = [word for word in self.user_query.split() if len(word) > 3]
            found_docs = self.knowledge_base.retrieve_documents(self.user_query, specific_keywords)
            next_state["current_references"].update(found_docs)
            reward += 0.8 if found_docs else 0
            next_state["browsing_progress"] += 1
            info["message"] = f"Performed specific search, found {len(found_docs)} documents."
        elif action.startswith("read_doc_"):
            doc_id = action.replace("read_", "")
            if doc_id in self.knowledge_base.documents:
                next_state["current_references"][doc_id] = self.knowledge_base.documents[doc_id]
                reward += 0.3
                info["message"] = f"Read {doc_id}."
            else:
                reward -= 0.5 # Penalty for reading non-existent doc
                info["message"] = f"Attempted to read non-existent {doc_id}."
            next_state["browsing_progress"] += 1
        elif action == "finish_browsing":
            reward += 1.0 if len(next_state["current_references"]) > 0 else -1.0
            done = True
            info["message"] = "Finished browsing."

        self.state = next_state
        return next_state, reward, done, info

class AnswerGenerationPhaseEnvironment:
    def __init__(self, user_query, collected_references):
        self.user_query = user_query
        self.collected_references = collected_references
        self.state = self.reset()
        self.available_actions = [f"add_sentence_from_{doc_id}" for doc_id in collected_references] + ["refine_answer", "finish_answer"]

    def reset(self):
        return {"query": self.user_query, "references": self.collected_references, "current_answer": "", "answer_progress": 0}

    def step(self, action):
        reward = -0.1 # Cost per action
        done = False
        info = {}
        next_state = self.state.copy()

        if action.startswith("add_sentence_from_"):
            doc_id = action.replace("add_sentence_from_", "")
            if doc_id in self.collected_references:
                # Simulate adding a relevant sentence from the document
                sentence = f"Based on {doc_id}: {self.collected_references[doc_id][:50]}... " # Take first 50 chars
                next_state["current_answer"] += sentence
                reward += 0.5
                info["message"] = f"Added content from {doc_id}."
            else:
                reward -= 0.5 # Penalty for using non-existent reference
                info["message"] = f"Attempted to use non-existent reference {doc_id}."
            next_state["answer_progress"] += 1
        elif action == "refine_answer":
            if len(next_state["current_answer"]) > 0:
                next_state["current_answer"] += "(refined) "
                reward += 0.2
                info["message"] = "Refined the answer."
            else:
                reward -= 0.2
                info["message"] = "Cannot refine empty answer."
            next_state["answer_progress"] += 1
        elif action == "finish_answer":
            reward += 2.0 if "product A" in next_state["current_answer"] and "pricing" in next_state["current_answer"] else -1.0 # Example answer quality check
            done = True
            info["message"] = "Finished answer generation."

        self.state = next_state
        return next_state, reward, done, info

class RLAgent:
    def __init__(self, config=None):
        self.config = config if config else {}
        self.browsing_policy = {}
        self.answer_policy = {}

    def select_browsing_action(self, state, available_actions):
        return random.choice(available_actions)

    def select_answer_action(self, state, available_actions):
        return random.choice(available_actions)

    def learn_from_experience(self, experience_batch):
        # In a real RL setup, this would update Q-tables or neural network weights
        # For this conceptual example, we just acknowledge the learning step.
        # print(f"Agent learning from a batch of {len(experience_batch)} experiences.")
        pass

# Main Training Loop (conceptual)
if __name__ == "__main__":
    knowledge_base = KnowledgeBase()
    agent = RLAgent()

    N_FULL_EPISODES = 5
    NUM_ANSWERING_ONLY_EPISODES = 3 # Additional answering episodes per full episode

    all_experiences = []

    print("Starting RL training with Reference Reuse strategy...")

    for i in range(N_FULL_EPISODES):
        user_query = f"Tell me about product A pricing and features (Episode {i+1})"
        print(f"\n--- Full Multi-Phase Episode {i+1} for query: {user_query} ---")

        # Phase 1: Browsing
        browsing_env = BrowsingPhaseEnvironment(user_query, knowledge_base)
        browsing_state = browsing_env.reset()
        browsing_done = False
        episode_browsing_experiences = []
        collected_references_for_answer = {}
        browsing_steps = 0

        while not browsing_done and browsing_steps < 5: # Limit browsing steps
            action = agent.select_browsing_action(browsing_state, browsing_env.available_actions)
            next_browsing_state, reward, browsing_done, info = browsing_env.step(action)
            episode_browsing_experiences.append((browsing_state, action, reward, next_browsing_state, browsing_done))
            browsing_state = next_browsing_state
            collected_references_for_answer.update(browsing_state["current_references"])
            browsing_steps += 1
            # print(f"Browsing step: {action}, Reward: {reward:.2f}, Info: {info['message']}")

        all_experiences.extend(episode_browsing_experiences)
        print(f"Browsing phase ended. Collected {len(collected_references_for_answer)} references.")

        # Phase 2: Answer Generation (after full browsing)
        answer_env = AnswerGenerationPhaseEnvironment(user_query, collected_references_for_answer)
        answer_state = answer_env.reset()
        answer_done = False
        episode_answer_experiences = []
        answer_steps = 0

        while not answer_done and answer_steps < 5: # Limit answer generation steps
            action = agent.select_answer_action(answer_state, answer_env.available_actions)
            next_answer_state, reward, answer_done, info = answer_env.step(action)
            episode_answer_experiences.append((answer_state, action, reward, next_answer_state, answer_done))
            answer_state = next_answer_state
            answer_steps += 1
            # print(f"Answer gen step: {action}, Reward: {reward:.2f}, Info: {info['message']}")
        
        all_experiences.extend(episode_answer_experiences)
        print(f"Answer generation phase ended. Final Answer: {answer_state['current_answer'][:100]}...")

        # Reference Reuse: Generate additional 'answering-only' episodes
        print(f"--- Generating {NUM_ANSWERING_ONLY_EPISODES} additional answering-only episodes ---")
        for j in range(NUM_ANSWERING_ONLY_EPISODES):
            print(f"  Answering-only episode {j+1} using previous references.")
            answer_only_env = AnswerGenerationPhaseEnvironment(user_query, collected_references_for_answer)
            answer_only_state = answer_only_env.reset()
            answer_only_done = False
            episode_answer_only_experiences = []
            answer_only_steps = 0

            while not answer_only_done and answer_only_steps < 5: # Limit steps
                action = agent.select_answer_action(answer_only_state, answer_only_env.available_actions)
                next_answer_only_state, reward, answer_only_done, info = answer_only_env.step(action)
                episode_answer_only_experiences.append((answer_only_state, action, reward, next_answer_only_state, answer_only_done))
                answer_only_state = next_answer_only_state
                answer_only_steps += 1
                # print(f"    Answer-only step: {action}, Reward: {reward:.2f}, Info: {info['message']}")
            all_experiences.extend(episode_answer_only_experiences)
            print(f"  Answering-only episode {j+1} ended. Final Answer: {answer_only_state['current_answer'][:100]}...")

    print(f"\nTotal experiences collected: {len(all_experiences)}")
    agent.learn_from_experience(all_experiences)
    print("Training complete. Agent has learned from all experiences, including reference reuse episodes.")