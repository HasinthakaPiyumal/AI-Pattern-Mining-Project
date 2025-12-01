
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# --- 1. Environment Simulation (ECommerceEnv) ---
class ECommerceEnv:
    def __init__(self, num_products=10, num_categories=3):
        self.num_products = num_products
        self.num_categories = num_categories
        self.products = self._generate_products()
        self.user_profiles = self._generate_user_profiles()
        self.current_user_id = None
        self.current_user_session = {}
        self.action_space_n = 5 # 0: popular, 1: new, 2: discounted, 3: filter_cat, 4: show_details

    def _generate_products(self):
        products = []
        for i in range(self.num_products):
            product = {
                "id": i,
                "category": random.randint(0, self.num_categories - 1),
                "price": round(random.uniform(10, 200), 2),
                "popularity": random.uniform(0.1, 1.0),
                "is_new": random.choice([True, False]),
                "is_discounted": random.choice([True, False])
            }
            products.append(product)
        return products

    def _generate_user_profiles(self):
        profiles = {}
        for i in range(5): # 5 distinct user profiles
            profile = {
                "preferred_category": random.randint(0, self.num_categories - 1),
                "price_sensitivity": random.uniform(0.1, 1.0) # Lower is more sensitive
            }
            profiles[i] = profile
        return profiles

    def reset(self):
        self.current_user_id = random.choice(list(self.user_profiles.keys()))
        self.current_user_session = {
            "viewed_products": set(),
            "cart": set(),
            "last_recommended_product": None,
            "last_action": -1 # No action yet
        }
        return self._get_state()

    def _get_state(self):
        user_profile = self.user_profiles[self.current_user_id]
        
        # State features
        state_features = [
            user_profile["preferred_category"] / (self.num_categories - 1), # Normalized category
            user_profile["price_sensitivity"],
            len(self.current_user_session["cart"]) / self.num_products, # Cart size normalized
            self.current_user_session["last_action"] / (self.action_space_n - 1) # Normalized last action
        ]
        
        # Add a one-hot encoding for the last recommended product's category if available
        last_prod_cat_one_hot = [0] * self.num_categories
        if self.current_user_session["last_recommended_product"] is not None:
            prod_id = self.current_user_session["last_recommended_product"]
            category = self.products[prod_id]["category"]
            last_prod_cat_one_hot[category] = 1
        state_features.extend(last_prod_cat_one_hot)

        return np.array(state_features, dtype=np.float32)

    def step(self, action):
        reward = 0
        done = False
        info = {}
        recommended_product_id = None
        user_profile = self.user_profiles[self.current_user_id]

        available_products = self.products

        if self.current_user_session.get("filter_category") is not None:
            available_products = [p for p in available_products if p["category"] == self.current_user_session["filter_category"]]
        
        if action == 0: # Recommend popular
            eligible_products = sorted([p for p in available_products if p["id"] not in self.current_user_session["viewed_products"]], key=lambda x: x["popularity"], reverse=True)
            if eligible_products:
                recommended_product_id = eligible_products[0]["id"]
        elif action == 1: # Recommend new
            eligible_products = sorted([p for p in available_products if p["is_new"] and p["id"] not in self.current_user_session["viewed_products"]], key=lambda x: x["popularity"], reverse=True)
            if eligible_products:
                recommended_product_id = eligible_products[0]["id"]
        elif action == 2: # Recommend discounted
            eligible_products = sorted([p for p in available_products if p["is_discounted"] and p["id"] not in self.current_user_session["viewed_products"]], key=lambda x: x["popularity"], reverse=True)
            if eligible_products:
                recommended_product_id = eligible_products[0]["id"]
        elif action == 3: # Filter by preferred category
            self.current_user_session["filter_category"] = user_profile["preferred_category"]
            reward -= 0.05 # Small penalty for filtering to encourage direct recommendation
        elif action == 4: # Show details (assuming it's for the last recommended product)
            if self.current_user_session["last_recommended_product"] is not None:
                prod = self.products[self.current_user_session["last_recommended_product"]]
                # Simulate user interest in details
                if random.random() < prod["popularity"] * (1 - user_profile["price_sensitivity"]) * 0.5: # Higher pop, lower price sensitivity leads to more interest
                    reward += 0.1 # Small reward for engaging with details
                else:
                    reward -= 0.05 # Mild penalty if details are not interesting
            else:
                reward -= 0.1 # Penalty for showing details without a product
        
        if recommended_product_id is not None:
            self.current_user_session["last_recommended_product"] = recommended_product_id
            self.current_user_session["viewed_products"].add(recommended_product_id)
            
            # Simulate user interaction with recommended product
            product = self.products[recommended_product_id]
            buy_prob = product["popularity"] * (1 - user_profile["price_sensitivity"] * (product["price"] / 200)) # Factor in price sensitivity

            if random.random() < buy_prob:
                self.current_user_session["cart"].add(recommended_product_id)
                reward += 1.0 # Significant reward for adding to cart
                if random.random() < 0.2: # Small chance to purchase immediately
                    done = True # Session ends on purchase
                    reward += 5.0 # Large reward for purchase
            else:
                reward -= 0.1 # Small penalty if recommendation is not acted upon
        
        self.current_user_session["last_action"] = action

        # End session after a few steps or if all products viewed
        if len(self.current_user_session["viewed_products"]) >= self.num_products or len(self.current_user_session["cart"]) > 2:
            done = True

        next_state = self._get_state()
        return next_state, reward, done, info

# --- 2. RL Agent (DQNAgent) ---
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=1e-3, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, replay_buffer_capacity=10000, batch_size=64):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

        self.q_network = QNetwork(state_size, action_size)
        self.target_q_network = QNetwork(state_size, action_size)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_q_network.eval() # Target network is not trained directly

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(replay_buffer_capacity)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.store_transition(state, action, reward, next_state, done)

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        transitions = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = transitions

        states_t = torch.FloatTensor(states)
        actions_t = torch.LongTensor(actions).unsqueeze(-1)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(-1)
        next_states_t = torch.FloatTensor(next_states)
        dones_t = torch.FloatTensor(dones).unsqueeze(-1)

        # Get current Q values from policy network
        q_values = self.q_network(states_t).gather(1, actions_t)

        # Get next Q values from target network
        next_q_values = self.target_q_network(next_states_t).max(1)[0].unsqueeze(-1)
        target_q_values = rewards_t + self.gamma * next_q_values * (1 - dones_t)

        loss = self.loss_fn(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._decay_epsilon()

    def _decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def update_target_network(self):
        self.target_q_network.load_state_dict(self.q_network.state_dict())

# --- 3. Recommender System (Integration & Training) ---
class RecommenderSystem:
    def __init__(self, num_products=10, num_categories=3):
        self.env = ECommerceEnv(num_products, num_categories)
        state_size = len(self.env._get_state())
        action_size = self.env.action_space_n
        self.agent = DQNAgent(state_size, action_size)

    def train(self, num_episodes, target_update_freq=10):
        rewards_per_episode = []
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            done = False
            step_count = 0
            while not done and step_count < 20: # Limit steps per episode
                action = self.agent.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.agent.store_transition(state, action, reward, next_state, done)
                self.agent.learn()
                state = next_state
                total_reward += reward
                step_count += 1
            
            rewards_per_episode.append(total_reward)
            
            if episode % target_update_freq == 0:
                self.agent.update_target_network()

            if episode % 100 == 0:
                print(f"Episode {episode}/{num_episodes}, Total Reward: {total_reward:.2f}, Epsilon: {self.agent.epsilon:.2f}")
        print("Training complete.")
        return rewards_per_episode

    def recommend(self, user_id, current_context=None):
        # For real-time recommendation, we would get the actual user state.
        # For this simulated example, we'll create a dummy state for a given user_id
        # or use the agent's internal state if we were continuing a session.
        
        # In a real system, current_context would be used to build the state representation.
        # For this example, let's just create a mock environment state for inference.
        
        # A simpler way for demonstration: reset env with a specific user (if available)
        # and then get the action.
        
        # For actual inference, the state would be derived from the current user's real-time interaction data.
        # Let's assume we can get a state vector for inference.
        
        # Mocking a real-time state for a user
        # In a real application, you'd fetch the user's actual session data
        # and construct the state vector similarly to _get_state in ECommerceEnv.
        # Here, we'll simulate a user's initial state for recommendation.
        self.env.current_user_id = user_id % len(self.env.user_profiles) # Map to existing profiles
        self.env.current_user_session = {
            "viewed_products": set(),
            "cart": set(),
            "last_recommended_product": None,
            "last_action": -1
        }
        initial_state = self.env._get_state()
        
        action = self.agent.select_action(initial_state) # Using current epsilon for exploration if any
        
        action_map = {
            0: "Recommend Popular Product",
            1: "Recommend New Arrival",
            2: "Recommend Discounted Product",
            3: "Filter by Preferred Category",
            4: "Show Details of Last Recommended Product"
        }
        
        # To actually get a product recommendation, we need to execute the action in the env
        # This is a bit clunky for a single 'recommend' call, as RL actions are sequential.
        # For demonstration, we'll return the chosen action type.
        
        # If the action is a recommendation, we'd need to simulate a step to get the product.
        next_state, reward, done, info = self.env.step(action)
        
        if self.env.current_user_session["last_recommended_product"] is not None and action in [0,1,2]:
            product = self.env.products[self.env.current_user_session["last_recommended_product"]]
            return f"Action: {action_map.get(action, 'Unknown')}, Recommended Product ID: {product['id']}, Category: {product['category']}, Price: {product['price']:.2f}"
        else:
            return f"Action: {action_map.get(action, 'Unknown')}"


if __name__ == "__main__":
    recommender = RecommenderSystem(num_products=20, num_categories=5)
    print("Starting training...")
    rewards = recommender.train(num_episodes=1000, target_update_freq=50)
    
    print("\n--- Inference Example ---")
    for i in range(3):
        user_id_to_recommend = i # Example user ID
        recommendation = recommender.recommend(user_id_to_recommend)
        print(f"For User {user_id_to_recommend}: {recommendation}")

